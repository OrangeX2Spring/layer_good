"""Remote CPU tests for isolated-worker dispatch and failure propagation."""

from argparse import Namespace
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import stream_cache_sweep as sweep


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        inputs = root / 'inputs'
        inputs.mkdir()
        (inputs / 'rgb.png').write_bytes(b'input-hash-fixture')
        frame = dict(rgb='rgb.png', rgb_sha256=sweep.digest(inputs / 'rgb.png'),
                     model_hw=[28, 28])
        (inputs / 'manifest.json').write_text(json.dumps(dict(frames=[frame, frame])))
        specification = root / 'sweep.json'
        specification.write_text(json.dumps(dict(isolate_conditions=True,
            geometry_export='final', conditions=[dict(name='recent8', policy={}),
                                               dict(name='random', policy={'eviction': 'random'})])))
        checkpoint = root / 'checkpoint'
        checkpoint.write_bytes(b'checkpoint-fixture')
        provenance = root / 'git'
        provenance.mkdir()
        for name in ('superproject', 'model'):
            (provenance / f'{name}_commit.txt').write_text('fixture')
            (provenance / f'{name}.patch').write_text('')
        self.args = Namespace(host='streamvggt', checkpoint=checkpoint, model_config=None,
            inputs=inputs, sweep=specification, out=root / 'run',
            git_provenance=provenance, worker=None)
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.object(sweep.sys, 'platform', 'linux'))
        stack.enter_context(patch.object(sweep.sys, 'argv',
                                        ['stream_cache_sweep.py', 'run', '--host', 'streamvggt']))
        stack.enter_context(patch.object(sweep.torch.cuda, 'is_available', return_value=True))
        stack.enter_context(patch.object(sweep.torch.cuda, 'get_device_name', return_value='fixture'))
        self.load = stack.enter_context(patch.object(sweep, 'load_model'))
        self.dispatch = stack.enter_context(patch.object(sweep.subprocess, 'run'))

    def test_fidelity_and_conditions_use_distinct_workers_without_parent_model(self):
        sweep.run(self.args)
        self.load.assert_not_called()
        self.assertEqual([call.args[0][-1] for call in self.dispatch.call_args_list],
                         ['__fidelity__', 'recent8', 'random'])
        self.assertTrue(all(call.kwargs['check'] for call in self.dispatch.call_args_list))
        self.assertTrue((self.args.out / 'inputs/rgb.png').exists())

    def test_failure_stops_later_conditions_and_final_evaluation(self):
        self.dispatch.side_effect = [None, subprocess.CalledProcessError(1, 'recent8')]
        with self.assertRaises(subprocess.CalledProcessError):
            sweep.run(self.args)
        self.assertEqual(self.dispatch.call_count, 2)
        self.assertFalse((self.args.out / 'camera_metrics.json').exists())
        self.assertTrue((self.args.out / 'provenance.json').exists())

    def test_worker_does_not_reinitialize_shared_output_or_repeat_fidelity(self):
        self.args.worker = 'recent8'
        self.args.out.mkdir()
        (self.args.out / 'sentinel').write_text('keep')
        with patch.object(sweep, 'run_condition') as condition, \
                patch.object(sweep, 'fidelity') as fidelity:
            sweep.run(self.args)
        condition.assert_called_once()
        fidelity.assert_not_called()
        self.dispatch.assert_not_called()
        self.assertEqual((self.args.out / 'sentinel').read_text(), 'keep')

    def test_precision_gate_records_payload_and_rejects_inactive_quantization(self):
        self.args.sweep.write_text(json.dumps(dict(isolate_conditions=True,
            geometry_export='final', precision_gate=True,
            conditions=[dict(name='recent8_int8', policy=dict(quant_bits=8))])))

        def complete_worker(command, check):
            if command[-1] == '__fidelity__':
                return
            result = self.args.out / command[-1]
            result.mkdir()
            rows = [dict(retained_frames=list(range(i + 1)), patch_indices=[0, 1, 2, 3],
                quant_bits=8, quantized_current_changed=True, refresh=False,
                aggregator_tokens=5 * (i + 1), aggregator_bytes=160 * (i + 1),
                analytical_packed_aggregator_bytes=60 * (i + 1),
                aggregator_dtype='torch.float32', camera_bytes=40, relative_pose_bytes=0,
                reference_bytes=0, position_bytes=20, token_index_bytes=40,
                feature_bytes=16, peak_allocated=1000) for i in range(2)]
            (result / 'events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))

        self.dispatch.side_effect = complete_worker
        sweep.run(self.args)
        report = json.loads((self.args.out / 'precision_gate.json').read_text())['recent8_int8']
        self.assertEqual(report['max_analytical_total_cache_bytes'], 236)
        self.assertEqual(report['packed_to_control_aggregator_byte_ratio_max'], 120 / 256)
        self.assertEqual([call.args[0][-1] for call in self.dispatch.call_args_list],
                         ['__fidelity__', 'recent8_int8'])

        original_complete = complete_worker

        def inactive_worker(command, check):
            original_complete(command, check)
            if command[-1] != '__fidelity__':
                path = self.args.out / command[-1] / 'events.jsonl'
                path.write_text(path.read_text().replace('"quantized_current_changed": true',
                                                        '"quantized_current_changed": false'))

        self.args.out = self.args.out.parent / 'inactive'
        self.dispatch.side_effect = inactive_worker
        with self.assertRaises(AssertionError):
            sweep.run(self.args)
        self.assertFalse((self.args.out / 'precision_gate.json').exists())

    def test_full_scene_checks_methods_without_control_workers(self):
        specification = dict(isolate_conditions=True, geometry_export='final',
            correspondence_full='scene', conditions=[dict(name='correspondence_p50',
                policy=dict(patch_policy='correspondence', patch_fraction=.5))])
        self.args.sweep.write_text(json.dumps(specification))

        def complete_worker(command, check):
            worker = command[-1]
            if worker == '__fidelity__':
                (self.args.out / 'fidelity.json').write_text('{}')
                return
            self.assertEqual(worker, 'correspondence_p50')
            result = self.args.out / worker
            result.mkdir()
            rows = [dict(retained_frames=[0], patch_indices=[0, 1, 2, 3],
                         evicted=[], matched_kept=0, refresh=False),
                    dict(retained_frames=[0, 9], patch_indices=[0, 2],
                         evicted=[1], matched_kept=1, refresh=False)]
            (result / 'events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))

        self.dispatch.side_effect = complete_worker
        sweep.run(self.args)
        report = json.loads((self.args.out / 'correspondence_full.json').read_text())
        self.assertTrue(report['matches_active'])
        self.assertTrue(report['fifo_active'])
        self.assertNotIn('cache_smaller', report)
        self.assertEqual([call.args[0][-1] for call in self.dispatch.call_args_list],
                         ['__fidelity__', 'correspondence_p50'])


if __name__ == '__main__':
    unittest.main()
