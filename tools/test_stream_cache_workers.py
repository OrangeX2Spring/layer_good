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


if __name__ == '__main__':
    unittest.main()
