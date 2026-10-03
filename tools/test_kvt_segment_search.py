"""CAMP contracts for search stopping and execution-failure propagation."""
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import kvt_segment_search as search


class SearchContracts(unittest.TestCase):
    def run_search(self, ratios, failure=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, out = root / 'work', root / 'out'
            work.mkdir()
            out.mkdir()
            for filename, member, content in (
                ('tum_26127_context.tar', './oracle_sources.json', {'checkpoint': {}}),
                ('tum_26127_oracle_maps_inputs.tar', 'office/manifest.json', {'inputs': [None] * 2585})):
                data = json.dumps(content).encode()
                with tarfile.open(out / filename, 'w') as archive:
                    info = tarfile.TarInfo(member)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            calls = []

            def execute(command, check):
                if '--run' in command:
                    result = Path(command[command.index('--run') + 1]).parent
                    ratio = 1.
                else:
                    if failure:
                        raise subprocess.CalledProcessError(1, command)
                    calls.append(command[command.index('--name') + 1])
                    result = work / 'runs' / calls[-1]
                    result.mkdir()
                    ratio = ratios[len(calls) - 1]
                    search.write_json(result / 'config.json', {'boundaries': [0, 2585]})
                metrics = dict(search.NATIVE, frames=2585, tracking_seconds=1.)
                for key in search.KEYS:
                    metrics[key] *= ratio[key] if isinstance(ratio, dict) else ratio
                search.write_json(result / 'metrics.json', metrics)
                search.write_json(result / 'summary.json', dict(metrics=metrics, timing={}, events=[]))
                search.write_json(result / 'environment.json', {})
                np.savez(result / 'evaluation.npz', **{key: np.array([0]) for key in
                    ('rgb_indices', 'gt_indices', 'timestamps', 'rpe_pair_start_indices', 'reference')})

            with patch.object(search, 'SOURCES', {}), patch.object(search.subprocess, 'run', execute), \
                    patch.object(search, 'archive_directory'), \
                    patch('huggingface_hub.snapshot_download', return_value=directory):
                args = SimpleNamespace(work=work, out=out, tag='test', max_trials=2)
                if failure:
                    with self.assertRaises(subprocess.CalledProcessError):
                        search.main(args)
                    self.assertFalse((work / 'JOB_OK').exists())
                    return
                search.main(args)
            return json.loads((work / 'runs/segment_search/comparison.json').read_text()), calls

    def test_stops_at_first_accuracy_pass(self):
        result, calls = self.run_search([
            dict(ate_m=1.4, rpe_translation_m=1.6, translation_p99_m=1.4), 1.4])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['stop_reason'], 'accuracy_target')
        self.assertEqual(result['passing']['trial'], 2)

    def test_trial_limit_stops_without_claiming_success(self):
        result, calls = self.run_search([1.6, 1.7])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['stop_reason'], 'trial_limit')
        self.assertIsNone(result['passing'])
        self.assertEqual(result['best']['trial'], 1)

    def test_execution_failure_is_not_an_accuracy_trial(self):
        self.run_search([], failure=True)


if __name__ == '__main__':
    unittest.main()
