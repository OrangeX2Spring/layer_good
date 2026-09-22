"""Run the ARCTIC and TUM correspondence gates in one CAMP container session."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--tum-out', type=Path, required=True)
    parser.add_argument('--arctic-out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent

    with (args.work / 'context' / 'packages.json').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'],
                       stdout=stream, check=True)
    subprocess.run([sys.executable, str(tools / 'test_kvt_tum.py')], check=True)
    subprocess.run([sys.executable, str(tools / 'test_kvt_correspondence.py')], check=True)
    arctic_run = subprocess.run([sys.executable, str(tools / 'kvt_correspondence_pilot.py'),
        '--tag', args.tag, '--scene', 'box_grab_01', '--stage', 'gate',
        '--out', str(args.work / 'arctic_review')])

    tum_run = subprocess.run([sys.executable, str(tools / 'kvt_correspondence_tum.py'),
        '--work', str(args.work), '--out', str(args.tum_out), '--tag', args.tag,
        '--scene', 'freiburg3_long_office_household', '--stage', 'gate'])
    (args.work / 'both_gate_processes.json').write_text(json.dumps(dict(
        arctic_returncode=arctic_run.returncode, tum_returncode=tum_run.returncode), indent=2) + '\n')
    arctic_run.check_returncode()
    tum_run.check_returncode()
    arctic = json.loads((args.work / 'arctic_review' / 'gates.json').read_text())
    assert arctic['evictions_after_budget'] and arctic['full_retention_exact']
    tum = json.loads((args.work / 'GATES_OK_freiburg3_long_office_household.json').read_text())
    assert tum['bounded_retention'] and tum['full_retention_exact']
    (args.work / 'both_gates.json').write_text(json.dumps(dict(arctic=arctic,
        tum=tum, arctic_out=str(args.arctic_out), tum_out=str(args.tum_out)), indent=2) + '\n')
    print('BOTH CORRESPONDENCE GATES COMPLETED; REVIEW RESULTS BEFORE FULL RUNS', flush=True)


if __name__ == '__main__':
    main()
