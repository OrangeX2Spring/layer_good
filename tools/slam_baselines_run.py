"""CAMP-only: VGGT-SLAM 1.0/2.0 and MASt3R-SLAM on TUM with their own published settings.

Stage 1 of tools/SLAM_BASELINES.md: reproduce each paper's TUM ATE with its own metric,
`evo_ape tum <gt> <est> -as` on the trajectory the program saves (keyframes only, after
its final optimisation). Runs inside the borrowed kvt container after
`tools/slam_baselines_env.sh install`. Per run it keeps the saved trajectory, the
timestamped stdout, GPU memory samples and the evo output; the ATE is not recomputed
by us here (stage 2 rescoring happens offline on these files).

Timing: each program's loop time, excluding model loading. MASt3R-SLAM prints its own
running FPS over processed frames (eval configs read every 2nd frame); VGGT-SLAM is
timed between its 'Found N images' line (printed after model loading) and its first
line after the frame loop. Memory: peak `nvidia-smi memory.used` of the allocated GPU,
which includes the CUDA context and allocator cache, so it is not torch's
max_memory_allocated.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

SRC = Path('/tmp/slam_src')
CKPT = Path('/mnt/projects/gr/3DRecon/slam_baselines/ckpt')
SEQUENCES = {'pilot': ('freiburg1_xyz',),
             'full': ('freiburg1_xyz', 'freiburg1_rpy', 'freiburg1_desk', 'freiburg1_desk2',
                      'freiburg1_room', 'freiburg3_long_office_household')}
# evals/eval_tum.sh of each VGGT-SLAM version, submap size 32 as in both papers' tables.
VGGT_V1 = ['--max_loops', '1', '--min_disparity', '50', '--conf_threshold', '25',
           '--submap_size', '32', '--log_results', '--skip_dense_log']
VGGT_V2 = VGGT_V1 + ['--lc_thres', '0.95']
# name: (checkout, vggt package checkout or None for MASt3R-SLAM, arguments, end marker)
ARMS = {'mast3r_calib': ('MASt3R-SLAM', None, ['--config', 'config/eval_calib.yaml'], None),
        'mast3r_nocalib': ('MASt3R-SLAM', None, ['--config', 'config/eval_no_calib.yaml'], None),
        'vggt_v1_sl4': ('VGGT-SLAM-v1', 'vggt', VGGT_V1, 'Total number of submaps in map'),
        'vggt_v1_sim3': ('VGGT-SLAM-v1', 'vggt', VGGT_V1 + ['--use_sim3'], 'Total number of submaps in map'),
        'vggt_v2': ('VGGT-SLAM-v2', 'vggt_spark', VGGT_V2, 'Total time:')}
# Published TUM ATE RMSE (m): MASt3R-SLAM arXiv 2412.12392 Table 1; VGGT-SLAM 1.0 arXiv
# 2505.12549v2 Table 2 (w=32); VGGT-SLAM 2.0 arXiv 2601.19887 Table I. Office: none.
PUBLISHED = {
    'mast3r_calib': dict(freiburg1_xyz=.009, freiburg1_rpy=.027, freiburg1_desk=.016, freiburg1_desk2=.024, freiburg1_room=.061),
    'mast3r_nocalib': dict(freiburg1_xyz=.020, freiburg1_rpy=.041, freiburg1_desk=.035, freiburg1_desk2=.055, freiburg1_room=.118),
    'vggt_v1_sl4': dict(freiburg1_xyz=.014, freiburg1_rpy=.030, freiburg1_desk=.025, freiburg1_desk2=.040, freiburg1_room=.102),
    'vggt_v1_sim3': dict(freiburg1_xyz=.016, freiburg1_rpy=.041, freiburg1_desk=.040, freiburg1_desk2=.055, freiburg1_room=.088),
    'vggt_v2': dict(freiburg1_xyz=.014, freiburg1_rpy=.026, freiburg1_desk=.025, freiburg1_desk2=.029, freiburg1_room=.063)}
MAST3R_SUBSAMPLE = 2  # dataset.subsample in both eval configs


def sample_memory(samples, stop):
    while not stop.is_set():
        used = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True).stdout.split()
        samples.append((time.monotonic(), int(used[0])))
        stop.wait(.2)


def run_arm(arm, sequence, data, out):
    checkout, vggt, extra, end_marker = ARMS[arm]
    scene = data / 'tum' / f'rgbd_dataset_{sequence}'
    cwd = SRC / checkout
    out.mkdir(parents=True)
    if vggt is None:
        command = ['python', 'main.py', '--dataset', f'{scene}/', '--no-viz',
                   '--save-as', f'{out.parent.name}/{arm}', *extra]
        env_path = f'{cwd}/thirdparty/mast3r:{cwd}/thirdparty/in3d'  # upstream installs both editable
    else:
        command = ['python', 'main.py', '--image_folder', str(scene / 'rgb'),
                   '--log_path', str(out / 'traj.txt'), *extra]
        env_path = f'{cwd}:{SRC / vggt}:{SRC / "salad"}'
    env = dict(os.environ, PYTHONPATH=env_path)
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(samples, stop))
    sampler.start()
    started = time.monotonic()
    lines = []
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            lines.append((time.monotonic(), line.rstrip('\n')))
    stop.set()
    sampler.join()
    (out / 'stdout.log').write_text(''.join(f'{t - started:.3f} {s}\n' for t, s in lines))
    (out / 'memory_mib.csv').write_text(''.join(f'{t - started:.3f},{m}\n' for t, m in samples))
    assert process.returncode == 0, (arm, sequence, process.returncode, lines[-5:])
    if vggt is None:
        saved = cwd / 'logs' / out.parent.name / arm / f'rgbd_dataset_{sequence}.txt'
        (out / 'traj.txt').write_bytes(saved.read_bytes())
        (out / 'reconstruction.ply').write_bytes(saved.with_suffix('.ply').read_bytes())
    evo = subprocess.run(['evo_ape', 'tum', str(scene / 'groundtruth.txt'), str(out / 'traj.txt'), '-as'],
                         capture_output=True, text=True, check=True).stdout
    (out / 'evo_ape.txt').write_text(evo)
    frames = len(list((scene / 'rgb').glob('*.png')))
    if vggt is None:
        rates = [float(m.group(1)) for _, s in lines if (m := re.match(r'FPS: (\S+)', s))]
        loop_fps = MAST3R_SUBSAMPLE * rates[-1]  # input frames per second
    else:
        start = next(t for t, s in lines if re.match(r'Found \d+ images', s))
        end = next(t for t, s in lines if s.startswith(end_marker))
        loop_fps = frames / (end - start)
    row = dict(sequence=sequence, arm=arm, command=command,
               ate_m=float(re.search(r'rmse\s+([0-9.]+)', evo).group(1)),
               published_ate_m=PUBLISHED[arm].get(sequence),
               keyframes=len((out / 'traj.txt').read_text().splitlines()), frames=frames,
               loop_fps_input_frames=loop_fps, wall_seconds=lines[-1][0] - started,
               peak_memory_mib=max(m for _, m in samples))
    if row['published_ate_m'] is not None:
        row['ratio_to_published'] = row['ate_m'] / row['published_ate_m']
    print('SLAM BASELINE', sequence, arm, 'ATE', round(row['ate_m'], 4), 'published', row['published_ate_m'],
          'keyframes', row['keyframes'], '/', frames, 'fps', round(loop_fps, 2),
          'peak MiB', row['peak_memory_mib'], flush=True)
    return row


def main(args):
    (SRC / 'MASt3R-SLAM' / 'checkpoints').symlink_to(CKPT)
    report = dict(mode=args.mode, arms=list(ARMS), sequences=SEQUENCES[args.mode], runs=[], complete=False,
                  protocol='own published settings; evo_ape tum -as on the saved keyframe trajectory; one run each')
    for sequence in SEQUENCES[args.mode]:
        for arm in ARMS:
            report['runs'].append(run_arm(arm, sequence, args.data, args.work / 'runs' / sequence / arm))
            (args.work / 'results.json').write_text(json.dumps(report, indent=1))
    report['complete'] = True
    (args.work / 'results.json').write_text(json.dumps(report, indent=1))
    print('SLAM BASELINES COMPLETE', args.mode, len(report['runs']), 'runs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=tuple(SEQUENCES), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    main(parser.parse_args())
