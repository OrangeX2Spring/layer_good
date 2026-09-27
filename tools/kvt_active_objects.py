"""Occlusion-setting active-keyframe runs in object mode: ARCTIC S01 and one
YCBInEOAT sequence. CAMP only, after kvt_active_keyframes.py --arms occlusion,
via `bash tools/kvt_tum.sbatch active-keyframes occlusion`.

Per sequence: native object-mode KV-Tracker with its own angular admission; then
every arm replays native's keyframe schedule, so arms differ only in query access:
all (fidelity), drop<j> for every native keyframe, and anchor plus newest versus
anchor plus most-visible keyframes at two and half. Visibility is a keyframe's own
SAM mask fraction at model resolution. Each arm is a fresh process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np

from kvt_active_keyframes import SUBSETS, audit
from kvt_tum_run import write_json

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = ROOT / 'kv_tracker'
ARCTIC = ('box_grab_01', 'ketchup_grab_01', 'espressomachine_grab_01')
YCB_SCENE = 'mustard_easy_00_02'
YCB_ARCHIVE = Path('/mnt/projects/gr/3DRecon/ycbineoat') / f'{YCB_SCENE}.tar.gz'
YCB_STAGE = Path('/tmp/data/ycbineoat')
RESIZE = 518  # ARCTIC protocol resolution, main.py's default.


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ycb_poses(scene_dir):
    """annotated_poses are object-in-camera; KV-Tracker estimates camera-in-object.

    Pose and RGB names differ (job 25949); like point_to_pose's YcbineoatReader,
    pair them by sorted order."""
    rgb = sorted((scene_dir / 'rgb').glob('*.png'))
    poses = sorted((scene_dir / 'annotated_poses').glob('*.txt'))
    assert len(rgb) == len(poses), (len(rgb), len(poses))
    ob_in_cam = np.stack([np.loadtxt(p).reshape(4, 4) for p in poses])
    assert np.isfinite(ob_in_cam).all()
    return ob_in_cam


def run(config_path):
    import cv2
    import torch
    import main as tracker
    import eval as kvt_eval
    from kv_tracker.active_keyframes import ActiveKeyframes
    from kv_tracker.eval_tools.evo_utils import align_pair
    from kv_tracker.sam_interface import SAMInterface
    from kvt_arctic_run import ArcticFrames, DATASET_DIR

    config = json.loads(config_path.read_text())
    result = config_path.parent
    os.chdir(CHECKOUT)  # SAM checkpoint and ARCTIC GT paths are relative to it.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    mask_dir = result / 'sam_masks'
    mask_dir.mkdir()

    class Arctic(ArcticFrames):
        def __iter__(self):
            for frame in super().__iter__():
                frame['visible_fraction'] = float(frame['resized_mask_np'].mean())
                yield frame

    class Ycb(SAMInterface):
        """YCBInEOAT RGB; SAM 2 initialised from the dataset's first gt_mask."""
        def __init__(self, device, **cfg):
            super().__init__(device, **cfg)
            scene_dir = Path(cfg['scene_dir'])
            self.rgb_paths = sorted((scene_dir / 'rgb').glob('*.png'))
            self.length = len(self.rgb_paths)
            first = self.get_rgb_frame(0)
            self.height, self.width = first.shape[:2]
            mask = cv2.imread(str(scene_dir / 'gt_mask' / self.rgb_paths[0].name), cv2.IMREAD_GRAYSCALE)
            assert mask is not None and mask.shape == first.shape[:2] and mask.any()
            self.init_models()
            self.init_SAM_w_mask(mask > 0)

        def get_rgb_frame(self, idx=0):
            bgr = cv2.imread(str(self.rgb_paths[idx]))
            assert bgr is not None, self.rgb_paths[idx]
            return bgr[:, :, ::-1]

        def get_segmentation(self, frame):
            mask = super().get_segmentation(frame)
            mask_np = mask.cpu().numpy()
            assert mask_np.any(), f'Empty object mask at frame {self.index}'
            cv2.imwrite(str(mask_dir / f'{self.index:05d}.png'), mask_np.astype(np.uint8) * 255)
            return mask

        def __iter__(self):
            for index in range(self.length):
                self.index = index
                frame = self.get_frame(index)
                frame['idx'] = index
                frame['visible_fraction'] = float(frame['resized_mask_np'].mean())
                yield frame

    if config['setting'] == 'arctic':
        source = Arctic('cuda:0', mask_dir, scene_dir=DATASET_DIR / config['scene'], offset=2,
                        obj_mode=True, resize_dim=RESIZE)
    else:
        source = Ycb('cuda:0', scene_dir=config['scene_dir'], obj_mode=True, resize_dim=RESIZE)
    assert source.length == config['frames'], (source.length, config['frames'])
    access = config['active_keyframes']
    active = None if access is None else ActiveKeyframes(**access)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        tracker.run_track3r(cfg=dict(results_path=str(result), que_size=1),
                            args=['--obj_mode', '--resize_dim', str(RESIZE)], frame_source=source,
                            keyframe_indices=config['keyframe_indices'], active_keyframes=active)
    finally:
        if active is not None:
            write_json(result / 'active_keyframes.json', active.events)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    traj = np.load(result / 'traj.npy')
    assert traj.shape == (config['frames'], 4, 4) and np.isfinite(traj).all()
    assert len(list(mask_dir.glob('*.png'))) == config['frames']
    metrics = dict(frames=config['frames'], seconds=elapsed,
                   keyframes=len(np.load(result / 'kf_idx.npy')),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if config['setting'] == 'arctic':
        gt = kvt_eval.load_gt_arctic(config['scene'])
    else:
        ob_in_cam = ycb_poses(Path(config['scene_dir']))
        gt = np.linalg.inv(ob_in_cam)
        # Convention check only: the wrong inverse should align far worse.
        metrics['ate_uninverted_gt_m'] = float(align_pair(
            dict(traj_gt=ob_in_cam, traj_est=traj, name=config['scene']), 'traj', ret_np=False)[0])
    ate, rpe_t, rpe_rot = align_pair(dict(traj_gt=gt, traj_est=traj, name=config['scene']),
                                     'traj', ret_np=False)
    # rpe_rot is evo's dimensionless rotation-part norm, as eval.py reports it.
    metrics.update(ate_m=float(ate), rpe_t_m=float(rpe_t), rpe_rot=float(rpe_rot))
    write_json(result / 'metrics.json', metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Internal: one arm config.json')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--tag')
    parser.add_argument('--ycb-only', action='store_true',
                        help='Rerun only YCBInEOAT (TUM and ARCTIC done in job 25949)')
    args = parser.parse_args()
    assert sys.platform == 'linux'
    if args.run is not None:
        return run(args.run)
    assert args.work.is_relative_to('/tmp') and (args.ycb_only or (args.work / 'TUM_OCCLUSION_OK').is_file())
    from kvt_arctic_run import DATASET_DIR, OUT as ARCTIC_OUT, stage_dataset

    os.chdir(CHECKOUT)
    inputs = {}
    if not args.ycb_only:
        manifest = json.loads((ARCTIC_OUT / 'initial_frames' / 'manifest.json').read_text())
        stage_dataset(manifest)
        inputs = dict(arctic_prepared=str(ARCTIC_OUT / 'prepared.tar'),
                      arctic_prepared_sha256=sha256(ARCTIC_OUT / 'prepared.tar'), init_masks={})
        for scene in ARCTIC:
            reviewed = ARCTIC_OUT / 'masks' / scene / 'init_mask.png'
            shutil.copyfile(reviewed, DATASET_DIR / scene / 'init_mask.png')
            inputs['init_masks'][scene] = sha256(reviewed)
    write_json(args.work / 'objects_protocol.json', dict(
        arctic=ARCTIC, ycbineoat=YCB_SCENE, resize_dim=RESIZE, tracker='object mode, no token drop',
        native='own angular admission; arms replay its keyframe schedule',
        subsets=[name for name, _ in SUBSETS], visibility='keyframe SAM mask fraction at model resolution',
        ycb_init='SAM 2 from first gt_mask', ycb_gt='inverse of annotated object-in-camera poses',
        inputs=inputs))
    sequences = [] if args.ycb_only else [
        ('arctic', s, DATASET_DIR / s, manifest['scenes'][s]['tracking_frames']) for s in ARCTIC]
    sequences.append(('ycbineoat', YCB_SCENE, YCB_STAGE / YCB_SCENE, None))
    results = {}
    for setting, scene, scene_dir, frames in sequences:
        if setting == 'ycbineoat':
            # Staged last so a layout surprise cannot cost the ARCTIC runs.
            YCB_STAGE.mkdir(parents=True, exist_ok=True)
            with tarfile.open(YCB_ARCHIVE) as packed:
                packed.extractall(YCB_STAGE)
            frames = len(ycb_poses(scene_dir))
            inputs.update(ycbineoat_archive=str(YCB_ARCHIVE), ycbineoat_sha256=sha256(YCB_ARCHIVE),
                          ycbineoat_frames=frames)
            write_json(args.work / 'objects_inputs.json', inputs)

        def arm(name, access=None, schedule=None):
            target = args.work / 'runs' / setting / scene / name
            target.mkdir(parents=True)
            write_json(target / 'config.json', dict(setting=setting, scene=scene, scene_dir=str(scene_dir),
                name=name, frames=frames, active_keyframes=access,
                keyframe_indices=None if schedule is None else schedule[1:]))
            print('RUN', setting, scene, name, frames, flush=True)
            with (target / 'run.log').open('w') as log:
                completed = subprocess.run([sys.executable, __file__, '--run', str(target / 'config.json')],
                                           stdout=log, stderr=subprocess.STDOUT)
            write_json(target / 'process.json', dict(returncode=completed.returncode))
            if completed.returncode:
                print((target / 'run.log').read_text()[-8000:], flush=True)
                raise RuntimeError(f'{setting}/{scene}/{name} failed; see run.log')
            metric = json.loads((target / 'metrics.json').read_text())
            if access is not None:
                metric.update(audit(target, schedule))
                # Upstream point-cloud dump, ~120 MB per run (25949: 6.4 GB); native keeps it.
                (target / 'pcd.npy').unlink()
            return target, metric

        native, baseline = arm('native')
        schedule = np.load(native / 'kf_idx.npy').tolist()
        assert schedule[0] == 0 and schedule == sorted(set(schedule)), schedule
        print('SCHEDULE', setting, scene, len(schedule), schedule, flush=True)
        metrics = dict(native=baseline)
        for name, access in [('all', dict(mode='all'))] + [
                (f'drop{j}', dict(mode='drop', drop=j)) for j in schedule] + SUBSETS:
            target, metric = arm(name, access, schedule)
            assert np.load(target / 'kf_idx.npy').tolist() == schedule
            np.testing.assert_array_equal(np.load(native / 'kf_poses.npy'), np.load(target / 'kf_poses.npy'))
            for mask in (native / 'sam_masks').glob('*.png'):
                assert mask.read_bytes() == (target / 'sam_masks' / mask.name).read_bytes(), (name, mask.name)
            # Identical to native's masks, which are kept; saves ~frames PNGs per arm.
            shutil.rmtree(target / 'sam_masks')
            # Tolerance of the existing ARCTIC prefix gate (kvt_arctic_run.py).
            untouched = None if access['mode'] != 'drop' else max(access['drop'], schedule[1]) + 1
            if name == 'all' or untouched is not None:
                np.testing.assert_allclose(np.load(native / 'traj.npy')[:untouched],
                                           np.load(target / 'traj.npy')[:untouched], rtol=1e-4, atol=1e-4)
            if name == 'all':
                print('OBJECT FIDELITY GATE OK', setting, scene, flush=True)
            metrics[name] = metric
        results[f'{setting}/{scene}'] = metrics
        write_json(args.work / 'comparison_objects.json', results)
        print('OBJECT SEQUENCE OK', setting, scene, flush=True)
    (args.work / 'JOB_OK').write_text('Occlusion setting complete; artifact review required.\n')


if __name__ == '__main__':
    main()
