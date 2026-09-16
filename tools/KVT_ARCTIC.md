# KV-Tracker: three-sequence ARCTIC pilot

## Status

**Run end to end 2026-09-15 on `muenchen`, at both 308 and 518.** At the settled
518: box **0.183**, ketchup **0.294**, espresso **0.150** against paper Table 4's
0.200 / 0.249 / 0.151. Numbers, the starved-geometry diagnosis and the resolution
decision: `tools/FINDINGS.md`, "KV-Tracker on ARCTIC". Archives on `/mnt` under
`kvt_arctic_out/`: `arctic_pilot_20260915T151239Z.tar` (308, 265 MB) and
`arctic_r518_20260915T152911Z.tar` (518, 646 MB). The five official source archives
are downloaded and checksum-verified (`docs/kv-tracker-cluster.md`).

Re-running any stage is the same three commands below; stages 1 and 2 are already
done and their outputs persist on `/mnt`.

| Stage | Tool | Where | Gate |
|---|---|---|---|
| 1. Prepare | `tools/arctic_prepare_subset.py` | `data`, system `python3` | 3x `ALIGNED`, `PREPARE OK` |
| 2. Initial masks | `kvt_run.sh arctic-mask` | `24g`, one GPU | 3x `MASK`, `MASK OK`, then **you** review the overlays |
| 3. Track, evaluate, visualise | `kvt_run.sh arctic-run` | `24g`, one GPU | 3x `ARTICULATION`, 3x `TRACKED`, finite ATE for all three, 3x `VIZ`, `VIZ OK`, `ARCHIVED`, `ARCTIC RUN OK` |

## Protocol

- Subject S01, camera 0 (the egocentric view), complete sequences `box_grab_01`,
  `ketchup_grab_01`, `espressomachine_grab_01`.
- Preserve the upstream loader's `offset=2`: SAM 2 is initialized on the **third**
  sorted image, and `eval.py` drops the matching two GT rows.
- Existing SAM 2.1 small and Pi3 from `kvt.tar`; upstream object-mode masking, the
  original 10-degree keyframe rule, `--resize_dim 518`. The HouseCat crop and the
  confidence-threshold experiment are deliberately **not** carried into this run.
- Evaluation uses the upstream GT transformation (`eval.load_gt_arctic`, which
  composes the egocentric extrinsics with the object pose into `T_obj2c`) and the
  upstream global Sim(3) alignment (`evo_utils.align_pair`). The runner only
  restricts the scene list; it does not reimplement either.
- [Paper Table 4](https://arxiv.org/html/2512.22581v1#S4.T4) reference ATE:
  box **0.200 m**, ketchup **0.249 m**, espresso machine **0.151 m**.
  Compare per sequence; this subset's mean is not the ten-sequence paper mean.
- Author initialization masks are absent from the checkout. Following the reported
  workflow with reviewed SAM masks does not guarantee identical results.
- Frames are fed synchronously (`frame_source`) instead of through
  `run_dataset.py`'s unbounded queue: same frames, same order, bounded memory, and
  every per-frame mask recorded. **The printed frames/s is not the paper's FPS
  benchmark** — mask export is synchronous.

### Resolution: settled 2026-09-15, report 518

Both resolutions were run on all three sequences. **518 is the protocol** —
`main.py`'s own default, and the README's `--resize_dim 308` appears only in its
camera-level TUM and 7-Scenes examples. At 308 the box fails badly (0.321 against
the paper's 0.200) because a texture-poor object brought up to the camera starves
Pi3 of confident points; at 518 it is 0.183. Numbers and the diagnosis:
`tools/FINDINGS.md`, "KV-Tracker on ARCTIC". `--resize-dim` now defaults to 518,
so stage 3 needs no extra flag; pass `--resize-dim 308` to reproduce the first run.

## Stage 1: prepare, on `data`

```bash
srun -p data --account=students --qos=students_normal --propagate=NONE --pty bash -l
cd /mnt/projects/gr/3DRecon/layer_good
git -c fetch.recurseSubmodules=false pull --ff-only
python3 tools/arctic_prepare_subset.py
exit
```

No container, no package installation, no GPU. It streams only the camera-0 JPEGs
and the six GT files into `/mnt/projects/gr/3DRecon/kvt_arctic_out/prepared.tar`,
**already in the layout `run_dataset.py` and `eval.py` expect**, so staging is a
plain extract plus one symlink. The three unchanged initialization JPEGs and the
manifest land in the `initial_frames/` sibling. Do not unpack the TAR on project
storage; it is meant for job-local `/tmp`.

`ALIGNED` proves image IDs map to all object-GT rows through the official S01
`ioi_offset`. If that assertion fires, the image/GT mapping is not what ARCTIC
documents — read the printed IDs, do not trim the sequence to make it pass.

Then on the Mac:

```bash
mkdir -p cluster_results/kvt_arctic
rsync -av chunquancheng@131.159.11.60:/mnt/projects/gr/3DRecon/kvt_arctic_out/initial_frames/ \
  cluster_results/kvt_arctic/initial_frames/
```

## Stage 2: initial masks, on one GPU

Read pixel coordinates off the three JPEGs, then:

```bash
srun -p 24g -w muenchen --account=students --qos=students_normal \
  --gres=gpu:1 --propagate=NONE --pty bash -l
cd /mnt/projects/gr/3DRecon/layer_good
bash tools/kvt_run.sh arctic-mask \
  box_grab_01:box:<x0>,<y0>,<x1>,<y1> \
  ketchup_grab_01:point:<x>,<y> \
  espressomachine_grab_01:point:<x>,<y>
exit
```

Prompt grammar: `<scene>:point:<x>,<y>[:<label>]` (label 1 foreground, default;
0 background; repeat the flag to add points) or `<scene>:box:<x0>,<y0>,<x1>,<y1>`.
One box **or** any number of points per scene, not both. All three sequences go in
one command.

These are hand-held objects in an egocentric view, so the failure to watch for is
SAM 2 returning object **plus hand**. A box around the object is the usual first
try; the refinement is a positive point on the object and a negative point on the
hand, e.g. `box_grab_01:point:410,300 box_grab_01:point:360,430:0`.

Results are written to `/mnt/.../kvt_arctic_out/masks/<scene>/` — `init_mask.png`
(what the loader reads), `init_overlay.png` (green object, blue contour, prompt
markers) and `prompt.json`. Re-running with different prompts overwrites them.

**Keep the allocation open while you review.** The podman store is job-local
(`/tmp/.local/share/containers/storage`), so the first `kvt_run.sh` in an
allocation pays a silent ~10-12 minute `podman load` of the 19 GB `kvt.tar`, and
leaving costs that again. Review from the Mac in a second terminal and run stage 3
in the same shell; the outputs themselves are on `/mnt` and survive either way.

```bash
rsync -av chunquancheng@131.159.11.60:/mnt/projects/gr/3DRecon/kvt_arctic_out/masks/ \
  cluster_results/kvt_arctic/masks/
```

**Do not continue until all three overlays show the object and only the object.**
The mask produced here is the exact mask the run starts from — it is written by
the same SAM 2 call the loader makes on its first frame.

## Stage 3: track and evaluate, on one GPU

```bash
srun -p 24g -w muenchen --account=students --qos=students_normal \
  --gres=gpu:1 --propagate=NONE --pty bash -l
cd /mnt/projects/gr/3DRecon/layer_good
bash tools/kvt_run.sh arctic-run --results r518
```

`--results` names both the per-scene results directory and the archive, so pick a
name that says what the run is: the two done runs are `pilot` (308) and `r518`
(518, the protocol). One invocation does all three sequences: stage `prepared.tar` into `/tmp`, symlink
it as `kv_tracker/datasets/arctic_data`, copy each reviewed mask in as the scene's
`init_mask.png`, track, evaluate, and write one archive. `--only-eval` re-evaluates
and re-archives results already in `/tmp` without re-tracking — the recovery path
if evaluation fails after the expensive part succeeded.

Every number is checked before it is reported: `traj.npy` must have exactly
`gt_count - 2` rows, one recorded mask per tracked frame, and GT and estimate
shapes must match before `align_pair` sees them.

### What it renders

The HouseCat6D layout, from `kvt_artifacts.py`'s `actual_history.mp4`, which is
also what the project page shows: one `viz/tracking.mp4` per sequence, 1280x800.

- **Left panel** — the frame the tracker saw with the mask it actually used on it,
  so segmentation quality is visible in the same video. Watch for the mask
  jumping to the hand once the grasp starts.
- **Right panel** — the reconstructed object as one-pixel depth-sorted points,
  with the ground-truth camera trail in green and the tracked one in red, growing
  frame by frame. Both trails and the cloud are carried into the GT metric frame
  by **one** Sim(3), and that transform is asserted to be the one evo used on the
  poses, so the gap you see between the trails is the error that is reported.
- **Text strip** — frame, object fraction, per-frame error, ATE RMSE and the paper
  Table 4 value.
- **Tail** — a 360-degree orbit of the frozen final cloud, labelled as such so it
  is not mistaken for tracking.

Plus `viz/object.ply` per sequence for MeshLab. No plots: the numbers are the
printed table and `metrics.json`.

`kvt_run.sh arctic-viz --results r518 --scenes <...> --metrics <path>` re-renders
from results still in `/tmp`, without re-tracking. The run leaves that file at
`/tmp/arctic_<results>_<UTC>/metrics.json`, the staging directory it built the
archive from.

### The articulation readout

The run prints `ARTICULATION <scene>: <range> deg` before tracking — the GT column
`load_gt_arctic` throws away. ARCTIC objects are articulated (box lid, ketchup cap,
espresso lever) and KV-Tracker tracks the masked region as **rigid**, so a large
range could mean a whole-object mask is the wrong initialization and the base part
should be masked alone.

**Measured 2026-09-15: box 1.86 deg, ketchup 3.43 deg, espresso 53.61 deg.** So
`grab` sequences are *not* uniformly near-rigid — the espresso lever really swings
through its clip — and yet espresso, with the lever deliberately inside the mask, is
the sequence that reproduces Table 4 exactly. Articulation does not drive the error
here. Keep the readout as a check; do not re-mask on the strength of it alone.

The archive `/mnt/.../kvt_arctic_out/arctic_<results>_<UTC>.tar` holds, per sequence,
the results (`traj.npy`, `kf_poses.npy`, `kf_idx.npy`, `pcd.npy`,
`keyframes.npz`), **every per-frame SAM mask**, everything under `viz/`, and the
initial frame, mask, overlay and prompt; plus `metrics.json` and a manifest with
both git revisions, the `kv_tracker` diff, torch and GPU, the tracker arguments,
the measured articulation and the preparation manifest. The raw input
pixels stay reconstructible from `prepared.tar`, which is already on `/mnt`.

```bash
rsync -av chunquancheng@131.159.11.60:'/mnt/projects/gr/3DRecon/kvt_arctic_out/arctic_*.tar' \
  cluster_results/kvt_arctic/
```

`--results` names the archive, so the 308 run is `arctic_pilot_*` and the 518 run
`arctic_r518_*`. Both are on `/mnt`; the 518 one is the protocol.

## What this pilot cannot claim

Three of ten sequences, one subject, masks that are ours rather than the authors'.
It is a per-sequence comparison against Table 4, not a reproduction of the paper's
ARCTIC mean, and it says nothing about inference FPS.
