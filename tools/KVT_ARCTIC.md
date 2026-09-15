# KV-Tracker: three-sequence ARCTIC pilot

## Status

The five official archives are downloaded and checksum-verified on CAMP
(`docs/kv-tracker-cluster.md`). The whole local tool chain is written and
statically checked; **nothing has been run on the cluster yet**, so no ARCTIC
tracking result or metric exists.

| Stage | Tool | Where | Gate |
|---|---|---|---|
| 1. Prepare | `tools/arctic_prepare_subset.py` | `data`, system `python3` | 3x `ALIGNED`, `PREPARE OK` |
| 2. Initial masks | `kvt_run.sh arctic-mask` | `24g`, one GPU | 3x `MASK`, `MASK OK`, then **you** review the overlays |
| 3. Track, evaluate, visualise | `kvt_run.sh arctic-run` | `24g`, one GPU | 3x `TRACKED`, finite ATE for all three, 3x `VIZ`, `ARCHIVED`, `ARCTIC RUN OK` |

## Protocol

- Subject S01, camera 0 (the egocentric view), complete sequences `box_grab_01`,
  `ketchup_grab_01`, `espressomachine_grab_01`.
- Preserve the upstream loader's `offset=2`: SAM 2 is initialized on the **third**
  sorted image, and `eval.py` drops the matching two GT rows.
- Existing SAM 2.1 small and Pi3 from `kvt.tar`; upstream object-mode masking, the
  original 10-degree keyframe rule, `--resize_dim 308`. The HouseCat crop and the
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

### The one open protocol question

The paper does not state the ARCTIC input resolution and `config/arctic.yaml`
carries none, so `--resize-dim` defaults to **308**, the resolution at which the
TUM baseline reproduced Table 1 exactly. If the ATE lands far from Table 4, 518
(`main.py`'s own default) is the first thing to vary, not the last.

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
markers) and `prompt.json`. They live on `/mnt`, so leaving the allocation to
review them costs nothing. Re-running with different prompts overwrites them.

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
bash tools/kvt_run.sh arctic-run --results pilot
```

One invocation does all three sequences: stage `prepared.tar` into `/tmp`, symlink
it as `kv_tracker/datasets/arctic_data`, copy each reviewed mask in as the scene's
`init_mask.png`, track, evaluate, and write one archive. `--only-eval` re-evaluates
and re-archives results already in `/tmp` without re-tracking — the recovery path
if evaluation fails after the expensive part succeeded.

Every number is checked before it is reported: `traj.npy` must have exactly
`gt_count - 2` rows, one recorded mask per tracked frame, and GT and estimate
shapes must match before `align_pair` sees them.

### What it renders

Per sequence, into `<results>/viz/`:

- `segmentation.mp4` — every frame the tracker saw with the mask actually used on
  it, the object's frame fraction printed per frame. This is the mask-quality
  evidence; watch for the mask jumping to the hand once the grasp starts.
- `trajectory.png` — three panels: the Sim(3)-aligned estimate against GT in 3D,
  the per-frame translation error with our ATE and the paper's Table 4 value drawn
  as lines and keyframes marked, and x/y/z against frame. The per-frame curve is
  recomputed from the aligned poses and **asserted equal to evo's ATE** to 1e-6,
  so the picture and the number cannot drift apart.
- `object.ply` + `object_turntable.mp4` + `object_views.png` — the object points
  from the last keyframe reconstruction, kept where `mask & confidence > threshold`,
  the same rule the HouseCat experiment used.

Plus one `summary.png` at the archive root: our ATE against Table 4, per sequence.

`kvt_run.sh arctic-viz --results pilot --scenes <...> --metrics <metrics.json>`
re-renders all of it from results still in `/tmp`, so a figure can be reworked
without re-tracking.

### The articulation readout

The run prints `ARTICULATION <scene>: <range> deg` before tracking — the GT column
`load_gt_arctic` throws away. ARCTIC objects are articulated (box lid, ketchup cap,
espresso lever) and KV-Tracker tracks the masked region as **rigid**, so a large
range would mean one whole-object mask is the wrong initialization and the base
part should be masked alone. `grab` sequences are expected to be near-rigid; this
is the check, not the assumption.

The archive `/mnt/.../kvt_arctic_out/arctic_pilot_<UTC>.tar` holds, per sequence,
the results (`traj.npy`, `kf_poses.npy`, `kf_idx.npy`, `pcd.npy`,
`keyframes.npz`), **every per-frame SAM mask**, everything under `viz/`, and the
initial frame, mask, overlay and prompt; plus `summary.png`, `metrics.json` and a
manifest with both git revisions, the `kv_tracker` diff,
torch and GPU, the tracker arguments and the preparation manifest. The raw input
pixels stay reconstructible from `prepared.tar`, which is already on `/mnt`.

```bash
rsync -av chunquancheng@131.159.11.60:'/mnt/projects/gr/3DRecon/kvt_arctic_out/arctic_pilot_*.tar' \
  cluster_results/kvt_arctic/
```

## What this pilot cannot claim

Three of ten sequences, one subject, masks that are ours rather than the authors'.
It is a per-sequence comparison against Table 4, not a reproduction of the paper's
ARCTIC mean, and it says nothing about inference FPS.
