# VGGT-SLAM and MASt3R-SLAM baselines on TUM

2026-10-12, user decision: check the open-source SLAM baselines ourselves instead of
trusting reported numbers. Scope chosen by the user: VGGT-SLAM 1.0 **and** 2.0,
MASt3R-SLAM calibrated **and** uncalibrated, the five fr1 sequences on CAMP plus fr3
office, first a reproduction with each paper's own protocol, then a rescoring against
our methods. No new container (user): `kvt.tar` is borrowed read-only.

## What the published TUM numbers measure (read from the code)

- **VGGT-SLAM** (`evals/eval_tum.sh`, both versions): `main.py --max_loops 1
  --min_disparity 50 --conf_threshold 25 --submap_size w` (2.0 adds `--lc_thres 0.95`),
  then `evo_ape tum <gt> <est> -as`. The estimate is `Map.write_poses_to_file`: one pose
  per **keyframe** (optical-flow disparity >= 50 px), after loop closure and the
  SL(4)/Sim(3) graph optimisation. 1.0's script averages 5 runs; 2.0's runs once.
- **MASt3R-SLAM** (`scripts/eval_tum.sh`): `config/eval_calib.yaml` or
  `eval_no_calib.yaml` (single-threaded, `dataset.subsample: 2`, i.e. every second
  frame), `evo_ape ... -as` on `save_traj`: **keyframes only**, after global
  optimisation and loop closure.

Both are therefore keyframe trajectories, revised after the fact. Our KV-Tracker
numbers score every frame's immediate (causal) pose, which is a stricter quantity. The
two are not comparable as-is; stage 2 makes them comparable.

## Arms and published targets (ATE RMSE, m)

| Arm | xyz | rpy | desk | desk2 | room | Source |
|---|---:|---:|---:|---:|---:|---|
| `mast3r_calib` | 0.009 | 0.027 | 0.016 | 0.024 | 0.061 | arXiv 2412.12392 Table 1 |
| `mast3r_nocalib` | 0.020 | 0.041 | 0.035 | 0.055 | 0.118 | same |
| `vggt_v1_sl4` (w=32) | 0.014 | 0.030 | 0.025 | 0.040 | 0.102 | arXiv 2505.12549v2 Table 2 |
| `vggt_v1_sim3` (w=32, `--use_sim3`) | 0.016 | 0.041 | 0.040 | 0.055 | 0.088 | same |
| `vggt_v2` (w=32) | 0.014 | 0.026 | 0.025 | 0.029 | 0.063 | arXiv 2601.19887 Table I |

fr3 long_office_household is not in either paper (no target; it is our main long
sequence). fr1 360/floor/plant/teddy are not on CAMP, so the papers' nine-sequence
averages are not reproduced, only these five per-sequence values.

## Environment (no image change)

`tools/slam_baselines.sbatch prepare` (once) runs `tools/slam_baselines_env.sh build`
inside `kvt.tar`: clones pinned upstreams (SHAs in the env script and
`slam_baselines/SHAS.txt`), compiles lietorch, the MASt3R-SLAM backend, curope, asmk
and pyimgui into wheels, installs the declared runtime dependencies under a
constraint file that forbids changing any package the image already has, and stores
every added package as a wheel. Each run installs those wheels offline into the
container's ephemeral layer and asserts Python, torch, CUDA, torchvision and numpy are
the image's own. mast3r and in3d run from source via `PYTHONPATH` (mast3r locates dust3r
relative to its own file; in3d loads resources beside its source), as upstream's
editable installs do. kvt.tar was chosen over optpose.tar because KV-Tracker's own
dependency list derives from MASt3R-SLAM's (pyrealsense2 + libusb, evo, open3d already
present) and because timing then shares KV-Tracker's environment.

Deviations from upstream, all recorded per run: the image's torch instead of the pinned
versions (VGGT-SLAM pins 2.3.1, MASt3R-SLAM documents 2.5.1); gradio not installed (only
demos import it); one run per arm (VGGT-SLAM 1.0 published a 5-run mean).

Storage (`/mnt/projects/gr/3DRecon`): `slam_baselines/` holds `vggt_1b/model.pt` (HF,
~5 GB), `ckpt/` (three MASt3R files, naverlabs), `torch_home/hub/` (dino_salad.ckpt,
DINOv2 hub code and weights, link to VGGT-1B), `wheels/`, `src.tar`, freeze/lock files.
Outputs: `slam_baselines_out/slam_<mode>_<jobid>.tar` (per run: saved trajectory,
timestamped stdout, GPU memory samples, evo output, MASt3R-SLAM `.ply`; `results.json`;
context incl. dataset hashes, SHAs, lock, container inspect, job log).

## Gates

1. **Prepare**: log shows `cuda available for the build`, `MASt3R-SLAM imports ok`,
   `VGGT-SLAM v1 imports ok`, `VGGT-SLAM v2 imports ok`, `ENV BUILD OK`,
   `dinov2 hub cached`, `SLAM BASELINES prepare JOB OK`. If the CUDA assertion fails on
   jena, rerun prepare on stuttgart (`-p 24g -w stuttgart`; the build is well under the
   two-hour idle-GPU limit).
2. **Pilot** (fr1_xyz, five arms): every arm completes with a non-empty keyframe
   trajectory and an `SLAM BASELINE` line; review ATE against the targets, keyframe
   counts, FPS and peak memory before the full run. No ratio threshold on one sequence.
3. **Full** (five fr1 + office): stage-1 reproduction per arm. Proposed criterion (mine,
   not the papers'): the arm's mean ATE over the five fr1 sequences is within 25% of the
   mean of its published values. An arm that fails is reported as not reproduced, with
   its per-sequence ratios; no tuning.

## Stage 2 (after gate 3, offline on the Mac)

- Rescore each baseline's saved keyframe trajectory with our evaluator's alignment
  (one Sim(3) over the trajectory) as a cross-check of evo.
- Score our saved per-frame trajectories (26520 native and segmented, same sequences)
  at each baseline's keyframe timestamps, so both sides are measured on the same frames.
  Ours stay causal; theirs stay optimised; that difference is reported, not removed.
- Cost: loop FPS over input frames (model load excluded) and peak `nvidia-smi` memory.
  KV-Tracker's recorded peak is torch-allocated memory, a different quantity; a matched
  memory comparison needs KV-Tracker measured the same way (decide after stage 1).

## Commands (CAMP head, from `layer_good`)

The pull runs inside each allocation (`--wrap`), as in the field notes; `-o` must be
`../slam_baselines_slurm-%j.log` because the script copies that log into its archive.

```
cd /mnt/projects/gr/3DRecon/layer_good
D=/mnt/projects/gr/3DRecon/slam_baselines/vggt_1b
sbatch tools/hf_download.sbatch facebook/VGGT-1B model $D model.pt
```

then, after `DOWNLOAD OK`:

```
O=../slam_baselines_slurm-%j.log
P='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$P && bash tools/slam_baselines.sbatch prepare"
G='--gres=gpu:1 --propagate=NONE -t 6:00:00'
sbatch -p 12g -w jena $G -o "$O" --wrap="$W"
```

Pilot and full use the same `O`, `P` and `G` with `pilot` / `full` in `W` and
`-p 24g -w stuttgart`. Each only after the previous gate has been reviewed.

## Known risks (unverified until prepare/pilot)

- kvt.tar's Python, torch and numpy are not recorded in our docs; prepare prints them.
  MASt3R-SLAM's TUM loader uses `np.unicode_`, removed in numpy 2.
- `gtsam-develop` is a nightly package; the version resolved at prepare is pinned in
  `wheels.lock`, and the SL(4) API it exposes may differ from what 1.0 was written for.
- `nvidia-smi` must exist inside the container for the memory sampler.
