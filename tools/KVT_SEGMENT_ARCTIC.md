# Fixed automatic segmentation/reanchor transfer: three ARCTIC objects

User selected 2026-10-04: S01 camera0 full box_grab_01, ketchup_grab_01,
espressomachine_grab_01. One fresh native object process and one fresh reanchor
process per scene; six tracking runs total. Resolution518, loader offset2,
existing reviewed SAM initialization masks, original angular native admission.
Fixed office-selected ORB25/max200 policy, no retuning, sweep or acceleration.

Fresh native uses kvt_arctic_run.py with no visualization. Its saved SAM masks
are reused to construct byte-hashed model inputs using the exact loader's
mask-before-resize operation. Reanchor's detector sees these masked pixels.
Object foreground segmentation (SAM) and temporal map segmentation (ORB) are
separate operations. Existing reanchor shared-point geometry is confidence
filtered; it has no explicit foreground-mask restriction. This is a transfer of
the current mechanism, not a new object-specific bridge implementation.

Native reference is original object-mode KV-Tracker, not TUM's fixed50/cap20
control. Both predictions are scored against load_gt_arctic camera-in-object GT
with the upstream full-run Sim(3). Save ATE, adjacent-pair translation RPE/p99,
raw poses, bridge/live-cut/refresh records and memory per sequence. Native's
original archive also includes evo metrics; adjacent-pair metrics are saved in
comparison evidence. Verify full frame count and identical pair indices.

Timing scopes differ: native includes live SAM and mask export; reanchor uses
saved masks and excludes SAM. Report separately; do not claim a direct speedup.
No rendering or new masks, thresholds or model checkpoints are requested.
No follow-on evaluation is automatically queued. Contracts run first; any failure
stops loudly. Raw inputs/masks/provenance are archived before allocation ends.

## CAMP submission

After parent publication, on head in /mnt/projects/gr/3DRecon/layer_good:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector arctic"
O=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

Inside the allocation: existing kvt.tar, ARCTIC prepared.tar and reviewed masks,
local Pi3/SAM checkpoints, plus previously verified26127 checkpoint context.
No installs or downloads. The wrapper runs detector/handoff/reanchor contracts.
Native archives: kvt_arctic_out/arctic_tum_JOB_segment_native_SCENE_UTC.tar.
Reanchor/input/comparison/context archives: kvt_tum_out/tum_JOB_*.tar.
Comparison records each exact native archive path. Transfer both sets before
review; extract each archive separately. Completion: three SEGMENT ARCTIC COMPARE
lines, SEGMENT ARCTIC COMPLETE and JOB OK. Interpret per scene, not paper mean.

Local preparation is static-checked only; object staging/evaluation/full GPU
tracking remains pending. No native baseline reproduction claim until reviewed.
