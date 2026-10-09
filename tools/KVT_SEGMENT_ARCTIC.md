# Fixed automatic segmentation/reanchor transfer: three ARCTIC objects

## Current object-aware comparison — 2026-10-09

User authorized detector implementation and explicitly selected **plan semantic
KV; do not implement semantic selection yet**. This supersedes the historical
three-object submission below as the next action. Existing26205 artifacts remain
preserved. Numerical audit/caveats: FINDINGS "Object detector audit26205".

The detector remains ORB25/max200 with the same matching, minimum age50,
five-frame persistence, unknown-texture behavior and map handoff. The sole
changed decision signal is coverage: `coverage_mode=anchor_support` divides
verified matches' occupied **anchor** cells by all initially occupied anchor
cells. The4x4 grid is normalized to the anchor ORB features' bounding rectangle
on the already SAM-masked RGB. A fraction below0.5 indicates low coverage.
This is an uncalibrated fixed hypothesis, not a fitted object-quality threshold.
It normalizes feature support rather than inferring a part, object pose or a
semantic mask. ORB extraction remains identical to the image-grid control,
including its susceptibility to mask-edge features. Absolute25-inlier threshold,
max200 cap, two-image bank and unconditional delayed connection are unchanged.

The audit found every actual cut had fewer than25 inliers, so fixing coverage
alone may not improve tracking. Different persistence/anchors can change later
decisions; saved logs cannot substitute for a causal rerun. Ketchup is selected
for the bounded diagnostic because it has the most coverage-only low flags in
the inspected logs, not because it promises the best tracking score. It is a
previously inspected sequence, not held-out validation. No threshold sweep.

Implementation: `kvt_segment_detector.py`, `kvt_reanchor_maps.py`,
`kvt_segment_arctic.py`; no model-fork change. The original `arctic` stage and
image-grid defaults remain available. New stages:

- `segment-detector arctic-object-contracts`: detector, handoff and reanchor
  contracts only. New contracts cover small-object support, current-view
  translation/scale, concentrated matches, persistence/unknown interruption and
  prefix causality. No tracking is queued after this stage.
- `segment-detector arctic-object`: contracts, then the full existing ketchup
  sequence. Fresh original object-mode native produces the masks/reference.
  A separate native replay uses exact hashed saved model pixels and resized
  masks, with original angular admission and no SAM model. Its raw trajectory
  must reproduce the reference at rtol=atol=1e-4 before either reanchor arm runs.
  Then independent image-grid and anchor-support reanchor processes use the same
  inputs, checkpoint environment and fixed handoff. Pair indices and finite full
  trajectories are checked. Stop on execution/fidelity failure; no automatic
  retry, threshold adjustment or other-object expansion.

Compared native replay and both reanchor arms exclude SAM/model load/evaluation/
final serialization; include saved-input I/O, hashing, upload and tracking.
Native needs resized masks; reanchor additionally pays detector/bridge/logging.
Native timing ends when its frame generator is exhausted, before tracker export.
The live-SAM reference timing is not used for speedup. Report per-arm scope,
memory and one ordered timing observation; this does not establish statistical
speedup or complete live-pipeline throughput. Native replay fidelity and timing
behavior remain CAMP runtime gates, not locally verified claims.

Quality success: ATE, translation RPE and p99 each within5% of the matched native
replay. Report old/new detector cuts and quality even if both fail. Fewer cuts
alone is not success. Audit changed boundaries, bridge/refresh behavior and
within-segment errors before choosing handoff/history work or another object.
No semantic labels, learned detector, loop decoder or new bridge policy in this
comparison. Follow-on plan: [OBJECT_SEMANTIC_KV_PLAN.md](OBJECT_SEMANTIC_KV_PLAN.md).

Outputs under `/mnt/projects/gr/3DRecon/kvt_tum_out/` include
`tum_JOB_segment_arctic_ketchup_grab_01_inputs.tar` (model RGB/masks + hashes),
`tum_JOB_arctic_native_replay_ketchup_grab_01.tar`,
`tum_JOB_arctic_reanchor_image_ketchup_grab_01.tar`,
`tum_JOB_arctic_reanchor_anchor_support_ketchup_grab_01.tar`, and
`tum_JOB_segment_arctic_object_comparison.tar`, plus context/all_runs.
The source live-SAM native archive remains in `kvt_arctic_out/` and is referenced
by hash. Partial runs are preserved by the existing wrapper. No new output is
claimed until the user returns runtime/archive evidence.

Publication/runtime status is recorded in docs/STATUS.md. After publication, the **first** user-run
chunk, from `/mnt/projects/gr/3DRecon/layer_good` on head, is contracts only;
pull and execution occur inside the allocation:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector arctic-object-contracts"
O=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE \
  -t 01:00:00 -o "$O" --wrap="$W"
```

Review detector/handoff suites, `OBJECT COVERAGE CONTRACTS OK`, archive exit0
and Slurm status before issuing the separate `arctic-object` command chunk.
Static Python3.10 AST, Bash syntax and whitespace checks pass; no local project
imports, tensor tests, tracking, installs or downloads performed.

## Historical fixed transfer — 2026-10-04

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
