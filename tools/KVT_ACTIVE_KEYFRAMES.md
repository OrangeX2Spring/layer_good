# Pro motion: control active keyframe count in KV-Tracker

User-selected scope, 2026-09-27: native KV-Tracker continues accumulating selected
keyframes and rebuilding their K/V; a controller chooses how many existing
keyframes each query attends to, from 1 to current K. Selection is the anchor plus
the newest n-1 admitted keyframes. Count may increase/decrease by one at a decision.
These are keyframes (native TUM interval 50), not recent video frames. This replaces
the incorrectly scoped StreamVGGT recent4/recent16 experiments as the active task.

## First concrete controller

This pilot uses an explicit **heuristic**, not the StreamVGGT learned regressor.
The signal is mean absolute change between current and previous image thumbnails:
uint8 model-input RGB, spatial stride 16, channel mean divided by 255. Camera/scene
motion, exposure and appearance all affect it; it is not geometric motion or GT.
At source frames 8,16,... compare current score with the selected quantile of up
to 64 preceding query scores (frames max(1,t-64)..t-1). Current score enters history
only after its decision. Above threshold adds one active keyframe; otherwise subtract
one, clipped to [1,K]. Ties decrease. Start n=1; hold count between decisions.
Quantiles are the user-requested .25/.5/.6/.75. They govern increase/decrease
frequency, NOT a target fraction of available keyframes. No learned weights,
threshold fitting, image skipping, eviction, RGB replay or new keyframe policy.

## Native bank and baseline

Use TUM camera mode, native interval50/cap20, width308 and seed0. Every arm uses
`policy=original`, full patches and original joint rebuilds. The baseline has no
active-cache object and uses all stored keyframes for every query. The query-only
hook temporarily replaces model.cache with gathered complete K/V frame slots;
post-RoPE values retain original positions, and full selection uses the original
dictionary/tensors without gathering. Restore full native bank immediately after
query, before admission/rebuild. Native cache tensors must preserve identity and
version. Query output must not modify the stored bank. No append-only intervention.

Native bootstrap stores [0,0]: retain both slots until native first rebuild removes
the duplicate. This is ONE unique keyframe, TWO physical slots. Logs preserve both
counts. Newly admitted frame t becomes available at t+1, never its own query.
All arms retain the same full bank: active-byte reductions are not persistent
storage reductions. Gathers temporarily add storage; CUDA peaks and total runtime
must be measured. Native rebuild cost is unchanged and cannot be claimed saved.

## Bounded pilot and gates

First 512 consecutive frames each of office, with-loop, no-loop, prepared by the
existing TUM loader. Native keys [0,49,99,...,499], 11 unique by clip end. K is the
number actually available, not a fixed 4/16 choice. This does not exercise a larger
bank or full-sequence saturation. Reused sequences mean exploratory diagnosis.

Per clip: native baseline; all-selected access fidelity control; forced alternating
+1/-1 access on 128 frames; q50 128-frame prefix; then one-keyframe, ceil(K/2), and
four quantile arms on 512 frames. The forced/prefix runs are correctness gates,
not additional accuracy candidates. Each is a fresh process. No full-run option.

Required gates: existing TUM tests plus actual Pi3-block tests for full equivalence,
subset equivalence, full-bank preservation/recovery, duplicate accounting and
causal step/dwell semantics. Before policy comparison, all-selected trajectory
must match native within atol1e-5/rtol1e-4; saved rebuilt keyframe poses exactly
match. Forced gate must exercise growth/shrink. Every arm has native insertion IDs,
identical rebuild count and final keyframe poses; every event reconstructs its
causal quantile, count step, anchor/newest IDs and active/stored byte relation.
Q50 independent prefix trajectory/events must match the full run's prefix.
GT validity/pair masks must match baseline. Abort on contract failure.

Report all arms: ATE, translation/rotation RPE and translation p99, fitted Sim(3)
scale and valid counts; active-count trajectory/range/changes, mean active/available
counts, active versus stored bytes, synchronized total runtime, query/rebuild calls
and times, CUDA peak allocated/reserved. Full-trajectory alignment, partial no-loop
GT and single-run timings limit interpretation. Count reduction is not evidence of
better allocation or lower peak VRAM. Motion can fail as an allocation signal;
report saturation or insufficient count range rather than retuning on evaluation.

## Execution and provenance

Publish fork code first, then parent gitlink/tools. Use existing kvt.tar read-only;
no model download, environment rebuild or dependency install. Entry:
`tools/kvt_tum.sbatch active-keyframes`. Pull/submodule update execute inside the
allocation. If the head checkout lacks this mode, bootstrap the pull in sbatch's
allocation wrapper before invoking it. All cluster operations remain the user's.

Wrapper saves parent/fork/Pi3 source, revisions/patches, container hash, package
list, tests/logs/exit. Driver archives exact model input pixels/manifests/ZIP hash
per scene at kvt_tum_out/tum_<job>_inputs_<scene>.tar. Existing EXIT trap archives
runs as tum_<job>_all_runs.tar and context/protocol/comparison as
 tum_<job>_context.tar. Partial failures are archived too. Never infer success from
archive existence: require tests/gates, JOB_OK, exit0 and artifact review. No new
full evaluation follows automatically.
