# Semantic and geometric KV patch selection: bounded overnight design

Design dated 2026-09-28. User selected strongest established reference per task,
limited later cross-host confirmation, and 12 hours on one GPU; on implementation
(same day) the user lifted the 12-hour cap. Implemented for KV-Tracker: see
"Implementation" at the end. No jobs are authorized to bypass the preflight
review below. Mass-correction branch and probe 25990 stay
closed. No new training, merging, mass bias, layer sweep or token-budget sweep.

## Host and tasks

Use KV-Tracker native as the primary object tracker and primary scene reference.
This is the best-supported common reference in our records, not a claim that it
dominates every metric/scene. FINDINGS cross-host synthesis and correspondence
review show no universal winner. StreamVGGT/LongStream are limited transfer checks
after selection, not ten-policy grids. Rank each model against its own controls.

Object: ARCTIC S01 box, ketchup and espresso, existing complete prepared sequences
and saved masks, width 518. Evaluate camera-in-object pose as in existing object
protocol; do not call streaming pose-head readout a native object tracker.
Scene: TUM office first 1150 consecutive frames, with-loop/no-loop first 1024,
width 308. Freeze frame IDs/timestamps/GT-valid masks before running. Industrial
GT is partial; report valid counts and time coverage, no interpolation. These
are previously inspected datasets: screening/confirmation, not a new benchmark
or an untouched held-out generalization claim.

## Shared intervention and budget

Change historical KV patch selection only. Keep current-frame queries/dense heads,
native rebuild/refresh semantics, original anchor and special tokens. Retain
original K/V and integer positions; never average, rescale or regenerate tokens.
Select exactly K=ceil(P/4) ordinary patches per non-anchor admitted frame, where P
is that host's ordinary patch count. Anchor stays dense. Fixed 25% fraction for
all candidates and sparse controls. Report actual bytes including selectors;
protected anchor/specials mean total memory does not fall by exactly 75%.

KV-Tracker: same causal native admission schedule across arms, obtained from the
reference and replayed as an explicit fixed-schedule intervention. Changes cannot
silently alter keyframe IDs. Report this as cache selection conditional on native
admissions, not an end-to-end learned admission policy. After each native dense
rebuild, gather matching raw K/V rows; release unused dense KV storage before
query inference. Rebuild peaks remain measured and may eliminate VRAM benefits.
Retain original RGB/masks needed by native rebuilds and count their storage.
Historical selections persist until eviction; no hidden dense shadow cache.

Signals use only frame t and already observed history, never future pixels, GT,
evaluation errors or a dense teacher trajectory. Selector descriptors/history
are charged as auxiliary state. Any confidence/depth/attention extraction not
already produced by native inference is timed, including synchronization.

All methods have deterministic score/ID tie breaking. Quota rounding uses largest
remainders, capacity clipping and redistribution. Duplicates are removed before
filling the remaining slots from the policy's ranked eligible patches. Report
score-flatness and selected-set overlap: numerically identical methods are not
independent evidence. Flat corner scores still produce an exact K selection and
are explicitly marked as weak keypoint support, not claimed as detected landmarks.

## Ten frozen candidates

All ten run on both tasks. Object masks are actual target semantics; encoder
feature groups are appearance regions, not semantic class labels. No new scene
segmentation model/checkpoint is assumed available.

| ID | Method | Exact selection idea and hypothesis |
|---|---|---|
| K1 | Keypoint-ranked patches | Rank patches by maximum Shi–Tomasi corner response (3x3 image-gradient structure tensor) within each patch; keep top K. Tests whether sparse landmark texture alone is sufficient. |
| K2 | Distributed keypoints | Same response, but divide image into 4x4 cells and allocate K proportional to cell patch count. Tests geometric coverage versus clustered corners. |
| K3 | Corners plus contours | Half the slots from K2; remaining slots ranked by patch-mean gradient magnitude, distributed over 4x4 cells. Tests surfaces/edges missed by corner-only ranking. |
| K4 | Region-balanced landmarks | Object: 75% target-mask patches, 25% other patches, corner-ranked within each group. Scene: eight normalized encoder-feature clusters, equal-capacity-adjusted quotas and corner ranking. Tests whether dominant texture crowds out useful regions. |
| K5 | Boundary and interior | Half slots on target-mask boundary patches (object) or four-neighbor feature-cluster boundaries (scene), half on nonboundary patches, ranked by corners and distributed spatially. Tests shape constraints versus textured interiors. |
| K6 | Feature diversity | Greedy farthest-point selection in normalized encoder-feature space; deterministic first point is maximum corner response, subsequent points maximize minimum cosine distance. Tests coverage of distinct appearance rather than raw texture. |
| K7 | Persistent landmarks | Reciprocal cosine matches to the most recent admitted frame; rank by causal track age, then similarity, then corner response. Require cosine >=0.9. Unmatched remainder uses K2. Tests repeatable observations. |
| K8 | Stable plus novel | Half K from K7; half by largest cosine distance to the closest retained historical descriptor. Only retained descriptors are stored. Tests stable anchors plus newly revealed content, distinct from pure persistence. |
| K9 | Read-demand selection | Rank current frame's candidate keys by mean attention received from camera/register queries at the final global block, on its actual causal pass; select after the output is emitted, for future reads. No dense shadow, attention-output distortion score or layer search. Tests actual read demand. |
| K10 | Geometry coverage | Normalize predicted per-frame 3D coordinates by median adjacent-point spacing; greedy farthest-point sampling, first point highest native depth confidence. Exclude nonfinite points and fill from K2 if needed. Tests 3D support rather than 2D/appearance coverage. |

Feature groups for K4/K5: eight cosine clusters, deterministic farthest-point
initialization, five assignment/update iterations, shared definition across arms.
K1–K3 use the exact model-input RGB, excluding mask-cut pixels from corner/gradient
scores so artificial black-mask edges are not counted as texture. K5 explicitly
uses the mask boundary as its signal. No original unmasked pixel side channel.
K7/K8 reset histories when the host resets its native cache; they cannot retain
unbounded descriptors. K9 needs a tested model hook and explicitly charged
attention materialization; it is not an unimplemented free statistic. K10 uses
native predicted geometry, never GT depth/poses. None is claimed novel or reliable
before this test. Prioritize K2/K7/K4/K6 in implementation and scheduled order.

## Controls and size

Per clip: ten candidates + native + all-kept adapter + spatial-uniform quarter
+ random-quarter seeds 17 and 29 = 15 conditions. Repeat native and all-kept once
at the end = 17 runs per clip. Six clips => maximum 102 screen runs. The repeated
controls check numerical/path sensitivity and drift in timings, not uncertainty
intervals for all candidates. Seeds and order are frozen before launch.

All-kept adapter must match native within established fidelity tolerances. If
native/all-kept repeats differ by >5% in RMS, p99 or ATE, label that clip unstable
and do not use it to establish a method win. Do not silently drop a difficult clip
and claim an all-task success; report the task inconclusive if fewer than two of
its three clips are interpretable. No extra repeat campaign that night.

## Execution budget and review boundary

1. Implement hooks/contracts and one preflight on espresso and office (128 frames,
   all-kept and the ten selectors), inside a user-run allocation. Verify actual
   gathering, K/positions, full-retention fidelity, causal prefix invariance,
   unchanged admitted IDs/rebuilds, finite outputs, selector storage and timings.
   LongStream later must cross its native refresh; no generic adapter assumption.
2. Review preflight archive BEFORE submitting the overnight screen. Existing short
   runtime evidence does not validate these new hooks. No full screen dependency
   behind an unreviewed pilot. Scientific preflight values do not select candidates.
3. Use measured synchronized run/staging cost to project the 102-run schedule.
   Reserve one hour for archive finalization and overhead; launch the complete
   plan only if its conservative estimate fits 11 hours of execution. This is a
   cap, not a claim of known runtime. If it does not fit, reduce scope explicitly
   before submission; do not silently shorten clips or discard controls.
4. Schedule all controls first for each clip, then candidates round-robin across
   tasks so a timeout cannot leave one task untouched. Archive each completed
   condition. At the deadline start no further inference; missing runs are
   incomplete, never negative results. No repeated expensive failure: stop the
   affected host on a correctness/OOM failure and preserve the first context.

One 12-hour/one-GPU overnight screen after a reviewed preflight. Preflight and
later confirmation allocations are additional, explicitly bounded steps; their
time must not be hidden inside the overnight estimate. No submission command is
provided until implementation and the runtime estimate exist.

## Fixed decisions the next morning

Report every condition, not just winners: ATE, translation/rotation RPE, translation
p99, halves, fitted scale, GT coverage, tracker hold/failure counts, query/rebuild/
selector/total seconds, actual persistent and peak allocated/reserved bytes.

Candidate eligibility on a stable clip requires ALL:

- RMS, p99, ATE and rotation RPE <=1.05 times same-job all-kept reference.
- Each temporal half RMS <=1.05 times its reference (halves of GT-valid pair
  times within the fixed clip; report empty halves as unevaluable, not passing).
- Total end-to-end inference time <=1.10 times native, INCLUDING selection.
- Persistent KV plus selector state <=0.60 times all-kept; report total/peak
  memory separately. A method that keeps the dense bank only saves active reads
  and cannot pass this persistent-cache criterion.

A semantic/geometric-selection claim additionally requires RMS <=0.95 times
EACH sparse control (uniform and both random seeds), p99 no worse than EACH,
and these gains on at least two of the three task clips. It must meet eligibility
on every interpretable clip. Otherwise any benefit is merely pruning/efficiency,
not evidence that the scoring signal helped. No ATE-only winner.

Select at most ONE candidate per task satisfying those conditions. Tie-break:
lowest worst-clip p99 ratio to all-kept, then total time, then method ID. If none
qualifies for a task, STOP this family for that task; no threshold/budget/layer
retuning or extra mechanistic diagnostic. An unstable task is inconclusive and
does not advance automatically. A shared winner is allowed but not required.

## Limited confirmation, after screen review

Only eligible task winners proceed, unchanged, to StreamVGGT AND LongStream.
Per successful task/host: winner, matched dense retained-history control,
spatial-uniform and random seed17, on the task's predeclared espresso or office
screen clip = four runs; maximum 16 runs for two task winners and two hosts.
Preserve host-native reset/rebuild mechanics, shared per-frame quarter patch
fraction and causal scoring rules. Re-run each host's full-retention fidelity and
verify object evaluation on the new path before scientific runs. Native retention
and bounded-control results must be labeled separately.

Use the same accuracy/resource gates and compare with both sparse controls.
Failure on a host ends transfer claims for that host; do not tune separately.
Success on both supports cross-host transfer ON THESE CLIPS, not unseen-scene
generalization. End this campaign after the one screen and one confirmation batch.
No automatic main-method development or unbounded rescue of a near miss.

## Source context

Internal evidence: FINDINGS cross-host synthesis, 25833 correspondence review,
25906/25907 numerical-path caveats and 25992 closed mass-path result. Prior methods
motivate hypotheses, not a novelty claim:
- SuperPoint distinguishes geometric interest points from semantic identity:
  https://openaccess.thecvf.com/content_cvpr_2018_workshops/w9/html/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.html
- DINOv2 supports using pretrained visual features; it does not validate our cache selectors:
  https://arxiv.org/abs/2304.07193
- Co-Me already studies confidence-guided token merging; this screen selects
  original KV rows and does not reopen merging/mass correction:
  https://arxiv.org/abs/2511.14751

## Implementation (KV-Tracker screen, 2026-09-28)

Code: fork `kv_tracker/patch_select.py` (policies, quota rounding, K9 hook,
gather) and a `patch_cache` hook in fork `main.py` (attach, begin/end query, after
bootstrap and after every native rebuild); `tools/kvt_patch_select.py` (one
condition per fresh process, instrumentation, evaluation, preflight, screen,
decision report); `tools/test_kvt_patch_select.py`; mode `patch-select` in
`tools/kvt_tum.sbatch`. Static checks only on the Mac; nothing here is
runtime-verified until the preflight archive is reviewed.

How the text above was made concrete. Items marked (confirm) change outcomes
and should be confirmed or overridden before the screen is submitted.

- **Halves (confirm).** Pairs split at the middle of the GT-valid pair (end) time
  span, read from "halves of GT-valid pair times". Splitting the clip's own span
  instead would leave no_loop's second half empty (its GT covers frames 0-228 of
  1024), so no scene candidate could ever pass. Change: `pair_summary`.
- **Time.** `tracking_seconds`: synchronized wall time from model load to the
  end of the run, including frame loading, SAM, selection and gathers. Native
  runs carry the same instrumentation (timing hooks, no selection), so the 1.10
  gate compares like with like. The user lifted the 12 h cap: the driver starts
  no inference after 22 h (`--hours`), because Slurm's limit is 24 h.
- **Persistent bytes.** Final gathered K/V plus selector state (descriptors,
  ages, kept patch IDs). All-kept counts K/V only. Rebuild inputs (RGB and masks
  kept for native rebuilds) are reported per run and are identical in all arms.
- **Signals.** Model-input RGB of the admitted frame (`kf_rgb_np`); corners are
  `cv2.cornerMinEigenVal(gray, 3, 3)`; mask-cut exclusion drops pixels within 3
  px of the mask edge (Sobel, structure window, resize interpolation). Encoder
  descriptors: normalized DINOv2 patch tokens from the admitted frame's own
  query pass (anchor: bootstrap). Target patch: mask covers at least half of it;
  object boundary patch: partially covered. K10 uses the patch-centre points and
  patch-mean confidence of the admitted frame from its native dense rebuild.
- **K7/K8.** Matching is against every patch of the most recent admission
  (bounded: one descriptor set). K8: `K - K//2` slots from K7's ranking, `K//2`
  by novelty against retained descriptors, the dense anchor included.
- **K9.** Final global block = decoder 35; the admitted frame's five register
  queries; head- and query-mean softmax mass on its own patch keys, from bf16
  q/k in fp32, with synchronization, charged as `read_demand_seconds` (inside
  the query's time).
- **Controls.** Uniform = the existing `spatial_pick` (4x4 cell quotas, even
  spacing inside a cell); random = `randperm` seeded `seed * 1e6 + frame`.
- **Stability.** A clip is unstable when `|repeat / first - 1| > 0.05` for RMS,
  p99 or ATE of the native or all-kept repeat. "Gains on two clips" counts
  interpretable clips; the tie-break's total time sums tracking seconds over
  them. KV-Tracker never resets its native cache, so K7/K8 never reset.
- **Preflight scope.** Beyond all-kept and the ten selectors, the preflight also
  runs native (schedule and fidelity reference), the three sparse controls, and
  a causal prefix run of every sparse arm ending right after an admission the
  full run continues past. These check correctness only.

### Preflight gates and command

Tests (CPU and GPU: K9 hook reproduces the real decoder-35 attention output;
gathered keys equal masked dense keys through the real block), then per clip
(espresso and office, 128 frames): the native schedule, `FULL RETENTION
FIDELITY OK` (ARCTIC rtol/atol 1e-4, TUM atol 1e-5/rtol 1e-4, keyframe poses
exact), every arm replaying the schedule with native's rebuild count and byte-
identical SAM masks, exactly K sorted unique patches per admission, token and
byte accounting, every query reading the gathered cache, finite outputs,
`CAUSAL PREFIX OK`, then `preflight.json` (time/persistent ratios, selector
seconds, state bytes, flatness) and `PROJECTION` (linear cost from the slowest
condition doubled; an estimate). Submit on CAMP head (the pull runs inside the
allocation):

```bash
getquota
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch patch-select preflight"
Q="-A students --qos=students_normal -p 24g -w stuttgart"
LOG=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log
sbatch $Q --gres=gpu:1 --propagate=NONE -o $LOG --wrap="$W"
```

Expected (amended arms): tests `OK` with `REGISTER ATTENTION` and `GATHER` error
lines, `SCHEDULE` x2, 67 `RUN OK`, `FULL RETENTION FIDELITY OK` x2, `CAUSAL PREFIX OK`
x2, `PATCH SELECT PREFLIGHT OK`, `JOB OK`. Outputs in `kvt_tum_out/`:
`tum_<job>_ps_<clip>_<arm>.tar` per run, `tum_<job>_ps_inputs_office.tar`,
`tum_<job>_context.tar` (protocol, comparison, preflight.json, logs, sources) and
the EXIT trap's `tum_<job>_all_runs.tar`. Well under 1 GB expected.

### Screen command (only after the preflight archive is reviewed)

`bash tools/kvt_tum.sbatch patch-select screen <reviewed preflight context tar>`
in the same allocation wrapper. It records that tar's SHA-256, runs the 141
conditions in the frozen order, archives each, writes `overlap.json`,
`decision.json` and `comparison.json`, prints `DECISION <task> <status> <winner>`,
and writes JOB_OK only if every run completed.

## Amendment 2026-09-29, after preflight 25993 (user-selected)

Preflight 25993 passed every mechanical gate, but its selections showed the
frozen screen could not support the intended conclusions (FINDINGS "patch-select
preflight 25993"): the object budget ceil(P/4) exceeded the object, so corner
arms kept every object patch and filled with background by patch index (K1 == K4),
controls dropped ~75% of the object (a win would reflect mask awareness), K7
tracked masked-black background, and the two random seeds differed by more than
the 5% margin while deterministic repeats cannot detect that. The user chose:

- **Object task: quarter inside the object.** Eligible patches are those touching
  the SAM mask (token_drop's definition). Every arm, controls included, chooses
  ceil(P_obj/4) of them by its own rule; every object arm then adds the same 16
  background patches, evenly spaced in raster order (`background_keep`, as bg16
  in 25906). Candidates differ only inside the object. Within-object versions:
  K4 = equal quotas over eight encoder-feature clusters of the object patches
  (the 75/25 target split is superseded by the shared background); K5 = partially
  covered (mask-edge) versus fully covered patches; K7/K8 match only object
  patches of the latest admission; K1-K3, K6, K9, K10 rank/sample only object
  patches; K2 fill likewise. Uniform = cell quotas plus even spacing over the
  object's patches; random = a random quarter of them. New reference arm
  `object_dense` (every object patch plus the shared background) is reported to
  separate background removal from object pruning; it is not a claim control.
- **Noise floor: eight random seeds** (17, 29, 41, 53, 67, 79, 97, 113) on every
  clip. The claim needs RMS <= 0.95 x and p99 <= each of uniform and all eight
  seeds on >= 2 interpretable clips. Seed spread per clip is reported in
  `decision.json`; native/all-kept repeats remain a determinism check.
- **Halves: GT-valid pair span** (as implemented).

Unchanged: scene budget ceil(P/4), anchor dense, eligibility gates (1.05 vs
all-kept, halves, 1.10 time vs native, 0.60 persistent), at most one winner per
task, confirmation rules. Scene selections are unchanged except uniform, which
now uses the shared cell-quota/even-spacing helper. Screen size: 141 runs (object
clips 12 controls, scene 11, ten candidates, two repeats per clip). Preflight:
espresso and office, 67 runs including prefixes of uniform, random17 and every
candidate. Preflight 25993 results do not carry over; the amended preflight must
pass and be reviewed before the screen.
