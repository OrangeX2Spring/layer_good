# Dynamic retention bias in causal KV compression

Initial plan, 2026-10-16; implementation subsequently requested by the user.
The source-audited executable protocol and its narrower within-source-frame
interventions are in [DYNAMIC_CACHE_BIAS.md](DYNAMIC_CACHE_BIAS.md), which supersedes
the implementation/next-action statements below. No CAMP run has been performed.
Success: a reproducible, budget-matched test of retention bias AND
its effect on camera pose, with a stop rule before method development.

## Evidence and scope

Primary sources reviewed on 2026-10-16:

- [InfiniteVGGT §3.3–3.4](https://arxiv.org/html/2601.02281): negative cosine
  similarity to the mean of normalized candidate keys, per layer/head, with
  protected anchors. Dynamic content having high scores remains a hypothesis.
  [Official code](https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT) exists;
  exact revision, checkpoint and executed selection path still need auditing.
- [Evict3R v2](https://arxiv.org/html/2509.17650v2): attention-based selection
  with exposure/length normalization, not the same diversity rule. Evaluates
  pose on Sintel and TUM-dynamics. Official code availability/revision pending.
- [FrameVGGT §5.3–5.4](https://arxiv.org/html/2603.07690): Bonn depth and
  TUM-dynamics pose; frame-coherent memory is a relevant alternative explanation
  to motion bias (fragmentation/coverage). These sections do not establish a
  dynamic-mask-conditioned retention mechanism.
- [GHOST §5 and limitations](https://arxiv.org/html/2605.15852): Bonn depth,
  geometry/confidence-based scoring; explicitly discusses static-scene assumptions
  and possible over-retention of dynamically changed regions. Thus the broad
  concern is already anticipated. No claim of an exhaustive novelty search.

Do not claim that dynamic benchmarks have been ignored. RayMap3R/MoRe and other
motion-aware approaches still need full-method review before novelty claims.
Do not transfer conclusions to KV-Tracker's object bank: foreground is useful
for object tracking, whereas this test targets camera pose in dynamic scenes.

## Hypotheses, separated

H1: the selectable cache over-retains moving-region patches relative to its
actual pre-eviction candidate distribution.
H2: changing which patches survive, at the same capacity, improves camera pose;
the effect exceeds an otherwise matched random intervention.
H3 (later): a causal predicted motion signal can recover that benefit at a net
memory/latency advantage. Oracle masks cannot establish H3.

## Gate 0: feasibility and frozen manifest

Start with InfiniteVGGT only. Inspect official source and callers; record revision,
checkpoint, preprocessing, anchor/special-token rules, insertion/eviction timing,
budget semantics, precision and attention backend. Instrument stable source-frame
and patch IDs through every gather, separately for each layer/head.

Audit available CAMP data through user-provided listings. Do not assume either
Sintel or Bonn supplies verified moving-pixel GT masks in the installed copy.
Object/semantic labels are not motion labels. If deriving masks from GT flow,
depth and camera motion, document coordinates, invalid/occluded pixels and a
fixed residual threshold; call these GT-derived masks. Predicted masks are not GT.
If Bonn has no trustworthy motion labels, retain it for pose/depth evaluation,
not the primary H1 test. Freeze exact sequence IDs before seeing model errors.

Target pilot: two Sintel sequences, two Bonn sequences and two ScanNet controls,
subject to mask/pose availability. Select by annotations and camera motion, not
by observed failures. Match input length, resolution and stride across arms.
Record dynamic coverage and camera motion; cross-dataset contrasts are descriptive,
not causal. Do not silently substitute a new dataset or add a sweep.

One 24-GB smoke run must show an eviction-active interval and a full-cache control
on the identical clip. Choose one capacity from source defaults and feasibility,
then freeze it; shorten all paired clips if needed. If no meaningful interval fits
with full cache, report this blocker instead of calling a sliding window full cache.
Environment/data preparation is outside the proposed one-to-two-day GPU budget.

## Gate 1: minimum instrumentation

For each eviction save candidate/retained counts, source IDs, scores and patch
motion coverage, layer/head, protected status and byte counts. Use exactly the
image crop/resize/padding transform for masks; pool valid coverage into patches.
Exclude special tokens, anchors and other protected tokens from H1 denominators;
report their motion coverage separately. Unknown labels stay unknown.

Let C be selectable candidates and K their survivors at one event. With patch
motion coverage d_i in [0,1], compute p_C=mean_C(d_i), p_K=mean_K(d_i), enrichment
E=p_K/p_C and excess p_K-p_C. E is undefined when p_C=0. Also report conditional
dynamic/static retention rates and resident-cache occupancy over time. Current
frame mask area alone is the wrong denominator for a historical cache.

Aggregate within sequence first; report all sequences and layer/head breakdowns,
stratifying source age/frame where possible. Repeated heads/tokens/events are not
independent samples. A tiny pilot supports a decision, not population significance.
Persist compact summaries and a small auditable event sample, not all K/V tensors.

Checks before pilot: unmodified versus logging-only trajectory equivalence within
declared numeric tolerance; retained IDs map to correct masks; identical K/V gathers;
capacity/protected-token invariants; prefix causality. Runtime checks run on CAMP.

## Gate 2: paired interventions at one budget

| Arm | Purpose |
|---|---|
| Full cache | Same backbone/checkpoint with eviction disabled, same clip |
| Native compression | Original ranking and budget |
| Oracle motion-aware | Static-first ranking, original score as tie-break, same capacity |
| Matched random replacement, three fixed seeds | Replace the same count in each layer/head/source-frame stratum, independent of motion |

Oracle arm only changes selectable historical cache entries. It preserves anchors
and special tokens and backfills by original score if static candidates cannot
fill capacity. No current RGB masking, extra frame dropping or pose GT input.
Record replacements and achieved coverage; no-intervention strata are explicit.
Random replacement selects outgoing/incoming candidates without motion preference
and keeps the oracle's replacement count and source-frame allocation where feasible;
record any infeasible strata rather than silently changing the control.

Six clips imply 36 runs (three deterministic arms plus three random runs per clip).
Run and REVIEW one dynamic clip first; only then run the remaining manifest.
If mask availability reduces the primary set, distinguish H1/H2 evaluation sets.

Report ATE (absolute trajectory error: translation RMSE after one sequence-wide
Sim(3) alignment), translation/rotation RPE at a fixed interval, valid pose fraction
and failures. No per-window realignment. Report native-minus-full absolute error
and ratios only where baseline error is non-negligible. Never drop failed runs.
Log actual KV bytes, peak allocated/reserved GPU memory, and latency with debug
logging disabled. Oracle timing excludes no hidden segmentation cost: it is only
an upper-bound diagnostic, not deployable efficiency evidence.

## Decision and stop rules

Proposed practical screening gates, fixed before runtime, not statistical claims:

- H1 signal: sequence-median E >= 1.2 and positive excess in both primary dynamic
  clips, not explained solely by protected anchors or source-frame allocation.
- H2 signal: oracle ATE improves >= 10% over native in both primary clips and beats
  the median matched random arm, without worse pose coverage or gross RPE damage.
  Inspect all random seeds, not just their median. Static controls should remain
  within 5% ATE or a predeclared absolute numeric tolerance near zero.
- Native-versus-full is required to claim compression-induced degradation. If
  native does not degrade, describe oracle gains as motion-aware selection benefit,
  not amplified pose contamination.
- H1 without H2: retention preference is not evidence of harmful pose contamination;
  stop the proposed causal story. H2 without H1: consider filtering utility only;
  do not keep the diversity-bias explanation. Neither: close this hypothesis.
- Both: repeat on Evict3R at matched byte capacity and additional held-out sequences
  before saying the defect is systematic across compression methods. GHOST and
  FrameVGGT become relevant method baselines, not automatic extra pilot runs.

## Only after the diagnostic passes

Implement one causal motion signal and eviction-only policy first, without training
the geometry model. Compare semantic-only versus actual-motion cues so stationary
objects are not removed merely for belonging to a movable class. Include signal
generation in end-to-end memory and timing. Do not add gating simultaneously:
historical eviction cannot undo already-contaminated current-frame inference, and
global shared-token suppression may harm dynamic geometry/object tracking. Test
gating as a separate ablation only if eviction leaves a demonstrated residual.

At a fixed budget, success means improved pose at similar memory; lower memory at
matched accuracy needs a later budget comparison. Neither speed nor memory gain
is guaranteed by swapping rankings. Integrate with semantic KV cache only after
this distinction and the camera/object task boundary are verified.

## Deliverables and next action

Save manifest, code/checkpoint identities, exact configs, source pixels/masks with
provenance, trajectories, compact retention logs, resource measurements and failures.
First GPU day: reviewed smoke and native/full diagnostics. Second GPU day: oracle
and random interventions if earlier gates pass. This is a conditional pilot estimate,
not a guarantee of a paper-ready conclusion within two days.

Next: local official-code/checkpoint/mask-source audit and frozen manifest, then
prepare minimal instrumentation in the model fork and a tracked CAMP driver.
All SSH, submissions and transfers remain user-operated. No commands are invented
before the wrapper exists. No runtime evidence exists for this plan yet.
