# Loop-transformer directions for KV-Tracker — 2026-10-08

Follow-up: user selected implementation of all seven. The prepared model paths,
fixed settings and current contract-first procedure are in
[LOOP_TRANSFORMER_IMPLEMENTATION.md](LOOP_TRANSFORMER_IMPLEMENTATION.md).
The research-stage selection/profiling recommendation below is historical;
the implementation protocol supersedes it. No runtime benefit is established.

Scope: research and implementation proposals, not a selected training run or
cluster submission. Primary papers inspected online on 2026-10-08; local source
and recorded outcomes inspected read-only. All proposed geometry adaptations
below are hypotheses, not results reported by those papers. Most source evidence
is language modeling; LoopVL is vision-language modeling, RAFT is optical flow.
None establishes a drop-in pretrained recurrent Pi3 pose tracker.

## Evidence and target accounting

Canonical experimental evidence is in [FINDINGS.md](FINDINGS.md), entries
26168/26170/26171 and the native 25996 timing split. Do not substitute paper
speedups for measurements of this tracker.

- LoopCD-inspired 26168 improved the 256-frame office prefix, but full-sequence
  26170 guidance produced 17.157 cm ATE versus native 6.190 cm (2.77x), with
  essentially unchanged FPS. Artifact review confirmed broad distortion, not a
  demonstrated single bad seam. Reduced24 queries in 26168 were substantially
  inaccurate. These were distinct-layer contrast/truncation analogues, not
  weight-shared recurrent models.
- Query26171: native 111.1778 s, query 98.6912 s (88.8%); graph timed
  99.9369 s, identical saved trajectories, one ordered timing observation.
  Native peak allocated 4,558,973,440 bytes; KV 1,415,577,600 bytes.
- Amdahl inference using those recorded totals: if all query time is accelerated
  by q, total speedup is 1/(0.1123 + 0.8877/q). Half query time gives about
  1.80x; 2x total FPS requires about 2.29x query acceleration, assuming unchanged
  non-query time. Decoder-only savings have a smaller ceiling; encoder,
  decoder and camera-head shares within query are not yet measured.
- Halving that KV alone removes 707,788,800 bytes, about 15.5% of native peak,
  assuming all other live allocations stay unchanged. It does not halve total
  GPU memory. Weight sharing, cache sharing and transient reductions must be
  accounted separately; teacher must be absent from deployment measurements.
- Object25996 queries occupied 57–59% of total time; outside-Pi3 costs occupied
  27–38%, including SAM. Object end-to-end doubling is much harder than doubling
  Pi3 query throughput. Timing scopes must include the same segmentation work.

## Source constraints in this checkout

`kv_tracker/kv_tracker/token_drop.py:forward_kept` alternates frame-local even
blocks and cached global odd blocks. It stores separate K/V in `model.cache[i]`
and concatenates the final two block outputs before camera/point/conf decoders.
`patch_select.py` explicitly asserts 36 decoder blocks. Archived native Pi3 source
at `cluster_results/kvt_tum/tum_25939_context/context/pi3.py` independently shows
36 distinct large-decoder blocks, 18 global cache slots and doubled-width head
inputs. That archive is historical evidence, not proof of a future runtime source.
Pi3 is imported from its environment by `pi3_utilts.py`; it is not vendored here.

Consequently, model surgery needs a pinned editable Pi3 source in the model-fork
workflow, plus native/kept-path and head compatibility checks. Do not represent
replacing a Python loop or aliasing stock cache tensors as a trained recurrence.
Preserve patch coordinates/RoPE, five register tokens and the two-state readout.

## Seven implementation ideas

### 1. Train an elastic-depth geometry decoder (first speed candidate)

Source: [LoopFormer](https://arxiv.org/html/2602.11451v1) trains time/step-size
conditioned shared blocks with long/short trajectory supervision and consistency.

Proposal: keep DINO and dedicated entry/exit layers initially frozen. Replace a
middle decoder region with a shared stack containing complete local/global pairs.
Condition each recurrence on normalized progress and step size. Train one full
schedule and one shorter schedule against frozen native Pi3 features and pose
outputs; retain supervised geometry where suitable data already exists. Distill
relative transforms and confidence-weighted geometry in the same gauge, rather
than independently Sim(3)-aligning every frame. Add a two-state readout adapter
for the existing heads if necessary.

Benefit: actually learns a short route, addressing the failed untrained24-block
truncation. Fewer block executions can reduce query time; shared blocks reduce
decoder weights. Equal executions alone do not save FLOPs.

Cost/risk: training required; intermediate features and gauge can drift. DINO
and heads limit both speed and memory savings. Pilot: one fixed short schedule
versus native and the trained full schedule, with identical inputs/keyframes.
Do not initially add routing, cache sharing or contrast guidance.

### 2. Relaxed recurrence: shared weights plus depth-specific low-rank residuals

Source: [Relaxed Recursive Transformers](https://arxiv.org/html/2410.20672v2)
uses tied weights plus depth-wise LoRA and studies initialization/distillation.

Proposal: group like-role local/global blocks across depth. Initialize each
shared projection from its group mean, and depth-specific low-rank residuals
from truncated SVD of original-minus-shared weights. Keep small depth-specific
normalization parameters. Distill with the original number of executed blocks
first, preserving all cache slots and the final two features.

Benefit: decoder checkpoint/weight memory reduction with a closer initialization
to the original model than pure tying. This is the strongest first candidate
when model memory is the priority.

Cost/risk: training and adapter kernels; original-minus-mean differences may not
be low rank. Separate adapters cannot be merged into separate full matrices at
deployment without losing the storage benefit. Same-depth FLOPs may increase;
this does not automatically reduce KV memory. Pilot: one predetermined sharing
layout/rank, quality and actual unique weight bytes, then measured latency.

### 3. Share historical KV across recurrences

Source: [Mixture-of-Recursions (MoR)](https://arxiv.org/html/2507.10524v1)
distinguishes recursion-wise caches from first-recursion KV shared across loops.
It also documents accuracy tradeoffs, especially with selective routing.

Proposal: after a trained shared decoder works, teach matched global modules
in successive recurrences to attend to one common historical K/V bank. Keep
current-frame hidden states evolving; preserve coordinates and register tokens.
Train using this memory representation in both rebuilds and queries. Start with
two recurrences sharing each bank, retaining dedicated entry/exit caches.

Benefit: approximately halves cache bytes in the affected paired recurrence
region, without deleting historical frames or patch coverage. Could reduce
rebuild work, but every query still reads memory at each recurrence.

Cost/risk: equal weights do not imply equal K/V because hidden states differ.
Direct stock-cache aliasing is a distribution mismatch. Whole-cache and total
memory reductions are smaller when untied slots remain. Pilot: trained same
student with separate versus shared caches, count unique underlying storage,
verify masks/RoPE/query-rebuild consistency and full-trajectory quality.

### 4. Frame-level adaptive recurrence with separately calibrated halting

Sources: [Universal Transformers](https://arxiv.org/abs/1807.03819) and
[Adaptive Depth](https://arxiv.org/abs/2607.20519). The latter separates the
quality of recurrent trajectories from the stopping/readout policy.

Proposal: first train useful short/full outputs, then calibrate a causal exit
decision using pose change between recurrences plus geometric residual against
the existing map. Easy frames take a short route; ambiguous/occluded frames
continue to the already-trained longer route. Keep bank updates at a fixed
validated schedule in the first pilot. An exit must represent a valid endpoint,
not an arbitrary intermediate feature fed to the stock camera head.

Benefit: lower mean query cost while preserving extra compute for difficult
frames; potentially better tails at matched average compute.

Cost/risk: a confidently wrong pose can appear converged. Repeated camera
readouts/SVD and host synchronization may erase savings. Use fixed graph variants
if justified by profiling. Pilot: compare adaptive execution with a fixed-depth
student at matched mean time; report maximum/p99 latency and drift as well as FPS.

### 5. Token-selective refinement, retaining exited tokens as context

Sources: [MoR](https://arxiv.org/html/2507.10524v1) and the recent preprint
[T-LoopFormer](https://arxiv.org/html/2609.15160v1). They allocate different
recurrence budgets to tokens and physically gather active rows to save work.

Proposal: all patches receive an initial geometry pass. Later loops update only
an active subset chosen for pose/geometry relevance, while retaining inactive
tokens' most recent states as attention context and for dense readout. Preserve
all five registers and explicit original patch indices. For objects, include
foreground boundaries and spatial coverage; for scenes, train relevance rather
than assume an object-mask rule transfers.

Benefit: skips repeated MLP/Q computation on easy patches without the previous
checkerboard strategy's permanent removal before the encoder. All output patches
retain a feature; memory savings require separately reducing later-loop cache
storage, not merely masking execution.

Cost/risk: proposed inactive-context retention differs from papers' exact routing
semantics and requires training. Gather/scatter and ragged kernels may outweigh
savings; masking without compact execution saves no dense GEMM work. Pilot: one
fixed active budget versus equal-cost uniform routing, with the same trained
backbone, and measured active-row execution rather than theoretical FLOPs only.

### 6. Small recurrent pose refiner between full geometry updates

Source: [RAFT](https://arxiv.org/abs/2003.12039) separates feature extraction
from recurrent updates over reusable correlation evidence. This is a related
recurrent-vision design, not a looped-transformer result.

Proposal: use Pi3 for anchor/map updates, then use current-frame encoded features,
map correspondences and a small shared transformer refiner for intervening pose
queries. Iteratively predict an SE(3) residual, initialized by a causal motion
estimate. Every frame still uses fresh image evidence. Train against native
tracking poses plus available geometry/GT; initially keep full updates at the
existing schedule to isolate the query change. Reuse local/sparse correspondence
evidence rather than introduce RAFT's dense all-pairs volume.

Benefit: strongest structural possibility for large FPS gains: ordinary frames
avoid most of the geometry decoder. A dedicated pose tracker may also improve
temporal stability. If the full Pi3 weights remain resident, there is no weight
memory saving; a resident full cache likewise limits memory reduction.

Cost/risk: largest subsystem change, training data and recovery behavior needed;
map occlusion and object foreground must be respected. Pilot: evaluate the
refiner inside tracking with fixed full updates, including update-frame cost and
the same SAM work. Offline pose matching alone cannot establish reduced drift.

### 7. Nested local/global geometry loops

Source: [LoopVL](https://arxiv.org/html/2609.38426v1) combines repeated local
module updates with a slower coupled high-level update. It is trained for VLM
tasks; transferring that hierarchy to geometry is our proposal.

Proposal: train a schedule with several cheap frame-local refinements followed
by one global map-attention update, repeating the cycle only as needed. Pi3's
existing local/global alternation supplies a natural boundary. Preserve query
and rebuild schedules consistently and train the readout for the new sequence.

Benefit: fewer expensive historical-attention visits and cache stages, while
local loops refine image evidence; may combine speed and memory benefits if
global stages really dominate and share storage.

Cost/risk: local loops cannot recover missing cross-view evidence; replacing
global passes can hurt scale/gauge and relocalization. Pilot: one fixed nested
schedule versus an alternating recurrent student at matched executed blocks;
measure global attention share first. This is more speculative than ideas1–3.

## Recommendation and review gates

For FPS: prioritize idea1; idea6 is the larger alternative if profiling shows a
shortened geometry decoder cannot reach the target. For memory: idea2 first,
then idea3 only after the trained recurrence passes quality. Idea4 follows
validated short outputs. Ideas5/7 are later options, not a proposed control grid.
Eventual combined candidate: trained short recurrence + shared weights + shared
historical memory, then graph execution. Do not assume gains multiply.

The smallest next implementation is a timing-only encoder/decoder/camera breakdown
at the existing query boundary, including upload, output readout, capture setup
and rebuild. No additional expensive evaluation is selected by this survey.
Use that breakdown and pinned Pi3 source to choose one architecture and a bounded
training-data/compute budget before preparing CAMP training.

For any chosen implementation: syntax/whitespace checks locally; Linux tensor
contracts and cached/uncached semantics on CAMP; then a small pilot reviewed before
full evaluation. Use the prior three-metric 5% non-regression rule for ATE/RPE/p99
against fresh native at identical pixels/masks/keyframes and GT associations.
An accuracy-oriented success must improve the intended metric without failing
the agreed others. Acceleration success is measured total-loop >=2x FPS with the
quality gate; memory success is <=50% peak allocated with reserved/process memory
also reported. Record smaller useful gains separately, not as passing those
targets. Measure warm and cold/capture-amortized timing with repeated paired runs
on the same GPU. Objects must include equal SAM work; sparse TUM GT cannot certify
unobserved route sections. Include one held-out scene/object after the pilot,
full-length office drift and raw transition evidence, not just a256-frame prefix.

## Designs not promoted to another immediate retry

[LoopCD, Liu et al.](https://arxiv.org/html/2610.02185v1) and
[LoopCD, Yu et al.](https://arxiv.org/abs/2609.24196) are distinct papers with
the same method name. They guide language predictions using earlier/later
recurrent states. Neither validates subtracting arbitrary Pi3 layer features.
Our full-sequence failure remains the relevant tracker evidence. Revisit only
after a trained geometry recurrence exists, with a bounded, geometry-supervised
correction and a separate full-trajectory gate; do not retry the same fixed
guidance on the strength of a language result.

[LoopSpec](https://arxiv.org/html/2609.17184v1) uses early states to draft
future language tokens and full-depth rejection-sampling verification. Continuous
poses have no corresponding exact token-acceptance rule, and future camera
frames have not arrived. A pose proposal verified geometrically is an approximate
tracking design, not a lossless transfer of speculative decoding. Multi-request
depth batching likewise targets throughput and does not demonstrate lower
single-stream frame latency.
