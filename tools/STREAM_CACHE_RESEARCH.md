# StreamVGGT: three cache-compression research pilots

User selected implementation of all three on 2026-09-28 and authorized publication
for CAMP. All three bounded pilots have now run and been reviewed: temporal 25976, context
25977, refresh 25978. Runtime gates passed; none meets its full scientific gate.
Results and caveats: tools/FINDINGS.md, dated 2026-09-28. No further run queued.
The next authorized run is the [context diagnostic](STREAM_CACHE_CONTEXT_DIAGNOSTIC.md),
followed by a main experiment only after reviewing its evidence.
Implementation audit supersedes broad method-rejection conclusions; see
[STREAM_CACHE_RESEARCH_AUDIT.md](STREAM_CACHE_RESEARCH_AUDIT.md) before rerunning.
Model code lives in the StreamVGGT fork at
`src/streamvggt/models/research_cache.py`; the parent only orchestrates experiments.
Native weights, native entry points and other hosts are unchanged.

## Completion and execution boundaries

Local completion: algorithms, explicit controls, archived metrics/provenance,
remote contract tests, pilot entry point, syntax/JSON/shell/whitespace checks.
Runtime completion requires successful CAMP tests and inspected pilot archives.
Scientific success is separate: an accuracy/resource advantage over the matched
controls. No full evaluation, dataset download, image modification, or automatic
second experiment is dispatched. Each submission runs one method on one sequence.

The original pilots used fork `380ace4` (base `05682bc`). The audit overhead fixes
were published on `main` at `9931fbe`; the current parent gitlink additionally
includes the passive context observer. The gitlink is the authoritative revision.
Pull the parent and initialize its pinned
submodule inside an allocation using the wrapper below. Recovery paths are the
fork commit and tracked `tools/stream_cache_research*`, not a temporary worktree.

## Shared model and measurement contract

- Frozen StreamVGGT; batch=sequence=1; FP32 as in the established sweep runner
  (not the demo’s outer mixed-precision autocast). New raw ragged K/V
  uses native Q/K normalization, integer 2D RoPE and SDPA. Special embeddings
  distinguish original frame zero from later frames, including historical refresh.
- Native camera-head history is retained and grows. Dense depth/point heads still
  execute. Bounding aggregator state is not bounding total GPU memory.
- Every condition runs in a separate worker. The existing independent native
  fidelity gate runs before conditions. Each research worker also compares its
  no-intervention path against the established sparse adapter for at least six
  frames with a three-frame budget, exercising eviction as well as appends.
- `state_bytes` includes K/V, positions, representative masses, descriptors and,
  for refresh, saved encoder inputs and last-context descriptors. These are actual
  tensor payloads, not quantization estimates. Python object overhead and allocator
  overhead are not tensor bytes; measured CUDA peaks are reported independently.
  K/V is materialized without retaining the unused Q projection through views.
- Report `aggregate_seconds` (encoder + grouping + alternating attention + unmerge),
  `head_seconds`, `allocation_seconds`, `refresh_seconds`, total/p50/p95 seconds,
  pre/post-maintenance state, camera payload, total persistent tensor payload,
  peak allocated and peak reserved. Synchronization makes boundaries explicit.
  Timing includes policy and oracle overhead, excludes input loading/output export.
  Single-run timing is diagnostic, not a publication-quality speed estimate.
- Predictions are emitted before maintenance/refresh. No old pose or camera-head
  entry is replayed or corrected. All refresh context has arrived by the current
  frame. Past representations can incorporate later *observed* frames for future
  queries; this is an intentional change from native immutable causal K/V.

## 1. Object-conditioned context tokens

Config: `stream_cache_research_context.json`. Default pilot: first 128 post-offset
espresso frames at width 518; ketchup and box are explicit alternative scene args.
Use the existing saved SAM masks and identical object-masked RGB in every arm.
These background encoder tokens therefore describe the masked input's learned
context, not unmasked background appearance. No new segmentation is performed;
mask generation cost is excluded equally in every arm. This is conditional on
available masks, not an end-to-end segmentation speed claim.

After the dense image encoder, retain every patch touching the foreground. Choose
64 spatial farthest-point seeds from background patches (or all if fewer exist).
Background tokens are assigned to their nearest seed using normalized squared
spatial distance, optionally plus 0.25 times cosine distance. Seeds own their
groups, guaranteeing exactly 64 nonempty groups. The assignment has no future
information and the coefficient/budget are frozen for this pilot.

| Arm | Background representation |
|---|---|
| dense8 | Every patch; no merge/unmerge |
| spread64 | Seed patches only; no averaging |
| spatial64 | Mean of spatially assigned background tokens |
| appearance64 | Mean with spatial + appearance assignment |

Sparse tokens enter *both* frame and global aggregator blocks. Group RoPE uses the
seed's real integer patch position. No attention-mass logit bias is added. Before
the existing heads, each dense patch receives its group's output plus its original
encoder residual relative to the group's input, independently for frame/global
features. This reconstruction is shared by the three sparse arms. It preserves
head shapes, not a guarantee of geometry fidelity; the encoder and heads stay dense.
Cache retains the sparse rows. All arms use anchor + recent history, budget eight.

Log exact dense-to-sparse maps, representative positions, processed patch counts
and stored bytes. The cross-condition gate requires identical sparse token counts
and persistent payloads between drop/spatial/appearance arms, and actual reduction
on at least one frame. This isolates averaging/assignment from token-count savings.

`--object-gt` staging is enabled only for this wrapper's context mode. It derives
camera-in-object transforms using the established ARCTIC algebra/units/offset,
reproduces the archived full KV-Tracker ATE/RPE before proceeding, and preserves
the raw GT files plus checksums in `run/inputs`. `object_metrics.json` reports
Sim(3) ATE, adjacent-frame translation RPE/RPE p99, and rotation Frobenius RPE
(not degrees). This is a masked pose-head readout, not world-camera tracking or
validated reconstruction geometry. Never invent camera timestamps for this GT.

Scientific gate: merging should outperform spread64 at identical tokens/bytes,
while keeping ATE, translation RPE and rotation RPE within +5% of same-job dense8
and reducing measured time and peak allocated memory. Inspect p99 as an additional
tail guard; no promotion if a gain hides substantially worse tails. Espresso first,
then review before ketchup/box confirmation. Any numerical-path sensitivity must
be characterized before interpreting small differences. Unseen-sequence and dense
geometry evaluation are required before a paper claim.

## 2. Temporal–spatial allocation

Config: `stream_cache_research_temporal.json`. First 256 office frames, width 308.
Byte ceiling equals eight complete native frame states, including auxiliary
tensors. All algorithms protect the two newest frames at full detail. Anchor
presence is protected but its spatial detail can be compressed.

When over budget, consider each older frame: halve its patch count down to a floor
of 16, or evict it when already at the floor (except the anchor). Group by spatial
farthest-point seeds, merge raw K/V with representative-mass-weighted averages,
and retain the seeds' integer RoPE positions. Special tokens are never merged.
Previous masses are carried forward so repeated compression does not give a small
group the weight of a large group. No full-resolution backup or expansion exists.

The coverage variant chooses the action minimizing
`(1 - max cosine(frame_descriptor, other_retained_descriptors)) / bytes_freed`.
Descriptors are normalized pooled encoder features. This is feature-coverage
redundancy, not measured geometric view coverage. The saved-byte denominator
includes the exact removed KV/position/mass payload; eviction also frees the
frame descriptor. Recompute candidate scores after each action until within budget.

Controls: recent8, oldest-first progressive compression (`temporal_fifo`), and
largest-patch-count-first compression (`temporal_uniform`, oldest tie-break).
They isolate progressive representation from feature-based budget allocation.
No threshold sweep, trained selector, packed quantization or unbounded seed bank.

Gate: identical byte ceilings; every post-maintenance state within ceiling;
recent detail and anchor preserved; matching KV/position/mass lengths; real
compression actions observed. Actual occupancy can differ because representations
are discrete. Report occupancy and camera/peak bytes; do not label equal ceilings
as exact equal occupancy. Transient arriving-frame and merge allocations are included
in measured peak, with pre-maintenance payload logged.

Scientific gate: coverage must improve the accuracy/resource frontier over FIFO,
uniform and recent8, including local-error tails around compression events. First
compare within the same tensor-byte ceiling; accept a paper efficiency claim only
with end-to-end time/peak evidence. A successful office pilot still requires
independent sequence validation and comparison with existing pruning/INT4 methods.

## 3. Budgeted contextual refresh

Config: `stream_cache_research_refresh.json`. First 256 office frames, width 308.
Native StreamVGGT does **not** jointly rebuild old frames. This experiment is a
new context-repair operation, not a faithful port of KV-Tracker append-vs-rebuild.
The group granularity in this first implementation is one historical frame's
tokens. Sub-frame refresh and learned selection are not implemented.

Retain anchor + recent history, budget four. Store each retained frame's original
post-encoder tokens including its correct special embeddings. Every eighth frame,
after predicting/evicting, recompute one retained historical frame through the
alternating blocks. At each layer, exclude that frame's old K/V and insert its new
K/V once; all other frame caches stay untouched. No DINO replay, dense recompute of
unselected frames, old-camera update, or duplicate historical slot. Full refresh
uses the same operation on every historical frame, oldest first.

Selection score:
`age_since_refresh * max(0, 1 - cosine(current, last_refresh_context)) * affinity`,
where `affinity = clamp((1 + cosine(current, frame_descriptor))/2, 0, 1)`.
The last context starts as the frame's own descriptor. This is a fixed causal
heuristic to test the localized-staleness premise, not an already learned method.

| Arm | Purpose |
|---|---|
| recent8 | Larger native-history accuracy/resource reference |
| recent4 | Native small cache, without refresh seed storage |
| refresh_none | Same query path, with seed storage; exposes auxiliary cost |
| refresh_random | One random historical frame per trigger; fixed seed |
| refresh_selective | One score-selected historical frame per trigger |
| refresh_oracle | Rank independently refreshed candidates by relative squared KV change |
| refresh_full | Refresh all retained historical frames per trigger |

The oracle scores candidates against the same frozen state before selecting;
candidate scores do not mutate the live cache. Its ranking is a **KV-change oracle,
not a pose oracle or guaranteed upper bound**. Scoring recomputes every candidate
and its full cost is included, so exclude this diagnostic from deployable-speed
claims. Selected groups are then recomputed/committed in ascending frame order.

Gate: refresh-none reproduces recent4 pose encodings; all refresh variants have
identical persistent payloads; selected-only mutations, no token growth, deterministic
random control and prefix invariance in remote CPU tests; triggers actually fire.
Scientific gate: first inspect whether full/oracle refresh benefits the small-cache
trajectory at all. If not, reject this refresh definition rather than training a
selector. If useful, selective must beat random and approach the larger-cache
quality while improving measured resources relative to recent8. It will usually
cost more compute than recent4; the trade-off must be measured, not assumed.

## Remote tests, evidence and first command

The existing job launcher discovers `test_*cache*.py`, including the original eleven CPU
contracts for weighted merging, context identity/shape, temporal bytes/positions,
causal prefixes, refresh isolation/oracle isolation, disabled fidelity and RNG.
Do not run project modules/tests on the Mac. The original tests passed as part of the 74-test suite in all three pilot jobs.
Four audit contracts (including a CUDA grouping parity check) are added locally;
their remote execution remains pending.
Inspect `RESEARCH FIDELITY OK`, each `RESEARCH CONTRACTS OK` / `RUN OK`, camera or
object evaluation, `RESEARCH CROSS-CONDITION GATE OK`, `JOB OK`, archived exit 0.
Read `research_gate.json`, every config/event/summary, per-frame pose errors,
input/GT hashes, source snapshots, checkpoint identity and source revisions.

After publication, the first user-operated pilot can be submitted from CAMP head
with an allocation-wrapped pull (no checkout updates on head):

```bash
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$W && bash tools/stream_cache_research.sbatch temporal"
sbatch -A students --qos=students_normal -p 24g -w stuttgart \
  --gres=gpu:1 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/cache_research-%j.log --wrap="$W"
```

Resource target follows existing jobs; queue availability is unverified. Return the
job ID before the next command. The next gate is remote CPU contracts + native and
research fidelity + bounded office pilot review, not a full evaluation. Alternative
wrapper methods are `context` (espresso default) and `refresh`; do not submit all
behind an unreviewed first pilot. Archives use existing
`/mnt/projects/gr/3DRecon/stream_cache_out/cache_streamvggt_<job>_*.tar` naming.
Transfer by the field-notes rsync route and extract each archive separately.
No runtime artifacts or environments have been created by this local milestone.
