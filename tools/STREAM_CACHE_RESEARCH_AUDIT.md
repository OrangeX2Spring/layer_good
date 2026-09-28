# StreamVGGT research audit: implementation before interpretation

## Context direction and prior-art correction (2026-09-28)

User now prioritizes improving context-token compression and identifying a
contribution beyond Co-Me. This supersedes fixed-anchor/fixed-recent refresh as
the immediate research direction. The user subsequently authorized the
[context diagnostic](STREAM_CACHE_CONTEXT_DIAGNOSTIC.md) followed by a main
experiment after evidence review. Diagnostic implementation is prepared; remote
contracts and scientific signal remain unverified. No job submitted yet.

Primary-source check rules out several easy novelty claims:

- [Co-Me §3](https://arxiv.org/html/2511.14751v2): predicted-confidence merging,
  dense splitting and attention-mass correction already exist.
- [LiteVGGT](https://arxiv.org/html/2512.04939v1): pixel gradients plus token
  variance preserve important geometry; cached merge indices reduce overhead.
  The earlier generic boundary/variance proposal overlaps this work.
- [HTTM](https://arxiv.org/abs/2511.21317): training-free head-specific temporal
  merging is already proposed.
- [Spark3R](https://arxiv.org/abs/2605.06270): asymmetric query/KV reduction is
  already proposed. Abstract inspected; implementation not audited.
- [RegimeVGGT](https://arxiv.org/abs/2606.18439): layer-dependent reduction with
  pose-path/spatial protection is already proposed. Abstract inspected only.
- [GRKV](https://arxiv.org/abs/2605.31105) and
  [Attention Matching](https://arxiv.org/abs/2602.16284): preserving attention
  outputs through training-free KV compaction is also prior art outside 3D.

Therefore neither no-training, protecting edges, protecting camera tokens nor
attention-output matching alone is an established new contribution. This is a
targeted literature screen, not an exhaustive novelty certification.

Candidate research question: can a small set of causal camera/register attention
probes predict which context merges damage subsequent pose stability, and guide
fixed-budget group refinement cheaply enough to retain the observed compute gain?
This differs from confidence ranking as an objective, but its novelty and utility
are unproven. Probe error is a local attention surrogate, not a guaranteed bound
on final pose or unseen future queries.

First diagnostic, before building a new selector: compare dense and compressed
camera/register attention outputs on the same available layer inputs and historical
state. Keep causal observations only. Relate the measured discrepancy to the
already recorded local pose failures; include simple feature-variance and confidence
controls. Dense probe computation is diagnostic overhead, not deployable speed.
It needs new remote instrumentation because existing archives do not contain Q/K/V
or intermediate camera-token activations. No future GT may enter online selection.

If the signal works, a candidate implementation would keep a short dense prefix,
then spend a fixed representative budget on splits that reduce measured probe
distortion, coarsening low-impact groups elsewhere. Measure the dense-prefix and
selection overhead and persistent state. It cannot retain our old architecture’s
speed by assumption; future queries and dense geometry require separate validation.
Do not retain a hidden full-resolution history as a free fallback.

Separately, missing log-mass correction is an implementation baseline to test,
not the proposed novelty. Keep it separate from partition/probe changes. Existing
saved-SAM masked-input setup does not establish mask-free generalization; benchmark
unmasked inputs and include segmentation cost before such a claim. Co-Me and the
relevant training-free methods must be compared at matched measured resources,
with pose tails and dense geometry, before claiming superiority.

User requested autonomous re-investigation on 2026-09-28. Scope: audit the three
published pilots, fix demonstrable implementation overhead without changing their
policies, then use bounded runtime gates to distinguish execution problems from
algorithmic limitations. Cluster execution remains user-operated. No new training,
dataset download, threshold grid or full evaluation is authorized by this audit.

## Evidence and revised conclusion

Reference parent `a3731a7`, model `380ace4`; archives:

- `cache_streamvggt_25976_tGhD4K`: temporal, office.
- `cache_streamvggt_25977_keHE7J`: context, masked ARCTIC espresso.
- `cache_streamvggt_25978_XUBmmf`: refresh, office.

Each archive and separate extraction is in `cluster_results/stream_cache/` on the
Mac. Runtime provenance and hashes were reviewed. Canonical numerical evidence is
in FINDINGS.md, especially **Implementation audit of 25976–25978**.

The old errors describe these implementations on these clips; no metric-arithmetic
failure was found. They do not establish the best achievable speed or disprove
temporal allocation, token merging or refresh in general. Earlier broad rejection
language is superseded. No new accuracy result follows from the fixes below.

## Confirmed implementation and measurement issues

| Finding | Evidence | Consequence / action |
|---|---|---|
| FIFO/uniform compute coverage scores they never use | `ResearchCache.maintain`, published fork | Unnecessary descriptor matrix operations and per-candidate GPU scalar reads contaminate control timing. Skip scoring for these policies. |
| Farthest-point selection reads a GPU scalar for every seed | `groups`, `int(distance.argmax())` | Keep seed indices on device; use `index_select` to avoid scalar tensor indexing. Same selection and tie order intended, CUDA parity gate required. |
| Random/full refresh compute unused selection scores | `ResearchCache.refresh` | Skip these scores. Their event `refresh_scores` becomes empty; selection/updates stay unchanged. |
| Every pilot uses FP32 | All archived `aggregator_dtype` values, `load_model/load_frame`, no outer autocast | Internally matched, but not a mixed-precision deployment benchmark. Demo uses BF16 on capable GPUs. Do not silently change dtype during the optimization check. |
| Diagnostics are partly inside measured time | Context `.tolist()` in `forward`; event construction in `maintain`; per-phase synchronization in runner | Total is instrumented model/maintenance latency, not video FPS. Existing logs cannot separate all host overhead. No claim of an optimized-kernel upper bound. |
| Dense outputs retained for all aggregator layers | `_aggregate`; DPT consumes selected layers, camera consumes last camera token | Possible transient-memory optimization; not implemented here because it changes allocation comparability and needs separate fidelity checks. |

These costs are not proven to explain every second of overhead. Coverage still
performs repeated candidate scoring, scalar synchronization and Python bookkeeping.
The seed loop still launches sequential GPU operations; removing host reads does
not make it an optimized fused kernel. Measured improvement remains pending.

## Representation issues, not yet causal diagnoses

**Temporal compression combines several interventions.** It changes historical
patch count, frame count, anchor detail, positional approximation and attention
weighting. Weighted means carry token multiplicity forward, but `_global` never
uses that multiplicity in attention. Special tokens remain singleton entries for
every retained frame, so their share can grow as patch detail shrinks. None of
these factors is isolated by FIFO/uniform/coverage alone.

For identical keys/values in a group of size n, replacing them with one entry
changes their softmax mass unless its logit receives log(n). That is an exact
limiting-case observation, not evidence that such a correction alone will fix
real data. Co-Me already uses this correction; it would not be new.

Raw keys are averaged *before* key normalization and RoPE, then assigned a seed
position. In general:

`RoPE(seed, norm(mean(K))) != mean(RoPE(pos_i, norm(K_i)))`.

The tests' affine positional fixture and identity key normalization do not model
this discrepancy. Disabled-path fidelity is not fidelity of lossy compression.
Do not add mass correction, change RoPE placement and protect the anchor in one
rerun; that would prevent attribution.

**Context compression is not a faithful Co-Me reproduction.** It uses a saved SAM
foreground mask, global background assignment to a fixed representative count,
no attention-mass correction, and an encoder-residual reconstruction. Its input
background is already masked. Thus it does not test preservation of real visible
background geometry, nor a segmentation-free end-to-end system. The mask originates
from a trained model even though this compression policy itself requires no training.

The encoder residual is added only to reconstructed patch features. CameraHead
reads the last layer's camera token (`tokens[:, :, 0]`), which is untouched by
that reconstruction. Changing residual unmerging alone cannot directly fix the
reported pose metrics. It can affect unmeasured dense geometry. This rules out
one proposed explanation without requiring another expensive run.

**Selective refresh collapses to anchor refresh in this pilot.** Its age-weighted
score selects frame zero at every trigger; random chooses it rarely. Comparing
them does not isolate context awareness from which historical frame is modified.
The archived local-error increase concentrates immediately after refresh. This
supports testing anchor-update discontinuity; it does not prove a gauge mechanism.
Past poses/camera history remain unchanged by design. Full refresh is sequential,
oldest-first, not a synchronous joint rebuild. The KV-change oracle is not a
pose-quality upper bound. No generalized conclusion about contextual repair follows.

## What the archived measurements establish

- Archived error-array RMS reproduces reported ATE and translation RPE for every
  arm. Source evaluation formulas and timestamp/GT staging were inspected; this
  was not an independent raw-pose evaluator rerun.
- Same-job inputs and source versions match; finite outputs, disabled fidelity,
  byte contracts and per-frame payload matches passed. No evidence of corrupted
  archives or arithmetic fabrication was found.
- Context's late-clip error concentration also exists in dense8. Earlier local
  errors still worsen under compression, so dropping the difficult tail would
  not justify declaring success. No cherry-picked replacement metric.
- Independent full-trajectory Sim(3) fits remove each arm's global gauge/scale.
  ATE gains and RPE losses can coexist. They do not mean the metrics contradict
  one another. Report local tails and scale alongside ATE.
- Single clips, single timing runs and one random seed cannot establish general
  rankings. Context has no per-frame dense-geometry validation; final export alone
  cannot support a geometry-preservation claim.

## Fixes and next gates

**25989 replay reviewed, 2026-09-28:** temporal optimization gate passed. Decisions
and payloads match 25976 exactly; pose/camera arrays agree within existing
atol/rtol, not bitwise for compressed arms. Timing improves; accuracy effectively
unchanged. See FINDINGS “Audit replay 25989 versus 25976” for numbers and caveats.
Gate 1 below is complete for temporal; do not repeat it. Context/refresh optimization
speed effects remain unmeasured. Gate 2 remains a proposal, not a queued run.

Implemented locally: device-resident seed selection; no unused coverage scores in
FIFO/uniform; no unused selective scores in random/full; explicit fidelity scope
and per-frame refresh-selection histogram/dtypes in research gate reports.
Policies, precision, budgets, merging math and prediction order are unchanged.
Fork fix published as `9931fbe` on main; parent pins this revision. No runtime
speedup or CUDA parity result has been measured for this fix yet.

Four added contracts cover seed order/ties, unused-score independence and CUDA
seed/assignment parity against published scalar selection. They must execute on
CAMP; the Mac is restricted to AST/shell/JSON/whitespace checks. The CUDA test is
skipped on CPU-only hosts, so a CPU pass alone is insufficient evidence.

1. **Optimization verification first.** Publish fork fix, pin the parent to that
   published revision and publish runner/tests/docs. User reruns the unchanged
   temporal pilot using `stream_cache_research.sbatch temporal`. Existing launcher
   runs the contracts before inference. Require the CUDA parity test to run, all
   fidelity/contracts and archived exit 0. Compare action schedules, retained
   counts, pose metrics and phase timing with 25976. Any numerical differences
   must be assessed against repeatability before interpreting a speedup. This is
   a bounded implementation check, not another scientific method sweep.
2. **After that archive is reviewed**, choose one mechanistic test. Strongest
   current diagnostic: fixed-anchor refresh versus fixed-recent refresh at the
   same trigger count and budget, with existing none/selective controls. This
   tests the selection confound. It is not implemented or submitted yet.
3. **For context/temporal merging**, first isolate the missing mass/position
   approximation using mathematical and small attention fixtures. Only then pick
   one matched pilot intervention. Do not transplant several Co-Me choices at once.
4. **Only after a useful mechanism survives**, evaluate the optimized deployment
   dtype and repeated timings, unseen sequences and Co-Me/training-free baselines
   under matched measured resources. No novelty or superiority claim before that.

Primary comparison: [Co-Me §3 and failure modes](https://arxiv.org/html/2511.14751v2).
Co-Me already occupies confidence-based merging with mass correction. The wider
training-free prior art must also be checked before naming a new contribution.

No algorithmic correction, new experiment result or confirmed speedup is claimed
by this audit. Publication and runtime status are maintained in docs/STATUS.md.
