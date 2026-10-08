# Seven loop-transformer implementations — 2026-10-08

User selected implementation of all seven researched directions. Research and
paper links: [LOOP_TRANSFORMER_RESEARCH.md](LOOP_TRANSFORMER_RESEARCH.md).
These are geometry prototypes inspired by the papers, not reproductions of
their language/VLM architectures or demonstrated tracker improvements.

## Current retry: relaxed recovery after 26348 — 2026-10-08

Supersedes the historical all-method submission instructions below. User
requested a paper-informed correction and retry. Research rationale and primary
sources: [recovery review](LOOP_TRANSFORMER_RESEARCH.md#recovery-review-after-26348--2026-10-08).

`train relaxed recovery` runs contracts and prepares exactly one new model:

- Two recurrences of eight complete local/global pairs, rank 64 depth-specific
  SVD residuals. Dedicated entry/exit pairs preserve all 36 executed blocks and
  all 18 cache slots. Frozen additive conditioning remains zero; depth-specific
  adapters/norms supply specialization. No short schedule or routing.
- Native teacher records bank and query states at decoder indices 1/17/33.
  Intermediate losses match 17/33, averaging normalized register/patch errors
  equally. They augment the existing final feature and geometry/pose losses.
- For optimizer indices 0–49, mix teacher states into the two loop inputs with
  weight `max(0, 1-step/50)`; indices 50–199 use only student states. This
  curriculum is a geometry adaptation inspired by RAPTOR, not its exact loss.
- 200 optimizer updates, each accumulating all eight train queries (1600 query
  examples including repeated examples), peak LR 1e-5, 20-update linear warmup,
  then cosine decay toward 0.1x peak. Before/after diagnostics use the same
  mixture and full-batch objective within each step; values across changing
  mixtures are not directly comparable. Component losses and scales are logged.
- Checkpoint selection uses the unchanged free-running mode every eight updates
  and at the end, including during the curriculum. Retain best measured state.
  Both fixed train mean and separate development-scene mean must improve over
  initialization for preparation eligibility. Development is a gate, not a
  gradient source. Selected development loss components are preserved.
- Scale normalization now differentiates student bank scale as well as bank
  features/origin. Diagnostic adaptive map points remain detached.

Contracts cover curriculum accumulation/development rejection, scale gradients,
real teacher capture, loop boundaries, inference without teacher states and
checkpoint reload; existing native/cache/gradient contracts still run first.
Local verification is Python 3.10 syntax, shell syntax and diff checks only.
GPU contracts, SVD initialization at rank 64, peak memory and loss behavior await
CAMP. Existing archives cannot be reused with changed source/driver hashes.

After publication, user runs from `layer_good` on head:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/loop_transformer.sbatch train relaxed recovery"
O=../loop_transformer_slurm-%j.log
sbatch -p 24g -w stuttgart --gres=gpu:1 --propagate=NONE \
  -t 04:00:00 -o "$O" --wrap="$W"
```

This is one bounded job, not a runtime estimate. Expected evidence: contracts
OK, one relaxed summary, TRAIN JOB OK, final `loop_JOB_train.tar` and Slurm 0:0.
Review free-running checkpoint history, matched descent after teacher mixing
ends, component scales and both train/development gates before proposing a
256-frame paired pilot. No pilot/full job is queued behind this training.
No teacher-assisted result is a deployment result; no automatic retry on failure.

## Authorized overnight experiment batch — 2026-10-08

The user now requests one job for all seven designs, with automatic full-office
promotion for feasible candidates. This explicitly supersedes the manual pause
between pilot and full for `overnight all` only. The old `train all` was an
engineering smoke, not seven completed scientific experiments. No repeated
single-method diagnostic cycle or tuning sweep is part of this batch.

`bash tools/loop_transformer.sbatch overnight all` first runs the contract suite,
including orchestration failure/gate tests. It then prepares and pilots each of
the seven variants, with fresh processes and independent checkpoints. Combined
is not an eighth experiment in this batch. Every feasibility attempt finishes
before any full-sequence evaluation begins.

| Design | Preparation in this batch |
|---|---|
| elastic | One bounded200-update AdamW attempt at1e-5, full/short budgets |
| relaxed | Pretrained group-mean projections + rank16 SVD residuals; no optimizer |
| shared_kv | Initialized shared decoder and paired historical banks; no optimizer |
| adaptive | Initialized recurrent endpoints, then disjoint-fit/validation halting calibration; no optimizer |
| token | Initialized decoder and fixed-seed untrained router; no optimizer |
| refiner | One bounded200-update AdamW attempt at1e-5 with causal history replay |
| nested | Initialized local/global schedule; no optimizer |

The200-update budget is a fixed engineering attempt for the two trained paths,
not a convergence guarantee. Their gate compares mean losses over the identical
fixed eight train queries and every trained budget before/after training. Both
lists are saved. Initialized candidates have `training_loss_decreased=null`;
they do not need to manufacture a loss decrease without optimization. Their
checkpoint provenance explicitly records initialized preparation and pilot
eligibility. Adaptive still cannot deploy unless calibration passes. Previous
checkpoints are not reused because source/driver hashes bind the preparation.

Direct tests of initialized weight sharing, token selection and nested schedules
are zero-update feasibility tests, not claims to reproduce learned paper methods.
The untrained token router is not learned pose relevance. Failures here cannot
rule out later trained versions. No trainable parameter is updated in these five
preparations, although frozen-teacher labels/losses are recorded for inspection.

Eligible candidates get256-frame office pilots against fresh native and compact
native, with two paired repetitions. Every tested budget/repeat must remain
within5% of native ATE, translation RPE and translation-RPE p99. All passing
candidates then automatically run the full office sequence with the same paired
controls/budgets. Passing quality is feasibility, not proof of2x speed or half
memory; those targets remain separate measured flags. Failure to pass means
skip, not retry. A child-process failure records phase/exit code and continues
with the remaining candidates; shared contract failure stops the entire job.
`overnight_results.json` distinguishes blocked preparation, blocked pilot,
process failure and completed full evaluation. Job completion is not a quality
claim. Full comparisons still need result review after the overnight run.

One final `loop_JOB_overnight.tar` preserves policy, all per-variant preparation,
checkpoints, pilots, full outputs and root provenance/logs. Nested checkpoint
archives are retained; redundant `reviewed_training` extractions are omitted.
No separate container build or new model download. Runtime remains CAMP-only.

After publication, from `layer_good` on head (pull runs inside allocation):

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/loop_transformer.sbatch overnight all"
O=../loop_transformer_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE \
  -t 12:00:00 -o "$O" --wrap="$W"
```

Return the job ID. This requests a12-hour allocation, not a runtime guarantee.
No assistant-operated submission. Completion evidence: contract success,
per-variant outcomes, `LOOP OVERNIGHT COMPLETE`, archive exit0 and Slurm status.

## Checkpoint selection correction after26337 partial results

The trained paths now measure the same fixed gate objective after each complete
cycle of eight training queries and after the final update (including a partial
cycle). They retain the lowest measured loss state, starting with initialization
as step0, and restore those trainable weights before final loss, development,
calibration and checkpoint serialization. The training update budget and learning
rate are unchanged. Frozen weights are not copied; the best trainable state is
held on CPU. `checkpoint_selection.json` records every measured query/budget
loss, mean and selected step. Summary/checkpoint provenance separately record
`selected_step` and `last_update_loss`; `final_loss` describes the restored state.
If no trained state improves, step0 is retained and training eligibility remains
false. Initialized paths retain their separate no-training policy.

This corrects last-update-only selection; it does not establish a cause or cure
for unstable optimization. There is no saved intermediate checkpoint from26337
with which to demonstrate that this would have rescued elastic. Training-set
selection is not held-out validation; unchanged pilot/full quality gates remain
required. Failed relaxed/shared-KV/token pilots are not repaired by checkpoint
selection because those candidates were not trained.

Every comparison now prints per-run quality ratios, speedup and memory ratio;
pilot gate JSON also records failed metrics by budget/repeat. This makes future
blocked_pilot results interpretable directly from the log without relaxing gates.
A regression exercises early improvement followed by regression, no improvement,
and a final partial query cycle, checking actual exported weights/provenance.
It is scheduled in CAMP contracts; local verification is syntax/whitespace only.

Job26337 remains on submitted source170fad9; do not pull into its checkout while
it runs or queue a duplicate. Review its remaining results and archive first.
This correction changes the driver hash, so26337 checkpoints cannot be promoted
using the corrected driver. Historical artifacts remain intact for inspection.

## Training gradient correction after the 26337 audit

26337 per-step logs (`updates.json`): elastic improved for 16 steps, then rose
almost monotonically on every frame (130→9600 from step ~80) while its gradient
norm grew with it; one refiner update raised frame-48 loss 0.03→8.4, while
frames replaying few history steps stayed near zero. Both trained against a
truncated objective: the rebuilt bank/origin (same weights) was `no_grad` and
detached, and the refiner's replayed history was cut at every pose. Training
now differentiates the bank, origin and the refiner rollout. Inference, pilots,
calibration and no-grad measurements are unchanged.

`train all` now trains every variant (no initialized-only arm) with the
overnight trained-variant policy: 200 steps, 1e-5, aggregate fixed-query gate,
best-measured checkpoint retention (4bf298f). It runs the contracts first; a new
contract checks that the bank and refiner history change the gradient.
`updates.json` adds `full_budget_before`/`full_budget_after` per step: the same
query re-measured with a fresh rebuild after the update. If steps still fail to
descend with the full gradient, the remaining cause is step size/conditioning,
not truncation. This is a cause hypothesis under test, not a demonstrated cure.
No pilot or full runs in this job; review training before `pilot all`.

From `layer_good` on head (pull runs inside the allocation):

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/loop_transformer.sbatch train all"
O=../loop_transformer_slurm-%j.log
sbatch -p 24g -w stuttgart --gres=gpu:1 --propagate=NONE \
  -t 06:00:00 -o "$O" --wrap="$W"
```

Evidence: `LOOP TRAIN CONTRACTS OK`, one `LOOP PREPARATION COMPLETE` per
variant, `LOOP TRAIN JOB OK`, archive `loop_<job>_train.tar`.

## Implemented architecture paths

Model code: `kv_tracker/kv_tracker/loop_models.py` in the model fork. Driver:
`tools/loop_transformer.py`. Entry: `tools/loop_transformer.sbatch`.
Native Pi3 source is imported from the existing environment and checked against
SHA256 `cbcf68b3c05baab7680f6e24afda42501dc0ba799e85ca18668ba3e8a5812979`.
No nested-source edit, container rebuild, dependency install or download.

All decoder variants retain the first and last local/global pair, the five
register tokens, original RoPE coordinates and the concatenated final pair
input to the native camera/point/confidence heads. Frozen encoder/heads remain
in eval mode during training. The middle has four shared local/global pairs,
normally repeated four times. Rebuilds use the full schedule; query budgets use
matching full-schedule history slots. Training rebuilds keep bank KV and origin
in the graph (inference banks are grad-free), with block checkpointing. These
pilots use dense resized scene RGB and the fixed native50-frame/cap20 schedule.
They do not use the stock token_drop/decoder-index-hook/cache-transform paths.

| Option | Actual implementation | Fixed prototype setting |
|---|---|---|
| `elastic` | Shared middle blocks with learned progress/step-size conditioning; long/short feature, pose and geometry distillation | Four/full and two/short recurrences; 36/20 executed decoder blocks |
| `relaxed` | Shared mean projection weights plus unmerged depth-specific low-rank residuals, with depth-specific norm/LayerScale parameters | Rank16 randomized truncated SVD of pretrained residuals; full36 execution |
| `shared_kv` | Matched global modules reuse a historical bank across paired recurrences; queries leave bank storage unchanged | Middle caches16→8, total slots18→10; separate current K/V every execution |
| `adaptive` | Prefix endpoint supervision; separately calibrated pose-change and projected-map depth-residual stopping | Earliest exit after two loops; up to four; calibration required |
| `token` | First loop all patches; later loops gather active Q/projection/MLP rows and scatter updates; inactive rows remain K/V context | Half patches, half top learned score and half spatial spread; all registers retained |
| `refiner` | Frozen native Pi3 rebuilds; fresh DINO query features plus a small recurrent transformer and map cross-attention; body-frame SE(3) residuals | Width256, three refinement steps,64 map samples per frame; causal constant-velocity warm start after two poses |
| `nested` | Two local refinements per global update; unused middle global modules removed | Four loops,28 decoder executions,10 historical cache slots |
| `combined` | Elastic short/full execution + relaxed projections + paired historical cache sharing | Same fixed settings as the corresponding separate options |

The combined path includes ideas1–3 only. Halting, token routing and refiner
execution remain separate scientific arms; their gains are not presumed additive.
`native` and `native_compact` are controls: the former uses inherited unchanged
Pi3 decode/heads; the latter additionally clones K/V into compact storage.
This distinguishes recurrence sharing from releasing native V's QKV backing
storage. Both must match native outputs exactly in the GPU contracts.

Specific limits: the token router learns a detached per-patch native-feature
error proxy, not demonstrated pose importance; inactive K/V are recomputed from
their retained state. Adaptive geometry readouts and host decisions are timed
and may outweigh early-exit savings. The refiner stores the full native decoder
and KV for rebuilds, so it does not claim weight/cache memory savings. Refiner
queries return poses only, not new dense reconstruction. Its smoke replays every preceding query since the last bank update with the
current model and backpropagates through that whole causal rollout (BPTT).
The nested schedule is a geometry adaptation of hierarchical recurrence, not LoopVL.

## Gates and reproducibility

1. **Contracts only.** `contracts` runs primitive low-rank/selection/SE(3) tests,
   native and native_compact exact fidelity, compact full-row equivalence,
   inactive-context gradients, all eight variant rebuild/query/head gradients,
   cache immutability, correct cache-stage counts and live checkpoint reload
   equality on small GPU tensors. It does not stage data or train/evaluate.
2. **Bounded engineering training.** After contract evidence is reviewed,
   `train all` stages64 frames each from read-only `/mnt/datasets/tum-rgbd`
   `freiburg1_desk` (train) and `freiburg3_structure_texture_far` (development).
   It creates native teacher targets for frames1/8/16/32/48/50/56/63 with causal
   banks `[0,0]` before49 and `[0,49]` afterward. Native source/weights, exact
   pixel and ZIP hashes, manifests and labels are archived. Eight fresh training
   processes each run 200 AdamW updates at 1e-5, aggregate gate. Still bounded:
   it is not a sufficient training budget for a scientific claim.
3. **Calibrate adaptive stopping within the engineering stage.** Fixed25th
   percentiles are fitted only to safe exits from the first four development
   queries. They must accept at least one exit in each of the fit and disjoint
   final-four validation partitions, with no unsafe accepted exit in either.
   Both eligible early endpoints (loops2 and3) receive direct supervision. Safety here means both
   map-scale-normalized translation and rotation chordal error≤.05 versus the
   frozen teacher. Unknown map support cannot trigger stopping. Calibration
   failure is recorded explicitly; an uncalibrated adaptive model cannot deploy.
   This small chronological holdout is still one scene, not independent-scene
   validation or evidence of safe full-trajectory behavior.
4. **Reviewed tracking pilot.** `pilot all <training.tar>` extracts the reviewed
   checkpoint archive into `/tmp`, stages256 office frames, and evaluates native,
   native_compact and each eligible implementation in fresh processes. There
   are two ordered paired repetitions. Elastic/shared_kv/combined each test
   their already-trained two/full-four schedules. Failed training smoke or
   calibration options are explicitly recorded in `blocked_variants.json`; they
   are not silently substituted. Selecting an ineligible single variant fails.
5. **Separate phase profile.** `profile <variant> <training.tar>` is the same
   pilot with CUDA events for encoder, local/global decoder, heads and refiner;
   compact token blocks are explicitly timed. Dynamic readout costs are included.
   Profiling is diagnostic, not the timed method performance.
6. **Reviewed full evaluation.** `full <variant> <training.tar> <gate.json>`
   requires that variant's passing pilot gate tied to the same checkpoint archive
   and exact driver/model source hashes; all expected repeats/budgets must exist.
   It runs the full office sequence, not a sweep or automatic successor. Inspect
   trajectory drift and both budget controls before claiming a result. Subsequent
   held-out scenes and ARCTIC/SAM transfer remain separate future validations.

Losses compare first-camera-relative poses and camera-local geometry normalized
by one fixed bank scale per episode, plus native final-pair features. No per-query
Sim(3) fit or GT training labels. This is not evidence of gauge preservation in
long live trajectories. Runtime uses the same existing MapHandoff native path
for gauge normalization,50-frame updates and cap20 in all arms. The first query
after each bank rebuild resets refiner state to that bank's last raw pose.

Changed-module checkpoints omit frozen pretrained encoder/head/entry/exit
weights. Deployment loads the same offline pretrained Pi3 and verifies source,
weight identities, exact training-driver hash and complete changed-state keys.
Inference does not load the teacher model. Checkpoints contain no episode caches or retained query graph.
Unique parameter, KV and auxiliary map-state bytes are reported separately;
actual peak allocated/reserved memory and device samples are also archived.

Evaluation uses identical pixels, keyframe IDs and GT/RPE associations. The
three-metric gate allows at most5% regression in ATE, translation RPE and
translation-RPE p99. A2x FPS or half-memory success additionally requires that
quality gate. Timing includes upload, queries, all bank updates and CPU pose/log
processing, but preloads verified uint8 RGB and is scene-only. Pixel conversion,
integrity hashing and final artifact serialization are outside the timer. Profile
or detailed-instrumentation runs cannot set doubled-FPS/half-memory success flags.
Do not call it complete video/SAM FPS.
Two paired observations are diagnostic evidence, not statistical significance.
Memory measurements include full resident model and refiner/map state; native
and compact controls expose differences in KV backing storage.


## Review corrections and design limits — 2026-10-08

Review found and corrected unsupervised third-loop adaptive exits, reuse of
threshold-fitting examples as the only validation, and isolated refiner training
that omitted deployed motion-history inputs. Adaptive execution now skips the
first-loop depth residual (ineligible exit) and all redundant final-depth checks.
The frozen final heads still run for the actual returned prediction; early-exit
camera heads are not free. Low-rank execution indexes the registered depth list
directly instead of constructing temporary ModuleLists for every pair.

Checkpoint round-trip verification loads on CPU, avoiding a second checkpoint
copy on GPU. Tracking keeps uint8 frames rather than float32 copies on the host.
Pose losses and calibration gauge transforms stay FP32 even under outer BF16
autocast. Compact attention follows native SDPA backend selection. Archive extraction
accepts regular files/directories only. Source-bound provenance deliberately
rejects checkpoints/pilot gates produced by the earlier implementation: retrain
after contracts; do not reuse an earlier checkpoint as a corrected result.

The short elastic schedule executes20 versus36 decoder blocks: its idealized
block-count speedup is1.8x before encoder, heads and full rebuilds. Thus skipping
these blocks alone cannot meet2x total FPS. Nested28/36 is a smaller reduction.
The refiner avoids the native query decoder and remains the strongest structural
speed candidate. Shared weights/KV are memory hypotheses, not half-total-memory
results. Trainable recurrent weights remain FP32 with BF16 autocast, whereas
native decoder projections are stored BF16; reported resident bytes include this
actual precision difference. No post-training precision conversion is applied.

Elastic training distills both budgets against the teacher; it does not implement
LoopFormer's additional long/short consistency objective. Paired history sharing
is a geometry adaptation, not an exact MoR reproduction. Token routing lacks an
equal-cost trained uniform-routing ablation; adaptive lacks a matched-cost fixed
exit ablation. These are necessary for later causal method attribution, but are
not added as an unrequested control sweep to the engineering smoke. None of the
seven paths has runtime-verified quality, speed or memory benefit yet.

New CAMP contracts cover both supervised adaptive endpoints, calibration success
and rejection on unsafe/uncovered validation, causal refiner history with stopped
history gradients, and invalid zero budgets. Existing native/cache/reload
contracts remain required. Local checks are AST/whitespace/Bash only.

## Contract failure26333 and retry

User-supplied job26333 log reports seven decoder variants failing at
`loss.backward()` in `pi3/curope/curope2d.py` with `tokens are not contiguous`.
The refiner subtest and other listed tests did not report failures, but the suite
ended `FAILED (errors=7)`: no training gate passed. The old unconditional Python
success print was misleading under unittest subtests. It is removed; the wrapper
now prints the all-variant success marker only after unittest exits successfully.
Archive completion/exit status was not included in the supplied log.

The pinned cuRoPE backward mutates its input gradient without materializing the
contiguous B,N,H,D layout required by the CUDA kernel. The model fork now installs
TrainingRoPE on the recurrent decoder and frozen readout decoders. Its custom
first-order autograd path clones an owned contiguous buffer before the in-place
forward/inverse kernels; no-grad inference delegates to original cuRoPE. Frozen
head module shells are copied while parameter/buffer tensors remain shared, so
the caller's native model is not patched or duplicated. No container or nested
Pi3 source modification. New GPU contracts cover FP32/BF16 native forward parity,
strided/broadcast backward, residual-branch gradient preservation and isolation
of native modules. Existing all-variant backward/reload contracts remain the
integration gate. This correction is locally syntax-checked, not GPU-verified.

Next is another contracts-only allocation using the chunk below, not training.

## CAMP operation

All commands below are user-operated; no assistant SSH/submission/transfer.
Wrapper pulls parent and updates the published model gitlink inside the allocation.
It borrows `kvt.tar` read-only with existing offline weights, mounts `/mnt` read-only
inside the container, and writes/extracts/trains only under job-local `/tmp`.
The host archives results to `/mnt/projects/gr/3DRecon/loop_transformer_out/`.
One archive per stage/job: `loop_JOB_contracts.tar`, `loop_JOB_train.tar`,
`loop_JOB_pilot.tar`, `loop_JOB_profile.tar` or `loop_JOB_full.tar`.
Partial logs/models are archived on failures. Original source PNGs are omitted,
but exact model_rgb pixels, manifests/implicit all-true scene masks and ZIP hashes
are preserved. Evaluation archives reference the separately preserved training
archive by SHA256 rather than duplicating its large extracted checkpoints.
No new artifact or environment is claimed to exist until a user-run log verifies it.

**First chunk**, from `/mnt/projects/gr/3DRecon/layer_good` on head, after code is
published. Pull and contracts execute inside the same allocation:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/loop_transformer.sbatch contracts"
O=../loop_transformer_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE \
  -t 02:00:00 -o "$O" --wrap="$W"
```

Return job ID, then `LOOP ALL VARIANT GPU CONTRACTS OK`, unittest OK,
`LOOP CONTRACTS JOB OK` and archive exit0, or the first failure and its context.
No training/evaluation is queued behind these unreviewed contracts.

After review, use the same allocation-wrapped pattern with `train all` as the
script arguments. Stop at `LOOP TRAINING ENGINEERING COMPLETE`, review
`training_complete.json`, per-variant updates/losses and adaptive calibration.
Only then select pilot/profile using the exact `loop_JOB_train.tar` path. The
full-stage gate is `pilot_gates/VARIANT.json` inside a reviewed pilot archive;
the single-variant pilot also writes top-level `pilot_gate.json`. Provide one
short submission chunk at a time, retaining each archive separately.

Local permitted checks: Python3.10 AST parsing, `bash -n`, whitespace checks.
All imports, tensor contracts, training, inference and actual resource/quality
verification require CAMP. The current source completion/publication and next
unpassed gate are recorded in `docs/STATUS.md`.
