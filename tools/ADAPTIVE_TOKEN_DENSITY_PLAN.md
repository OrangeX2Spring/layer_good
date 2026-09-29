# Adaptive token density for KV-Tracker: one method for objects and scenes

Plan dated 2026-09-29, from the user's direction after the patch-select screen
(25996) closed. Stage 0 + gate 1 are implemented for user-run CAMP verification.
Training remains blocked until the gate archive is reviewed. The user confirmed
this scope and retained Co-Me strictly as a comparison on 2026-09-29. Numbers are quoted from `tools/FINDINGS.md`, which carries their
caveats.

## 1. Why this, and why not the alternatives

- **Patch selection of the cache is closed.** 25996: ten hand-designed selectors
  at a quarter of patches did not beat uniform or eight random seeds on either
  task, and every quarter-budget arm failed the temporal-halves gate. Pruning after
  a dense forward never lowered peak memory.
- **The rebuild is not the lever.** Native split (FINDINGS "per-frame time and
  memory split, 25996"): rebuild is 3-16% of tracking time and adds 0.3-0.9 GiB
  over steady state. Queries are 57-65%; work outside Pi3 (loading, SAM) 26-38%.
  The user declined rebuild work on this basis.
- **Computing fewer tokens does work on objects.** Token drop bg64 (25907):
  object patches plus 64 evenly spaced background patches in every query and
  rebuild; tracking time -28% / -46% / -28%, peak allocated -24% / -52% / -36%
  (box / ketchup / espresso); accuracy passes box and espresso, fails ketchup
  translation RPE (+9.03%). Kept keyframe counts changed (ketchup 21 to 14).
- **Mass weighting hurts here.** A2 (25906): the log-mass bias on kept background
  degraded every object, overshot the dense background attention share by ~1.3x
  and collapsed the hold threshold. StreamVGGT context mass correction (25992) was
  stopped. Spread original samples without a bias (bg16/bg64) were what worked.

Goal (user, 2026-09-29): one method, or one family, that applies to objects and
scenes, cutting query time and peak memory; a small trainable module is welcome
(it would be the project's first training).

## 2. The method: adaptive token density (ATD)

Each frame is computed with dense tokens where the image matters for tracking and
a sparse, spatially spread subsample of original tokens elsewhere:

- **Dense region D_t.** Objects: patches touching the SAM mask. Scenes: patches
  whose informativeness score is high (Stage 1: a pixel heuristic; Stage 2: a
  learned score).
- **Sparse region.** Everything else (object mode: the zeroed background; scenes:
  homogeneous areas such as blank walls and floor) keeps an evenly spaced
  subsample of original patches, so region extent and position survive.
- **No averaging, no mass bias, no token regeneration.** Kept tokens keep their
  DINOv2 position embedding and decoder RoPE position.
- **Applied where the cost is:** to the query and every rebuild through the fork's
  token-drop path (`kv_tracker/token_drop.py: forward_kept`), so the encoder, the
  decoder, the heads and the stored cache all see only kept tokens.
- **Scoring happens before the encoder** (pixels, gradients, mask), so the whole
  encoder is saved as well.
- **Objects reduce to bg64** (background is perfectly homogeneous), so the object
  side starts from a measured, nearly passing configuration.

## 3. Difference from Co-Me (arXiv 2511.14751; read 2026-09-29)

| | Co-Me | ATD |
|---|---|---|
| Setting | Multi-view batch forward (online only as 4-image segments); no KV cache | Streaming tracking with a persistent keyframe cache; long causal sequences |
| Token operation | Average 3x3 spatial groups, `log n` attention bias, replicate for dense outputs | Keep original tokens: dense region plus spread subsample; no averaging, no bias |
| Signal | Distilled confidence (uncertainty) from layer-15 features, ranking loss | Tracking relevance: SAM mask / homogeneity, later a module trained on pose sensitivity |
| Where scoring runs | Layer 15; the first 15 encoder layers see all tokens | Before the encoder |
| Tasks evaluated | Depth, pose AUC, point clouds | Tracking ATE/RPE, temporal halves, drift, objects and scenes |

The difference must be shown, not claimed: every accuracy stage includes a
**Co-Me-style comparison arm** (not part of ATD): confidence-ranked 3x3
groups averaged after encoder block 15, `log n` bias through the decoder A2
path, features replicated for dense outputs. Confidence comes from a separate
dense Pi3 teacher, not Co-Me's learned predictor. The final group is partially
merged to hit the ATD count when possible; otherwise all groups are retained and
the larger actual count is recorded. In particular, 3x3 groups cannot always fit
object background into 64 tokens. No patches are dropped to hide that mismatch.
This is a diagnostic reimplementation, not a faithful Co-Me reproduction;
encoder attention has no mass bias and the teacher cost is charged separately. Before any novelty statement, survey other token
reduction work (ToMe, DynamicViT and any VGGT/streaming-specific pruning or
merging papers); that survey has not been done.

## 4. Stages and gates

Each stage is one user-run job whose archive is reviewed before the next. No
stage is queued behind an unreviewed one.

### Stage 0: implementation prerequisites (local, static checks only)

- Fork: expose `run_track3r(token_policy=...)` for both modes; its explicit
  policy supplies keep masks to the same `forward_kept` path. The existing
  object-only CLI and bg64 selection remain unchanged; no policy means the
  original path. All-kept fidelity is a runtime gate.
- Fork: a keep-set builder shared by both modes: dense set + evenly spaced
  subsample of the rest to an exact per-frame count (reuse `background_keep` and
  `patch_select.even`).
- Fork: sensitivity oracle (below) and the Co-Me-style arm.
- Parent: driver reusing `tools/kvt_patch_select.py` infrastructure (fresh process
  per arm, object/scene evaluation, eight random seeds, with timing around
  `pi3_inference` because the direct sparse forward bypasses model hooks,
  per-arm archives) and a `kvt_tum.sbatch` mode; contract tests.
- Contracts: all-kept via the new path matches native within the established
  tolerances; kept tokens have exact count, original positions and no duplicates;
  cache holds only kept tokens; queries read that cache; causal prefix run.

### Stage 1 (gate 1): feasibility, homogeneity and oracle headroom — one job

1. **Timing microbenchmark (no tracking).** Query and rebuild wall time and peak
   allocated at 308 and 518, kept fraction {1, .75, .5, .25}, cache of 1, 10, 20
   keyframes, synchronized, repeated. Question: does fewer tokens speed up the
   16 ms, 475-token scene query at all, or is it overhead-bound?
2. **Homogeneity statistics.** Per clip, the share of patches below candidate
   gradient-energy thresholds (distribution, not a choice).
3. **Oracle headroom at a frozen budget.** Sensitivity oracle: for each frame, a
   dense forward with a multiplicative gate g=1 on every patch token after the
   encoder; importance = sum over the six se(3) pose components of |d xi / d g_p|
   (six backward passes, first-order). It uses no GT, only an unaffordable dense
   pass. Sparse original-token arms have equal kept count per frame; dense
   references and explicitly flagged Co-Me count mismatches are exceptions:
   dense (all-kept), ATD-heuristic, oracle, uniform spread, random x8 seeds,
   Co-Me-style comparison, plus a dense repeat (determinism check).
   Budgets: scenes keep ceil(0.5 P) patches per frame; objects keep every
   mask-touching patch plus 64 background patches (the arms choose which 64;
   Co-Me-style merges background groups, with actual count recorded if 64
   background groups cannot be reached without dropping source patches).
   Clips: ARCTIC espresso/ketchup/box (full, 518) and TUM office 1150,
   with_loop/no_loop 1024 (308). Corrected count: 14 arms x 6 = 84 main runs,
   plus six native schedule/fidelity references and six causal-prefix runs =
   **96 tracking runs**, plus GPU contracts and the microbenchmark. The earlier
   ~1 h estimate is withdrawn: dense teacher gradients were not priced into it.

Gate 1 decisions (frozen before submission):
- **Scene speed:** if the 308 query at kept 0.5 is not at least 20% faster than
  at 1.0 at **each** of 1, 10 and 20 cached frames, the scene speed gate fails; the scene side then moves to
  exact engineering (CUDA-graph replay of the query, skipping unused heads,
  overlapping loading/SAM) and ATD stays object-only.
- **Headroom:** the oracle must have RMS translation RPE <= 0.95 x the best of
  uniform and the eight seeds, and p99 no worse, on >= 2 of 3 clips of a task.
  Require p99 no worse than **each** control as well. If it does not, stop the
  trainable-module plan for that task at this budget. This tests this first-order
  oracle; a failure is not a proof that every possible learned selector fails.
- **Co-Me:** report ATD-heuristic and oracle against the Co-Me-style arm per clip.

### Stage 2: the small trainable module (only for tasks passing gate 1)

- **Model:** a per-patch scorer, < 1 M parameters: a few convolutions over the
  model-input RGB, gradient energy and (objects) the SAM mask at patch resolution,
  optionally one early DINOv2 block's features if timing allows. Output: a keep
  score; selection = top scores in the dense budget plus spread context.
- **Labels:** the sensitivity oracle computed offline on training sequences
  (self-distillation from dense Pi3; no GT).
- **Loss:** ranking loss on patch pairs within a frame plus a spatial-spread
  regulariser; Pi3 stays frozen. End-to-end soft gates through frozen Pi3
  (straight-through / Gumbel top-k) only if the offline labels plateau.
- **Training data, never the evaluation clips:**
  - scenes: TUM zips on `/mnt/datasets/tum-rgbd` other than office, with_loop and
    no_loop (static sequences first; dynamic `walking_*` excluded initially);
  - objects: open decision (below).
- **Compute and storage:** one 24 GB GPU on stuttgart; label generation is the
  dominant cost (dense forward + six backward passes per frame); outputs and
  checkpoints under `/mnt/projects/gr/3DRecon/atd_out/`, archived per run.
- **Contracts:** held-out sequence loss, deterministic inference, scorer time per
  frame reported and charged to the tracker's time.

### Stage 3: evaluation of the trained module

Same six clips, same arms as gate 1 with the trained scorer added, eight seeds,
dense repeat. Decision per task, frozen beforehand: within 5% of dense on RMS,
p99, ATE and rotation RPE and on each temporal half; lower tracking time and peak
allocated than dense; RMS <= 0.95 x the best random seed and uniform with p99 no
worse on >= 2 of 3 clips; reported against the Co-Me-style arm. One winner
configuration, no post-hoc threshold or budget sweep; failure stops the module.

## 5. Decisions confirmed 2026-09-29

| Decision | Options considered | Selected |
|---|---|---|
| Oracle gate before any training code | yes / no | yes |
| Co-Me-style arm in gate 1 | yes / no | yes |
| Object training data | YCB-V real sequences (BOP on `/mnt/datasets`, GT masks, no download) / HouseCat6D / more ARCTIC subjects (download) | YCB-V real |
| Lead task | objects / scenes / both in one job | both in gate 1; objects lead Stage 2 (lab goal) |
| Scene dense budget | fixed kept fraction / score threshold | fixed 0.5 in gate 1 (comparable to random at equal count) |

## 6. Risks

- Scene queries may be overhead-bound (inferred from 16 ms for 475 tokens, not
  measured); gate 1 item 1 decides.
- Selection may simply not matter at practical budgets (25996's lesson); gate 1
  item 3 decides before training.
- bg64 still fails ketchup; the object side is not yet a pass.
- Tracking is chaotic: kernel-path differences alone moved box ATE 0.198 to 0.387
  (FINDINGS, 25906); only seed ensembles and several clips can separate signal.
- Training infrastructure is new to this project (data pipeline, checkpoints,
  held-out validation); budget time for it.

## 7. Execution rules (unchanged)

Edit locally, static checks only on the Mac; publish fork then parent gitlink; the
user submits with the allocation-wrapped pull command (stuttgart) and transfers
archives from the Mac repo root; review each archive (provenance, contracts, what
each arm actually selected, seed spread) before the next stage.


## 8. Implemented gate protocol and next command

Entrypoint: `bash tools/kvt_tum.sbatch adaptive-tokens`. Fork implementation:
`kv_tracker/adaptive_tokens.py`, extended `token_drop.forward_kept`, and the
explicit policy hook in `main.py` / `pi3_utilts.py`. Parent driver:
`tools/kvt_adaptive_tokens.py`; contracts: `tools/test_kvt_adaptive_tokens.py`.
Only Stage 0 + gate 1 are implemented; no scorer or training pipeline exists yet.

Frozen details resolving the draft's underspecified mechanics:

- Scenes: heuristic keeps the highest gradient-energy half of its kept budget,
  then fills the rest spatially using the existing 4x4-cell uniform allocator.
  Energy is mean squared adjacent RGB differences **within** each 14x14 patch,
  on [0,1] model-input pixels. No threshold is selected. Objects use exactly
  the existing raster-spread bg64 heuristic. Uniform controls use the shared
  cell-spread builder; random controls keep the same mandatory object region.
- Oracle: six backward passes for each arriving frame, on full encoder outputs
  and a separate dense, causal teacher cache. Score is the sum of absolute
  derivatives of local body translation (Pi3 units) and rotation (radians).
  The differential of `log(T0^-1 T)` at detached `T0=T` avoids the singular
  rotation log at zero. No GT, future images or fitted scale weighting enters
  selection. Oracle keeps the highest scores at the exact budget; it is a
  sensitivity diagnostic, not an optimized subset oracle or deployable method.
- Each arrival's choices are frozen and reused at every rebuild. All arms replay
  the native admission schedule, including the duplicated anchor bootstrap.
  This isolates token choice; it does not establish natural-admission accuracy.
- The teacher cache is swapped out before student computation. Its cost is
  included in wall time and peak allocation; oracle/Co-Me memory is **not** a
  deployable memory result. Student-forward, teacher and selection times are
  recorded separately. The microbenchmark has no teacher or selector and uses
  actual office/espresso input grids, two warmups and five measured repetitions.
  Scene queries are camera-only; object queries and all rebuilds include heads.
- All-kept/native **per-forward poses** use the existing token-drop GPU
  contract (`test_kvt_token_drop.py`): max absolute difference <1e-2. Full
  trajectory/keyframe-pose differences are reported, not asserted equal: bf16
  per-frame encoder kernels can amplify into different trajectories (already
  observed for token drop). The much tighter patch-cache tolerances apply only
  to same-path causal prefixes. The all-kept repeat must reproduce exactly;
  all-kept is the accuracy reference for sparse original-token arms. Any failed contract stops the job, not a scientific negative result.
  Prefixes on espresso/office cover the first real admission and compare complete
  event records and poses for heuristic, oracle and random17.
- Events preserve every frame's exact indices, scores, grid, target/actual count,
  homogeneity quantiles/shares and comparison group membership. Runtime checks
  assert cache token counts and that queries preserve persistent cache storage.
  No selected patch can lose its original ATD position. GPU tests exercise dense
  fidelity, sparse query/rebuild, oracle backward, and group replication.
- The Co-Me comparison is reported with its count mismatch and teacher cost;
  it never determines ATD's training gate. It runs only as a comparison, as the
  user explicitly confirmed. No novelty claim is made.
- Archives include exact source images, SAM masks, model-input hashes for TUM,
  selected indices, source revisions, checkpoint hashes, container provenance,
  configs, logs, trajectories, metrics, and each run's return code. Completed
  runs are archived immediately; the wrapper preserves failed/partial runs.
  A 22 h start-deadline stops new tracking runs; a timeout is incomplete, not STOP.

Outputs use the existing `/mnt/projects/gr/3DRecon/kvt_tum_out/tum_JOB_atd_*`
namespace (gate only); future training would use `atd_out/`. No new environment,
model download or dataset download is required. Static validation on the Mac:
Python compilation, `bash -n`, and both repositories' `git diff --check`.
Numerical tests and runtime verification remain pending on CAMP.

After publishing fork and parent, submit from CAMP head (only the allocated job
pulls or computes); check quota first. One command chunk:

```bash
getquota
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch adaptive-tokens"
Q="-A students --qos=students_normal -p 24g -w stuttgart"
LOG=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log
sbatch $Q --gres=gpu:1 --propagate=NONE -o $LOG --wrap="$W"
```

Return the job ID, then completion lines or the first failure with context.
Required evidence: tests OK, `ATD GPU CONTRACTS OK`, twelve `ATD DENSE COMPARISON`
messages, six `ATD DENSE REPEAT OK`, `ATD BENCHMARK OK`, six prefix messages, 96 archived runs,
`ATD GATE OK`, `JOB OK`, then archive review. Do not start Stage 2 until that
review confirms provenance, contracts, dense repeat, actual budgets, seed spread,
speed and task-specific oracle headroom.


### Job 26007: oracle backward boundary fix

The user-supplied log reports 17 existing tests passing, then the ATD GPU
contract failing at `sensitivity()` with `RuntimeError: tokens are not contiguous`
in `pi3/curope/curope2d.py` backward. The gate driver had not started; this is an
implementation failure, not evidence about token selection or training headroom.
Inspection of the archived Pi3 source shows backward applies its in-place CUDA
kernel directly to the incoming gradient. `kv_tracker/oracle_rope.py` now uses a
private contiguous gradient copy at that boundary. Sensitivity installs the
adapter on shared RoPE references; no-grad forwards still use upstream RoPE.
A GPU regression checks strided incoming gradients, inverse-rotation derivatives,
unchanged caller gradient storage and forward equality. The existing six-pass
oracle contract remains the end-to-end gate. Local syntax/whitespace checks only;
resubmit the same gate command and review its runtime evidence before proceeding.
