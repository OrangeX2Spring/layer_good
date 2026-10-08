# Seven loop-transformer implementations — 2026-10-08

User selected implementation of all seven researched directions. Research and
paper links: [LOOP_TRANSFORMER_RESEARCH.md](LOOP_TRANSFORMER_RESEARCH.md).
These are geometry prototypes inspired by the papers, not reproductions of
their language/VLM architectures or demonstrated tracker improvements.

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
matching full-schedule history slots. Rebuild KV is detached; gradient training
is through current-frame computations, with block checkpointing. These initial
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
queries return poses only, not new dense reconstruction. Its smoke learns
individual bank-conditioned queries, not full temporal backpropagation. The
nested schedule is a geometry adaptation of hierarchical recurrence, not LoopVL.

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
   processes each run20 AdamW updates at1e-4. This tests training plumbing;
   it is not a sufficient training budget for a scientific claim.
3. **Calibrate adaptive stopping within the engineering stage.** Fixed25th
   percentiles from safe development exits are accepted only if they select at
   least one exit and no unsafe development exit. Safety here means both
   map-scale-normalized translation and rotation chordal error≤.05 versus the
   frozen teacher. Unknown map support cannot trigger stopping. Calibration
   failure is recorded explicitly; an uncalibrated adaptive model cannot deploy.
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
   requires that variant's passing pilot gate tied to the same checkpoint archive.
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
weight identities and complete changed-state keys. Inference does not load the
teacher model. Checkpoints contain no episode caches or retained query graph.
Unique parameter, KV and auxiliary map-state bytes are reported separately;
actual peak allocated/reserved memory and device samples are also archived.

Evaluation uses identical pixels, keyframe IDs and GT/RPE associations. The
three-metric gate allows at most5% regression in ATE, translation RPE and
translation-RPE p99. A2x FPS or half-memory success additionally requires that
quality gate. Timing includes upload, queries, all bank updates and CPU pose/log
output, but preloads RGB and is scene-only. Do not call it complete video/SAM FPS.
Two paired observations are diagnostic evidence, not statistical significance.
Memory measurements include full resident model and refiner/map state; native
and compact controls expose differences in KV backing storage.

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
