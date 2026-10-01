# StreamVGGT historical-KV depth pilot

This is the accepted bounded second-backbone/longer-history follow-up to the
reviewed Pi3 26086/26087 archives. Numerical findings and their caveats live in
FINDINGS.md. No Pi3 layer indices transfer. No weights, blocks, FFNs, current
tokens, heads or model defaults change. The new fork controller is opt-in.
Controller revision: StreamVGGT `18b3d23` (published main).

## Fixed protocol and gates

Use the existing strict-load StreamVGGT checkpoint
`/mnt/projects/gr/3DRecon/streamvggt_ckpt/checkpoints.pth`, FP32 weights/activations/
KV with no autocast, and the borrowed read-only `optpose.tar` image. The existing
stream_cache_job.py dependency bootstrap installs only absent inference packages
inside the disposable job container, constrains all existing package versions,
archives before/after inventories and the pip report, and verifies full model
import before contracts or data preparation. Setup and all pilot subprocesses
share one container. No saved-image changes, model/data downloads, training,
budget sweep or automatic next job.

Job26088 stopped before calibration because the original wrapper bypassed this
bootstrap and optpose lacks transformers. The corrected wrapper uses the same
constrained setup as established StreamVGGT jobs; the scientific protocol is
unchanged. Its package installation/import and all pilot gates require remote
verification. The supplied failure tail does not establish final archive exit
status or the earlier contracts' result.

1. Two GPU contract tests cover untouched-native equivalence, zero-storage
   first-layer history, current-only attention equivalence, preservation of
   subsequent-frame special embeddings despite empty history, matching K/V/ID/
   RoPE gathers, special preservation, exact equal bytes and FIFO eviction at
   divisible and non-divisible per-layer token budgets.
2. On the loaded checkpoint, compare every aggregator intermediate over the
   first three desk frames with untouched upstream cached attention at
   atol/rtol 1e-5. A failure stops before calibration/evaluation.
3. Calibration uses desk source 0..39, with 32 consecutive native bank frames
   and eight subsequent queries. Each single-layer history omission restores
   the identical native bank **and camera-head cache**. Query camera memory
   evolves within its eight-frame intervention; aggregator query frames are
   not admitted. All global attention calls execute. Scores use native c2w
   translation squared divided by bank RMS translation radius, plus rotation
   Frobenius squared/6. GT is not used. Freeze the lowest eight of 24 scores,
   with layer-index tie breaking. No joint-additivity assumption.
4. Evaluate office (`freiburg3_long_office_household`) and with-loop
   (`freiburg2_large_with_loop`) source 256..511, 256 consecutive frames each.
   These are separate windows from calibration, but are previously explored
   sequences: this is an engineering generalization pilot, not held-out test
   set evidence. Each of six arms starts in a fresh process. A discarded
   first-image warmup resets both aggregator and camera state before scoring.
5. All arms admit frames 0,4,...,252 and retain the last 32 admitted frames.
   Current tokens are always dense in every layer. Frame 128 onward has 32
   historical slots before every query. Native means native KV precision and
   dense historical rows under this common fixed-admission/FIFO protocol,
   **not upstream unbounded all-frame inference**.
6. `omit` stores no history at the frozen eight global depths and full history
   at the other 16. `uniform_special` keeps all five special rows in every
   admitted frame at every depth, plus evenly spaced original patch rows.
   Across 24 depths, integer per-frame budgets sum to exactly 16 times the
   dense frame token count. The pattern per depth/frame remains fixed so an
   older sparse frame never needs discarded rows restored.

## Representation and accounting

StreamVGGT caches raw pre-normalization/pre-RoPE K/V. Each ragged layer retains
original integer RoPE coordinates and token IDs alongside its paired rows.
Normalization, rotary embedding and native SDPA run when queried. Native
frame/global blocks and all normal inference heads execute; tracking head is
inactive because no query points are supplied, as in upstream camera inference.

Physical empty pairs have layout B,H,1,0,D. The non-None frame-axis sentinel
preserves upstream subsequent-frame special-token selection even when global
layer zero has no historical rows. This sentinel allocates no tensor storage.
All gathers materialize compact storage; unused Q backing storage and dense
historical shadows are absent in live arms. A dense teacher exists only in the
separate calibration process.

Every post-query write asserts actual unique untyped-storage bytes equal the
exact row budget times the K/V + two position coordinates + one int64 token-ID
cost. Omit/control must match and equal exactly two thirds of native at every
frame. Camera-head cache semantics remain unchanged, including its refinement
iteration history. Its physical bytes must agree across arms and are reported
separately. Python bookkeeping is not tensor byte accounting. Transient dense
current queries/concatenations remain included in whole-process peak CUDA
allocated/reserved bytes. A persistent-cache saving does not imply peak-VRAM
or speed improvement.

## Review criteria and artifacts

Full-window and steady-history (frames128..255) ATE, translation p99 and
translation/rotation RPE are reported. Each arm fits one full-window Sim(3),
including the steady subset; GT association is <=20ms, adjacent RPE never
bridges invalid GT or a >100ms interval. Raw c2w trajectories, matching
GT/pixels/manifests, saved alignment and per-frame errors permit review.

Query timing is synchronized aggregator + camera/depth/point heads + c2w
conversion through the common instrumented reader. Retention costs are timed
separately; report steady query p50/p95 and query+write p50. Image IO is outside
these timers. This is one run per arm in one allocation, not repeated statistical
speed evidence. No thresholds/sweeps are selected by the wrapper.

Runtime success: both contracts pass, native fidelity passes, frozen calibration
exists, all six records and exact-byte/admission/camera comparisons pass,
summary.complete=true and archive exit_status=0. Scientific outcome is reviewed
afterward: compare omission with equal-byte uniform on both scenes and with
native, explicitly record mixed tails or accuracy loss. Only then select any
next experiment; no full evaluation is queued behind this pilot.

Archive: `/mnt/projects/gr/3DRecon/stream_depth_out/stream_depth_JOB.tar`.
Includes source pixels and prepared model pixels, manifests/GT, checkpoint hash,
model-source tar, source identities/patches, image identity, contract/fidelity
logs, calibration scores/raw poses, configs, six trajectories/evaluations,
per-frame storage/timing events, summaries, partial failure output and exit
status. All construction stays in job-local /tmp; /mnt mounts read-only.

## Submit on CAMP head

Use the existing Q/W/R setup: W updates the parent inside the allocation,
then this wrapper updates the pinned model fork there. Do not pull on head.

```bash
sbatch $Q --gres=gpu:1 --time=01:00:00 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/stream_depth_slurm-%j.log \
  --wrap="$W && bash $R/tools/stream_depth.sbatch"
```

Return job ID, first failure with context, or completion/archive lines. Transfer
the single archive and Slurm log from Mac repo root into a separate
`cluster_results/stream_depth_JOB/` directory using the field-notes rsync
procedure. A source-only static check does not verify any runtime gate.
