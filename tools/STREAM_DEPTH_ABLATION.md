# StreamVGGT historical-KV depth pilot

## Overnight diagnostics and separately gated full validation

User requests broader overnight experiments after26089 review; anchor pilot
is already submitted as26090. Do not duplicate it. `overnight` is an independent
FIFO diagnostic suite; serialize it after26090 to prevent allocation-time pulls
from changing the shared checkout while another driver's child processes read it.
No anchor result is selected, trusted or promoted automatically by this suite.

One sequential GPU allocation,4h cap,40 fresh-process runs,14848 frame queries:

| Question | Fixed cases | New runs |
|---|---|---:|
| Is the layer choice meaningful? | Worst eight calibration layers and one random eight (seed17), two exact26089 windows, equal byte budget |4|
| Does the timing advantage repeat? | Native/omit/special-uniform, three process repeats per window, shuffled order seed29 |18|
| How do bytes/time/errors scale with history? |8/32/64 slots, all three arms, office/loop source0..511 |18|

Teacher[0..7] stays frozen. Reverse selection derives once from the archived
single-layer scores, random choice derives once from a declared seed. These
are negative/equal-byte controls, not a search that selects a new winning policy.
No new calibration/model/training/quantization/threshold sweep. All architecture,
precision, admission stride4 and camera memory semantics are unchanged. Original
ref/native/omit/uniform outputs are copied as labelled historical evidence;
the first two groups read byte-identical archived model inputs. Scaling cases
share one prepared512-frame input per sequence. For capacity C, steady metrics
start at4*C, when every query has C history slots. Full and steady GT-valid
metrics, actual bytes/peak GPU usage, all raw poses/events and first-eviction
RPE at4*C->4*C+1 are saved. Scaling uses FIFO and cannot establish an anchor
quality result; its event alignment tests the existing eviction hypothesis.

Every repeat must reproduce its reviewed FIFO raw trajectory at atol/rtol1e-5.
Time summaries keep all three process medians/min/max, rather than reporting the
best run. No cross-job historical timing claim. Each independent subprocess
failure is logged with traceback and retained in the incremental summary;
other independent cases continue. The suite finishes nonzero if any case failed.
Do not automatically retry failed negative controls or hide them as successes.
Setup/reference/source or orchestration assertion failures stop immediately.

Archive `stream_depth_out/stream_depth_overnight_JOB.tar` includes the declared
plan, incremental/final overnight_summary.json, old references, exact pixels,
all new configs/trajectories/evaluations/events/logs and source/package/container
provenance. Shared images are stored once, no saved environment changes.
Runtime pass requires contracts/reference gates and40 recorded successful cases,
complete=true and archive exit0. Scientific conclusions require raw review.

Submit on CAMP head after publication, using existing Q/W/R:

```bash
sbatch $Q --gres=gpu:1 --time=04:00:00 --propagate=NONE \
  --dependency=afterany:26090 \
  -o /mnt/projects/gr/3DRecon/stream_depth_overnight_slurm-%j.log \
  --wrap="$W && bash $R/tools/stream_depth.sbatch overnight"
```

The dependency serializes source updates; afterany is intentional because these
FIFO diagnostics do not depend on a successful or reviewed anchor result. The
recorded students_normal limit permits one running and two submitted jobs, so
one sequential suite fits beside26090. Actual scheduling remains user-observed.

**Prepared but not submitted:** `full` stage runs native/omit/special-uniform
with32 anchor slots on complete office2585/with-loop5182/no-loop3359 frames,
nine runs33378 queries. It requires manually supplied REVIEWED_ANCHOR_ARCHIVE
and REVIEWED_ANCHOR_SHA256 from a scientifically reviewed exit0 anchor pilot,
checks SHA/completion/six prefix+byte comparisons, and records the review identity.
Missing fields stop before full-data preparation. There is no automatic
diagnostic/pilot-to-full chain and no full submission command for the unreviewed
26090. AGENTS.md requires reviewing pilot evidence before full evaluation.
After that review, supply one short full-stage submission chunk separately.
Output would be stream_depth_out/stream_depth_full_JOB.tar; none exists yet.

## Accepted anchor comparison after reviewed26089

Entry: `bash tools/stream_depth.sbatch anchor`. Controller revision `b5b4f05`.
This is one fixed intervention selected by the user after archive review. The
reviewed FIFO protocol below remains the reference, not a run to repeat.

- Preserve the first admitted frame0 plus the latest31 other admitted frames.
  Capacity remains32, admission every fourth frame, all layers/current tokens/
  weights/heads unchanged, frozen omissions[0..7]. Compression remains exactly
  two thirds of native historical KV+positions+IDs at every frame.
- Pin the existing `stream_depth_out/stream_depth_26089.tar` SHA256
  `389b561efc782d3aab6fc7465d16cc4a2371ea7a51423991a4b91beabb27e205`.
  Require saved exit0, six complete records and original fork18b3d23. Extract
  exact source/model pixels, manifests/GT, selection/raw teacher/fidelity and
  labelled historical FIFO results. No data preparation, teacher calibration
  or native-fidelity rerun. Copied teacher/fidelity records are historical.
- Verify current checkpoint SHA equals the archived identity and current
  dependency package/version inventory equals the reviewed post-setup inventory.
  Reuse the established constrained setup in one disposable container; no saved
  image or model/checkpoint download/change.
- Run the affected two existing GPU contracts plus one new contract verifying
  original anchor K/V/positions/IDs remain bit-identical across admissions and
  skipped frames, with exact fixed-capacity/equal-byte three-arm budgets.
- Six new fresh-process anchor runs use office/with-loop source256..511 and
  native/omit/special-preserving uniform. Assert actual retained IDs match the
  anchor schedule after every query, and every frame's physical history/camera
  bytes match the corresponding FIFO reference. All three new arms must match
  each other's frame IDs and the existing two-thirds budget.
- Check first129 raw poses against their FIFO reference at rtol/atol1e-5:
  query128 precedes the first differing post-query eviction, query129 follows it.
  A failure stops this pilot; do not waive it or queue another run automatically.
- Report all full/steady quality and current-job timing/peak metrics, and pair
  128->129 translation/rotation RPE, squared-error fraction, raw translation
  step, and translation RPE under the **fixed old FIFO alignment**. This prevents
  a new full-trajectory scale fit alone from explaining the event improvement.
  Historical FIFO timing is descriptive only; compare new arms within this job.

Runtime gate:3 contracts pass, reference/checkpoint/packages pass, six runs,
prefix and byte comparisons pass, complete summary with six paired comparisons
and labelled historical records, final archive exit0. Scientific success is
reviewed separately: does the anchor shrink the targeted discontinuity under
the fixed alignment and improve overall errors without unacceptable regression?
The common-FIFO diagnostic is a hypothesis, not proof before this intervention.

New archive: `stream_depth_out/stream_depth_anchor_JOB.tar`. Includes original
shared pixels/teacher and FIFO reference outputs once, six new results,
reference identity, package/source provenance and pair comparisons. Construction
is job-local; old archive/image/checkpoint remain read-only. No sweep/training/
automatic successor. On CAMP head, with existing Q/W/R:

```bash
sbatch $Q --gres=gpu:1 --time=01:00:00 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/stream_depth_anchor_slurm-%j.log \
  --wrap="$W && bash $R/tools/stream_depth.sbatch anchor"
```

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
