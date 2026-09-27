# StreamVGGT learned dynamic history: minimal two-implementation pilot

Selected 2026-09-27. Frozen StreamVGGT, native precision, all incoming frames and
all patches processed, width 308. No other host, bit-width experiment, training of
the backbone or full sequence is included. Local verification is static only;
all tests, preparation, controller fitting and inference execute on CAMP.

## Question and two implementations

Can a tiny causal controller allocate history more usefully than a fixed budget
or a predetermined alternating schedule? The action chooses **4 or 16 past
frames**, including frame zero as anchor and the newest budget-minus-one frames.
The current frame is additional. Thus maximum attention is 5 or 17 frames,
consistent with the established runner's post-prediction recent4/recent16 caches.
Decisions occur every 8 frames, starting at frame 16; dynamic arms use budget16
during warmup. Neither variant skips incoming frames or invents lost history.

1. **Active history:** retain anchor plus 15 recent frames in native aggregator
   K/V. Gather selected K/V and their matching positions before normalization,
   RoPE and attention. Persistent raw K/V retains all 16 frames. Increasing active
   history immediately reuses available entries, whose deeper representations
   were produced under their original attention histories. This may reduce
   attention work; it does **not** reduce persistent aggregator storage.
2. **Stored history:** physically prune aggregator K/V and positions to the chosen
   budget. When growth requests missing entries, rebuild from anchor plus recent
   **past** prepared RGBs with chronological aggregator-only inference, then
   predict the current frame. Never replay the camera head: that would duplicate
   observations in its native causal history. Rebuilt K/V has recomputed context,
   so this is not semantically identical to restoring the active-history bank.
   Report recovery frames, time, attention work, resident bytes and CUDA peak.

Both share the same frozen controller and attention-selection code. Camera-head
history remains native and grows. Replay reads archived prepared RGB from local
disk; its loading/recomputation cost is included in per-frame and total time.
This is an explicit storage/replay trade-off, not a free-memory-recovery claim.
Input archives contain the full clip for reproducibility, but recovery may only
read indices smaller than the current frame. Source RGB storage is not KV storage.

## Learning and fixed data split

- Calibration: first 256 consecutive office frames, start zero, stride one.
- Evaluation: first 256 with-loop frames, then first 256 no-loop frames, each from
  a clean model/cache. These familiar TUM sequences are not an unseen benchmark.
- At each calibration frame after warmup, branch low/high attention from exactly
  the same high-history teacher state. Snapshot mutable aggregator/camera cache
  lists and positional metadata; discard the low branch and commit the high one.
  The full committed trajectory must match an independently run fixed16 trajectory.
- Target: low-versus-high current-pose disagreement, using translation normalized
  by calibration teacher RMS motion and rotation normalized by one degree. Average
  the one-step counterfactual losses over each next 8-frame decision block. This
  produces 30 training samples, not 240 independent block examples.
- Features: current grayscale thumbnail change (mean/p90), current spatial gradient,
  **previous** confidence mean/std in log scale, previous translation/rotation step.
  No GT or current-frame tracker output is available to the decision. Features at
  block start predict its subsequent loss; future frames enter training labels only.
- Fit standardized linear ridge regression to log1p(loss), fixed regularization 1.
  Threshold at the median calibration prediction, aiming for half high actions on
  calibration. Save means, scales, weights, threshold and feature names. Freeze and
  hash this file before evaluation. Test action rates are not forced to 50%.

This is **supervised fidelity distillation**, not learning which history has better
GT accuracy. High history is a teacher, not guaranteed ground truth. A controller
trained on high-history states also faces distribution shift in closed-loop use,
especially after stored-cache rebuilds. The pilot measures that failure mode; a
negative result cannot rule out stronger controllers, other features, or GT-aware
training. No reinforcement learning or threshold search is included.

The calibration screening report compares hindsight top-half high allocation with
expected uniformly allocated high actions at the same count. It is a **teacher-state
one-step diagnostic**, not a closed-loop oracle or a bound on GT tracking quality.
Both its labels and the learned training fit are in-sample; do not report them as
evaluation accuracy.

## Controls and implementation gates

Before training, use a separate 32-frame office implementation clip:

- Existing stream_cache_sweep.py recent4/recent16 as independent controls.
- Active and stored fixed4/fixed16 must match those camera extrinsics/intrinsics
  with atol 1e-5, rtol 1e-4. Existing native adapter fidelity runs too.
- Forced low/high alternation must shrink and grow stored memory and execute
  recovery; active history must never replay. Tensor tests also check selected
  attention versus physical pruning, matching positional indices, persistent
  cache preservation, snapshot isolation, causal replay and camera-cache isolation.
- Calibration counterfactual branching must preserve the independent fixed16
  trajectory at the same tolerances. Failure blocks controller fitting/evaluation.

On each evaluation clip run seven fresh workers: active fixed4, fixed10, fixed16;
active alternate, stored alternate; active learned, stored learned. Fixed-budget
backend identity is established by the gate, rather than repeating all fixed
controls under both backends for the entire pilot. Alternation uses low/high
8-frame blocks with the same warmup and decision cadence as learning.

Every prediction logs requested budget, exact attended/stored/replayed IDs,
latest available prediction index, features, score, cache bytes, time and peaks.
Contracts verify anchor/FIFO frame identities, no future replay, decision dwell
and actual recovery. The output records whether a learned policy ever switches;
a constant policy is **inactive adaptation**, not an implementation crash or
evidence that dynamic history cannot work.

## Evaluation and completion

Report per clip GT Sim(3) ATE, translation RPE p99, rotation RPE, fitted scale and
valid GT/pair counts; exclude gaps under the existing 20 ms/100 ms conventions.
Also report closed-loop relative-pose disagreement versus fixed16, average attended
past frames, active/stored aggregator and camera bytes, actual CUDA peaks, replay
count/time, total synchronized time and attention-pair counts including replay.
Attention-pair counts are an operation proxy, not measured FLOPs/speed. Total time
includes signals, loading, gathering, replay and prediction; excludes model loading
and final artifact serialization. Single-run timings need caution.

Fixed10 is a nominal midpoint reference, **not automatically an equal-cost
comparator**. Compare actual cost: warmup, high-action fraction, gathers, replay
and unmodified camera caches all matter. Learned-versus-alternating comparisons
also need their actual budgets; frozen thresholds may yield unequal action counts.
Do not tune thresholds on evaluation to manufacture a matched comparison.

Technical success: all remote tests and native/implementation/counterfactual
identity gates pass; all seven evaluation arms produce finite outputs; both
evaluation contract/comparison reports exist; controller hash is unchanged; final
DYNAMIC HISTORY PILOT OK, JOB_OK and archived exit status 0. Runtime success does
not itself establish the research hypothesis. Inspect the measured accuracy/cost
trade-off before selecting any longer run or changing the controller. No automatic
full run is scheduled. Report implementation failure, inactive policy and actual
negative accuracy/cost evidence separately.

## CAMP execution and artifacts

Entry point: tools/stream_cache_dynamic.sbatch. Publish parent task files first;
no model-fork change. Bootstrap via allocation-wrapped git pull, never pull on head.
Review current quota/queue/node availability before supplying a submission command.
Borrow existing optpose.tar and StreamVGGT checkpoint read-only; no new image or
model download. The usual stream_cache wrapper installs only its existing missing
dependencies, captures revisions/patches, runs cache tests and archives failures.

Output: /mnt/projects/gr/3DRecon/stream_cache_out/cache_streamvggt_<job>_*.tar.
Log: /mnt/projects/gr/3DRecon/dynamic_history-<job>.log when specified at submission.
The archive includes prepared exact pixels/manifests, source/protocol, checkpoint
and dataset hashes, controller and screening report, independent reference/gate
runs, events, metrics, comparisons, tests/run.log and exit status. Working files
remain job-local /tmp. Follow logs with tail -n 100 -f; Ctrl-C stops watching.
Transfers are user-operated through the field-notes rsync route, with one extraction
directory per archive. No submission or runtime outcome is implied by this file.
