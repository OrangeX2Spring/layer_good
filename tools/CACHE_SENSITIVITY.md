# A3 follow-up: bounded compression sensitivity diagnostic

Recovered from interrupted local implementation on 2026-09-27. This is separate
from the completed LongStream precision/count experiment in STREAM_CACHE_PRECISION.md.
Hosts: KV-Tracker, LongStream, StreamVGGT. No full-sequence evaluation is included.
Local syntax checks do not establish runtime success. Publish the KV-Tracker fork
change in main.py first, update the parent gitlink, then publish the task tools.

## Fixed protocol recovered from the draft

Use consecutive TUM RGB frames from offset zero at width 308. Calibration is
256 office frames. Evaluation uses the first 512 frames of with-loop and no-loop,
with fresh models/cache for every condition. These are diagnostic clips, not a
held-out benchmark claim. Industrial GT coverage must be reported.

Calibration has 18 conditions: unmodified reference, native identity/profile,
uniform INT4, uniform INT2, and INT2 with each of six contiguous layer groups,
four head groups or four channel groups restored to native precision, K and V
together. Groups partition the cached aggregator layers, heads or head channels;
they do not change weights, queries or head caches. Layer groups cover 1/6 of
layers; head/channel groups cover 1/4 of their respective axes. Compare budgets
within each family, not across families as though identical.

For each family, freeze three group choices after calibration: minimum distortion,
maximum token variance, and seeded random (NumPy seed 0). Ties use the first group.
The distortion score is the mean squared relative translation error normalized by
native RMS step plus squared relative rotation error normalized by one degree.
No GT or candidate trajectory alignment enters selection. Variance sums population
variance over at most 128 evenly spaced tokens per fresh cache entry/rebuild.
Evaluation has 13 conditions: reference, native, INT4, INT2, plus the three frozen
choices for each family. Coincident choices are retained, not resampled.

Streaming hosts retain up to 32 frames with anchor-protected FIFO and all patches;
LongStream's existing refresh limits history to 25 frames. Only fresh entries are
quantized, including refresh seeds. KV-Tracker retains its periodic schedule
(interval 50, cap 20; these clips do not reach the cap), including native duplicated
bootstrap frame. Its freshly rebuilt persistent Pi3 cache is transformed after
each dense rebuild. This is not append-only KV-Tracker and is not a cross-host
equal-history comparison.

Quantization reuses asymmetric grouped affine reconstruction: K over tokens,
V over channels, group size 32, FP32 scale/minimum. Actual cache dtype/storage
stays native. Analytical bytes count the complete low-bit payload, additional
native overwrite values for the protected group, and one 8-byte group descriptor
per resident cache. This deliberately includes redundant low-bit codes beneath
overwrites. It is not a packed kernel or a measured memory/speed saving.

## Required gates and evidence

- Remote cache tests, including test_cache_sensitivity.py and the INT2 scalar/CUDA
  quantization contracts. KV-Tracker also runs its existing TUM tests.
- Native identity versus unmodified reference on every clip (atol/rtol 1e-4),
  plus the existing streaming adapter fidelity checks.
- Identical retained frame/layer IDs, dtype and actual byte counts across arms;
  check streaming pre-refresh and post-refresh payloads. Quantized arms must
  actually change values; identity must not. Each family's group arms must have
  equal analytical budgets at every checked prefix/rebuild.
- Choices saved before evaluation and unchanged afterward. Report native-gauge
  distortion separately from GT-aligned ATE/RPE and scale. No scientific pass
  threshold was present in the interrupted draft; report comparisons rather than
  inventing one after results arrive.
- SENSITIVITY CONTRACTS OK for all three clips, SENSITIVITY DIAGNOSTIC OK,
  JOB_OK file and archived exit status 0 per host. Review failures before retrying.

The driver archives its source/protocol and frozen choices. Streaming archives
contain exact pixels, manifests, model/checkpoint provenance, per-condition events,
predictions and metrics. KV-Tracker preserves three input archives, all_runs and
context (including source, choices, protocol and exit status). Output roots remain
/mnt/projects/gr/3DRecon/stream_cache_out and kvt_tum_out; /tmp work is transient.
No new image, checkpoint or download is needed.

## Execution boundary

Entry point: tools/cache_sensitivity_overnight.sbatch with argument kvt or
streaming (LongStream then StreamVGGT sequentially). Two user-operated submissions,
only after publishing both repositories and reviewing current quota/resources.
The wrapper pulls inside the allocation; do not execute it on head. Use an
allocation-wrapped pull/bootstrap so an old CAMP checkout can obtain the new file.
No submission has been made or inferred from local preparation.

Run the bounded diagnostics first. Review all three host archives before choosing
any longer sweep. Follow logs with tail -n 100 -f; Ctrl-C stops watching.
