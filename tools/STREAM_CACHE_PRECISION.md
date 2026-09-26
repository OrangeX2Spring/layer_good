# A3: cache precision versus retained frame count

Selected 2026-09-26: StreamVGGT and LongStream; office pilot first, then three
TUM sequences only after review. Local implementation is not runtime evidence.

## Fixed experiment

`stream_cache_precision.json` has exactly three new conditions:

| Condition | Frame budget (anchor included) | Quantization | Archived control |
|---|---:|---|---|
| recent8_int8 | 8 | 8-bit | uniform_p50, eight frames |
| recent16_int8 | 16 | 8-bit | recent8 |
| recent32_int4 | 32 | 4-bit | recent8 |

All patches and special tokens are kept. Admission is every frame; FIFO eviction
protects the segment anchor and newest frame. Resolution 308, start 0, stride 1,
native heads, keyframe stride 8 and refresh 4 remain as in the archived sweeps.
LongStream resets every 24 frames: its budget-32 arm reaches at most 25 frames
immediately before refresh, not 32. Do not disable refresh to fill that budget.
StreamVGGT's native camera-head cache still grows; the pilot does not establish
full-run memory safety for the larger dequantized aggregator caches.

The quantizer follows [KIVI's asymmetric axes](https://proceedings.mlr.press/v235/liu24bz.html):
keys group 32 tokens per channel, values group 32 channels per token, separately
for each layer/head. Affine min/max quantization uses FP32 minimum and scale,
round-to-nearest codes, and unpadded short final groups. Constant groups reproduce
the minimum exactly. It is **KIVI-style frame-entry fake quantization**, not a
reproduction of KIVI's residual-window algorithm or kernels.

After prediction and retention, quantize/dequantize only the arriving frame's
aggregator K/V, including its special tokens and anchor. Older entries are never
requantized. Current-frame attention uses native K/V; subsequent frames consume
the restored cache. Refresh replay creates and quantizes a fresh seed. StreamVGGT
keys are raw/pre-normalization/pre-RoPE; LongStream keys are normalized/post-RoPE.
Head caches, positions, weights, query computation and reference state are unchanged.
No rotation arm or quantization-parameter sweep is included.

## Bytes and success criterion

The survey's byte matches assumed bf16 and omitted metadata. The actual runner
loads FP32 weights without an outer autocast. Keep that established computation;
record the actual cache dtype instead of silently changing precision.

`aggregator_bytes` and CUDA peaks are **actual native-dtype storage**. The separate
`analytical_packed_aggregator_bytes` counts ceil(group elements * bits / 8), plus
eight bytes per group for FP32 scale/minimum, for every retained frame/layer/head.
The analytical total adds unchanged head/reference/position/index/feature payloads;
it excludes allocator overhead and is not a measured VRAM saving. Temporary
quantizer tensors and measured simulation latency do not demonstrate packed-kernel
speed or memory. Events preserve pre-refresh and post-refresh accounting.

`precision_gate.json` reports analytical aggregator-byte ratios against a same-dtype
recent8 control (or uniform_p50 for recent8_int8), at every prefix including warmup.
The half-patch control keeps the anchor complete and all special tokens. These are
analytical control payloads, not new control inference. Before drawing conclusions,
verify the archived control's dtype, input hashes, checkpoint, model source,
resolution, head/refresh settings and GT/pair masks against the new provenance.
Use the same-host controls indexed by `cluster_results/scoreboard_25833/rows.json`;
read their per-frame arrays to compute translation-RPE p99, not the scoreboard's RMSE.

Scientific gate, per host: the **same fixed arm** must improve both full-trajectory
Sim(3) ATE and translation-RPE p99 on at least two of office / with-loop / no-loop,
without exceeding its comparator's analytical aggregator-byte budget. Report actual
ratios; do not label unequal sizes exactly matched. If it costs more bytes, the
result is a trade-off and does not pass this gate. No per-sequence arm selection.
Report rotation RPE, fitted scale, valid GT/pair counts, actual peaks, native head
growth and simulation timing as diagnostics. Industrial GT gaps remain excluded;
this is exploratory reuse of existing sequences, not held-out evidence.

## Gates and user-operated execution

Publish the parent changes before CAMP execution. No model fork change is needed.
Pull only inside an allocation. On CAMP head, after publication:

```bash
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$W && bash tools/stream_cache_long.sbatch streamvggt,longstream precision-pilot"
Q='-A students --qos=students_normal -p 24g -w stuttgart'
LOG=/mnt/projects/gr/3DRecon/stream_cache_precision-%A_%a.log
sbatch $Q --array=0 --gres=gpu:1 --propagate=NONE -o "$LOG" --wrap="$W"
```

This runs 256 office frames per host and condition, sequentially, with isolated
workers. It borrows `optpose.tar` read-only and existing checkpoints; no new image
or download. Each host produces its own `stream_cache_out/cache_<host>_<job>_*.tar`.
Existing wrapper traps preserve failures, exact pixels, source/config, checkpoint
hashes, events, predictions, metrics, tests in run.log, and exit status.

Review all cache contract tests (including the new scalar-reference, axes,
short-group, constant-group, old-entry preservation, eviction, attention-consumption
and CUDA-dtype checks), native fidelity, all three RUN OK records,
CAMERA EVALUATION OK, PRECISION CONTRACT GATE OK, JOB OK and archived exit 0
for **each host**. Inspect finite metrics, actual dtype, byte ratios, observed
retention counts and LongStream refresh. The native fidelity worker is unquantized;
it validates the adapter, not accuracy preservation under quantization.

Only after reviewing the pilot, use stage `precision` in place of
`precision-pilot` for the same host list. That runs office (2585), with-loop
(5182) and no-loop (3359), each as a separate archive, for 18 condition runs.
Do not chain or submit the full stage behind an unreviewed pilot. Review quota
and pilot archive sizes first. Runtime contracts are repeated for every scene.
Follow logs with `tail -n 100 -f <log>`; Ctrl-C stops watching. Transfers remain
user-operated through the field-notes rsync route; extract each archive separately.
