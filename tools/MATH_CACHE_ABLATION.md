# Mathematical cache implementations and first-stage ablation

User authorized all potential mathematical approaches and ablations,2026-10-01.
Model implementation is `kv_tracker/kv_tracker/math_cache.py`; parent tools
capture and replay attention inputs. No learned module or model fine-tuning.
Success for implementation: all readers and paired controls implemented,
causal calibration/evaluation separated, real state/timing reporting, numerical
contracts prepared, permitted static checks completed. Runtime remains pending.

## Methods and comparisons

| Cases | Mathematical intervention | Scope/limits |
|---|---|---|
| exact | Full original historical K/V | FP32 reference reader plus separate native-dtype SDPA timing |
| uniform / probe_balance | Weighted subset / greedy joint numerator-denominator balancing | Same retained count; finite calibration probes, not a BalanceKV reproduction or theorem |
| kernel | Positive Gaussian random-feature sufficient statistics | Persistent omega/S/z/shift; exact current suffix; additive history updates; variance may make a case invalid |
| centroid / moments | Key clustering; projected key covariance and value-key cross-covariance | Same groups, different actual bytes; second cumulant and first value correction; not full MuSe |
| lowrank / query_lowrank | Ordinary / query-metric weighted key factorization | Same ranks, original head scale; value basis fitted independently; no full-vector reconstruction |
| quant4 / waterfill4 | Packed uniform / query-weighted integer bit allocation | Same nominal total code-bit budget; per-channel packing padding/metadata make actual bytes differ |
| *_corrected | Empirical mean and second-cumulant key-noise correction | Additional covariance state; shared historical mass correction versus exact current keys, not token-specific unbiased softmax |

These are prototype matrix readers. There is no claim of new algorithms or
fused kernel performance. Waterfilling uses the separable distortion model in
the original coordinate basis, without AATC whitening or downstream-output
projection weights. Probe balancing uses sequential CPU construction, counted
as write cost. Quantized arrays are genuinely uint8 packed; decoding creates
transient full tensors during reads, included in reader time/workspace. Kernel,
moment and low-rank readers do not retain a dense shadow cache.

Input contract: one head's2-D matrices after normalization/RoPE, all eligible
history visible, no per-query historical masks, exact current K/V optional.
Caller owns masking/causality, updates and rebuilds. Kernel supports additive
append; other methods fit at a cache construction/rebuild boundary. No existing
tracker forward, admission policy or upstream source is changed. Closed-loop
model integration is a later gate, not implied by trace replay.

## Fixed first-stage design

`math_cache_ablation.json` selects one setting for each mechanism,12 arms total;
no automatic parameter search. Each source uses the first8 frames as the bank,
frame8 as calibration and frames9–12 as held-out future queries. Every frame's
query observes only current/past input. Three global layers and first/last heads
are captured with64 query rows: the first five special queries plus evenly
spaced remaining queries, with indices archived. This is a bounded operator screen,
not whole-frame throughput: later efficiency work must time complete query batches.
Source clips: Freiburg1 desk and Freiburg3 structure without texture far.
They are previously available engineering data, not a new held-out benchmark.

Capture observes actual SDPA inputs without replacing its returned output.
Pi3 cached attention bypasses its module call, so hooks scope enclosing decoder
blocks. Pi3's imported SDPA alias and PyTorch F alias are restored on exit.
StreamVGGT uses the existing sparse adapter with no pruning; its native raw
cache is transformed by the original norm/RoPE before capture.

Each head fits once using calibration history/Q only. Every evaluation confirms
the fixed historical prefix is byte-identical; later StreamVGGT cache entries
and current tokens are an exact suffix, common to all cases. Future Q never
enters fitting. Other backbones can supply the documented trace manifest and
tensor dictionary to replay. This API generality is not measured cross-model
success. The default job tests Pi3; StreamVGGT is a separate user-selected host.

## Evidence and review

Reports record error against FP32 dense attention, per-query p99 relative error,
native SDPA fidelity, finite status, actual persistent tensor-storage bytes,
construction time/extra peak, eager read median time and extra read workspace.
Decoder maps, random matrices, counts, scales, offsets, packed padding and
correction statistics count as persistent state. Python object overhead does not.
Shared dense teacher fixtures remain on GPU during replay but are excluded from
method state and incremental peak; these are not whole-process VRAM results.
Saved estimate/reference tensors, including nonfinite results, support
independent error review. Loaded-model tensor-content identity is also recorded.
`summary.json` provides a compact case comparison; completion means execution
finished, not that any candidate passed scientific or efficiency requirements.

Native SDPA uses captured native dtype; prototypes use explicit FP32 reads.
Do not infer production speed from asymptotic work or prototype timing. Paired
cases hold the named parameter fixed; they are not all matched in total bytes.
Compare the recorded error/resource frontier, not just labels or nominal bits.
No thresholds select a winning method automatically. A nonfinite approximation
is recorded as failed, with no fallback; execution/shape failures stop the job.
Exact-reader fidelity is required; partial reports are saved after each case.

Runtime contracts test real packing at every width, waterfilling against a tiny
exhaustive optimum, full-rank identities, original scaling, current/history
mass merging, kernel split-append equivalence, moment response on an analytic
example, empty clusters, coreset mass, input immutability and capture restoration.
They run on CAMP only. The Mac performs AST/JSON/Bash/whitespace checks only.

## User-operated CAMP entry

Publish parent/fork revisions before submission. Code updates occur inside the
allocation; no pull or compute on head. From CAMP head:

```bash
sbatch /mnt/projects/gr/3DRecon/layer_good/tools/math_cache.sbatch
```

If the new wrapper is absent on CAMP, use the existing head-side `sbatch --wrap`
pattern below; this updates the checkout inside the allocation, then executes
the new wrapper. The fixed one-hour GPU allocation runs tests and the two short
operator captures/replays, and archives even a failed/partial run.

```bash
R=/mnt/projects/gr/3DRecon/layer_good
Q="-A students --qos=students_normal -p 24g -w stuttgart"
W="git -C $R -c fetch.recurseSubmodules=false pull --ff-only"
sbatch $Q --gres=gpu:1 --time=01:00:00 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/math_cache_slurm-%j.log \
  --wrap="$W && bash $R/tools/math_cache.sbatch"
```

For StreamVGGT set `MATH_CACHE_HOST=streamvggt` and
`MATH_CACHE_CHECKPOINT` to the verified existing checkpoint file before submission;
the wrapper borrows `optpose.tar` instead of `kvt.tar`. No downloads/installations.
Do not queue this second host behind unreviewed first-host results automatically.

Output `/mnt/projects/gr/3DRecon/math_cache_out/math_<host>_<job>.tar` contains
source revisions/patches, container identity, contract log, pixels/manifests,
Q/K/V/native traces with SHA256, replay outputs and partial/complete reports.
The wrapper never launches tracking evaluation or training afterward.
Review finite status, actual byte reductions and paired errors first, then
decide whether any method merits a live reader integration and further run.
