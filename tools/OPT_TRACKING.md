# OPT-Pose frozen-cache tracking comparison

Updated 2026-09-10. **Implementation and local static checks only; the revised
pipeline has not run on remote Linux. No new FPS, VRAM or accuracy result exists.**
The user runs all cluster commands and transfers. The CAMP skill, local
`docs/camp-cluster-field-notes.md` and `docs/opt-pose-cluster.md` remain operational
references; this tracked document reaches the cluster.

## Why revise Step 1b?

The lost log cannot be identified uniquely. The 2026-09-04 local, gitignored record in
`tools/FINDINGS.md`, “Step 1a”, documents these historical GPU failures:

- A 32-reference cache claimed 8.05 GiB but pinned 16.1 GiB: value tensors were
  views into the fused Q/K/V projection. The existing `clone()` fix stays.
- Cache construction retained all 24 concatenated layer outputs, plus underlying
  frame/global tensors. The aggregator-only fix discarded all outputs.
- The ordinary comparison exhausted VRAM before the cached path did. Existing
  `--cache_only` was intended for those large-reference timing sweeps.

Static inspection of the current `opt_pose` base `fdb19bd` found additional faults:

1. Full-model cache construction passed that **empty output list** into the camera
   head, which indexes `[-1]`. This is a deterministic code-path defect, not an
   inferred cause of the lost OOM log.
2. Construction would otherwise continue into reference dense decoding and the
   NOCS/Sonata/object-pose branch; these are unnecessary for camera tracking.
3. The old end-to-end benchmark kept the real cache resident during baselines and
   allocated another full cache for its zeroed control. It ignored `--cache_only`,
   reported a fabricated `(0, 0)` REPEAT, and omitted its precision control.
4. Checkpoint loading put the checkpoint on CUDA while moving model weights there,
   creating a second GPU copy. The loader now maps checkpoint storage on CPU with
   `mmap=True` and releases it before `model.to(device)`. Checkpoints must be the
   existing torch-save zip-format files; there is no silent format fallback.

The earlier prose mentioning `c951337` is historical; the local submodule HEAD
inspected for this revision was `fdb19bd`.

## Implementation and boundaries

`OPT.forward_tracking` supports three attention paths with identical requested
outputs: ordinary bidirectional inference, recomputed causal readout, and one-frame
cached readout. `OPT.forward(..., collect_cache=...)` and `opt_cache=...` dispatch
into it; `tracking_mode="camera"|"geometry"` explicitly selects outputs.

- **Build:** jointly process reference frames; store the 24 global blocks' K/V
  and normalized final camera tokens. Retain only final camera token 0, never
  reference maps. No Sonata, NOCS, metric-scale head or object-pose solver runs.
- **Camera query:** retain only final camera token 0 and run the camera head. Its
  small trunk recomputes gated reference/query pose tokens over four iterations.
- **Geometry query:** retain layers required by the depth and point heads (currently
  4, 11, 17, 23), preserving layer indices with `None` placeholders elsewhere.
  Run camera, depth and point heads. Confidence outputs are preserved too.
- **Checkpoint compatibility:** no parameters or state-dict keys change. Ordinary
  full inference still retains all its outputs. The verification compares selected
  outputs against this original full forward, including its object branch.
- **Precision:** bf16 autocasts the aggregator, preserving existing fp32 head
  behavior. No cache quantization, token pruning, lower input resolution or training
  is introduced. The measured historical cache size is **193.2 MiB/reference in
  bf16**, not 128.8: K remains fp32 after normalization and V is bf16.

The experiment explicitly caps references at three, plus one query, within the
checkpoint's trained 2–4-frame window. It processes arbitrarily many query frames
one at a time using that **same frozen reference set**. Default: 50 queries, one
object sequence, three initial consecutive references, 518×518 input.

This bounds cache storage with respect to video duration and keeps its reference
frame fixed. It does not remove O(reference-count) cache storage or the quadratic
cost of jointly building many references. Estimated K/V payload for three
references is about 580 MiB in bf16; **total peak VRAM still needs measurement**.
Host data memory is bounded by a reference set plus a query and temporary outputs;
only small per-frame statistics accumulate in RAM.

This is the frozen-memory part of KV-Tracker, not its complete live application:
no online keyframe selection/replacement, segmentation tracker, relocalization or
object-frame alignment across cache rebuilds. Cache refresh would need a jointly
rebuilt bounded set that retains frame 0, with pose continuity evaluated before
claiming a working adaptive tracker. A fixed small cache can lose useful coverage
at large viewpoint changes; the long-clip comparison measures that limitation.
The per-frame pose is the camera head's pose in model coordinates, **not** the
released object head's frame-0-referenced 9-DoF object pose.

## Comparisons and success criteria

`test_tracking_housecat6d.py` runs each method in a **separate process**. The wrapper
executes `prepare`, `verify`, `cached`, `original`, `readout`, then renders a report.

| Method | Work per query | What the comparison means |
|---|---|---|
| Original | Bidirectional references + query; selected camera/geometry heads | Training-time attention behavior on the same frames |
| Readout | Same frames and heads, references gated from query | Directionality control; reference computation repeated |
| Cached | Query only, attending to immutable reference K/V | Savings from avoiding reference computation |

“Original” is deliberately a **matched-output** baseline. It gets the same output
retention optimization and skips the same unused object branch. Thus the reported
speedup is not inflated by making the original solve additional tasks. The full
legacy forward is an untimed verification reference, not the speed denominator.
The old `tools/opt_pose_kvcache.sh ... e2e` and direct `--end_to_end` harness now
fail explicitly with migration instructions; aggregator-only historical sweeps
remain available and are not the new streaming experiment.

Remote gates must pass before the result is accepted:

1. Prepared frame IDs exactly match requested IDs: the dataset loader sometimes
   substitutes random valid frames. Such substitutions fail loudly.
2. Output retention matches the full original forward on the first query.
3. Cached output matches recomputed causal readout per output key. At fp32 the
   relative max-error floor is 1e-4. At bf16/tf32 use the larger of 1e-4 and the
   measured readout-versus-fp32 difference for that key. This is an arithmetic
   fidelity gate, not an accuracy threshold.
4. A separately rebuilt cache repeats the output within 1e-4. The first cache is
   freed before the second build. Zeroing K/V and camera tokens **in place** must
   worsen max absolute error by at least 100× the fidelity error, per key.
5. A bytewise SHA-256 fingerprint of every cache tensor stays unchanged after
   different queries and after the complete streaming run. Each tensor is copied
   to CPU individually for this untimed check. Tensor storage bytes must equal
   payload bytes, catching the old projection-view retention bug.
6. The report checks cached-versus-readout fidelity for **every recorded query**,
   checks finite predictions, and decodes each complete MP4 to verify its frame
   count and 1400×800 dimensions. Remote visual inspection of PNGs/videos is still
   required; a successful decode does not establish good visual layout.

If a gate fails, stop and inspect its log and saved `checks.json`. Do not turn up
its tolerance or run larger jobs blindly. After roughly three failed attempts,
record the attempts and best hypothesis before further changes.

## Timing, memory and statistics

Per-query time is synchronized host wall time around model inference, including
CPU dispatch. Inputs are already on the GPU. It excludes loading, cropping,
segmentation, CPU↔GPU transfers, archive writing and visualization. It is **model
throughput**, not live-video pipeline FPS. Disk-backed replay adds gaps between
calls, so this is not a continuously saturated GPU throughput benchmark.

Each method warms up on its first query. Cached construction also warms up with
independent builds, freeing each before the next. The first cold-build timing is
recorded separately; summaries charge one warmed build. Model/checkpoint startup
is excluded. Model weights and unused heads remain loaded equally for both
methods; this revision avoids a second CUDA checkpoint copy but does not strip
checkpoint modules.

For query latencies `original[i]`, `cached[i]` and measured build time `B`:

```text
steady speedup = sum(original) / sum(cached)
net seconds saved = (sum(original) - B - sum(cached)) / 1000
speedup including build = sum(original) / (B + sum(cached))
break-even = first query count j with sum(original[:j] - cached[:j]) > B
```

A negative net saving is retained, not hidden. A separate median-based break-even
estimate is labeled as an estimate; observed first crossing may not remain
positive. No cache rebuilds occur, so there is no unmeasured refresh cost in this
protocol. Original-versus-cached includes the change in attention direction;
readout-versus-cached isolates reuse with the same visibility rule.

Reports include median/p95 latency, model throughput, time saved with construction,
cache payload, phase-specific CUDA peak allocated/reserved bytes, and Linux
process-lifetime peak host RSS. `nvidia-smi` and `ulimit -m` are recorded at startup.
Torch memory excludes driver/context allocations and other processes. Host peak
RSS is process-lifetime, not an instantaneous per-frame delta; Slurm `MaxRSS`
should also be inspected when diagnosing a host OOM.

## Artifacts and visual interpretation

The unique run directory contains:

- `manifest.json`: arguments, frame IDs and original paths, input SHA-256 hashes,
  checkpoint SHA-256, model commit, GPU, torch and CUDA versions.
- `seq*/references.npz`, `query*.npz`: exact normalized model images, original crop
  pixels, masks, cropped intrinsics, chosen indices, NOCS/depth inputs, crop boxes
  and original/symmetry-adjusted extrinsics. The revised tracking path consumes
  only RGB by default; the other inputs preserve the legacy verification record.
- `verify/`: per-key fidelity, precision, repeat and zeroed controls.
- `cached/`, `original/`, `readout/`: query predictions, per-frame timing/memory
  JSONL (flushed after each frame) and method summaries.
- `report/summary.{json,csv}`, per-sequence frame CSVs, MP4 comparisons and first/
  last PNGs. Camera mode shows translation and rotation differences; geometry
  mode shows same-scale depth maps and fixed-scale absolute difference maps.
  Point-map masked distances and camera differences are also in the CSV.

Differences are **prediction discrepancies**, not ground-truth errors. Translation,
depth and point-map distances are in model units; no metric alignment is silently
applied. Bidirectional inference can change the reference coordinate gauge as the
query changes, so raw discrepancies may include gauge shifts. Quaternion angular
difference is sign-invariant. No ATE, pose mAP, reconstruction quality or 9-DoF
accuracy claim follows from these plots. HouseCat6D annotation masks supply crops;
this does not benchmark a front-end segmentation tracker.

The wrapper saves environment details, exact relevant source/config files in
`source.tar.gz`, patches and stage logs alongside `run/`, then stages everything
as **one uniquely named tar** under `/mnt/projects/gr/3DRecon/optpose_tracking_out/`.
Normal errors trigger the archive trap as well. SIGKILL, node loss and allocation
termination cannot guarantee a trap runs; the external Slurm log persists. Results
are never placed inside a git checkout. Run names prevent overwriting old evidence.

## Publish local changes

The model revision was committed and pushed as `271be55` on 2026-09-10.
The superproject pin and tools are published separately. For subsequent revisions,
commit the model submodule first, then the superproject pin/tools. These commands
enumerate this task's files so unrelated working changes remain outside the commits:

```bash
git -C opt_pose add opt/models/aggregator.py opt/models/vggt.py
git -C opt_pose add test_abs_housecat6d.py test_kvcache_housecat6d.py
git -C opt_pose add test_tracking_housecat6d.py
git -C opt_pose commit -m "Bound OPT tracking memory and add streamed cache comparison"
git -C opt_pose push

git add opt_pose tools/opt_pose_kvcache.sh tools/OPT_TRACKING.md
git add tools/opt_pose_tracking.sh tools/opt_pose_tracking.sbatch
git add tools/opt_pose_tracking_report.py
git commit -m "Document and package OPT frozen-cache tracking comparison"
git push
```

## Run on CAMP (user executes)

Check current allocation eligibility from `head` first:

```bash
sinfo -p 24g -N -o "%.10N %.9T %.15G"
sacctmgr show user chunquancheng format=Defaultaccount%30,Defaultqos%30
```

`muenchen` is a documented sm_86 choice for bf16; it has not been checked live in
this session. The script uses one node/GPU and the account's default QoS, as the
previous working jobs did. `24g` has utilization auto-stop policies; the short
comparison requires native bf16, but its revised memory use may fall below the
policy threshold. Inspect the job outcome; do not add dummy memory allocations.

Update code **inside an allocation**, not on `head`. One interactive allocation
can perform the update and the entire comparison, avoiding a second queued job:

```bash
srun -p 24g -w muenchen --propagate=NONE --gres=gpu:1 --pty bash -l
hostname
cd /mnt/projects/gr/3DRecon/layer_good
git pull --recurse-submodules
bash tools/opt_pose_tracking.sbatch camera 5 1
```

Calling the `.sbatch` file with `bash` inside an allocation runs its body; the
`#SBATCH` directives are comments. Start with this five-query smoke test. Require
`VERIFY OK`, `CACHED OK`, `ORIGINAL OK`, `READOUT OK`, `REPORT OK`, and `JOB OK`.
Then, in the same allocation, run the longer comparison and geometry visuals:

```bash
bash tools/opt_pose_tracking.sbatch camera 50 1
bash tools/opt_pose_tracking.sbatch geometry 50 1
```

These reload the saved container but install nothing. Extraction is job-local and
uses the existing extractor. After code is updated, the alternative unattended
submission from `head` is:

```bash
cd /mnt/projects/gr/3DRecon/layer_good
sbatch tools/opt_pose_tracking.sbatch camera 50 1
# Submit geometry after the camera job finishes if using students_normal.
sbatch tools/opt_pose_tracking.sbatch geometry 50 1
```

From `head`, inspect the actual job ID returned by Slurm:

```bash
sacct -j JOB_ID --format=JobID,Elapsed,MaxRSS,State
jobstats JOB_ID
```

From the Mac, sync the tar(s) and Slurm log after the run:

```bash
mkdir -p cluster_results/optpose_tracking
rsync -avz camp:/mnt/projects/gr/3DRecon/optpose_tracking_out/ \
  cluster_results/optpose_tracking/
rsync -avz 'camp:/mnt/projects/gr/3DRecon/optpose_tracking_slurm-*.log' \
  cluster_results/optpose_tracking/
```

Use the actual filename printed by `Artifact:` to unpack one result on the Mac:

```bash
tar -xf cluster_results/optpose_tracking/ACTUAL_RUN_NAME.tar \
  -C cluster_results/optpose_tracking
```

Open the already-rendered MP4/PNG and inspect CSV/JSON locally. Do not run the
report renderer or model code on the Mac under current repository rules. Remote
rerendering from an extracted artifact uses the existing container:

```bash
python tools/opt_pose_tracking_report.py /tmp/ACTUAL_RUN_NAME/run
```

The report directory must not already exist: use a fresh extraction without its
old report directory, preserving the original tar. This intentionally avoids
silently mixing old figures with new results.

## Verification record

Locally: AST parsing of modified/new Python source (no project imports), `bash -n`
on the launch scripts, and `git diff --check` in the superproject and model
submodule. Functional gates, model-memory measurements and rendered-figure visual
QA remain pending the user's Linux run. Historical Step 0/0.5/1a results stay
historical; they do not verify this revision.
