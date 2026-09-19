# Streaming cache sweep

2026-09-19 implementation handoff. **Locally syntax-checked only; no new runtime
result.** Hosts: StreamVGGT, LongStream, STream3R. 4RC remains an offline diagnostic
and is not given a fictitious KV-cache adapter. This supersedes the threshold-only
proposal for these three hosts; it does not change the separate KV-Tracker TUM job.

Job **25690** passed the STream3R model import and all 24 contract tests, then
prepared ten example frames. It stopped before model loading/fidelity because the
container has no Git executable. The wrappers now capture commits and binary diffs
on the allocated host and pass `--git-provenance` to the runner. No Git install is
needed in the borrowed image. This fix is locally syntax-checked; fidelity and the
sweep still await the next remote run.

## Questions and controlled comparisons

Separate the three interventions:

1. **Feature and admission:** encoder patch tokens or the first frame-local block
   before global attention; global pooled cosine, 2x2/4x4 spatial pooling, centred
   cosine, patch coverage, symmetric patch Chamfer, or patch-novelty q90.
2. **Frame retention:** anchor plus latest query, with FIFO, redundancy, or seeded
   random eviction among the other retained frames. Default budget eight includes
   the anchor and recent frame. Novelty controls admission to longer history;
   rejected queries still remain for one step as recent context.
3. **Patch retention:** keep all, spatial-row uniform, random, highest confidence,
   highest patch novelty, or equal-weight confidence/novelty rank fusion. Half of
   each non-anchor frame's patches for the first controlled comparison. All special
   tokens survive; the segment anchor remains complete. Optional `mask` priority
   fills the same fixed count, using background when the mask is smaller; it is
   explicitly not the old variable-size object-only cache.

The checked-in JSON has 21 exploratory conditions and a **32-frame maximum** for
the first integration pilot. It rejects longer inputs rather than truncating them.
This is not the long-sequence result. After the pilot's memory and fidelity checks,
create a separate frozen JSON for long clips; an unrestricted causal arm can exceed
VRAM. Hold resolution, dtype (float32), input order, masks, weights, and head
behavior fixed. Do not pick the best threshold per test sequence.

`recent8` is the common aggregator-budget control, not a claim to reproduce a
host's native window. STream3R's native window is anchor + five recent frames;
LongStream's window counts the anchor inside the limit. The legacy token helper
now exposes that distinction explicitly.

Scores are exploratory, not calibrated semantics. Defaults: novelty > 0.05;
coverage > 0.1 with cosine floor 0.9. Comparisons against admission-all can have
different actual retained counts even under the same cap; use logged bytes/counts,
not just the nominal cap, when interpreting accuracy.

## Exact feature and state contracts

Features are observed through a removable forward hook. The encoder feature is
`patch_embed`'s `x_norm_patchtokens`; `frame0` is `frame_blocks[0]` after removing
special tokens. Both precede cached global attention. No model source is edited.

Spatial pooling averages a CxHxW token map to gxg, then flattens and L2-normalizes.
Centred cosine subtracts the first segment frame's mean patch vector before
normalization. A zero centred vector has similarity zero, hence novelty one.
Coverage uses each current patch's best cosine over all retained historical patch
descriptors. Chamfer averages current-to-history and history-to-current mean cosine
distances; q90 uses the current-to-history distribution. Similarities are computed
in chunks of 128 query patches. Full patch distributions are computed and timed
for every arm, so this runner measures an **instrumented experiment**, not an
optimized deployment FPS. Descriptors are CPU float32; retained feature bytes are
reported separately. Similarity work grows with the retained patch bank.

Admission and eviction occur **after** the current prediction, affecting the next
query. Newly encoded tokens retain the causal context they were computed with;
eviction does not re-encode old tokens. Patch selection occurs only at insertion,
not by resurrecting previously deleted patches. Random frame eviction protects
anchor/latest and is not uniform reservoir sampling over the entire past.

STream3R and LongStream store RoPE-transformed keys, so gathering flat token
indices preserves their positions. StreamVGGT caches raw keys before normalization
and RoPE: its adapter retains those raw K/V plus per-layer key positions and gathers
both identically. Its attention override is installed on global blocks only and
removed after each condition. Its native camera head remains unchanged, including
four cache appends per input frame. Batch size and streaming chunk length are one.

**Only aggregator K/V is pruned.** Camera and relative-pose heads retain native
causal history identically across arms. LongStream's `_frame_info`, reference-token
dictionary, scale token, stride-eight references, and refresh every three keyframe
intervals (refresh=4) stay intact. The refresh boundary frame is predicted, then
replayed as a fresh anchor without emitting a duplicate output, as upstream does.
Its cache/feature selection state resets at the same boundary. The original YAML's
default batch-refresh is a different execution mode; this study explicitly uses
its implemented **streaming_refresh** path.

LongStream cannot demonstrate retention across refresh boundaries in this protocol.
Do not disable its refresh just to manufacture long memory: that is a separate
pose experiment. STream3R/StreamVGGT head caches still grow, so bounded aggregator
bytes are **not bounded total memory**. KV tensor payload, positions, CPU token
indices, CPU features, relative-head and reference-token payloads are recorded
separately; Python object overhead is not included. CUDA allocator peaks include
transient concatenation/gather copies. Previous dense predictions do not accumulate
on GPU in the sweep (the small native fidelity oracle does retain its prefix).

## Inputs and provenance

The preparation CLI takes a JSON manifest with ordered frames. Paths are relative
to that manifest; absolute paths also work. Example schema, not a terminal snippet:

```json
{
  "sequence": "sequence-name",
  "mask_source": "none or exact causal/oracle mask provenance",
  "frames": [
    {"rgb": "rgb/000001.png", "timestamp": 0.0},
    {"rgb": "rgb/000002.png", "timestamp": 0.033}
  ]
}
```

Optional per-frame fields: `mask`, `intrinsics` (3x3), `gt_c2w` (4x4 camera-to-world,
translation in metres), and `gt_timestamp`. Missing GT is `null`, never a nearest
pose propagated across a gap. Preserve source frame IDs and visibility/occlusion
annotations as additional fields; preparation carries them through unchanged.

Preparation implements the common width resize / centre-height crop used by the
STream3R and StreamVGGT loaders, with patch-aligned height. The LongStream comparison
uses these same pixels, deliberately not its different default preprocessing.
Masks use nearest-neighbor interpolation. Pixel-centre transforms and transformed
intrinsics are recorded. No mask is applied to RGB automatically. Save a separately
labelled input set if testing masked RGB. No GT depth or object model enters inference.

The runner copies exact prepared RGB, masks, manifest, sweep, source tools, model
config, commits/diffs and checkpoint hashes into its output. The wrapper additionally
saves container inspection, package versions, GPU inventory, memory grant, logs and
exit status. Outputs go to job-local `/tmp` and are archived even on failure into
one tar under `/mnt/projects/gr/3DRecon/stream_cache_out/`, outside the checkout.
Each frame stores full predictions in NPZ, including confidence and scale factors;
camera poses/intrinsics are also saved as `camera.npz`. Check disk capacity before
long dense runs; there is intentionally no silent dropping of geometry.

## Gates and interpretation

1. CPU contract tests on **remote Linux**: pooling, host window conventions,
   admission/rejection, budget invariants, protected anchors, deterministic random
   controls, and sparse StreamVGGT key/position agreement with an independent
   attention computation.
2. Native streaming output twice, then the all-retained adapter on the same prefix.
   Compare every camera/depth/point/confidence tensor with explicit atol=1e-5 and
   rtol=1e-4. Native entry points are `StreamSession.forward_stream`,
   `run_streaming_refresh`, and `StreamVGGT.inference`. LongStream's gate spans a
   refresh when the input is long enough. Any mismatch stops the sweep. The gate
   writes repeat noise and adapter errors; do not relax tolerance without inspection.
3. Run the short sweep. Require finite outputs and inspect event logs for actual
   nontrivial admissions, evictions, differing patch choices, and refresh boundaries.
   A short smoke clip cannot establish long-term memory utility.
4. On frozen long-clip settings, report every configuration's camera ATE/RPE,
   actual cache/feature bytes, peak memory, latency distributions, and selection
   history. Inspect errors on frames after eviction and refresh, using the logged
   frame IDs. Confidence quantiles are diagnostics, not accuracy ground truth.

`evaluate` uses timestamp-valid GT (default tolerance 20 ms), full-trajectory
Sim(3) camera-centre ATE, and consecutive-frame translation/rotation RPE. It excludes
pairs spanning missing GT or >100 ms. A collinear/stationary predicted path is
reported as degenerate. Full-trajectory alignment is an offline camera diagnostic;
these are **not object-pose or absolute orientation accuracy claims**. Geometry and
occlusion/reappearance conclusions still need their dataset-specific GT/readout.

Compare fixed-budget random seeds as a distribution. Feature novelty alone does not
prove object identity or semantic understanding. Thresholds, confidence controls,
and 4RC's offline selection remain separate questions; no learned module is added.

## Remote command interface

### Unattended CAMP entry point (2026-09-19)

`tools/stream_cache.sbatch` now supplies the allocation lifecycle: 24g/muenchen,
one GPU, students/students_normal, no explicit wall-clock limit. It pulls inside the allocation,
initializes the selected host and StreamVGGT (used by the small attention tests),
loads the selected image, and runs `stream_cache_job.py` inside one disposable
container. One host per job, including when different images are needed. It does
not rebuild or overwrite the saved container tar. Missing inference dependencies
are installed as wheels only, constrained to preserve all already-installed package
versions; package inventories and the pip install report are archived. Import and
contract-test failures stop the job before inference.

Default is a **STream3R example smoke sweep**: existing `optpose.tar`, existing
`stream3r_ckpt`, the ten shipped `static_room` images, width 308, no GT claims.
This completes environment and integration checks, not the long-sequence study.

Optional exported job settings: `CACHE_HOST`, `CACHE_IMAGE`, `CACHE_IMAGE_TAR`,
`CACHE_CHECKPOINT`, `CACHE_INPUT_TAR`, `CACHE_MANIFEST`, `CACHE_SWEEP`, `CACHE_WIDTH`.
For other hosts provide the verified checkpoint path and an input archive.
`CACHE_MANIFEST` defaults to `manifest.json` relative to the extracted archive.
The archive must contain the source manifest and its referenced RGB/masks, using
relative paths. Existing large dataset archives stay packed on project storage.

**User requirement:** all extraction, prepared frames, and intermediate outputs live
under job-local `/tmp`. `/mnt` is mounted read-only inside the container. The outer
job saves only a single result/provenance tar to `stream_cache_out`, containing the
exact model/visualization inputs copied by the runner; no loose extracted datasets
are written to network storage. EXIT/TERM/INT handling archives partial results.

The new script must first reach the cluster via a pull **inside** an allocation.
From `head`, use the established short-variable bootstrap (after publication):

```bash
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=0 pull --ff-only && bash tools/stream_cache.sbatch'
O=/mnt/projects/gr/3DRecon/stream_cache_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

The bootstrap inherits the verified students/students_normal defaults. Once the
file is present, `sbatch tools/stream_cache.sbatch` uses its explicit directives.
Never run the pull or container setup directly on `head`.

All commands are user-executed. No new container, checkpoint location, or dataset
location is assumed for StreamVGGT or LongStream. `CACHE_IMAGE` must name a locally
loaded image whose imports have passed; STream3R previously ran in `localhost/optpose`.
Do not install anything on the Mac. Use one host per wrapper invocation.

First command, inside the existing remote container from the updated repo root:

```bash
python -m unittest discover -s tools -p 'test_*cache*.py'
```

Preparation interface (supply paths for the chosen licensed dataset):

```text
python tools/stream_cache_sweep.py prepare --manifest SOURCE.json --out /tmp/cache_inputs --width 308
```

Wrapper interface, from the GPU allocation outside the container:

```text
bash tools/stream_cache_run.sh HOST CHECKPOINT /tmp/cache_inputs tools/stream_cache_sweep.json [LONGSTREAM_YAML]
```

Direct inference interface is `stream_cache_sweep.py run --host ... --checkpoint ...
--inputs ... --sweep ... --out ... --git-provenance ...`, with `--model-config`
required for LongStream. Both wrappers create the Git provenance directory on the
allocated host; it contains `superproject_commit.txt`, `model_commit.txt`, and the
corresponding `superproject.patch` / `model.patch` files.
Use `longstream/configs/longstream_infer.yaml` for its architecture. Loading is strict;
no missing weights are silently accepted and no download fallback is added.

When the input manifest supplies camera GT, the runner evaluates automatically
before the wrapper archives the run. Without GT it writes an explicit not-evaluated
status. To re-evaluate, run `stream_cache_sweep.py evaluate --out RUN_DIRECTORY`
inside the remote container, including after extracting an archive on remote Linux.
Keep its resulting metrics with that run's archive.

Sync follows the established route, one extraction directory per archive:

```bash
rsync -av chunquancheng@131.159.11.60:'/mnt/projects/gr/3DRecon/stream_cache_out/cache_*.tar' cluster_results/stream_cache/
```

Local verification: AST parsing without importing project modules, JSON/schema
inspection, `bash -n`, and `git diff --check`. Unit tests, fidelity gates, GPU memory,
and quality metrics remain unverified until their remote executions succeed.
