# Streaming cache sweep

## Reduced StreamVGGT sweep — 2026-09-21

The LongStream archives have been reviewed. User authorized preparing the reduced
StreamVGGT sweep while preserving exploratory variety: different strategies are
valuable at this stage; a common paper method can be formulated later.

`stream_cache_streamvggt_reduced.json` contains 17 conditions:

| Group | Conditions |
|---|---|
| Frame controls/selection | recent8, redundancy8, random_frames_s0/s1/s2 |
| Half-patch controls/selection | uniform_p50, confidence_p50, confidence_novelty_p50, random_p50_s0/s1/s2 |
| Pooled admission candidates | pooled_encoder_t1/t2: thresholds 0.00034 / 0.00087 |
| Centered admission candidates | centered_encoder_t1/t2: thresholds 0.15 / 0.3 |
| Coverage admission candidates | coverage_f099_t01/t05: similarity floor 0.99, thresholds 0.1 / 0.5 |

**These thresholds are provisional LongStream-derived candidates.** First run all
17 arms on the first 256 consecutive office frames, start 0, stride 1, width 308,
RPE gap 0.1 s. Fresh processes, native fidelity, remote contract tests and final-only
geometry exports remain enabled. Unrestricted causal attention stays excluded.
The existing runner rejects clips exceeding max_frames=256 before model loading;
this guards against accidentally using provisional settings for full runs.

After pulling this revision inside an allocation, submit from the repository root:

```bash
sbatch --array=0 tools/stream_cache_long.sbatch streamvggt calibrate
```

Do not use a multi-task array: the observed student QoS rejected it. Source updates
must be pulled inside an allocation before submitting the wrapper, since sbatch
copies the wrapper at submission. The in-job pull does not update that copied code.

Calibration review must inspect fidelity, scores, actual admission fractions,
retained counts/bytes, distinct frame/patch histories, memory and timings. ATE on
this prefix is diagnostic only; do not select thresholds by its best ATE. If an
admission pair collapses to all-admit/all-reject or indistinguishable behavior,
adjust that pair and repeat only the necessary short arms before freezing. The
short office prefix cannot establish calibration across every scene. This is an
exploratory dataset reuse, not held-out validation or promised admission rates.

Once reviewed, commit the frozen thresholds and raise max_frames to 5182 in the
same JSON; full office/loop/no-loop then use the same frozen configuration with
`--array=0`, `--array=1`, `--array=2`, one at a time, without `calibrate`.
The wrapper now selects this reduced JSON for StreamVGGT; LongStream and STream3R
retain their existing configurations. Full runs remain gated until calibration.

No policy implementation changed. StreamVGGT's head cache still grows even when
its aggregator cache is bounded. A successful short calibration does not establish
memory safety for the longest sequence. All poses/events/shared pixels/provenance
and final-frame geometry are archived as before.

## Full-sweep handoff — 2026-09-20 (historical)

This supersedes the unrun-host and next-command statements below. StreamVGGT's
full-office recent8 probe is archived and read; LongStream's corresponding run
reached JOB OK in the user-supplied terminal log. Evidence and limitations are in
tools/FINDINGS.md. The new isolated-worker path has only local static verification;
its tests run in the container before the next inference job.

User approved camera metrics plus final-frame geometry: every pose encoding,
converted camera, cache event, shared prepared RGB/mask input and manifest is kept;
only the final frame retains dense predictions. Each archive also keeps GPU samples,
source, sweep config, model YAML where required, checkpoint hashes and provenance.
This supports camera/cache analysis, not per-frame geometry accuracy evaluation.

`stream_cache_long.sbatch HOST` selects all three full TUM sequences as array tasks,
start 0 / stride 1 / width 308, RPE gap 0.1 s. The bounded JSON has 37 conditions;
LongStream's JSON adds native segment-causal retention, for 38. recent8 runs first.
Fidelity and every condition use fresh processes; a failure stops that task before
later conditions and evaluation. Other array tasks are independent. LongStream
keeps native refresh. StreamVGGT/STream3R exclude unrestricted causal retention;
their head caches still grow. The longer sequences remain resource-unverified.
Thresholds are an exploratory grid held fixed across sequences, not a held-out
winner or a claim of calibrated admission rates on the new hosts.

The wrapper must be present on the cluster before direct sbatch submission. Pull
inside an allocation, never on head. Submit one host at a time with array concurrency
one. The batch container runs all cache tests (including worker failure propagation)
before preparing inputs and running fidelity. Do not claim these new tests passed
until the remote log confirms them. No new image build is needed.

## Repair handoff — 2026-09-20 (historical)

STream3R job 25691 passed smoke fidelity; job 25692 completed the scored pilot.
Seven admission arms collapsed to anchor plus latest, and RPE had no eligible pairs.
The detailed corrected read is in `tools/FINDINGS.md`. Earlier handoff below is historical.

`stream_cache_sweep.json` now holds 38 calibration conditions: the 13 admission-all
controls, seven score families at three explicit thresholds each, and coverage at
two stricter similarity floors crossed with two thresholds. The original 21 arms
are preserved in `stream_cache_sweep_pilot.json`. Policy defaults are unchanged.
Thresholds bracket the corresponding saved score distributions; coverage floors
are exploratory because the original coverage score was identically zero.
Do not claim prescribed admission rates or choose a held-out winner on this clip.
Inspect `summary.json` admission_diagnostics and events: all-reject/all-admit arms
are calibration outcomes, not evidence of useful selective admission. Compare
actual retained counts/bytes, not just the cap, before claiming a matched budget.

User chose the same 32 frames, start 0, stride 4, width 308. Set
`CACHE_MAX_PAIR_GAP=0.15`; both wrappers pass it through to automatic evaluation.
Reports now record pair counts and thresholds and explicitly flag empty RPE.
Other jobs retain the 0.1-second default. No remote execution of this repair yet.
The existing pilot predictions can also recover stride-4 RPE without new inference:
use `evaluate --out EXTRACTED_RUN --max-pair-gap 0.15` on remote Linux and preserve
that evaluation separately from the original archive, with its source revision.
The follow-up still needs inference because it changes admission policies.

After publishing these changes, the next user-executed command chunk on head is:

```bash
cd /mnt/projects/gr/3DRecon/layer_good
export CACHE_HOST=stream3r CACHE_TUM_COUNT=32 CACHE_TUM_START=0 CACHE_TUM_STRIDE=4 CACHE_WIDTH=308 CACHE_MAX_PAIR_GAP=0.15
export CACHE_TUM_ZIP=/mnt/datasets/tum-rgbd/rgbd_dataset_freiburg3_long_office_household.zip
unset CACHE_INPUT_TAR CACHE_SWEEP
W='git -c fetch.recurseSubmodules=0 pull --ff-only && bash tools/stream_cache.sbatch'
O=/mnt/projects/gr/3DRecon/stream_cache_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

StreamVGGT and LongStream have implemented adapters, not verified host runs. Their
checkpoint availability is unconfirmed remotely; last recorded status is absent.
Their checked-out READMEs specify `lch01/StreamVGGT` / `checkpoints.pth` and
`NicolasCC/LongStream` / `50_longstream.pt`. Use the existing hf_download.sbatch
interface when staging them, then run one host per job with CACHE_CHECKPOINT.
The job wrapper already supplies LongStream's inference YAML. The runner loads
only its model section and calls streaming_refresh directly; it does not call
core/infer.py's output stage, so sky masking/ONNX download is outside this path.
Imports, strict checkpoint loading and native fidelity in optpose.tar remain gates.
Downloads may be queued separately but the recorded students_normal QoS serializes
GPU jobs. They need not wait for scientific success; transfer claims do require a
frozen rule and held-out evaluation, including matched random and confidence controls.

## Long-sequence feasibility audit — 2026-09-20

Source-inspected, not runtime-verified. Neither host has a model-level 32-frame
limit; that ceiling is in our pilot JSON. Full TUM source streams are appropriate
for both hosts, with the following distinctions and gates.

- **StreamVGGT:** the adapter prunes aggregator K/V and the matching RoPE positions.
  The native camera head appends at every refinement iteration and is not pruned;
  memory and query cost therefore still grow with sequence length. An unrestricted
  aggregator causal baseline is unsuitable for the full TUM streams on the existing
  GPU. Use bounded aggregator conditions and a short native fidelity prefix, then
  a full-length resource probe. A bounded aggregator is not bounded total memory.
- **LongStream:** keyframe_stride=8 and refresh=4 reset caches every 24 source frames
  in our streaming-refresh protocol. `clear_cache_only` clears aggregator, camera,
  relative-pose K/V, reference tokens and frame metadata. The adapter uses
  `record=False`, exports each prediction, and composes the saved relative poses
  along global keyframe references afterward. Thus a full trajectory does not
  require a full-history attention cache. Its all-retained arm is all-retained
  *within each refresh segment*, not a global causal baseline. This experiment
  cannot establish semantic memory across refresh boundaries.
- The native fidelity oracle runs only a prefix, including a LongStream refresh;
  it does not load the entire long stream into GPU memory. The experimental loop
  loads one frame at a time and retains only pose chunks on CPU. Per-frame dense
  NPZ exports still grow on disk for every condition. Budget /tmp and persistent
  archive space before a multi-condition long run; do not silently omit provenance.

For comparison to KV-Tracker, use all source RGB frames, start=0, stride=1, the same
three TUM archives, GT association <=20 ms, and RPE gap <=100 ms. The stride-4 /
0.15-second choice applies only to the short calibration repair. Full sequence
counts are recorded in tools/FINDINGS.md, KV-Tracker job 25680.

**Resolution is not yet matched:** KV-Tracker's `pi3_resize_image` interprets 308
as a square pixel-area budget with aspect-preserving patch alignment and linear
interpolation. This runner interprets it as width and uses bicubic resizing plus
centre-height cropping. For a controlled cross-host comparison, freeze shared
prepared pixels (and rerun KV-Tracker if needed), or explicitly report each host's
native preprocessing and actual dimensions; do not label both simply resolution 308.

Before submission: checkpoint/import/strict-load and native fidelity gates for each
host; a separate long-run JSON without the 32-frame ceiling or unrestricted
StreamVGGT causal arm; full-count stride-1 input settings; a full-length memory,
latency and output-size probe; then frozen policy comparisons. Existing frame caps
are not automatically byte-matched across models or to KV-Tracker's keyframe cap.
These checks support feasibility, not a claim that either host already ran on CAMP.

## Original implementation record

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

### Operational record

The first submission used job 25690 on `muenchen` with `localhost/optpose` and
failed only at provenance collection. The order before failure was:

```
MODEL IMPORT OK stream3r
Ran 24 tests ... OK
PREPARED 10 frames: /tmp/cache_stream3r_25690_.../prepared
FileNotFoundError: ... executable 'git'
```

The image is intentionally immutable and has no Git. Commit `d84e4ba` moved
`rev-parse` and binary `diff` capture into the outer Slurm script. Commit `5a3ad7d`
removed the explicit `--time=04:00:00`; the job now requests no project-level
wall-time limit. Any scheduler or partition limit remains authoritative.

The canonical unattended entry point is `tools/stream_cache.sbatch`. The older
`tools/stream_cache_run.sh` is a lower-level in-allocation wrapper for manual use.
Each host gets a separate job and container setting; do not combine StreamVGGT,
LongStream, and STream3R when their dependency closures or images differ.

All extraction, preparation, intermediate predictions, and condition outputs stay
under job-local `/tmp`. The only persistent run artifact is one tar under
`/mnt/projects/gr/3DRecon/stream_cache_out/`, containing logs, package inventories,
Git provenance, exact model-input pixels, masks, manifests, predictions, metrics,
and visualization inputs. Large source archives remain packed on project storage.

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

The original `stream_cache_sweep_pilot.json` has 21 exploratory conditions and a **32-frame maximum** for
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
The default `geometry_export=all` stores every frame in NPZ. The approved long-run
JSONs explicitly select `geometry_export=final`: final-frame dense predictions, all
pose encodings in `pose_encodings.npz`, and all camera poses/intrinsics in
`camera.npz`. The selected export mode is recorded in each condition summary.

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
`CACHE_CHECKPOINT`, `CACHE_INPUT_TAR`, `CACHE_MANIFEST`, `CACHE_SWEEP`, `CACHE_WIDTH`,
and for TUM `CACHE_TUM_ZIP`, `CACHE_TUM_COUNT`, `CACHE_TUM_START`, `CACHE_TUM_STRIDE`.

### TUM needs no staged input archive (2026-09-19)

`/mnt/datasets/tum-rgbd` is read-only and costs no quota, so `CACHE_TUM_ZIP` cuts the
clip straight into job-local `/tmp` instead of staging ~1 G on project storage.
`tools/stream_cache_tum.py` reads the ZIP, writes the selected frames and a source
manifest carrying `timestamp`, original-resolution `intrinsics`, `gt_c2w` and
`gt_timestamp`; `prepare` then derives `model_intrinsics` from its own pixel
transform. GT association mirrors `kvt_tum_run.associate_gt` -- nearest timestamp
within 20 ms, `null` beyond it, never an extrapolated pose -- so this sweep and the
KV-Tracker TUM sweep agree on which frames are scorable. It runs inside the image
because a bare allocated node is not guaranteed to have `python3`.
`CACHE_TUM_ZIP` and `CACHE_INPUT_TAR` are mutually exclusive.

`tools/test_stream_cache_tum.py` is standard library only, so it runs on the Mac and
is also picked up by the job's `test_*cache*.py` discovery -- a broken clip stops the
allocation before the model loads. It covers the rotation contract, the GT-hole
branch, stride, and the two loud failures (too few frames, existing output directory).

**`max_frames` is a ceiling, not a truncation.** `stream_cache_sweep.py:360` raises
`ValueError` when the prepared clip is longer than the sweep JSON allows, by design.
`CACHE_TUM_COUNT` must therefore be <= `max_frames`, currently 32. That 32 is sized to
what the all-retained `causal` condition can hold: STream3R measured 17.05 GiB
allocated / 19.31 GiB reserved at 30 frames on a 24 GB card. A longer clip is a
different experiment and needs `causal` bounded or dropped first.
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
