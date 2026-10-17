# InfiniteVGGT dynamic-cache diagnostic on CAMP

Implementation prepared 2026-10-16; no CAMP execution yet. Research motivation and
stop rules: [DYNAMIC_CACHE_BIAS_PLAN.md](DYNAMIC_CACHE_BIAS_PLAN.md). User steering:
run `pilot /absolute/path/manifest.json` as the first job, without a separate
synthetic smoke run. Pilot retains contract tests and real-clip logging fidelity.
The selected `pilot-tum` mode generates the manifest and SAM3 person masks in the
same allocation. SAM3 setup/download authorized 2026-10-17; the new setup wrapper
builds the missing environment and continues directly into this pilot.
Review the pilot before any full evaluation.

User steering, 2026-10-16: **TUM dynamic first**, rather than waiting for
Sintel/Bonn. Dataset identity is not essential to the first retention-bias test.
Selected `freiburg3_walking_xyz`: user CAMP listing confirms its ZIP on 2026-10-16.
ZIP inspection confirms RGB and depth/camera metadata but no mask/label/segment
filenames. User authorized SAM3 predicted person masks. `sitting_xyz` is a possible lower-motion
comparison (not a static control). Official descriptions:
https://webshare.cvg.cit.tum.de/g/rgbd/dataset/ . The near-stationary-camera
`walking_static` can test retention but is a poor choice for this pilot's
nondegenerate translation-ATE gate. Sequence availability does not establish
pixelwise motion labels. Inventory now includes
TUM/walking/sitting names. No substitute annotations are claimed; the runner's
dataset-independent RGB/GT/motion-mask contract is unchanged. Semantic/predicted
person masks support the explicitly labelled exploratory foreground mode below;
they are never relabelled as motion GT.

## First run: TUM with SAM3 person masks

### SAM3 setup and pilot in one job (authorized 2026-10-17)

The user listings found only kvt.tar/optpose.tar and no SAM3 entry in the documented
HF cache. They do not prove when or whether SAM3 was deleted. User authorized a
new environment/checkpoint. First review `getusage /mnt/projects/gr/3DRecon` and
`getquota /mnt/projects/gr/3DRecon` before submission: checkpoint is about 3.45 GB,
and image size is unmeasured. Allow at least 20 GiB additional quota headroom for
the image, checkpoint and pilot archives; this is a planning reserve, not a size
measurement. Existing image archives are preserved. Builds/install files use
job-local `/tmp` and Podman's job-local store.

After storage review and publication, from CAMP head in `layer_good`:

```bash
W='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$W && bash tools/sam3_setup.sbatch"
sbatch -A students --qos=students_normal -p 24g -w stuttgart \
  --gres=gpu:1 --time=04:00:00 --propagate=NONE \
  -o ../sam3_setup-%j.log --wrap="$W"
```

`sam3_setup.sbatch` uses the existing `hf_download.sbatch` inside this allocation
for `facebook/sam3`, files `sam3.pt`, `LICENSE`, `README.md`, destination
`/mnt/projects/gr/3DRecon/sam3_ckpt`. It uses the documented `.hf_env` identity;
access errors stop the job. The model is gated: https://huggingface.co/facebook/sam3 .
No duplicate safetensors checkpoint or new dataset is downloaded.

Build: `python:3.12-slim-bookworm`, PyTorch 2.10.0/torchvision 0.25.0 CUDA 12.8
wheels (https://pytorch.org/get-started/previous-versions/), pinned SAM3 submodule,
numpy 1.26.4, setuptools 80.9.0 for its pkg_resources import, declared inference
dependencies and transitive scipy/einops/pycocotools/psutil imports. No optional
flash-attn/cc_torch compilation. Source audit found HF fetching in the builder;
pilot passes the explicit checkpoint and `load_from_HF=False`. Package versions,
source revision and base image identity are recorded. `pip check` and actual
builder/processor imports, CUDA/bfloat16 availability and SDPA API checks must
pass before saving. No model construction or synthetic inference in build checks.

The image is saved immediately after commit as `sam3_JOB.tar` with tag
`localhost/sam3:job-JOB`; successful save, archive listing and SHA256 are recorded.
`dynamic_cache_out/sam3_JOB_env.sh` records the exact environment for reuse.
Then the SAME allocation calls `dynamic_cache_bias.sbatch pilot-tum`.
No full evaluation follows the pilot. Success markers include `SAM3 IMPORTS OK`,
`SAM3 IMAGE SAVED`, the real-pilot markers below and `SAM3 SETUP AND REAL PILOT OK`.
Setup/failure evidence lives in `dynamic_cache_out/sam3_setup_JOB.tar`.
Container builds run with Bash command tracing. On build failure the setup
archive also includes `container.log` and `container_inspect.json`, and the
wrapper preserves the build exit code. Job 26814 failed with exit 100 after
56 seconds; both supplied logs ended at the Podman startup warning, so its
root cause remains unresolved. Review these additional diagnostics on a retry.
Pilot evidence lives in `dynamic_cache_JOB_pilot-tum.tar`. A pilot failure does not erase the
saved environment. Review evidence before a retry; reuse the saved image rather
than rerunning setup. Runtime, gated access and image size remain unverified.

### Reusing a saved SAM3 environment

`bash tools/dynamic_cache_bias.sbatch pilot-tum` runs preparation and the real
pilot in one allocation, without synthetic smoke. Export three verified paths/tag
before submitting: `SAM3_IMAGE_TAR` (saved working SAM3 image), `SAM3_IMAGE`
(its Podman tag), and `SAM3_CHECKPOINT`. The wrapper initializes the existing
`third_party/sam3` submodule and records its revision. Checkpoint must be under
`/mnt` or `/tmp`, which are
mounted into the container. Existing SAM2/KV-Tracker or SAM 3D images are not
assumed to provide SAM3. These three values remain unverified; do not guess them.
Historical documentation audit: STATUS's September 23 correspondence handoff explicitly
deferred SAM3; its SAM swap section says not started. The source submodule is
documented, but no SAM3 container/checkpoint path was found in the working docs,
storage inventory, protocols or build scripts. Verify actual stored assets next.

The wrapper loads the supplied image read-only, with network disabled, and runs
`dynamic_cache_tum.py` against the verified walking_xyz ZIP. It selects 32 RGB
frames starting five seconds after the first RGB timestamp, stride two, before
inspecting masks/results. GT uses nearest association within 20 ms, TUM xyzw
quaternions and camera-to-world translations in metres. Missing GT stays null.
Only selected RGB files and metadata are copied into job-local `/tmp`.

SAM3 runs independently per frame with text `person`, confidence >0.5 and mask
probability >0.5, using bfloat16 autocast. All detected person masks are unioned:
255 means predicted person, 0 means predicted non-person. No detections produces
an empty mask, recorded with its instance count; the pilot rejects an entire clip
without both foreground/background patches. No motion inference is claimed.

After the SAM3 process exits, the existing optpose container runs the contracts
and eight pilot arms on this real clip. `--predicted-person-masks` explicitly
enables the new provenance kind; default `pilot` still requires motion GT.
The existing `oracle` arm name now denotes hindsight person suppression for this
mode. Enrichment measures person retention; non-person pixels are not verified
static, and segmentation mistakes can confound results. Summary JSON records
these limits and the mask source. Fidelity is measured on the real clip.

The archive includes source RGB/masks, manifest, original metadata, SAM3 source,
image/checkpoint hashes, container identity, per-frame scores/counts/coverage and
all existing pilot evidence. Output is `dynamic_cache_JOB_pilot-tum.tar` under
`/mnt/projects/gr/3DRecon/dynamic_cache_out/`. Success requires `TUM PERSON INPUTS
OK`, both contract suites, both real-data fidelity checks, `METRICS OK`,
`DYNAMIC CACHE PILOT COMPLETE`, `DYNAMIC CACHE JOB OK pilot-tum`, archive exit=0
and Slurm COMPLETED 0:0. Review outputs before expanding beyond this clip.

## Container and source

Borrow `/mnt/projects/gr/3DRecon/optpose.tar` (`localhost/optpose`) read-only.
Recorded build: Python 3.11, torch 2.5.0+cu124, numpy 1.26.4, CUDA architectures
7.5/8.6. This image already serves the repository's StreamVGGT jobs. Reuse
`stream_cache_job.py --setup-only --host infinitevggt`: add missing inference
dependencies inside the disposable container, pin all previously installed
packages, record before/after/install reports, and fail if any existing version
changes. No torch replacement, requirements.txt bulk install, image commit/save,
model download, or persistent environment. Other shared images (`kvt.tar`,
`genrecon.tar`) are not needed.

Checkpoint: `/mnt/projects/gr/3DRecon/streamvggt_ckpt/checkpoints.pth` (existing
StreamVGGT weights). Official InfiniteVGGT's README specifies these weights.
Actual archive/checkpoint hashes and loaded image ID are recorded at runtime.
Compatibility is a source/record-based choice until prepare succeeds.

The `infinitevggt` gitlink is based on official
`AutoLab-SAI-SJTU/InfiniteVGGT` commit
`7f9a5a26c2e2f9603151d1dcc61fd402ef3f6a27`. Its dedicated branch is
`codex/infinite-dynamic-cache` in our existing `OrangeX2Spring/StreamVGGT`
repository. It has a separate checkout and does not change the existing
`streamvggt` gitlink or its branch. Upstream source and license remain intact;
model changes live in the model repository, drivers in parent `tools/`.

Source audit corrections to the initial plan:

- Upstream evicts BEFORE attention, including the incoming frame in candidates.
- Only the entire first frame is protected. Later camera/register tokens can
  be evicted. Native retains this behavior; they are excluded from H1 statistics.
- Source ranks by negative dot product with the mean normalized candidate key.
  This gives the paper's cosine ranking within a head, but source layer-score
  aggregation is retained exactly, not replaced with a paper reimplementation.
- Interventions replay native layer budgets, per-head source-frame patch quotas,
  nonpatch decisions and current-frame survivors. Only historical ordinary
  patch choices change. This is a controlled offline diagnostic, not a deployable
  causal policy; quota replay deliberately removes source-age/capacity changes.

## Optional prepare mode (superseded as the first submission)

The command below is retained for reference, not the current next action.

From CAMP head, in `/mnt/projects/gr/3DRecon/layer_good`, after publication:

```bash
W='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$W && bash tools/dynamic_cache_bias.sbatch prepare"
O=../dynamic_cache_slurm-%j.log
sbatch -A students --qos=students_normal -p 24g -w stuttgart \
  --gres=gpu:1 --time=02:00:00 --propagate=NONE -o "$O" --wrap="$W"
```

The in-allocation pull handles a new wrapper absent on head. All cloning/fetching,
image loading and compute happen in the allocation. No SSH/transfer by the agent.

Prepare records bounded inventories of `/mnt/datasets`, project root and known
project data folders. Historical `docs/stream3r-cluster.md` said Sintel/Bonn were
absent; only NRGBD and NYU-v2 downloads were recorded. This job checks current
names, not annotation validity. It does NOT download or extract a dataset.

Required evidence:

1. `MODEL IMPORT OK infinitevggt` and both contract suites `OK` (4 + 2 tests).
2. Strict checkpoint load, 12 synthetic 56x56 CUDA frames, eviction active,
   native-versus-logged pose fidelity at atol=1e-6 / rtol=1e-5.
3. `ENVIRONMENT SMOKE OK` and `PREPARE ENV OK; DATA INVENTORY ONLY`.
4. `DYNAMIC CACHE JOB OK prepare`, archived exit=0 and Slurm COMPLETED 0:0.
5. Review `context/dataset_inventory.txt`; inspect actual candidate annotation
   contents before making a real manifest. Synthetic success proves no H1/H2 claim.

## Input contract and one-clip pilot

`pilot /absolute/path/manifest.json` accepts one preselected 16..64-frame clip.
The manifest and its referenced files must be accessible in the container through
`/mnt` or `/tmp`. If datasets are packed, prepare the selected files in job-local
`/tmp` with a dataset-specific extractor only after the archive layout is known.
No default path, sequence, automatic truncation, mask invention or dataset change.
Dataset-specific extraction/download remains pending current inventory evidence.

Top-level JSON fields:

| Field | Required meaning |
|---|---|
| `sequence` | Dataset and exact sequence identity |
| `role` | `primary_dynamic` or `static_control` |
| `pose_units` | `metres`; convert explicitly before this stage |
| `mask_source` | Object with `kind` = `gt_motion` or `gt_derived_motion`, nonempty `description` and `reference` |
| `frames` | Ordered list, 16..64 entries, strictly increasing timestamps |

Every frame has `rgb`, `motion_mask` (paths relative to the manifest, or absolute),
`timestamp` (seconds), `gt_c2w` (4x4 rigid camera-to-world matrix, or null) and
`gt_timestamp` when GT exists. Association tolerance is 20 ms; at least three
valid GT poses and a nondegenerate camera path are needed. Preserve source-frame
IDs and all derivation parameters in the manifest as additional fields.

In the default motion-GT mode, masks must be original-RGB-sized grayscale PNG:
0 static, 255 moving, 128 unknown.
Semantic/person/instance masks are not accepted as GT motion. A GT-derived mask
must record derivation, coordinate conventions, thresholds and invalid/occlusion
handling in its description/reference; do not call it direct GT. No mask generator
is invented before data inspection.

RGB is resized to width 308 (height rounded to patch 14), with center-height crop
if necessary. Masks receive the same geometry with nearest interpolation. Motion
coverage is pooled over 14x14 patches; any unknown pixel makes that patch unknown.
Original RGB/masks, transformed inputs, manifest and hashes are archived. This is
a reduced-resolution diagnostic, not a reproduction of published benchmark scores.

With a reviewed input manifest, the SAME wrapper's `pilot` mode runs eight fresh
processes at FP32, with TF32 disabled and a fixed total of eight dense-frame
equivalents summed over global layers (head multiplicity is accounted in bytes):

1. Full global cache (large non-binding budget, zero evictions asserted).
2. Native unlogged and native logged, checked for pose fidelity.
3. Native shorter prefix, checked against full native poses and exact decisions.
4. Oracle: within each frozen native source-frame quota, exchange retained moving
   patches (coverage >=0.5) for unretained static patches, ranked by native score.
   Unknown labels retain their score-ranked decision. Backfill to the same capacity.
5. Three random seeds: identical exchange counts per layer/head/source-frame from
   the oracle trace, selecting removals/additions independent of motion labels.

This is an exchange intervention with exact strata matching, a stricter/narrower
oracle than unrestricted global static-first eviction. Failure rejects this
within-stratum mechanism; it cannot rule out all possible motion-aware policies.
Native special/current decisions and first-frame anchors are identical across
interventions. All model weights, input pixels, camera head and geometry heads
remain unchanged. Separate layer budgets are replayed, avoiding adaptive-budget
confounds. No gating or Evict3R implementation is included in this first pilot.

The next submission command is supplied only after prepare/data evidence; no
automatic promotion or full-manifest batch is implemented.

## Results and limitations

`summary.json`: one sequence-wide Sim(3) translation ATE, adjacent-frame translation
and rotation RPE (<=100 ms, no bridging missing GT), GT coverage, compression-minus-
full ATE, oracle relative gain and random median. Each raw arm remains visible.
Failure/OOM stops the pilot with nonzero exit and preserves completed/partial runs;
no retry at a changed resolution/budget and no silent removal of failed arms.

Native per-layer/head eviction JSONL records candidate-normalized enrichment,
excess, dynamic/static conditional retention, score means and source-frame/age
strata. Both all-patch and historical-only statistics are present. Primary H1 is
historical selectable patches; anchors and special tokens are excluded. First-event
NPZ samples retain IDs/scores/labels for mapping audits; occupancy is logged over
time. Undefined ratios stay null. Sequence medians are screening summaries, not
independent-token significance tests.

Resource reports contain actual global K/V logical bytes, CPU token-ID/label
metadata, total peak CUDA allocation/reservation, and wall time. Global K/V totals
exclude the unchanged camera-head bank, allocator slack and model/activations;
CUDA peaks include GPU allocations. Instrumented CPU logging, output copies and
file IO are timed. Do not derive a speedup from these runs or call oracle masks a
free motion estimator. Net efficiency requires a later uninstrumented causal method.

Outputs: `/mnt/projects/gr/3DRecon/dynamic_cache_out/`
`dynamic_cache_JOB_prepare.tar` or `dynamic_cache_JOB_pilot.tar`; Slurm log at
`dynamic_cache_slurm-JOB.log`. EXIT/TERM/INT traps archive exit code, logs, source
revisions/diffs/archive, dependency state and input/result artifacts before `/tmp`
disappears (SIGKILL/node loss cannot guarantee final archival). Do not delete
shared images/checkpoints or data. No output existence is claimed before a run.

Local checks: AST parsing and bash syntax plus diff whitespace only. Contract
tests, checkpoint compatibility, actual 24-GB peak and dataset readiness must all
be verified on CAMP. A successful prepare is not a successful research pilot.
