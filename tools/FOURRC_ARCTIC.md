# 4RC on the KV-Tracker ARCTIC pilot

## Status, 2026-09-16

Fork `OrangeX2Spring/4RC`, submodule `4rc`, model commit `a08ca0a`.
GenRecon's saved image is reused without installing packages. User-executed checks
passed on muenchen: Python 3.11.11, torch 2.6.0+cu124, A5000, CUDA available;
model, inference and image-loader imports passed. Default `Luo-Yihang/4RC` weights
downloaded to `/mnt/projects/gr/3DRecon/4rc_ckpt/model.safetensors` (5.7 GB).
No config.json was downloaded; constructor defaults successfully loaded the weights.
Three bundled robot-arm frames ran at 504x280, reported inference 0.88 s, and saved
`4rc_out/smoke.npz`. This is a smoke test, not an ARCTIC result or FPS benchmark.

**ARCTIC comparison: runner published, not runtime-verified yet.** The next gate is
the 30-frame box run on the current `muenchen` allocation; see `docs/4rc-cluster.md`
for the complete handoff and remaining work.

## Protocol

User requested both original RGB and SAM-masked RGB on box, ketchup and espresso.
Select the same 30 evenly spaced frames over each complete post-offset sequence,
including first and last. Read pixels from `kvt_arctic_out/prepared.tar` and the
per-frame masks from `arctic_r518_20260915T152911Z.tar`. Mask index 0 corresponds
to prepared frame row 2 (the third image), as in KV-Tracker's loader. Assert both
archives' preparation manifests agree. No segmentation rerun or object bbox crop.
Masked pixels are zeroed before resizing. Original and masked PNGs are lossless.

The unchanged upstream CLI loads at size 512 / patch 14, uses bf16, uses the middle
sample as the tracking query, and retains frame 0 as the coordinate anchor.
Its 30-frame cap does not resample our already selected <=30 images.
The mask export reproduces its resize/crop coordinates with nearest-neighbour
label sampling; model-grid dimensions are checked against predictions.

This is offline reconstruction from sampled views, not full-sequence tracking.
Original input includes background; both outputs use the SAME SAM masks to select
object points. Each PLY is a single time instant's visible object geometry in
model units, with raw confidence stored as a vertex property and no confidence
filter. **Do not concatenate these PLYs:** the object moves. No motion alignment,
fusion, metric scale, ground-truth shape accuracy or paper-comparable ATE is claimed.
KV-Tracker's fused object cloud and these observed surfaces have different coverage.
Full NPZ predictions retain motion tracks for subsequent motion-aware analysis.

## Run, one command chunk at a time

User executes on an allocated GPU, after pulling the superproject and initializing
`4rc`. `localhost/genrecon` must already be loaded. No new image or package install.
First run both conditions on the box to establish the 30-frame memory requirement:

```bash
bash tools/fourrc_arctic.sh box_grab_01
```

Require `PREPARE OK`, two `EXPORT OK` lines, `FOURRC ARCTIC OK`, and `ARCHIVED`
with exit status 0. Then run the remaining two scenes:

```bash
bash tools/fourrc_arctic.sh "ketchup_grab_01 espressomachine_grab_01"
```

Three-frame success does not establish that 30 fits 24 GB. On OOM, stop and inspect
the log; do not silently reduce one condition's frame count. An explicit second
argument changes the count for BOTH conditions and is recorded in a fresh archive.
Each inference is a separate process to avoid retaining GPU state across runs.

## Artifacts and verification

Work stays under a unique `/tmp/fourrc_*` directory. The shell stages one tar under
`/mnt/projects/gr/3DRecon/4rc_out/` on success or ordinary failure. SIGKILL/node loss
cannot guarantee archival; the allocation must remain alive until `ARCHIVED`.
The tar contains exact selected source JPEGs, PNG model inputs before resizing,
SAM masks, GT files, selected frame IDs/GT rows, input hashes, upstream NPZs,
per-frame PLYs, actual model-grid RGB/masks, confidence summaries, logs, checkpoint
SHA-256, package versions, container ID, code commits, model source and patch,
and the exact runner sources. No thousands of result files are left on `/mnt`.

Success requires finite points, confidence, tracks and camera matrices for all
frames and the expected frame count in every output. These are integrity checks;
visual inspection after syncing is still required to judge reconstruction quality.
Confidence quantiles describe predictions, not accuracy or calibrated probability.

No inline Python/heredocs or one-off check scripts for terminal paste. These two
runner files form the reusable experiment workflow. Local verification is syntax
and whitespace only; runtime verification requires returned CAMP logs.
