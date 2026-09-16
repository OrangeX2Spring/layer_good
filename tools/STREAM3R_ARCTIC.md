# STream3R on the ARCTIC pilot

Completed 2026-09-16, job 25621 on muenchen, code c06d548. All 12 runs
(three scenes, two samplings, two RGB conditions) passed the finite-output gates
and both archives finished with exit status 0. Synced to the Mac and extracted
separately under cluster_results/stream3r_arctic/{box,ketchup_espresso}/.
Local inventory verified 12 summaries and 360 frame-condition exports, with
source frames, masks, model-grid inputs and NPZ prediction files present.
Prediction contents have not been independently evaluated for accuracy.

The synchronized 30-frame forward took 19.528-20.073 seconds, with peak PyTorch
allocated memory 17.046 GiB and reserved memory 19.311 GiB in every run.
These are batched inference measurements, not streaming FPS.

Archives:
- stream3r_arctic_20260916T190534Z_25621_yALo7o.tar (box)
- stream3r_arctic_20260916T190931Z_25621_eKETrF.tar (ketchup and espresso)

## Protocol

Run S01 camera-0 box_grab_01, ketchup_grab_01 and espressomachine_grab_01.
For each scene run both 30 evenly spaced frames over the complete post-offset
sequence and the first 30 consecutive post-offset frames. Each selection has
original RGB and RGB zeroed outside the saved KV-Tracker SAM mask before resize.
No segmentation rerun or bounding-box crop. These are subsets of KV-Tracker's
full-sequence input; only the even selection matches the completed 4RC pilot.

Reuse fourrc_arctic.prepare for exact frame selection, preparation-manifest
agreement, two-frame loader offset, mask indexing, source hashes and GT staging.
Its default behavior remains unchanged for 4RC. STream3R metadata replaces the
4RC-specific inference fields before running the model.

Inference uses the existing optpose image and stream3r_ckpt, batched causal mode,
float32 parameters and the upstream crop loader (width 518, patch size 14).
This matches the previously verified STream3R inference path. It is not an
incremental StreamSession benchmark. Neither window nor full attention is run.
At the pilot's 840x600 source size, the model grid is 518x364; 4RC used 504x364.
Identical source pixels and frame IDs do not imply identical model grids.

## Run

User executes all cluster operations. Pull the published code inside an existing
GPU allocation, with localhost/optpose loaded. Use a 24 GB GPU for this initial
batched pilot: the previous 32-frame STream3R run exceeded 16 GB. ARCTIC's actual
peak is still unverified. No new dependencies, image build, or downloads are needed.

First run the box, both sampling choices and both conditions:

```bash
bash tools/stream3r_arctic.sh box_grab_01
```

Require two STREAM3R PREPARE OK lines, four EXPORT OK lines,
STREAM3R ARCTIC OK, and ARCHIVED with exit status 0. Inspect that log before
running the remaining scenes:

```bash
bash tools/stream3r_arctic.sh "ketchup_grab_01 espressomachine_grab_01"
```

The second command requires eight EXPORT OK lines. Stop on failure; do not
silently lower one condition's frame count. Each inference uses a fresh process.

## Record and interpretation

Each run stages one uniquely named stream3r_arctic_*.tar under
/mnt/projects/gr/3DRecon/stream3r_out/. Work stays in job-local /tmp. The EXIT
trap archives ordinary failures too; node loss or SIGKILL cannot guarantee staging.

The archive includes source JPEGs, lossless original/masked PNG inputs, masks,
frame IDs and hashes, input manifests, KV-Tracker baseline metrics, ARCTIC GT,
model-grid RGB and masks, per-frame object PLYs with raw confidence, and full
float32 NPZ predictions (pointmaps, depth, both confidences, pose encoding,
extrinsics and intrinsics). It also includes runner sources, model source and
diff, commits, checkpoint hash/config, container identity, package versions and logs.

Every prediction array must have the expected frame count and finite values.
Object masks must match the model grid and contain pixels. No confidence threshold
is applied; identical SAM masks select object points in both conditions.
One PLY represents one instant: do not concatenate moving-object world clouds.

Summaries record one synchronized forward's duration and peak PyTorch allocated
and reserved memory. Loading/export are excluded, no warmup is performed, and
these values are not streaming FPS or total device memory.

Geometry remains in model units. Camera extrinsics are camera-from-world.
The matching KV-Tracker pose-head readout is their inverse: KV-Tracker exports
Pi3 camera-to-world poses from masked inputs, recentred and aligned across cache
updates. This supplies the object-centric trajectory interpretation without a
separate Kabsch object fit. Original RGB is a control, with no guaranteed
object-centric interpretation. Metric geometry accuracy and fused completeness
are separate questions.

## Pose evaluation (completed)

The evaluator ran in job 25621 with
localhost/optpose loaded. It used CPU only, read the two completed archives plus
the original KV-Tracker archive, and saved
`stream3r_arctic_eval_20260916T193217Z_25621.json` under stream3r_out.
No inference, dataset extraction or new packages are required.

The evaluator reproduces upstream ARCTIC GT construction and evo's Sim(3)
alignment, translation ATE, adjacent-sample translation RPE and rotation-part
Frobenius RPE (not degrees). All three full-sequence KV-Tracker metrics must agree
with the recorded values within 1e-5 before results can be accepted. It then scores
KV-Tracker and both STream3R conditions on identical selected timestamps. It fixes
the pose convention from code; it never selects an inversion using GT error.

Reports include translation spread, alignment scale, covariance rank and the ATE
of a stationary mean-position baseline. Rank-deficient alignment is reported as
a failure with no ATE. Short-window scores must be read beside GT motion spread;
small motion can produce small errors without useful tracking. For even sampling,
KV-Tracker has processed intervening frames that STream3R never saw: this is an
output-at-matched-timestamps comparison, not identical temporal input history.
ATE by itself does not establish a rigid reconstruction or shape accuracy. The
consecutive 4RC follow-up and its evaluation are recorded in
`cluster_results/fourrc_arctic/consecutive/eval.json` and `tools/FINDINGS.md`.

Sync on the Mac from the repository root, after the run:

```bash
mkdir -p cluster_results/stream3r_arctic
rsync -av chunquancheng@131.159.11.60:'/mnt/projects/gr/3DRecon/stream3r_out/stream3r_arctic_*.tar' cluster_results/stream3r_arctic/
```

Extract each returned archive into its own directory to preserve provenance.
