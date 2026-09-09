# KV-Tracker reconstruction history and handle occupancy

Status: implemented and locally syntax-checked; no remote execution yet. A specific
HouseCat6D object/clip and empty handle region have not been selected or verified.

The scripts run on Linux, with inference on an allocated CAMP GPU. Do not run
preparation, inference, evaluation, or Blender generation on the local Mac.
The implementation adds optional offline-frame and capture callbacks to
`kv_tracker/main.py`; normal invocations keep their existing stream. It also fixes
the initial low-confidence fallback pose from two centred poses to one raw first-camera
4x4 pose, matching the destination tensor and the subsequent coordinate transform.
The experiment keeps the default `sim3=False`, object-mode
keyframe selection, SAM 2, resize logic, confidence thresholds, and the existing
disabled keyframe-rejection behaviour. Export timing is not a speed benchmark.

## What is saved

- `capture/queries/`: every processed query frame's XYZ map in the current tracking
  coordinate system, confidence, RGB, mask, pose and source index. These maps are
  archival; the geometry evaluation uses keyframe reconstructions. Query maps
  must not be concatenated across cache updates without additional alignment.
- `capture/keyframes/`: every full keyframe reconstruction, including the duplicated
  bootstrap frame. XYZ `[N,H,W,3]`, RGB uint8 `[N,H,W,3]`, mask bool `[N,H,W]`,
  confidence `[N,H,W]`, poses `[N,4,4]`, source indices `[N]`, fixed threshold and
  full latest RGB. Per-view maps retain their original pixel grids.
- `manifest.json`: ordered RGB paths/IDs, object identity, mask mode, commits,
  tracker working diff, script hashes, torch/GPU and forced schedule if used.
- `capture_complete.json`: written only after the expected query count is saved.
  Each NPZ is reloaded during capture to check archive readability.
- `artifacts/snapshots/`: aligned combined PLY/NPZ for each actual snapshot, with
  saved Sim(3), anchors and residual. `final_views/`: separate PLY/NPZ per final
  keyframe, all in the same reference system. These are the Blender inputs.
- `metrics.csv`, `metrics.json`, `panels/*.png`, `actual_history.mp4`, and
  `playback.json`. Video uses one second per keyframe update, then a six-second
  frozen-final-cloud orbit. It is a summary of updates, not real-time playback.

Snapshots recompute earlier views. The separate `reveal` playback only reveals the
final reconstruction's clouds in acquisition order; it is explicitly not a record
of those earlier predictions. Duplicate bootstrap views are removed for metrics
and exported playback. Confidence filtering is fixed to the initial tracker
threshold for all evaluated snapshots, including bootstrap. The upstream viewer
does not confidence-filter bootstrap; this intentional display difference avoids
changing the evaluation filter halfway through the sequence. Raw arrays preserve it.

## Remote setup and first verification

On head, inspect `sinfo -N -o "%.10N %.6P %.9T %.15G"` and
`sacctmgr show user "$USER" format=Defaultaccount%30,Defaultqos%30`. Select a
compatible GPU and current account/QoS; the existing baseline used a 24g RTX 3090.
Do not use `--mem*`, `--cpus-per-task` or `--exclude`. Do not pull on head.

Commit/push the `kv_tracker` submodule change first, then commit/push its parent
pin and these tools. Both are required for the callback interface to reach CAMP.
Inside an allocation, pull the parent and submodules as documented in the cluster
notes. GPU-free preparation/evaluation can run on `data`; inference requires a GPU.

```bash
cd /mnt/projects/gr/3DRecon/layer_good
git pull --recurse-submodules
bash tools/kvt_run.sh check
```

The check exercises a known rotated/scaled/translated synthetic cloud, bootstrap
deduplication, a single intrusion into an empty ROI, separation of old/new views,
PLY exports, and decoding every generated MP4 frame. Expected final line:
`SYNTHETIC CHECK OK`. This must pass remotely before treating the pipeline as
runtime-verified. No packages are installed by the wrapper. An encoder failure is
reported rather than silently omitting the video.

## Select and prepare a clip

Extract selected HouseCat6D scenes into allocation-local `/tmp`. The existing
`tools/opt_pose_extract_housecat.sh /tmp/data/housecat6d test_scene1` can prepare a
candidate, but its extraction stamp is shared across scene selections: use a fresh
destination when changing the selected scene set. Inspect the labels for actual
instance IDs and model names. Do not assume any example scene contains a suitable
handle. The preparation command requires explicit frame selection, instance ID,
and verified depth units; no object or metric depth scale is guessed.

```bash
bash tools/kvt_run.sh prepare \
  --scene /tmp/data/housecat6d/test/SELECTED_SCENE \
  --instance-id SELECTED_ID --start START_INDEX --stop EXCLUSIVE_STOP \
  --stride 1 --resize-dim 308 --depth-units-per-metre VERIFIED_SCALE \
  --out /mnt/projects/gr/3DRecon/kvt_out/prepared
```

`--start/--stop` refer to sorted RGB file indices, not numeric filename values.
The contact sheet overlays the selected instance across up to 12 frames. Review
it along with the reference mesh and RGB opening. Preparation checks that instance
identity remains the same. The loader follows the existing HouseCat6D code's
OpenCV instance channel **2** and encoded-depth channel convention. Confirm these
against the displayed masks and known object scale before inference.

`reference.npz` is first-frame GT depth unprojected to camera coordinates in metres,
resized to the model grid with nearest-neighbour depth and pixel-centre-aware
intrinsics. No GT depth enters KV-Tracker. `reference_object.ply` and
`reference_rgb.png` support alignment-anchor and ROI selection. This point cloud is
only a visible surface, **not proof that the entire handle volume is empty**:
verify the region using the scanned mesh and multiple RGB views. Persist that
evidence with the experiment. Paths in `input.json` must remain accessible; after
an allocation ends, re-extract to the same `/tmp` path before capture or control.

## Capture

```bash
bash tools/kvt_run.sh capture \
  --input /mnt/projects/gr/3DRecon/kvt_out/prepared/input.json \
  --out /mnt/projects/gr/3DRecon/kvt_out/baseline
```

Use a fresh output directory each run. SAM 2 receives only the initial instance
mask, then propagates on the selected RGB sequence. `--stride 1` preserves all
frames for propagation. Start with a short clip to assess VRAM and output size;
saving all keyframe histories grows quadratically with keyframe count, and saving
all query maps also consumes substantial storage. Do not force 30 keyframes just
to match a target number. Record actual counts. Sample `nvidia-smi` during capture
and inspect delayed `jobstats` from head; use a smaller eligible partition if
the measured footprint permits it.

## Define evaluation inputs and produce artifacts

Paint an `anchor_mask.png` on the exact `reference_rgb.png` grid: nonzero pixels
must be reliable object surfaces away from the opening, depth discontinuities,
and occlusion boundaries. The same pixels align every snapshot using Sim(3);
anchors are not reselected based on confidence or fitted to the empty ROI.
This separates geometry error from the model's arbitrary scale and coordinate
system but can still be affected by first-view depth error. Inspect fit residuals.

Create `evaluation.json` beside `reference.npz`, with the following fields. Values
below marked `null` require object-specific choices and are deliberately not defaults:

```json
{
  "reference": "reference.npz",
  "anchor_mask": "anchor_mask.png",
  "roi_to_reference": null,
  "roi_extent_m": null,
  "voxel_size_m": null,
  "max_alignment_rmse_m": null,
  "empty_region_evidence": "",
  "render_reference_to_view": null,
  "render_bounds_m": null
}
```

- `roi_to_reference`: rigid 4x4 transform from the box centre/axes into first-camera
  coordinates. `roi_extent_m`: full box widths `[x,y,z]`. Leave clearance from all
  real surfaces. An arbitrarily oriented box is allowed.
- `voxel_size_m`: fixed grid cell size. Occupancy is occupied cells divided by
  `prod(ceil(extent / voxel_size))`; boundary cells can be smaller. Choose extents
  divisible by the voxel size for equal-volume cells.
- `max_alignment_rmse_m`: preselected acceptable anchor fit error. Rows above it
  are retained and explicitly flagged invalid, not counted as reliable evidence.
- `empty_region_evidence`: actual mesh identity, inspection result and supporting
  frame IDs. The script requires a note but cannot verify the geometric claim.
- `render_reference_to_view`: fixed rigid 4x4 display transform. Identity views
  along first-camera +Z, with +Y downward. `render_bounds_m`: `[xmin,xmax,ymin,ymax]`
  for the orthographic view; use equal X/Y spans to preserve aspect and include the
  object throughout the orbit. Point radius is exactly one pixel in the MP4.

```bash
bash tools/kvt_run.sh artifacts \
  --capture /mnt/projects/gr/3DRecon/kvt_out/baseline \
  --evaluation /mnt/projects/gr/3DRecon/kvt_out/prepared/evaluation.json \
  --out /mnt/projects/gr/3DRecon/kvt_out/baseline_artifacts
```

Metrics include retained/ROI point counts and fraction, voxel occupancy, mean
confidence inside the ROI, alignment error, old/new-view ROI counts, and displacement
of corresponding old-view pixels. The raw confidence maps remain available for
histograms. The video shows occupancy values and a fixed `[0,1]` occupancy plot;
invalid-alignment samples are red. Measurements are computed for the same saved
snapshots used for playback; they do not need a separate inference run. Evaluation
follows capture rather than running in a competing GPU process.

Counts alone grow with the number of views; inspect occupancy, fractions, old-view
changes, segmentation, and alignment together. No monotonicity or positive finding
is assumed. A usable numerical result needs acceptable alignment and independently
verified empty space. Without both, treat the outputs as visual evidence only.

## Matched segmentation control

```bash
bash tools/kvt_run.sh capture \
  --input /mnt/projects/gr/3DRecon/kvt_out/prepared/input.json \
  --mask-mode annotation \
  --keyframes-from /mnt/projects/gr/3DRecon/kvt_out/baseline \
  --out /mnt/projects/gr/3DRecon/kvt_out/annotation_control
```

This diagnostic uses annotation masks on every frame and the baseline's keyframe
schedule. It is not the unmodified method. Evaluate with the same reference, ROI,
anchors, voxel size and view; each run uses its own initial native confidence
threshold, which is recorded. For strict cross-run filtering comparisons, compare
the saved raw confidence maps under a separately declared common threshold.

## Blender playback (remote Linux with Blender 4.x available)

```bash
blender --background --python tools/kvt_blender.py -- \
  --artifacts /mnt/projects/gr/3DRecon/kvt_out/baseline_artifacts \
  --mode reveal --point-radius-m CHOSEN_RADIUS \
  --out /mnt/projects/gr/3DRecon/kvt_out/reveal.blend
```

Repeat with `--mode actual` and a new output path for actual history. The generated
scene uses RGB-coloured octahedra at point positions and one timeline step per
view/update. `reveal` accumulates objects; `actual` replaces the visible snapshot.
The script saves, reopens and checks vertex counts and visibility at every step.
Use Home / Frame All in the 3D viewport to frame the object, then scrub and orbit.
`--display-stride N` reduces Blender memory only and is recorded in the scene;
default 1 includes every filtered point. Point radius affects perceived filling:
use raw PLY and one-pixel MP4 alongside Blender when judging geometry.

Blender is not assumed to exist in `kvt.tar`; no installation is attempted. PLY,
NPZ and MP4 generation use the existing numerical/OpenCV stack. Copy finished
artifacts back with rsync; outputs remain outside the repository on CAMP.
