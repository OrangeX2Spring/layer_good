# KV-Tracker reconstruction history and handle occupancy

Status: **executed on CAMP 2026-09-09/10**. `SYNTHETIC CHECK OK` passed remotely,
five captures and one matched control ran, and the object, clip and empty region
were selected and verified. Results and every number are in `tools/FINDINGS.md`
(last section); cluster-specific traps are in `docs/kv-tracker-cluster.md`. This
file is the workflow only — do not restate results here.

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

The `kv_tracker` callback commit (`c444af7`) and its parent pin are pushed. Inside
an allocation, pull the parent and submodules as documented in the cluster notes.
`prepare`, `scan`, `roi` and `artifacts` are GPU-free; `capture` needs a GPU.

```bash
cd /mnt/projects/gr/3DRecon/layer_good
git pull --recurse-submodules
bash tools/kvt_run.sh check
```

The check exercises a known rotated/scaled/translated synthetic cloud, bootstrap
deduplication, a single intrusion into an empty ROI, separation of old/new views,
PLY exports, and decoding every generated MP4 frame. Expected final line:
`SYNTHETIC CHECK OK`, preceded by `ARTIFACTS OK: 2 snapshots, 96 video frames;
2 valid alignments`. It passed remotely for the first time 2026-09-09; run it again
after any change to `kvt_artifacts.py`. No packages are installed by the wrapper. An encoder failure is
reported rather than silently omitting the video.

## Select and prepare a clip

Extract selected HouseCat6D scenes into allocation-local `/tmp`. The existing
`tools/opt_pose_extract_housecat.sh /tmp/data/housecat6d test_scene1` can prepare a
candidate, but its extraction stamp is shared across scene selections: use a fresh
destination when changing the selected scene set. Inspect the labels for actual
instance IDs and model names, then run `scan` — it reports, per frame, how many
pixels the mask *encloses* without covering, which is the opening. Do not assume any
example scene contains a suitable handle, and do not pick a reference frame by eye:
an object's handle can hide behind its body for stretches of a clip, and the frame
with the largest opening is rarely the one you would guess.

```bash
bash tools/kvt_run.sh scan --scene $SCENE --instance-id ID --stride 5
```

**Check the viewpoint arc first, before anything else.** `--arc-only` reads the
camera's position in the object's frame (`-R.T @ t`) straight from the annotations,
so it needs only `labels/` — no images, no capture, no GPU, and a whole test split
unzips in 12 MB. `--all-instances` reports every object in one pass.

```bash
bash tools/kvt_run.sh scan --scene $SCENE --all-instances --arc-only --stride 5
```

The number that matters is the **max pairwise angle**, which caps how many separated
keyframes any clip of that scene can ever yield. Ignore the azimuth span when the
path straddles +/-180 (it reads ~358 deg inside a 70 deg arc), and do not read the
10-degree cell count as a viewpoint count — a wandering trajectory lights many bins
while staying close to itself. HouseCat6D's whole test split spans 48-91 deg;
nothing in it orbits an object. The preparation command requires explicit frame selection, instance ID,
and verified depth units; no object or metric depth scale is guessed.

```bash
bash tools/kvt_run.sh prepare \
  --scene /tmp/data/housecat6d/test/SELECTED_SCENE \
  --instance-id SELECTED_ID --start START_INDEX --stop EXCLUSIVE_STOP \
  --stride 1 --resize-dim 518 --depth-units-per-metre VERIFIED_SCALE \
  --crop-margin 1.8 \
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

`--crop-margin` cuts a square window that follows the object, sized once from its
largest extent over the whole clip so the focal length is fixed and only the
principal point moves, taken at source resolution before the resize. Without it the
object arrives as a small island in an otherwise black frame and the reconstruction
suffers badly for it. It writes two files per frame under `--out`, so put `--out` on
job-local `/tmp` and `tar` the directory to `/mnt` before the allocation ends — one
file against the quota instead of hundreds. `capture` must then run in the same
allocation, since `input.json` points into `/tmp`.

## Capture

```bash
bash tools/kvt_run.sh capture \
  --input /mnt/projects/gr/3DRecon/kvt_out/prepared/input.json \
  --out /mnt/projects/gr/3DRecon/kvt_out/baseline
```

### Keyframe schedule

The tracker's own rule is defective and will not use the arc that exists.
`check_if_keyframe` (`main.py:136-141`) takes the minimum elevation difference and
the minimum azimuth difference over keyframes **independently**, so a keyframe fires
only when the view is >10 deg from *every* keyframe in one axis alone, and the two
minima may come from different keyframes. It saturates after ~5 keyframes however
much arc remains.

```bash
--keyframe-arc 10     # the same 10-degree intent, by true angular distance
--keyframe-arc 6      # deliberate over-sampling; say so when reporting
--keyframe-every 8    # a frame counter; arbitrary, prefer --keyframe-arc
```

Both compute a schedule offline and inject it through the override `main.py:480`
already supports for `--keyframes-from`. `--keyframe-arc` reads the annotations, so
its schedule is identical whatever `--mask-mode` is used: the segmentation control
is matched by construction and needs no `--keyframes-from`. Any of these is a
**forced schedule, not the method** — report which was used and why.

Cost grows with the square of the keyframe count, since `main.py:490` re-runs
`pi3_inference` over every keyframe each time one is added. 24 keyframes of 518x518
fit in 24 GB; step upward rather than jumping if you go further.

Use a fresh output directory each run. SAM 2 receives only the initial instance
mask, then propagates on the selected RGB sequence. `--stride 1` preserves all
frames for propagation. Start with a short clip to assess VRAM and output size;
saving all keyframe histories grows quadratically with keyframe count, and saving
all query maps also consumes substantial storage. Do not force 30 keyframes just
to match a target number. Record actual counts. Sample `nvidia-smi` during capture
and inspect delayed `jobstats` from head; use a smaller eligible partition if
the measured footprint permits it.

## Define evaluation inputs and produce artifacts

`roi` derives the evaluation inputs from the reference and writes them beside it:

```bash
bash tools/kvt_run.sh roi --reference $PREPARED/reference.npz --out $PREPARED
```

It must run after `prepare` and before `artifacts`: the ROI lives in the first
camera's frame and the anchor mask is indexed on that reference's pixel grid, both
of which change when the crop changes. It produces `anchor_mask.png`,
`evaluation.json` and `evaluation_tight.json`, and refuses rather than guessing
when the reference cannot support a measurement.

**Anchors** are the eroded, depth-continuous object surface, held clear of the
opening. The same pixels align every snapshot by Sim(3); they are never reselected
on confidence or fitted to the empty region. This separates geometry error from the
model's arbitrary scale and pose, but remains subject to first-view depth error, so
inspect the residuals.

**The box** is centred on the opening's centroid at the surrounding surface's median
depth, and oriented along the **line of sight through the opening** — not axis-aligned
in camera coordinates, which for an off-axis hole leaves the opening with depth and
runs into the object. Its lateral size is a fraction of the measured opening and its
depth is the free run between the nearest sensed surface in front of and behind it,
less one voxel: the space through a handle is a tube, not a cube. Of the candidate
fractions, the two largest distinct voxel-rounded sizes whose nearest GT point is at
least `--min-clearance-m` away are emitted, so the pair is a robustness check —
though when they land one voxel apart, that check is weak and should be reported as
such.

The fields it writes, and what they mean:

- `roi_to_reference`: rigid 4x4 transform from the box centre/axes into first-camera
  coordinates. `roi_extent_m`: full box widths `[x,y,z]`. Leave clearance from all
  real surfaces. An arbitrarily oriented box is allowed.
- `voxel_size_m`: fixed grid cell size. Occupancy is occupied cells divided by
  `prod(ceil(extent / voxel_size))`; boundary cells can be smaller. Choose extents
  divisible by the voxel size for equal-volume cells.
- `max_alignment_rmse_m`: preselected acceptable anchor fit error, default 5 mm.
  Rows above it are retained and explicitly flagged invalid, not counted as reliable
  evidence. **Choose it before the run and do not move it afterwards.**
- `empty_region_evidence`: written for you from what was measured — hole size, the
  density of GT depth on the object, and the clearance to the nearest GT point.
  A missing depth return inside the opening is **not** proof of free space, so
  confirm visually that you can see through the loop before trusting it.
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
