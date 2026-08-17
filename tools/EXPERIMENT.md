# Experiment: is the geometry failure Pixal3D's prior, or our crop camera?

## Question

Occluded tabletop objects reconstruct badly (a thin box comes out as a thick
cuboid). Two candidate causes are currently confounded:

- **C1 — crop camera.** We crop a tight square around the object and let
  Pixal3D estimate FOV with MoGe-2. For YCB-V (`fx ≈ 1067 px`) a ~200 px crop
  has a true horizontal FOV near **11°** — nearly orthographic — while MoGe-2,
  fed a context-free object crop, typically returns 40–60°. Told to expect
  perspective foreshortening that is not there, the model can only explain the
  image by adding depth extent. Inflated thickness is the signature. (Upstream
  hints at this: `inference.py --fov` help says "try 0.2 rad ≈ 11.5° if you
  notice distortion".) The crop is also off-centre, so its principal point is
  not at the crop centre, which Pixal3D's centered-pinhole model assumes.
- **C2 — generative prior.** Pixal3D is trained on Objaverse-like assets and
  may simply be biased toward volumetric objects, in which case no amount of
  better 2D completion will fix thin geometry.

If C2 dominates, the whole complete-in-2D-then-lift strategy has a ceiling and
we should move to constraining the 3D stage with depth instead.

## Design

A 2×2 over **input condition** × **camera**, per object instance.

| | MoGe FOV (current) | True crop FOV (`--fov`) |
|---|---|---|
| **control** (unoccluded real view) | upper bound, today's camera | upper bound, fixed camera |
| **completed** (occluded + pix2gestalt) | today's pipeline | fixed camera |

`visible` mode (occluded, no completion, visible-mask alpha) is a useful lower
bound to add as a fifth cell — it says how much the 2D completion buys at all.

The control is a *real photograph* of the same physical object from a frame
where it happens to be unoccluded, not a render, so control and occluded
conditions differ only in occlusion.

### Success criteria

Stated before running, on `flatness_inflation` (generated thin/long extent
ratio over ground truth; 1.0 is correct, higher means too thick):

- **C1 confirmed** if fixing the camera drops `flatness_inflation` materially
  in *both* rows — i.e. the current numbers are largely a camera artefact.
- **C2 confirmed** if the **control + true FOV** cell still shows
  `flatness_inflation ≳ 2`. A perfect real input with a correct camera leaves
  only the prior to blame, and the 2D-completion path is capped.
- If control is fine and completed is not, the bottleneck is pix2gestalt
  quality, and better/duller completion is the right next investment.

`sensor_residual_mm` (median distance from the visible depth points to the
reconstruction) is the secondary metric: it says whether the output explains
what the camera actually saw, and it is the quantity the depth-constrained
sampling idea would directly optimise.

## Running it (remote Linux only)

```sh
cd /path/to/3d_generative_reconstruction/tools
YCBV=/mnt/4T/chunquan/datasets/ycbv
RUNS=$HOME/runs/occlusion_experiment
```

**1. Pick the occluded instance** (already done by `run_pix2gestalt_ycbv.py`;
its output dir holds `metadata.json` with `scene_id`/`image_id`/`gt_id`/`obj_id`).

```sh
COMP=outputs/pix2gestalt_ycbv/scene_000054_image_000001_gt_000003   # adjust
python -c "import json;print(json.load(open('$COMP/metadata.json')))"
```

**2. Find the matching unoccluded control view** for the same `obj_id`:

```sh
python find_unoccluded_view.py --ycbv-root $YCBV --obj-id <OBJ_ID>
```

**3. Build inputs.** `--padding` must match the pix2gestalt crop (default 0.2).

```sh
# occluded + completion
python prepare_pixal3d_input.py --ycbv-root $YCBV \
  --scene-id 54 --image-id 1 --gt-id 3 \
  --mode completed --completion-dir $COMP \
  --output-dir $RUNS/completed

# occluded, no completion (lower bound)
python prepare_pixal3d_input.py --ycbv-root $YCBV \
  --scene-id 54 --image-id 1 --gt-id 3 \
  --mode visible --output-dir $RUNS/visible

# unoccluded control, from step 2
python prepare_pixal3d_input.py --ycbv-root $YCBV \
  --scene-id <S> --image-id <I> --gt-id <G> \
  --mode control --output-dir $RUNS/control
```

Each prints the exact crop FOV. Sanity check: it should be in the 8–20° range
for a typical YCB-V object. If it is not, the crop is wrong.

**4. Run Pixal3D both ways** for each condition:

```sh
for COND in completed visible control; do
  FOV=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['fov_x_rad'])" $RUNS/$COND/meta.json)
  python ../inference.py --image $RUNS/$COND/input.png \
    --output $RUNS/$COND/moge.glb --low_vram
  python ../inference.py --image $RUNS/$COND/input.png \
    --output $RUNS/$COND/truefov.glb --low_vram --fov $FOV
done
```

**5. Score everything:**

```sh
for COND in completed visible control; do
  for CAM in moge truefov; do
    python eval_reconstruction.py --mesh $RUNS/$COND/$CAM.glb \
      --input-dir $RUNS/$COND --models-dir $YCBV/models \
      --output $RUNS/$COND/$CAM.json
  done
done
```

Repeat over ~10 instances (several objects, a spread of `visib_fract`) before
drawing conclusions; single-instance differences will be inside the noise of
diffusion sampling. Fix `--seed` across conditions.

## Notes

- `prepare_pixal3d_input.py` supersedes `compose_pix2gestalt_ycbv.py` for
  building Pixal3D inputs: it rectifies rather than slices, and emits the FOV
  and the depth point clouds. The old script is left in place untouched.
- Depth is eroded (`--depth-erosion`, default 3 px) before back-projection;
  depth sensors are worst exactly at occlusion boundaries, and flying pixels
  there would poison the residual metric.
- `eval_reconstruction.py` searches the 24 axis-aligned rotations of the PCA
  frame because Pixal3D's canonical output frame is not tied to the object
  pose. `flatness_inflation` needs no alignment at all and is the number to
  trust if the ICP looks unstable.
