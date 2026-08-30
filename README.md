# layer_good

Orchestration repository for a set of 3D reconstruction experiments. It holds no
model code of its own: each experiment line is a fork pinned as a submodule, and this
repository contributes the scripts that build and run them on a SLURM cluster.

## Experiment lines

| Submodule | Upstream | What it is |
|---|---|---|
| `pixal3d/` | `TencentARC/Pixal3D` | Single-image image-to-3D (mesh + PBR), TRELLIS.2 backbone |
| `kv_tracker/` | KV-Tracker | Monocular tracking / SLAM, evaluated on TUM RGB-D |
| `opt_pose/` | OPT-Pose | Category-level object pose, evaluated on HouseCat6D |
| `genrecon/` | `kasothaphie/GenRecon` | Posed multi-view scene reconstruction, TRELLIS.2 prior |
| `sam3d/` | `facebookresearch/sam-3d-objects` | Single-image shape + texture + layout, built for occlusion |
| `third_party/sam3` | `facebookresearch/sam3` | SAM 3, for mask generation |

Clone with submodules:

```bash
git clone --recursive git@github.com:OrangeX2Spring/layer_good.git
```

Submodules are independent forks. Nothing here modifies them in place; changes to a
model belong in that model's own repository.

## `tools/`

Two kinds of script, deliberately kept together:

- **Cluster environment and run scripts** — `*_build.sh` builds a container image for
  one line, `*_shell.sh` enters it, `*_run.sh` / `*.sbatch` run a job. Each resolves
  its own submodule, so they are independent of one another.
- **Pixal3D experiment scripts** — `prepare_pixal3d_input.py`, `eval_reconstruction.py`,
  `measure_mesh_metric.py`, `verify_camera_mapping.py`, `find_unoccluded_view.py`,
  `run_pix2gestalt_ycbv.py`, `compose_pix2gestalt_ycbv.py`, `ycbv_bop.py`. These drive
  Pixal3D from the outside rather than being part of it, which is why they live here.
  `prepare_pixal3d_input.py` writes the directory that the submodule's
  `inference.py --depth_dir` consumes.

## History

Until 2026-08-30 this repository *was* the Pixal3D fork, with the other lines added as
submodules alongside it. The Pixal3D source has since been split out into
`OrangeX2Spring/Pixal3D`, which carries upstream's history plus the depth-constraint
work, and is referenced here as a submodule like every other line. Commits before that
date therefore contain Pixal3D's own file history.

Working notes, cluster documentation and experiment findings are kept out of version
control by design; see `.gitignore`.
