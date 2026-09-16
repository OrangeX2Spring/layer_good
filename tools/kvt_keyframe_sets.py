"""Derive keyframe index sets for the ARCTIC keyframe-selection replay experiment.

No GPU, no model, no container, no cluster: this reads a completed run's own outputs
and writes one JSON holding every policy's frame indices per scene. The sets are
replayed with `kvt_arctic_run.py --keyframes-from`, which hands them to run_track3r's
existing `keyframe_indices` argument (`kv_tracker/main.py:196,480`), so KV-Tracker
itself is never modified.

Policies, all encoder-free:

  baseline              the run's recorded kf_idx. Replaying it is the fidelity gate:
                        it must reproduce the run it came from, or every replayed
                        number below is void.
  arc<T>                insert when the view direction is more than T degrees of
                        *true angular distance* from every keyframe already held.
                        This is what `check_if_keyframe` says it does and does not:
                        that function takes the elevation minimum and the azimuth
                        minimum independently, over what may be different keyframes
                        (`kv_tracker/main.py:115`), with azimuth as asin(x/z) rather
                        than atan2 (`:100`). Same rule as kvt_capture.py's
                        --keyframe-arc, which exists for the same reason.
  uniform_match_<P>     evenly spaced, as many frames as policy P chose.
  random_match_<P>_s<S> drawn without replacement, same count, seed S.

The matched pair are the null controls. Without them, "policy P beats the angular
rule at matched count" cannot be separated from "any set of that size does".

Every emitted set lists the frames to *insert*. Frame 0 is the bootstrap keyframe,
is always cached, and is never re-added (`main.py:360-365`), so it is excluded.

Run from the repo root, on the Mac against synced results:

    python3 tools/kvt_keyframe_sets.py \
      --run-root cluster_results/kvt_arctic/r518 --out /tmp/kf_sets.json

or inside the allocation against results still in /tmp, where the per-scene results
directory is named by --results rather than "results":

    python3 tools/kvt_keyframe_sets.py \
      --run-root datasets/arctic_data/data/cropped_images_grab_only_subset/s01 \
      --results-subdir r518 --out /tmp/kf_sets.json
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

SCENES = ("box_grab_01", "ketchup_grab_01", "espressomachine_grab_01")


def object_center(results_dir):
    """The masked mean of the latest keyframe reconstruction.

    `LatestKeyframes` writes it after the scene-origin shift and the Sim(3), which is
    the frame traj.npy is also in, so camera positions and this centre are comparable.
    """
    with np.load(results_dir / "keyframes.npz") as snapshot:
        xyz, masks = snapshot["xyz"], snapshot["masks"].astype(bool)
    assert xyz.shape[:-1] == masks.shape, (xyz.shape, masks.shape)
    assert masks.any(), results_dir
    return xyz[masks].mean(axis=0)


def view_directions(poses, centre):
    """Unit vectors from each camera position towards the object, as check_if_keyframe
    forms them (`main.py:118`) before it discards the angle for two per-axis ones."""
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4), poses.shape
    directions = centre[None, :] - poses[:, :3, 3]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    assert (norms > 1e-9).all(), "camera coincides with the object centre"
    return directions / norms


def greedy_arc(directions, arc_deg):
    """Insert once the view direction is more than arc_deg from *every* keyframe held."""
    assert 0 < arc_deg < 180, arc_deg
    limit = np.cos(np.radians(arc_deg))
    chosen = [0]
    for index in range(1, len(directions)):
        if (directions[chosen] @ directions[index]).max() < limit:
            chosen.append(index)
    return chosen


def uniform_set(count, n_frames):
    """`count` frames spread evenly over the trackable range, excluding the bootstrap."""
    if count == 0:
        return []
    spaced = np.linspace(1, n_frames - 1, count).round().astype(int)
    return sorted(set(spaced.tolist()))


def random_set(count, n_frames, seed):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.arange(1, n_frames), size=count, replace=False).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Directory holding <scene>/<results-subdir>/")
    parser.add_argument("--results-subdir", default="results",
                        help="Per-scene results directory name; 'results' in a synced "
                             "archive, the --results value inside the allocation")
    parser.add_argument("--out", type=Path, required=True, help="JSON to write")
    parser.add_argument("--scenes", nargs="+", default=list(SCENES), choices=SCENES)
    parser.add_argument("--arc", type=float, nargs="+", default=[10.0],
                        help="Angular thresholds in degrees; 10 is the tracker's own")
    parser.add_argument("--match", nargs="+", default=["baseline"],
                        help="Policies whose per-scene keyframe count the uniform and "
                             "random controls reproduce")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Seeds for the random control; one run is not a control")
    args = parser.parse_args()

    policies = {}
    centres, frame_counts = {}, {}

    def record(name, scene, indices):
        indices = sorted(int(i) for i in indices)
        assert indices == sorted(set(indices)), (name, scene, "duplicate index")
        assert all(1 <= i < frame_counts[scene] for i in indices), (name, scene, "out of range")
        policies.setdefault(name, {})[scene] = indices

    for scene in args.scenes:
        results_dir = args.run_root / scene / args.results_subdir
        traj = np.load(results_dir / "traj.npy")
        kf_idx = np.load(results_dir / "kf_idx.npy")
        frame_counts[scene] = int(traj.shape[0])
        centre = object_center(results_dir)
        centres[scene] = centre.tolist()

        record("baseline", scene, set(kf_idx.tolist()) - {0})

        directions = view_directions(traj, centre)
        for arc in args.arc:
            record(f"arc{arc:g}", scene, set(greedy_arc(directions, arc)) - {0})

    for target in args.match:
        assert target in policies, (target, sorted(policies))
        for scene in args.scenes:
            count = len(policies[target][scene])
            record(f"uniform_match_{target}", scene,
                   uniform_set(count, frame_counts[scene]))
            for seed in args.seeds:
                record(f"random_match_{target}_s{seed}", scene,
                       random_set(count, frame_counts[scene], seed))

    names = sorted(policies)
    width = max(len(n) for n in names) + 2
    print(f"\n=== keyframes inserted, excluding the bootstrap frame 0 ===")
    print(f"{'policy':<{width}}" + "".join(f"{s.split('_')[0]:>18}" for s in args.scenes))
    print("-" * (width + 18 * len(args.scenes)))
    for name in names:
        print(f"{name:<{width}}" +
              "".join(f"{len(policies[name][s]):>18}" for s in args.scenes))
    print("-" * (width + 18 * len(args.scenes)))
    print(f"{'frames':<{width}}" +
          "".join(f"{frame_counts[s]:>18}" for s in args.scenes))

    args.out.write_text(json.dumps({
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(args.run_root),
        "results_subdir": args.results_subdir,
        "frames": frame_counts,
        "object_center": centres,
        "arc_degrees": args.arc,
        "seeds": args.seeds,
        "policies": policies,
    }, indent=2) + "\n")
    print(f"\nSETS OK {args.out}: {len(names)} policies x {len(args.scenes)} scenes")


if __name__ == "__main__":
    main()
