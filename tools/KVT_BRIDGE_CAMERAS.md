# Bridge cameras: why job 26127's connections disagree at b+49

User-approved diagnostic, 2026-10-02. Context: [KVT_ORACLE_MAPS.md](KVT_ORACLE_MAPS.md)
and the 26127 section of FINDINGS. All five bridges failed the camera-translation
check. Refitting the Sim(3) on held-out points or on any image half never passes.
So the disagreement lies in the two cameras compared at b+49:

- the old frozen bank's query, extrapolated 49 frames past its own keyframes;
- the new bank's joint-reconstruction camera, not a query of its cache.

This run separates the two. It is **diagnostic only**: no tracking pass, no oracle
reproduction, no GT, and no change to any acceptance rule or threshold.

## Method

Pi3 queries only read the KV bank (`use_cache` passes `ret_kv=False`). Each local
map's frozen state therefore depends only on two inference calls: the bootstrap of
its first image and the `[0, 49]` rebuild. The driver rebuilds each of the six maps
in turn from the 26127 input pixels (hash-checked against the manifest).
It then records:

- **D1, shared frame b.** The old map's query camera at b, compared with the new
  map's rebuild camera at b (and its same-cache query at b) through the Sim(3)
  recomputed exactly as 26127 did. No extrapolation is involved.
- **D2, route at b+49.** The new map's same-cache query at b+49, compared with its
  rebuild camera. It is also compared with the old query through the Sim(3)
  (query against query).
- **D3, curves.** Old and new cameras for every frame b..b+49. GT scoring is done
  offline on the Mac, using the 26127 segment alignments.

Distances are fractions of the old shared pointmap RMS extent, the gate's unit.
Fork change: the oracle rebuild also keeps `anchor_pose` (its camera at b), and
nothing else changes. Local frame 49 of the new map is read with the dense query,
because `step()` at 49 would rebuild again.

## Fidelity gates (checked after archiving)

The rebuilt maps must be 26127's maps:

- Bootstrap pose, emitted b+49 pose, local 50 and the last ten frames of every
  segment versus 26127 `local_traj.npy`: maximum absolute difference ≤1e-4.
- Bridge evidence `old_pose`, `new_pose` and `mapped_pose` ≤1e-4. Sampled
  `old_points` and `new_points` ≤1e-3.
- Identical bridge check booleans; camera-translation value within 1e-3.

The three input archives must match their reviewed SHA256 values. The Pi3
checkpoint must match 26127's `oracle_sources.json`. A failure exits nonzero, and
the results archive is already saved.

## Interpretation, fixed before the run

| Result | Reading | Candidate fix (user selects; not automatic) |
|---|---|---|
| D1 passes 5%, but b+49 fails | Disagreement grows over the 49-frame delay | Check cameras at b, where both maps have support |
| D2 route difference is large, and query-vs-query passes | The rebuild camera is the inconsistent estimator | Compare query with query |
| D1 fails | Points and camera are not consistent with each other in Pi3 | Fit points and the camera at b jointly |

D3 against GT then attributes the drift to the old or the new side. Boundary 1400
also fails its geometry checks, so no camera-side fix is expected to accept it.
Any changed gate needs its own pre-registered `oracle-maps` run.

## CAMP procedure

The job requires these files in `kvt_tum_out/`: `tum_26127_oracle_maps_inputs.tar`,
`tum_26127_oracle_maps.tar` and `tum_26127_context.tar`. Refresh the parent inside
an allocation. Then, from `/mnt/projects/gr/3DRecon/layer_good`, submit with the
default time limit:

```bash
sbatch tools/kvt_tum.sbatch bridge-cameras
```

The log prints one `BRIDGE CAMERAS <b>` line per boundary, then
`BRIDGE CAMERAS COMPLETE` and `JOB OK`. Output: `kvt_tum_out/tum_<JOB>_bridge_cameras.tar`
(`cameras_<b>.npz/json`, `fidelity.json`, `summary.json`, `inference.jsonl`, `config.json`)
and `tum_<JOB>_context.tar`. Transfer with the field-notes rsync route into
`cluster_results/kvt_bridge_cameras_<JOB>/`, one extraction directory per archive.
