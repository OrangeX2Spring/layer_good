# Manual local maps with bounded overlap and history retirement

User-selected revision, 2026-10-02: implement the idea across the full office
sequence and target at least the saved oracle accuracy. **No reproduction runs**,
new native arm, sweep or replay inference. One pass over 2585 frames. Job26115
archives are read only as pixel/checkpoint references and post-run quality targets.
Supersedes the 26118 fixed-anchor/latest-frame policy for this research direction;
preserve that pilot and its negative findings.

## Tracking contract

Boundaries are exactly 0,275,575,700,975,1400,2585. Each local map duplicates its
first image for bootstrap, rebuilds once at local49 using [0,49], then freezes
that pair. No global frame0 retention, periodic replacement or inherited scene
centering. One shared frozen model; each new map starts with an empty KV bank.
The same model is reused in eval mode; no stochastic operation is introduced.

At boundary b, the old frozen bank queries b without rebuilding. A second,
dense query obtains b's geometry in the old map; its camera pose must match the
camera-only query, and cached tensor identities must remain unchanged. The old
origin_offset and old scene-centering transform are applied exactly once.
Start a fresh local map at b and retain the old bank temporarily.

At b+49, first emit the normal local query and rebuild exactly [b,b+49]. Query
b+49 with the saved old bank, then restore the new cache. Fit candidate-to-old
Sim(3) using b's shared pointmap; validate disjoint points and candidate b+49
camera agreement with the old-bank query. Reuse the fixed geometric criteria
from [KVT_HANDOFF_PILOT.md](KVT_HANDOFF_PILOT.md); do not loosen after failure.
Compose with old-to-global Sim(3); never insert scale into camera rotation.
The new map's bootstrap scene centering is applied to both bridge geometry and
candidate camera, so the fitted transform maps the actual local output coordinates.

This is a **49-frame delayed-output diagnostic**: the first50 local poses are
buffered conceptually until the connection is available. The driver saves outputs
after tracking, and never claims immediate global poses for those frames.
No future frame beyond b+49 enters that connection. No GT enters tracking/fitting.
Following local49, the local bank and coordinate transform remain fixed.

Retain at most two banks (four image slots, four unique images once the second
new keyframe arrives). The shared model is not duplicated. Temporary activations,
pointmaps, CPU trajectory diagnostics and archive inputs are additional storage.
Measure peak allocated/reserved GPU memory and dual-cache bytes. Assert old/new
KV storage disjointness, then old KV tensor and old image weakref death.

On rejection, record **tracking loss** and continue an explicitly disconnected
local component to measure all six segments. Retire the old state; do not restore
the global-frame0 policy, force a transform, or score disconnected maps as one
trajectory. Later accepted edges cannot repair a previously broken chain.
All five boundaries are attempted once; no automatic retries.

## Acceptance and reporting

Before inference, require the saved26115 input/result/context archives under
`kvt_tum_out/`. Verify current checkpoint hashes against archived checkpoint.json,
and every source/model pixel hash and timestamp against archived manifests.
Record reference archive hashes. No model downloads or dependency installs.

One new tracking pass supplies every score; no reference model inference:

1. **Local accuracy:** evaluate each local output against exactly the oracle GT
   frames/pairs using the same independent alignment. ATE, translation RPE and
   p99 must each be no worse than saved unrounded values, allowing only
   `oracle * 1e-4 + 1e-6 metres` numerical tolerance. Report raw trajectory maximum
   difference for diagnosis, not a claim of bit identity. All six must pass.
2. **Connections:** all five bridges must accept, old state retirement must pass,
   and no disconnected component may be labelled a continuous trajectory.
3. **Continuous accuracy:** if connected, fit one global Sim(3), then measure
   each oracle window under that same fit. Require the same three oracle bounds
   in every window; do not independently refit these continuous windows.
4. **Seams:** report translation/rotation RPE at each cross-boundary pair. Require
   each translation error <= the larger oracle translation p99 of its two adjacent
   segments, with the same numerical tolerance. All five pairs must be GT-valid.
   This is a new engineering gate: the independent oracle has no seam measurement.

Only all four gates establish the requested objective. Local parity alone does
not. Five geometry passes alone do not. Since the last two oracle segments were
weak, oracle parity is not a claim of uniformly good long-sequence accuracy.

`JOB OK` means execution/contracts and the local accuracy gate passed; the printed
continuous gate and summary determine whether the objective was achieved. A local
regression exits nonzero after saving results. Rejected connections can finish
normally, but `objective_achieved` remains false and full `traj.npy` is absent.

## Checks, artifacts and CAMP procedure

Local: Python AST, Bash syntax, whitespace and source/caller review only.
CAMP: existing TUM/bridge contracts, plus frozen1185-frame admission schedule,
shared-query coordinate conversion, delayed accepted/rejected connections and
image/KV retirement tests. No extra inference replay is scheduled.

Publish fork and parent gitlink; user refreshes parent inside an allocation.
From `/mnt/projects/gr/3DRecon/layer_good`, submit with default time limit:

```bash
sbatch tools/kvt_tum.sbatch oracle-maps
```

Outputs: `kvt_tum_out/tum_<JOB>_oracle_maps_inputs.tar`,
`tum_<JOB>_oracle_maps.tar`, context and partial-run safety archives.
Result contains `local_traj.npy`, per-segment metrics/evaluation arrays, five
bridge JSON/NPZ evidence pairs, events, inference rows, summary/config and, only
if connected, full `traj.npy`/evaluation arrays. Context records source revisions,
oracle/checkpoint identities and remote contracts. Log uses the existing
`kvt_tum_slurm-<JOB>.log` path. Preserve separate extraction directories on transfer.

Review first failure or returned artifacts before changing the bridge, boundary
placement or pair policy. No promised runtime outcome before that review.
