# Reanchored local maps: the original idea, reimplemented

User-selected, 2026-10-02. It replaces the oracle-maps connection policy
([KVT_ORACLE_MAPS.md](KVT_ORACLE_MAPS.md)) and keeps that run's results as history.

The idea, in the user's words: one continuous run; "when we enter the second
segment, we can reanchor the first frame and delete all previous contents"; about
two keyframes per segment; connect segments with a good Übergang; cut history
instead of keeping it.

## What oracle-maps (26127) got wrong

- It kept the old bank alive for 49 frames, then judged the connection by the old
  map's camera at b+49. That camera is extrapolated past its keyframes and drifts
  (FINDINGS, bridge cameras 26133).
- Gates rejected every connection, so the run produced no continuous trajectory.
- Each segment kept its frozen b/b+49 pair, so long segments (975, 1400) degrade.
- The target required each window under one alignment to match independently
  aligned oracle windows. Native KV-Tracker itself fails that in 4 of 6 windows.

## Mechanism

Boundaries stay 0, 275, 575, 700, 975, 1400 and 2585. At boundary b:

1. The old bank tracks b as usual (camera query) and reads b's dense geometry.
2. Old KV and old images are **deleted immediately**; weakrefs must be dead.
   Only b's old camera and pointmap survive (CPU), as transition state.
3. A new map bootstraps on b. Inside a segment it keeps b plus the latest keyframe,
   rebuilt every 50 frames (`MapHandoff` mode `reanchor`, the same schedule as
   `fixed`). Every bank has at most two images.
4. At the first rebuild [b, b+49], the Übergang commits, always.
   - Rotation and position pin the new camera of b exactly onto the old camera of b.
   - Scale comes from the robust Sim(3) fit on b's shared pointmap.
   - The fit's geometric checks are saved as diagnostics, not used as a gate.
   - Global poses for b..b+48 are a 49-frame delayed output.

No GT enters tracking. One shared model; seed 0; same inputs as 26115/26127.

## Target

The run is scored as one continuous trajectory with the standard evaluator. ATE,
translation RPE and translation p99 must each be ≤1.05× native 25680 on the same
2583 GT frames: 6.19cm, 1.19cm and 3.26cm. Seams (pairs b-1→b) are reported.

Posthoc expectation, from saved 26127 arrays with frozen pairs and this
connection: better than native on 0–700, but worse once the weak 975/1400
segments join (full ATE 141.9cm). The keyframe refresh is the untested part that
must fix those segments. No outcome is promised.

## Correction after job 26135: scale at refresh rebuilds

Job 26135 tracked all 2585 frames and committed all five connections, then crashed
building the global trajectory (zip over 7 boundaries; fixed in parent bbf7cf8).
Offline scoring of its saved arrays (FINDINGS, reanchor 26135) showed the refresh
working between rebuilds but jumping at each one. The median distance from the
anchor camera changed 8.4% across a rebuild, against 0.9% otherwise. Each
[b, latest] rebuild re-normalizes Pi3's scale about camera b. Now every later
rebuild is rescaled to the first rebuild's scale. The scale is the median
norm ratio of the anchor's confident pointmap: both maps are expressed in camera
b's frame, so only scale differs. Queries and dense queries apply that scale.
`update_total` logs it (1.0 in other modes, whose poses are unchanged).

## Checks and CAMP procedure

Local: Python AST 3.10, Bash syntax, whitespace. CAMP contracts:
`test_kvt_handoff.py` and `test_kvt_reanchor.py` (anchor+latest schedule;
retirement at b; pose-anchored connection with checks that do not gate).

The job requires these files in `kvt_tum_out/`: `tum_26127_oracle_maps_inputs.tar`
and `tum_26127_context.tar`. Their hashes and the checkpoint are verified first.
Refresh the parent inside an allocation. Then, from
`/mnt/projects/gr/3DRecon/layer_good`, submit with the default time limit:

```bash
sbatch tools/kvt_tum.sbatch reanchor-maps
```

The log prints five `REANCHOR <b>` lines, then `REANCHOR MAPS COMPLETE` with the
metrics against native, then `JOB OK`. `JOB OK` means execution only; the native
verdict is in `summary.json`. Output: `kvt_tum_out/tum_<JOB>_reanchor_maps.tar`
(`traj.npy`, `local_traj.npy`, `evaluation.npz`, `summary.json`, `bridge_*.json/npz`,
`transforms.json`, `inference.jsonl`, `config.json`) plus the context tar.
