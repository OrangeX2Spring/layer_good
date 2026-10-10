# Segmentation + reanchor evaluation on every tracked sequence

2026-10-10. User request: test both redesigns of the cut module on all object and TUM
sequences tested before. Evidence for the redesign is in FINDINGS ("What the cut
detector measures, and where segment tracking degrades"): on the office, frames
tracked against the segment anchor alone (local < 49) matched native per-frame error,
while all excess error came after the in-segment two-image rebuild; ORB fails on
masked objects (median 6 matches) and costs about 29 s on the office.

## Sequences (11) and arms (4): 44 tracking runs

| Group | Sequences | Frames |
|---|---|---:|
| tum-short | fr1_xyz, fr1_rpy, fr1_desk2, fr1_desk, fr1_room, fr3_long_office_household | 798, 723, 640, 613, 1362, 2585 |
| tum-long | fr2_large_with_loop, fr2_large_no_loop | 5182, 3359 |
| arctic | box_grab_01, ketchup_grab_01, espressomachine_grab_01 | 732, 652, 663 |

Arms, each a fresh process on the same saved pixels, no GT in tracking:

1. `native`: KV-Tracker unchanged (camera: cam_only, 308; objects: obj_mode, 518, saved
   SAM masks). TUM natives record the job 25680 ATE beside them, not asserted.
2. `segmented`: the current design: live ORB cuts (25 inliers, persistence 5, minimum 50
   frames), two-image segment banks, pose-pinned point-fit connection at b+49. TUM uses
   the 26159 office policy (no maximum); objects the 26458 anchor-support policy
   (maximum 200, anchor coverage 0.5).
3. `anchor_schedule`: `kv_tracker.anchor_segments.AnchorSegments` with cut 'schedule':
   each segment tracks against its anchor alone (bootstrap [b, b], never rebuilt), cut
   every 48 frames, connection made immediately at the cut (old map's dense geometry of
   the cut frame versus the new anchor's, pose pinned at the cut, point-fit scale).
4. `anchor_signal`: the same, cut by a tracker-side signal: TUM when fewer than 0.7 of
   the anchor's confident points lie in the current frustum; objects when the viewing
   direction about the masked object centre has turned more than 20 degrees.

48, 0.7 and 20 are fixed, uncalibrated values (no sweep). For ARCTIC the group first
runs `kvt_segment_arctic.py --object-coverage --coverages anchor_support` (live native
with SAM to produce masks, masked inputs, matched native replay and the segmented
control), then the two anchor-only arms.

## Running

Three independent jobs, each from layer_good: `bash tools/kvt_tum.sbatch segment-eval
tum-short`, `... tum-long`, `... arctic`. The driver's deadline is the job's own Slurm
EndTime; choose `-t` per job. It stops before starting a run that would not finish
(15 minutes kept for archiving) and records `stop_reason`. Contract suites
(segment_detector, handoff, reanchor 24 tests) run first.

Markers: `SEGMENT EVAL <scene> <arm> ATE/RPE/p99 ... fps ... peak GiB ... cuts ...` per
run, then `SEGMENT EVAL COMPLETE <group> <runs>` and `SEGMENT EVAL <group> JOB OK`
(tum-short 24 runs, tum-long 8, arctic 12). Outputs in kvt_tum_out: per-run archives,
`<tag>_segment_eval_<group>_comparison.tar` (all metrics, timing, FPS, peak memory,
cut counts, ratios to native and to the segmented control), TUM input archives
(model_rgb and manifests), context and all_runs.

Success, per sequence: an anchor-only arm improves on `segmented` in all three of ATE,
RPE and p99; the overall target remains native within 5%. Report speed and memory with
accuracy. Object RPE is measured after Sim(3) alignment: read alignment_scale and
rotation RPE with it.
