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

Each group is one job from layer_good: `bash tools/kvt_tum.sbatch segment-eval <group>`
with group tum-short, tum-long, arctic or tum-long-arctic (both in one job). CAMP allows
one running and two submitted jobs per user, so splitting gives no parallelism: run
tum-short and tum-long-arctic. The driver's deadline is the job's own Slurm
EndTime; choose `-t` per job. It stops before starting a run that would not finish
(15 minutes kept for archiving) and records `stop_reason`. Contract suites
(segment_detector, handoff, reanchor 24 tests) run first.

Markers: `SEGMENT EVAL <scene> <arm> ATE/RPE/p99 ... fps ... peak GiB ... cuts ...` per
run, then `SEGMENT EVAL COMPLETE <group> <runs>` and `SEGMENT EVAL <group> JOB OK`
(tum-short 24 runs, tum-long 8, arctic 12, tum-long-arctic 20). Outputs in kvt_tum_out: per-run archives,
`<tag>_segment_eval_<group>_comparison.tar` (all metrics, timing, FPS, peak memory,
cut counts, ratios to native and to the segmented control), TUM input archives
(model_rgb and manifests), context and all_runs.

Success, per sequence: an anchor-only arm improves on `segmented` in all three of ATE,
RPE and p99; the overall target remains native within 5%. Report speed and memory with
accuracy. Object RPE is measured after Sim(3) alignment: read alignment_scale and
rotation RPE with it.

## Cut-placement control: does ORB's placement matter? (group cut-control)

Question (user, 2026-10-11): does the ORB detector work in the segmented design on
with_loop/no_loop, or is the result a coincidence? Existing runs cannot answer: the
anchor-only arms also changed the bank and the connection. This group keeps the
segmented design exactly (two-image segment banks, pose-pinned point-fit connection at
b+49) and changes only where the cuts fall.

Sequences: fr3 office, fr2 large_with_loop, fr2 large_no_loop. Arms (8 per sequence,
24 runs): native; segmented with live ORB cuts (26159 policy); the same number of cuts
evenly spaced (`segmented_uniform`); and placed uniformly at random with every cut at
least 50 frames from the previous one and from the end (`segmented_random0..4`, seeds
0-4). Runner `--cut-frames <json>` gives fixed boundaries without a detector; cut files
are saved in the comparison archive.

Decision rule, fixed before the run: ORB placement "works" on a sequence if the ORB arm
is better than the uniform arm and better than all five random placements on ATE, and
not worse than their median on RPE and p99. If ORB lies inside the random range, its
result there is consistent with chance placement. Five seeds give a coarse range only.
Caveat: GT covers 24% of with_loop (none between frames 636 and 4529) and 19% of
no_loop, so ATE there scores only the start and end of each route.

Entry: `bash tools/kvt_tum.sbatch segment-eval cut-control` from layer_good. Markers:
`SEGMENT EVAL <scene> <arm> ...` per run, `SEGMENT EVAL COMPLETE cut-control 24 runs`,
`SEGMENT EVAL cut-control JOB OK`.

## Context-probe diagnostic: does the model's own attention say when the anchor stops helping? (group probe)

User direction (2026-10-11): an automatic cut/loop signal from the KV cache's own context,
replacing ORB. Earlier "semantic/context" signals did not measure what the tracker uses:
25680 keyframe novelty pooled frame-local decoder[0] features (before any cross-frame
attention); 25996 K9 measured a keyframe's own register-to-patch attention at admission;
K7 tracked masked-black background on objects; 25978 context mass was uncorrected for
attention sinks. This probe measures the query itself.

`kv_tracker.context_probe.ContextProbe` hooks the 18 cross-frame decoder blocks (odd
indices) and, for each cached query, recomputes the softmax of the query's patch tokens
(object patches only on ARCTIC) over [cached bank keys, own keys] exactly as
forward_w_cache does. Per layer: support = cached-patch / (cached-patch + own-patch) mass
(register sinks excluded), top1 = share of (head, patch) whose strongest key is a cached
patch, object_support (anchor object patches only), and register (sink) masses. It reads
only; probe time is logged and included in query time.

Run: fr1_desk, fr1_room, fr3 office and the three ARCTIC objects; arms native, segmented
(ORB control) and anchor_probe (anchor-only segments cut every 150 frames so support can
decay, probe on). 18 runs. Entry `bash tools/kvt_tum.sbatch segment-eval probe`.

Analysis fixed before the run (Mac, saved arrays, GT for diagnosis only): per sequence and
layer, rank correlation of support/top1 with per-pair excess error over native and with
frames since the anchor; partial correlation controlling for frames since the anchor; and
whether the signal separates the worst 10% excess pairs (AUC) better than frames since the
anchor alone. A signal is promising only if it adds information beyond anchor age on most
sequences, objects included. Then it becomes the cut rule and is tested against random
placement as in 26524.

## Carry-over pilot: does keeping part of the old map help the connection? (group carry)

User direction (2026-10-12): segment reset plus selective carry-over of old-segment tokens,
pilot on fr2 and office. Prior work (docs/streaming-model-landscape.md §8b, gitignored):
OVGGT keeps the top 5% confident tokens of overlap-triggered anchors but never resets the
coordinate frame; LongStream resets but is trained; the training-free combination was not
found. Arms chosen by the user (four-arm option).

Why the connection matters (FINDINGS 26159): segment starts track at native level and
seams carry < 1% of squared RPE; about 90% of the excess squared ATE is between segments,
the chain of connections pinned on the old map's pose of b, its least-covered frame.

Design. Same segmented design and live ORB cuts (every arm must cut where the control
did; asserted). At each cut the old map's latest rebuild keyframe L is copied to the CPU and
a quarter of its patches (ceil(P/4); 25% is Claude's choice, the 25996 budget) joins the new
map's bootstrap and every rebuild (`MapHandoff(carry=...)`, Pi3 forward over kept patches;
the cache holds them) until the next cut replaces it. Old KV and images are still deleted.

| Arm | Patches of L | Connection |
|---|---|---|
| segmented | none | pinned at b, point-fit scale (control) |
| carry_confidence | top quarter by the old rebuild's confidence of L (patch mean) | unchanged |
| carry_confidence_pin | same | rotation/position pinned on L: old map's rebuild pose vs the new map's first joint rebuild containing L; point-fit scale at b |
| carry_random0/1/2 | uniformly random quarter (seeded per cut) | unchanged |

Plus native. 21 runs on office, with_loop, no_loop. carry_confidence is the plan as
proposed; carry_random is its selector control (25996: selectors did not beat random);
carry_confidence_pin targets the measured cause. 26518 pinned on L via one query and was
worse; here L is a member of the new map's joint rebuild.

Primary readout: `segment_split` in the driver, per run and native on the same GT-valid
frames: squared ATE under the one global Sim(3) minus squared ATE with one Sim(3) per
segment (segments with < 3 GT frames skipped) = between-segment part; reported as excess
over native (cm^2), with the within-segment excess. Also ATE/RPE/p99, FPS, peak memory.
Proposed reading (Claude's, before the run): an arm is promising if on office its excess
between-segment error is at least 25% below the control's and below all three random
seeds, without worse fr2 ATE. fr2: GT covers 24%/19% of frames, so its readout is coarse.

Entry: `bash tools/kvt_tum.sbatch segment-eval carry` from layer_good. Markers: contract
suites OK, `SEGMENT EVAL <scene> <arm> ...` and `SEGMENT SPLIT <scene> <arm> ...` per run
(native has no split line), `SEGMENT EVAL COMPLETE carry 21 runs`, `SEGMENT EVAL carry JOB OK`.
