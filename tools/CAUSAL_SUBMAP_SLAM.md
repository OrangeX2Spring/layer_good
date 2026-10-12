# Causal submap SLAM on KV-Tracker: direction note

2026-10-11, user decision: close the gap between causal per-frame tracking (KV-Tracker,
the user's segmentation + reanchor design) and keyframe SLAM with a back-end (VGGT-SLAM),
by borrowing VGGT-SLAM's mechanisms in a causal form. Target: beat native KV-Tracker, and
if possible VGGT-SLAM, on the immediate per-frame trajectory, with bounded GPU memory.

## Why this gap exists (evidence)

- VGGT-SLAM (arXiv 2505.12549v2) processes frames in order but outputs keyframe poses in
  submap batches (w = 32 keyframes), revises past poses by global SL(4)/Sim(3)
  optimisation with loop closures, reports no FPS/latency/memory, and lists real-time
  operation as future work. TUM fr1 ATE mean on five shared sequences 0.042 m versus
  native KV-Tracker 0.114 m and the segmented design 0.161 m (FINDINGS "External
  baselines on TUM"; protocols differ).
- KV-Tracker gives every frame an immediate pose at ~20 FPS. The segmented design tracks
  locally at native level and keeps memory constant (2.7 GiB TUM), but drifts globally
  across segment connections (FINDINGS 26159 analysis, 26520/26522).

## Causal contract

The pose of frame t is output at time t (or within a stated fixed latency) from frames
<= t only and is never rewritten. A back-end may correct the current submap's transform
when new evidence (a loop closure) arrives; the correction applies to frames from that
time on. Report both the immediate trajectory (primary) and, for comparison with
VGGT-SLAM's protocol, the back-end-optimised trajectory.

Closest published causal method: Anchor3R (CoRL 2026), current-frame anchors with a
median-fused relative-pose graph and offline loop averaging; paper-level comparison in
`tools/ANCHOR3R_COMPARISON.md`.

## Mechanisms to borrow (VGGT-SLAM) and their causal form

1. Submaps: a segment is a submap with its own Pi3 gauge (already the user's design).
2. Submap alignment: VGGT-SLAM estimates SL(4) homographies from the shared frame's
   points instead of Sim(3), absorbing projective ambiguity of uncalibrated
   reconstructions. Causal form: the connection at a cut (no change in timing).
3. Loop closure: retrieve a past keyframe similar to the current view (VGGT-SLAM: SALAD
   descriptors), reconstruct the current anchor with the retrieved image, add a
   relative constraint, optimise the submap graph. Causal form: store past anchor images
   and descriptors on the CPU (no old KV); on a detection, optimise and apply the new
   current-submap transform forward.
4. Keyframing by image motion (disparity) rather than a fixed schedule or ORB.

## Open choices (user)

- Retrieval descriptor: Pi3's own encoder features (no new model, no download) or SALAD
  as in VGGT-SLAM (needs a model download through the HF job and a dependency).
- Baseline: decided 2026-10-12, VGGT-SLAM 1.0/2.0 and MASt3R-SLAM run by us in the
  borrowed kvt image: `tools/SLAM_BASELINES.md`.
- Sequences: loops matter most on fr3 office, fr1_room, fr2 with_loop; 7-Scenes (used by
  VGGT-SLAM) availability on CAMP unchecked.

## Reconstruction is untested (user, 2026-10-11)

None of our runs has evaluated 3D reconstruction; all comparisons are poses. The
KV-Tracker paper (arXiv 2512.22581) also reports only pose metrics (TUM, 7-Scenes,
ARCTIC ATE; OnePose recall) and shows reconstruction (fused keyframe pointmaps,
low-confidence keyframes pruned, threshold not given) qualitatively. VGGT-SLAM reports
dense reconstruction on 7-Scenes (accuracy, completion, Chamfer). Earlier we ran one
GT-assisted ketchup surface diagnostic (cluster_results/kvt_arctic/visualizations_25632/
ketchup_redundancy, ARCTIC mesh template), not a method comparison.

Needed for a reconstruction evaluation, all on CAMP:
- Export: native already saves keyframe pointmaps (pcd.npy, final_scene.npz); bounded
  runs save no geometry yet: save each segment anchor's confident pointmap and its
  segment transform (the causal map), plus the final fused cloud.
- GT: TUM depth images with GT poses (prepare() currently extracts RGB only; depth would
  be extracted for evaluation only, never for tracking); ARCTIC object mesh templates
  with articulation and object poses; 7-Scenes availability unchecked.
- Metrics: accuracy, completion, Chamfer, F-score at a fixed threshold, after the
  trajectory's Sim(3) alignment (no ICP for the primary number). The confidence
  threshold for fusion must be fixed per method before scoring (see the earlier lesson
  that apparent hallucination can be an uncalibrated retention threshold).

Pending input: cut-placement control 26524 (does ORB placement matter).
