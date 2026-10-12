# Anchor3R versus our KV-Tracker designs: paper-level comparison

2026-10-12, user decision: compare Anchor3R with our results from what the paper states.
A matched run is not possible: the code (github.com/polar-explorer/Anchor3R) and weights
(huggingface.co/polar-explorer/Anchor3R) linked from the project page were not publicly
accessible on 2026-10-12 (GitHub 404, HF 401).

Source: Tao et al., "Anchor3R: Streaming 3D Reconstruction with Transient Anchors for
Long-Horizon Visual Mapping", arXiv 2606.05035v2 (14 Sep 2026); "Accepted at CoRL 2026"
on polar-explorer.github.io/Anchor3R-project-page. Numbers: FINDINGS "Anchor3R versus
ours, from the paper".

**Trust status (user, 2026-10-12): the paper's numbers are unverified claims** (no
public code or weights; baselines below their own papers' values). Do not use them as a
target or as evidence about our methods. The mechanism description is kept as related work.

## Mechanism

| | Our segmented design (`tools/KVT_SEGMENTATION_OVERVIEW.md`) | Anchor3R (Sec. 3, App. A-B) |
|---|---|---|
| Model | Pi3 inside KV-Tracker, training-free | VGGT-initialised, 1.2B, retrained ~12 days on 32 A800; pose-query tokens (1 pose + 31 registers per frame) |
| Anchor | First frame of a segment, kept until the next cut (ORB, >= 50 frames) | The current frame, every step; window W = 10 |
| Cache | Segment bank (anchor + latest keyframe), old images and KV deleted at a cut | Image-side KV of the window reused; pose-query states discarded (caching them hurt, Table 5) |
| Output per frame | Pose in the segment gauge, connected to the global chain | W-1 relative poses to frames in the window, plus the current local pointmap |
| Connection | One bridge per cut: pose pinned at the cut frame, scale from a point fit | Every frame: Lie-algebra median of W-1 candidate rotations, coordinate-wise median of W-1 candidate centres from already-fixed poses |
| Scale | Fitted explicitly at each bridge | Not modelled; one shared scale per window in training; limitation: "assumes reasonably consistent relative scales" |
| Causal output | Yes, never rewritten | Online: yes, never rewritten. Offline: revises all poses |
| Back-end | None yet (planned, `tools/CAUSAL_SUBMAP_SLAM.md`) | Offline: NetVLAD keyframes every 5 frames, top 3 reinserted as anchors; IRLS rotation + L1/ADMM centre averaging (11.9 s for 12K frames) |
| Past geometry | Not updated | Not updated (stated limitation) |
| Evaluation | TUM (8), ARCTIC objects (3); ATE, RPE, p99, FPS, peak memory | KITTI, VBR, TUM, Oxford Spires, Waymo (ATE); 7-Scenes, TUM (CD, F1@0.25); FPS and memory curve |

## What the comparison says

1. Short TUM sequences: through the shared baselines, Anchor3R-Online is about 0.74x
   VGGT-SLAM and 1.1x MASt3R-SLAM, while native KV-Tracker is 2.1-2.7x and the segmented
   design 3.0-3.8x on the five sequences we share with those baselines. This assumes the
   ratios carry over between the two protocols, which differ (their baselines score
   1.4-2.3x worse than in their own papers).
2. Cost: ours uses about 4x less peak memory (2.7 GiB vs 11.9 GB) at 1.5x the FPS, but
   at 308 px against an unstated resolution (training at <= 518) and unstated GPU.
3. No shared long-horizon sequence, no reconstruction numbers on our side, no
   object-centric results on theirs.
4. Both papers name the same cause of drift: a persistent early anchor (their Fig. 3
   attention sink; our 10-10 frame-0 drift). Their fix is a dense, robustly fused
   connection every frame; ours connects with one bridge per segment, and about 90% of
   our extra squared ATE is between segments (FINDINGS 26159 analysis).

## What would make it a matched comparison

The released code and weights, run on our prepared TUM inputs and scored with
`kvt_tum_run.evaluate` (it takes any `traj.npy`, N x 4 x 4 camera-to-world, manifest
order), on one GPU type with our FPS and peak-memory definitions; and a reconstruction
harness for CD and F1@0.25. Recheck the two links before relying on this note.
