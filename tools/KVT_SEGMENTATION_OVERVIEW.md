# KV-Tracker segmentation + reanchor: overview (as of 2026-10-11)

One-page entry point for the user's segmentation + reanchor line on KV-Tracker. Numbers
are quoted from `tools/FINDINGS.md` (section names given); procedures are in the linked
protocols; current job state is in `docs/STATUS.md`. All results are one run per arm
unless stated.

## 1. Goal and idea

KV-Tracker (Pi3 backbone) tracks every frame causally by attending to a KV cache of
keyframes. Native keeps up to 20 keyframes (camera) or all novel views (objects) in GPU
memory and re-solves them jointly. The user's idea: **segment the stream, reanchor each
segment on its first frame, delete the old images and KV, and connect consecutive
segments**, so GPU memory stays bounded while tracking continues. Target: accuracy within
5% of native, with less memory and time.

Requirements added along the way:
- 3D reconstruction results are required by the tutor, although the KV-Tracker paper
  reports only poses (`docs/STATUS.md` 2026-10-11; memory note).
- Direction (2026-10-11): a *causal* submap method that closes the gap to keyframe SLAM
  with a back-end (VGGT-SLAM) while keeping per-frame output (`tools/CAUSAL_SUBMAP_SLAM.md`).

## 2. What is implemented (fork `kv_tracker`, parent `tools/`)

| Component | Where | Notes |
|---|---|---|
| Segmented reanchor (the user's design) | `kv_tracker/reanchor_maps.py` (ReanchorMaps) + `map_handoff.py` | Segment bank = anchor + latest keyframe (or + previous, `--local-keyframe-cap 3`), rebuilt every 50 frames; connection at b+49: rotation/position pinned at the cut frame b, scale from a point fit of b's pointmaps; old KV released. |
| Cut detector (ORB) | `tools/kvt_segment_detector.py` | ORB 1000 features, ratio 0.75, RANSAC 2 px, < 25 inliers for 5 frames, minimum segment 50 frames. Objects add anchor-support coverage. |
| Connection variants (rejected) | `reanchor_maps.py` options | `overlap_stride` (scale from shared displacements), `two_way` (two-frame weighted Sim(3)), `latest_anchor` (pin on the old latest keyframe). |
| Anchor-only segments | `kv_tracker/anchor_segments.py` (AnchorSegments) | Bank = segment anchor only, never rebuilt; immediate connection at the cut; cuts every N frames, by anchor covisibility (TUM) or object view angle (ARCTIC). |
| Context probe | `kv_tracker/context_probe.py` (ContextProbe) | Read-only: per query, attention of the query's patch tokens to cached anchor patches in all 18 cross-frame layers, sinks separated. |
| Runner | `tools/kvt_reanchor_maps.py` | Flags for every option above; `--cut-frames` for fixed placements. |
| Evaluation driver | `tools/kvt_segment_eval.py`, wrapper mode `segment-eval <group>` | Groups: tum-short, tum-long, arctic, tum-long-arctic, cut-control, probe. Deadline = the job's Slurm EndTime. Protocol: `tools/KVT_SEGMENT_EVAL.md`. |

## 3. Evaluation protocol and its limits

- Sequences: TUM fr1_xyz, fr1_rpy, fr1_desk2, fr1_desk, fr1_room, fr3_long_office_household,
  fr2_large_with_loop, fr2_large_no_loop (308 px); ARCTIC S01 box, ketchup, espressomachine
  grab (518 px, SAM-masked).
- Metrics: ATE (one Sim(3) over the trajectory), translation RPE and its p99 between
  consecutive frames, rotation RPE; native within 5% on all three translation metrics.
- Limits that change conclusions: RPE is computed after the same Sim(3) fit, so a
  trajectory whose fitted scale shrinks shows artificially low RPE (seen in 26500); GT
  covers only 24% of fr2 with_loop (none between frames 636 and 4529) and 19% of no_loop;
  placement alone moves fr2 ATE by about 0.4 m (26524), so smaller single-run differences
  there are noise. Timings exclude model load and evaluation.

## 4. Main results

**All 11 sequences** (FINDINGS "Segmentation evaluation on all 11 sequences, 26520/26522"),
geometric mean of ratios to native:

| Method | ATE | RPE | p99 | Lowest bounded ATE on | TUM FPS | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---|
| Native | 1.00 | 1.00 | 1.00 | — | 20.0 | 3.7-4.3 GiB (objects 4.3-8.6) |
| **Segmented (user's design)** | **1.30** | 1.01 | 1.21 | **7 of 11** | 18.6 | 2.7 GiB (objects 3.0) |
| Anchor-only, every 48 frames | 2.16 | 0.64 | 0.74 | 1 | 23.0 | 2.7 GiB (objects 3.0) |
| Anchor-only, tracker signal | 1.94 | 0.52 | 0.47 | 3 | 22.4 | 2.7 GiB (objects 3.0) |

Office best variant: segmented with three-image banks, ATE/RPE/p99 0.0823/0.0143/0.0481 m
versus native 0.0619/0.0119/0.0326 (FINDINGS 26515).

**Long sequences**: native degrades with a fixed bank; segmentation keeps memory constant.

| | with_loop ATE / RPE | no_loop ATE / RPE | Peak memory |
|---|---|---|---|
| Native, 20 keyframes | 1.142 / 0.126 | 2.117 / 0.424 | 4.9 GB |
| Native, 48 keyframes spread over the clip (uses clip length) | 0.578 / 0.161 | 0.838 / 0.201 | 7.6 GB |
| Segmented | 0.926 / 0.059 | 0.735 / 0.066 | 2.9 GB |

(25680 sweep; 26520/26522.)

**Does ORB's cut placement matter?** (FINDINGS "Cut-placement control 26524"): same design,
same number of cuts, placed by ORB, evenly or randomly (5 seeds). Office: ORB ATE 0.084
versus 0.129 even and 0.147-0.246 random, so placement matters. with_loop and no_loop: ORB
lies inside or worse than the random range, so its result there is chance; every
placement beats native on both, so the fr2 gain comes from segmenting at all.

**Efficiency** (FINDINGS 26159 review, 26471/26472 audits): office KV 1.42 to 0.14 GB
(-90%), peak 4.56 to 2.91 GB (-36%), but 0.79x native speed because ORB costs about 29 s;
query time is unchanged at 308 px. Objects: 1.87x pipeline speed, 1.66-1.75x with matched
heads, peak 8.62 to 3.03 GiB.

**External reference** (FINDINGS "External baselines on TUM"): VGGT-SLAM reports mean ATE
0.042 m on the five shared fr1 sequences, versus native KV-Tracker 0.114 and the segmented
design 0.161. Protocols differ: VGGT-SLAM is incremental keyframe SLAM with global
optimisation, outputs keyframe poses in submap batches, revises past poses, and reports no
FPS, latency or memory. VGGT-Long reports no TUM or indoor results.

## 5. What was tried and what it taught (chronological)

| Period / jobs | Change | Outcome | Lesson |
|---|---|---|---|
| 10-02..10-04 (26136, 26159, 26198, 26200) | User's segmented reanchor, ORB cuts | Office ATE 8.38 cm vs native 6.19; fr2 beats native | The idea works; gap = between-segment consistency |
| 10-10 (26471-26504) | One map anchored on frame 0 forever, refresh every 50 (pin, scale transfers, novelty, retrieval bank) | Office ATE 0.53-1.54 m | **Drift away from the user's idea** (no segmentation); scope correction in FINDINGS. Memory note: keep the user's mechanism in every arm |
| 26514 | Overlap connection (scale from shared displacements) | ATE 1.09 m | The old map is biased on frames leaving its coverage |
| 26515 | Three-image segment banks | Small gain on all three (office 0.0823) | Within-segment noise is only part of the gap |
| 26516/26517 | Two-way, two-frame weighted scale (VGGT-Diff-inspired) | No camera gain; ketchup gain not attributable | Connection scale is already consistent to ~2% |
| 26518/26519 | Pin connection on the old latest keyframe | Office ATE 0.143 | The new map's view of a past frame is the weaker side |
| 26520/26522 | All 11 sequences, anchor-only variants | Table above | Local accuracy vs global drift trade-off |
| 26524 | Cut-placement control | ORB matters on office only | Every cut rule needs a random-placement control |

Read-only analyses on saved 26159 arrays (FINDINGS "Segmented reanchor 26159", "What the
cut detector measures"): about 90% of the extra squared ATE is between segments; frames
before a segment's first rebuild (local < 49) track at native level (1.22 vs 1.22 cm),
all excess error comes after it; cuts come a median 8 frames after overlap is lost
(those tails carry 50% of excess RPE); ORB "overlap lost" occurs while 77% of the
anchor's points are still in view, so ORB measures appearance-matching failure, not what
Pi3 uses. On masked objects ORB finds median 6 matches, so object cuts are effectively a
fixed 54-frame schedule.

Earlier semantic/context signals did not measure the query's use of the cache: keyframe
novelty used frame-local `decoder[0]` features (25680); the K9 selector measured a
keyframe's own register attention at admission (25996); K7 tracked masked background;
the 25978 context mass ignored attention sinks. This motivates the context probe.

## 6. Current plan

1. **Context-probe diagnostic (job 26554, running)**: does attention of the query's patches
   to the cached anchor predict where tracking degrades, beyond the trivial "frames since
   anchor"? Analysis rule fixed in `tools/KVT_SEGMENT_EVAL.md` (probe section).
2. If promising: make it the automatic cut rule (and loop-retrieval score), validated
   against random placement as in 26524.
3. Reconstruction harness (tutor requirement): export the causal map, score against TUM
   depth and ARCTIC meshes (accuracy, completion, Chamfer, F-score), confidence threshold
   fixed per method before scoring.
4. Causal back-end (`tools/CAUSAL_SUBMAP_SLAM.md`): loop closure over stored anchors and a
   pose graph whose corrections apply forward; report the immediate trajectory first.
   Open choices: loop descriptor (Pi3 features vs SALAD), running VGGT-SLAM ourselves as a
   matched baseline.

## 7. Where things are

- Results and caveats: `tools/FINDINGS.md` (sections named above).
- Current state, job IDs, next action: top of `docs/STATUS.md`.
- Protocols: `tools/KVT_SEGMENT_EVAL.md` (evaluation, cut control, probe),
  `tools/KVT_HISTORY_GENERALIZATION.md` (10-10 gates incl. the drift period),
  `tools/KVT_REANCHOR_MAPS.md` (original design), `tools/CAUSAL_SUBMAP_SLAM.md` (direction).
- Local archives: `cluster_results/kvt_segment_26159/` (reference segmented run),
  `kvt_segment_eval_26520/` (27 archives), `kvt_segment_eval_26522/` (24 archives; both
  transferred, not yet audited or extracted), per-gate
  folders `kvt_camera_history_*`, `kvt_object_history_*`.
