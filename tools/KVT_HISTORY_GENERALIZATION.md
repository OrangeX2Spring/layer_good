# Bounded history accuracy fix and generalization

2026-10-10. User requests one further accuracy fix, then evaluation across all
available object and camera tracking sequences, short and long, followed by a
tutor report. This supersedes repeating only the original ketchup matrix.
Cluster actions remain user-run. No automatic full run behind an unreviewed pilot.

## Fixed change and first gate

Keep the first anchor and the previous and newest admitted keyframes, at the
existing 50-frame refresh interval. Maximum three images/KV frame slots after
the second refresh; bootstrap remains two duplicate slots. Compare against the
existing two-image no-retirement map. No new thresholds, smoothing, training,
semantic masks, resolution change or threshold sweep. The additional view may
improve geometry when the anchor becomes dissimilar; the reviewed ketchup result
is mixed (FINDINGS Artifact audit26472).
Memory can rise with the extra frame; this is an explicit capacity tradeoff.

Entry: `bash tools/kvt_tum.sbatch object-history three-frame`, two-hour ceiling,
same timeout/packaging rules as KVT_OBJECT_OVERNIGHT. Full saved ketchup inputs,
three fresh processes: native, no_retirement (two images), three_frame (three).
Native must match26458; two-image control must match26471's hash-pinned archive.
Both bounded maps retain [0,N] boundaries, no retirement events and the same
detector decisions. Assert input/image-bank caps, finite full trajectories and
identical evaluation pairs. Contracts test the exact three-view schedule and
scale restoration. Source/checkpoint/pixel assertions are retained.

Report ATE, translation/rotation RPE, p99, native ratios and three/two ratios.
`improves_two_image` means all three translation metrics strictly improve;
native5% is the separate existing quality gate. Poor quality is recorded, not
converted to an execution error. Runtime/fidelity failure blocks expansion.

The pilot is necessary before the full sweep. It is not the requested full
evaluation. The camera gate below ran as job26476 and was reviewed; dataset
adapters remain pending work; do not claim all-dataset readiness from two sequences.

## Timing audit and measurement

Paper: https://arxiv.org/html/2512.22581v1, Table5 and section4.3. It reports
27FPS at308 and16FPS at518 in its object table, with RTX4090 as default hardware.
Section4.3 uses synthetic308-square inputs and cached-query inference. Pilot26471
used RTX A5000, resize518, and a whole-sequence synchronized replay timer.
The native displayed FPS is a rolling mean of reciprocal query durations; its
endpoint precedes map updates. It is not frames divided by sequence wall time.

Additional confound: native object queries compute dense points/confidence and
apply its confidence rule; MapHandoff queries normally compute camera poses only.
The old1.87x/1.96x ratios describe these different pipelines, not a cache-only
speedup. Keep accuracy protocols intact; do not disable native confidence merely
to improve a timing number. Exact causes of the remaining hardware/runtime gap
need measurements, not an assumed GPU multiplier.

The three-frame pilot profiles all arms, separately saving:

- `forward_timing.json`: synchronized model.forward calls, mapping versus query,
  dense versus pose heads, actual dimensions, GPU-event and wall seconds.
  Includes first calls; hooks add overhead. Keep profiled pipeline timings
  distinct from unprofiled26471 timings.
- `cached_head_benchmark.json`: after trajectory/evaluation, reuse the last
  frame and final frozen cache; five warmups and20 forwards each for pose-only
  and dense heads. Require the predicted poses to agree. Input transfer, SAM,
  mapping and postprocessing excluded. Compare matching heads across methods.
  This is a fixed-cache microbenchmark, not online FPS or a reproduction of the
  paper's308/4090 experiment. Cache sizes differ by method; record them.
- Existing tracking timer: all input handling/tracking/rebuilds, with scope
  recorded; subprocess wall also includes setup/evaluation and the new benchmark.
  Do not call subprocess wall pure inference time.

## Dataset coverage from repository records

Do not request a new broad cluster listing: start from the recorded inventory.
Run complete chronological sequences, no frame sampling to make long clips fit.
Save frame counts/duration, sequence/split IDs and task, and group results by
dataset and length. Never average camera and object tasks into one headline.

| Dataset | Recorded availability and scope | Integration status |
|---|---|---|
| TUM RGB-D | `/mnt/datasets/tum-rgbd`; job26045 training_dataset_inventory.json explicitly lists25 ZIPs (older docs say26) | Existing full-frame preparation and timestamp/GT-hole evaluator; use every listed ZIP, including short/static/dynamic/long cases |
| ARCTIC S01 | box_grab_01, ketchup_grab_01, espressomachine_grab_01; prepared.tar plus reviewed initial masks | Existing SAM/native and saved-mask replay; all three full clips,518 |
| YCBInEOAT | mustard_easy_00_02.tar.gz only | Existing kvt_active_objects.py staging/SAM/GT inversion; adapt saved-mask replay without changing convention |
| HouseCat6D | Complete release on `/mnt/datasets/housecat6d`; scene archives, test_scene.zip, val_scene.zip | Existing preparation/capture; full tracking evaluator and per-instance sequence manifest still needed; distinguish train/val/test |
| YCB-V | `/mnt/datasets/bop/ycbv/ycbv_test_all.zip`; test data used by prior experiments | Existing extraction/GT tools; chronological object instance identities, mask protocol and pose evaluator must be fixed before running all sequences |
| Neural RGB-D | `/mnt/projects/gr/3DRecon/stream3r_data/neural_rgbd.tar.gz` | Images and poses available; add camera replay/pose convention gate. Old depth_filtered/depth issue blocks reconstruction evaluation, not necessarily RGB pose evaluation; do not substitute depth silently |
| KITTI and ScanNet | Shared raw copies recorded in docs/stream3r-cluster.md | Validate temporal sequence/split/pose eligibility using their recorded loaders before a pose sweep; reconstruction-only frames are not a tracking sequence |

7-Scenes, OnePose/OnePose-LT and NOCS are recorded absent; no downloads are
authorized by this plan. NYU-v2's downloaded reconstruction samples and BOP
single-image pose sets are not assumed chronological tracking sequences.
Record ineligible/missing-GT entries explicitly rather than silently skipping.
No new synthetic occluder sweep or prior token-policy arms.

Inventory sources: docs/kv-tracker-cluster.md (paper dataset availability),
docs/stream3r-cluster.md (shared and downloaded data), docs/opt-pose-cluster.md
(complete HouseCat), tools/KVT_ACTIVE_KEYFRAMES.md (YCBInEOAT), and
cluster_results/atd_review_26043_26045/26045/training_dataset_inventory.json.

Full comparison: native, existing two-image no-retirement control, proposed
three-image no-retirement method. Preserve the reanchor26471 result as historical
context, not an extra all-dataset arm. Same pixels/masks, task resolution, GPU,
precision and evaluation associations within each sequence. Report complete
pipeline time plus like-for-like head timing, memory and all failed cases.
Pilot runtime informs allocation chunks and output storage; do not assume the
previous eight-hour request is enough for every shared dataset. No full job
command until manifest/adapters and reviewed pilot evidence are ready.

## Tutor report after evaluation

Provide a concise Markdown report with method diagram/description, exact dataset
coverage and exclusions, per-sequence accuracy/time/memory, short/long summaries,
head-matched timing, failure cases and limitations. Distinguish verified results
from the three-frame hypothesis. Explain anchor age, refresh scale and object
confidence differences. Ask whether broader viewpoint coverage or a better
geometric connection is the more useful next direction. Do not send the report
to the tutor without explicit instruction.

## Camera gate after reviewed26472

26472 execution/fidelity and timing artifacts pass review; its accuracy does not
meet the all-three-metrics improvement criterion. Retain all three arms; do not
promote three-image as an overall accuracy winner. See FINDINGS Artifact audit26472.

Entry: `bash tools/kvt_tum.sbatch camera-history`, two-hour allocation. Fixed full
freiburg3_long_office_household,2585 chronological frames, resize308; no sampling,
new policy thresholds or dataset expansion. kvt_camera_history.py reuses the
hash-pinned26127 office inputs/context and25680 original trajectory/evaluation
archive from kvt_tum_out. All three source archives and Pi3 checkpoint are checked
before inference. No fresh staging/download, SAM tracking or environment change.
Native saved-pixel replay uses stock camera admission and must match25680 within
rtol/atol1e-4. The original object replay remains the default for existing callers.

Fresh processes run native, two-image no retirement, three-image no retirement.
The fixed ORB25/max200 detector is logged in bounded arms; its decisions cannot
retire maps. Both retain boundaries[0,N], no events, capped banks, equal detector
records. All arms must have finite full trajectories and identical historical
RGB/GT indices, timestamps, reference poses and adjacent RPE pairs. Benchmark
input shape, GPU, software and precision must match. Each arm saves separated
forward/head timings; native camera queries already use pose-only heads.

Existing TUM, detector, handoff, reanchor and history contract suites run first.
Deadline clips to Slurm EndTime; subprocess/outer timeouts reserve30 minutes for
packaging. Comparison records an active arm before launching, completed archive
hashes and quality ratios. Runtime/fidelity failure exits nonzero; quality failure
is recorded without changing the three-arm matrix. Expected markers: three
CAMERA HISTORY ARM lines, CAMERA HISTORY COMPLETE 3 2585, CAMERA HISTORY JOB OK.
Then verify Slurm terminal status and six archives before transfer/review:
`tum_JOB_camera_history_{native,no_retirement,three_frame}.tar`,
`tum_JOB_camera_history_comparison.tar`, `tum_JOB_context.tar`, `tum_JOB_all_runs.tar`.
The wrapper preserves partial runs/context on ordinary failure; sources remain
immutable and need not be duplicated. A scheduler hard kill can prevent packaging.

Local preparation checks: Python3.10 AST, Bash syntax, diff whitespace. No local
project imports/runtime tests. CAMP contract/fidelity/runtime checks are pending.
Publish only kvt_camera_history.py, kvt_segment_arctic.py, kvt_reanchor_maps.py,
kvt_tum.sbatch and this protocol. FINDINGS and docs remain local canonical
records (gitignored). Model remains269a943.

User submits from CAMP head in /mnt/projects/gr/3DRecon/layer_good using the
allocation-wrapped pull/run procedure, selected24g/muenchen/GPU1, two hours.
Do not pull on head. Review camera artifacts before all-sequence evaluation;
manifest/adapters for the remaining table entries and the tutor report remain
pending. No full-dataset job or automatic successor is prepared by this gate.

Outcome, 2026-10-10: job26473 failed before inference because the25680 reference
archive was missing from kvt_tum_out; after it was restored, job26476 at22afc3a/
269a943 completed. Contracts, native fidelity, pairs, detector and no-retirement
checks pass, and both bounded arms fail the native quality gate. Results and the
refresh-step analysis are in FINDINGS (Artifact audit26476, Handoff audit). Before
any rerun of this gate, confirm the25680 reference is present in kvt_tum_out.
Next: a read-only diagnosis of the refresh connection, not a rerun.

## Pinned-refresh camera gate after reviewed26476

Diagnosis from saved26476 files (no new run): without retirement every refresh
rebuilds [0, previous, latest] and Pi3 re-solves the latest camera relative to
frame0; only scale is restored. When frame0 faces away (office frames749-1449)
that solution flips: three-image refresh pairs carry99% of squared translation
RPE, with isolated steps of metres and several degrees. User-selected fix:
continuity pin. After each rebuild the new solution is rigidly moved so its pose
of the refresh frame equals the outgoing bank's pose of that frame; anchor-based
rebuild_scale is kept. No thresholds. Each rebuild logs pin_rotation_deg and
pin_position_step (the two banks' disagreement), which is the runtime diagnostic.
Pinning is single-map only (asserted); it removes the anchor's absolute
correction, so slow drift can accumulate. That is the tradeoff under test.

Entry: `bash tools/kvt_tum.sbatch camera-history pinned`, two-hour allocation,
same inputs, contracts, deadline and packaging as the camera gate above. Fresh
processes: native (25680 fidelity), three_frame (unpinned control; must reproduce
the reviewed tum_26476_camera_history_three_frame.tar trajectory within rtol/atol
1e-4, hash-pinned in kvt_tum_out), three_frame_pinned. Same pair/detector/no-
retirement/environment checks. comparison.json adds refresh_pairs for every arm,
unpinned_ratios/improves_unpinned and the 51 per-rebuild pins for the pinned arm.
Expected markers: three CAMERA HISTORY ARM lines, CAMERA HISTORY COMPLETE pinned
3 2585, CAMERA HISTORY JOB OK. Archives: tum_JOB_camera_history_{native,
three_frame,three_frame_pinned}.tar, _camera_history_comparison.tar, context,
all_runs. Stage gate (default) keeps the original 26476 matrix unchanged.

Review: the pin works mechanically if refresh_pairs share drops toward native's
and pins are large exactly where unpinned steps were large. Quality is judged by
the existing native5% gate and versus unpinned on all three translation metrics;
report ATE drift plainly. Object ketchup and two-image pinning follow only after
this review. No full-dataset job is prepared by this gate.

Outcome, 2026-10-10: 26480 failed before inference (tar member passed to np.load;
fixed in37f362d). 26481 completed: control fidelity and pin contracts pass; the
pin improves all three translation metrics over unpinned three-image but fails
the native5% gate. Remaining error is mainly scale drift between rebuilds
(FINDINGS Artifact audit26481). No follow-up gate is prepared yet.

## Scale-continuity and pin-generality gates after reviewed26481

User asked to try both follow-ups. (a) Scale continuity: `pin_scale` (requires the
rigid pin) replaces the anchor-pointmap rebuild_scale. Before each rebuild one
dense read-only query of the refresh frame on the outgoing bank gives its points
in its own camera; rebuild_scale is the median ratio of those depths to the new
rebuild's depths of the same frame (both maps' confidences at or above median).
No new thresholds. (b) Generality: the current pin on two-image banks and on
the ketchup object sequence. No other policy changes.

Camera: `bash tools/kvt_tum.sbatch camera-history scale`, two hours. Arms native,
three_frame_pinned (control; must reproduce the reviewed
tum_26481_camera_history_three_frame_pinned.tar trajectory within 1e-4),
three_frame_pinned_scale, no_retirement_pinned (two-image). Markers: four CAMERA
HISTORY ARM lines, CAMERA HISTORY COMPLETE scale 4 2585, CAMERA HISTORY JOB OK.
The unpinned two-image control is the reviewed26476 arm, compared offline.

Object: `bash tools/kvt_tum.sbatch object-history pinned`, two hours. Full saved
ketchup; arms native (26458 fidelity), three_frame (control; must reproduce the
reviewed tum_26472_object_history_r0_three_frame.tar within 1e-4),
three_frame_pinned, three_frame_pinned_scale. Markers: four OBJECT HISTORY ARM
lines, OBJECT HISTORY COMPLETE pinned 4, OBJECT HISTORY JOB OK.

Both: forward profiling and head benchmarks as before; per-rebuild pins and
rebuild_scale in comparison.json; compared_ratios against the arm each variant
modifies (pinned vs unpinned, scale vs rigid pin). Same contracts, detector,
pair and no-retirement checks. Quality uses native5% and the compared arm on all
three translation metrics. Review both before any further dataset expansion.

## Shared-keyframe scale camera gate after reviewed26487/26488

26487 showed refresh-frame scale (pin_scale) reaches native RPE/p99 but rises about
2.5% per refresh relative to the anchor scale (39 of 50 steps), inflating ATE.
Hypothesis: a single-image query's depths and a rebuild member's depths are not
like-for-like. Test: `shared_scale` (three-image, requires the rigid pin, excludes
pin_scale) takes each rebuild's scale from the previous keyframe, which is a
rebuild member of both the outgoing and the new bank; depths in its own camera,
both maps' confidences at or above median, scale compounded from the outgoing
bank's. The first rebuild has no shared keyframe and keeps the anchor scale. No
extra forward, no thresholds.

Entry: `bash tools/kvt_tum.sbatch camera-history shared`, two hours. Arms native,
three_frame_pinned (control; must reproduce the reviewed
tum_26481_camera_history_three_frame_pinned.tar within 1e-4),
three_frame_pinned_shared. Markers: three CAMERA HISTORY ARM lines, CAMERA HISTORY
COMPLETE shared 3 2585, CAMERA HISTORY JOB OK. Review per-refresh rebuild_scale
against the anchor scale (bias), windowed Sim(3) scales and the native5% gate.
Object is not rerun: its deficit is between refreshes (FINDINGS audit26487/26488).

## Fused-scale camera gate and object novelty-refresh gate after reviewed26493

User said go ahead on both candidates after26493. (1) Camera: `fuse_scale` takes the
geometric mean of the shared-keyframe chain and the absolute anchor-pointmap scale
at each three-image refresh (26493: chain unbiased but random-walks; anchor
bounded). Equal weight is the one new parameter, chosen as the simplest default,
not tuned. Entry `bash tools/kvt_tum.sbatch camera-history fused`: native,
three_frame_pinned (control; reproduce26481 within1e-4), three_frame_pinned_fused.
Markers: CAMERA HISTORY COMPLETE fused 3 2585, CAMERA HISTORY JOB OK. Both scale
sources are logged per refresh.

(2) Object: ketchup error is bursts when the object turns away from all three bank
images (FINDINGS ketchup diagnosis). `novelty_refresh` adds a rebuild after frame49
whenever native's own object keyframe rule (main.py check_if_keyframe, unchanged
10-degree elevation/azimuth thresholds about the masked object centre) fires against
the current bank poses; same three-image cap, rigid pin, anchor scale. Masks are the
saved hash-checked model_masks. Entry `bash tools/kvt_tum.sbatch object-history
novelty`: native, three_frame_pinned (control; reproduce
tum_26488_object_history_r0_three_frame_pinned.tar within1e-4),
three_frame_pinned_novelty. Markers: OBJECT HISTORY COMPLETE novelty 3, OBJECT HISTORY
JOB OK. comparison.json records novelty_refreshes and per-rebuild triggers; extra
rebuilds cost time, so report speed with accuracy.

Outcome, 2026-10-10: 26497 failed the fused contract (log dict reassigned; fixed in
fork fd90576), 26498 cancelled. 26499 (camera fused) and 26500 (object novelty) ran
at parent 9ba9929/fork fd90576 with both controls reproduced byte-for-byte. Fused
scale stays within a few percent of the anchor scale and matches the anchor-scale pin
(ATE 0.5293 vs 0.5250), so continuity scale is closed. Novelty refresh fired 112 extra
rebuilds on ketchup, 15 pins exceeded 90 degrees, and the Sim(3) alignment scale
collapsed (0.066 vs 0.328). Object translation RPE is measured after that alignment,
so its drop is not evidence of better tracking. Numbers are in FINDINGS (gate and
audit 26499/26500).

## Retrieval-bank gate after reviewed 26499/26500

Method review (user: stop the continuity-tuning loop). Every gate since 26476 changed
only how one re-solve's gauge is carried to the next through one or two shared
images; native's frozen prefix banks (26183: first 750 frames, 16 keyframes, ATE
0.0596 over all 2585 frames) show that spatial coverage with fixed keyframe poses
is what tracks the loop. User approved the redesign with a four-image bank.

`kv_tracker.retrieval_bank.RetrievalBank` (runner `--retrieval-bank 4`, requires
`--disable-retirement`): every admitted keyframe keeps image, global pose and
pointmap on the CPU, never re-solved. Admission is native's rule: every 50 frames
for the camera (uncapped), and for objects main.py check_if_keyframe against all
stored keyframes. The GPU bank is the newest keyframe plus the three stored
keyframes with the best coverage: the fraction of their confident points inside
the current frustum (scene), or the closest viewing direction about the object
centre (object). Each rebuild is registered to the global frame by one robust
Sim(3) over all shared members' same-pixel pointmaps (bridge() on stacked maps);
its validation checks are logged, not used to reject. No frame-0 anchor, no pin,
no rebuild_scale.

Entries, each from layer_good:
`bash tools/kvt_tum.sbatch camera-history retrieval`: native, three_frame_pinned
(control; reproduce 26481 within 1e-4), retrieval_bank. Markers: CAMERA HISTORY
COMPLETE retrieval 3 2585, CAMERA HISTORY JOB OK.
`bash tools/kvt_tum.sbatch object-history retrieval`: native, three_frame_pinned
(control; reproduce 26488 within 1e-4), retrieval_bank. Markers: OBJECT HISTORY
COMPLETE retrieval 3, OBJECT HISTORY JOB OK.
comparison.json records per-admission bank ids, coverage scores, fit scale and
validation. Success: retrieval improves ATE over the pinned control on both
sequences with RPE/p99 not worse; the target remains native within 5%. Report
speed and peak memory with accuracy. Object RPE is measured after Sim(3) alignment,
so read it together with alignment_scale (26500 audit).

Outcome, 2026-10-10: 26503 (camera) and 26504 (ketchup) ran at parent 86044ce/fork
be99bfb with both controls reproduced byte-for-byte. The retrieval bank is worse
than the pinned control on camera (ATE 1.0389) and gains no rotation accuracy on
ketchup (rotation RPE 29.1 degrees, alignment scale 0.191). Registrations of
independent re-solves fail at low-overlap transitions (office 749 onwards), and
fixed stored poses keep those errors. Rejected; numbers in FINDINGS (gates and
audit 26503/26504).

## Overlap-connection gate on the user's segmentation design (after the scope correction)

Gates 26471-26504 ran without segmentation (FINDINGS scope correction). This gate
returns to the user's design: live ORB cuts at 25 inliers (26159 policy, no maximum
segment length), the old map retired at each cut, two-image segment banks. The
read-only 26159 analysis puts about 90% of the extra squared ATE in consistency
between segments; each connection takes its scale from one pointmap comparison.

Change (ReanchorMaps overlap_stride=5, runner --overlap-bridge): the old map and its
KV are kept, swapped out of the model, until the new map's first rebuild at b+49
(at most four images), and the old map also locates frames b+5, ..., b+45. The
connection keeps rotation and position pinned at b and takes scale by least squares
from the ten shared camera displacements from b in both maps. The point-fit scale is
still computed and logged (point_fit_scale, overlap_scale). Old KV/images are
released at b+49. Stride 5 is the single new parameter, chosen for about 10% extra
queries, not tuned.

Entry from layer_good: `bash tools/kvt_tum.sbatch camera-history overlap`: native,
segmented (control: cuts must equal 26159's; trajectory difference from 26159 is
printed and recorded, not asserted, because the tracker changed since 26159), and
segmented_overlap. Needs tum_26159_segment_reanchor_inliers25.tar (c6617206...)
in kvt_tum_out. Markers: SEGMENTED CONTROL max abs difference from 26159, CAMERA
HISTORY COMPLETE overlap 3 2585, CAMERA HISTORY JOB OK. Success: segmented_overlap
lowers ATE versus the in-job segmented control without worse RPE/p99; native within
5% remains the target. Report extra query time and peak memory.
