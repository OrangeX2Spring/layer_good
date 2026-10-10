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
improve geometry when the anchor becomes dissimilar; accuracy is untested.
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
evaluation. Camera integration and dataset adapters are separate pending work;
do not claim all-dataset readiness from a ketchup marker.

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
