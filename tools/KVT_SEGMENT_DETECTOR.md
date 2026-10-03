# Online segmentation points: anchor-overlap diagnostic

## Optimization after 26159: sequential anchor tracks (flow pilot)

For the user-selected overnight batch with independent graph and query-half
experiments, use [KVT_QUERY_ACCELERATION.md](KVT_QUERY_ACCELERATION.md):
`segment-detector overnight`. This still reviews flow boundaries before tracking;
the independent query runs keep native cache/keyframes and use no segmentation.

26159 does not establish that segmentation cannot improve. It tested raw ORB
match counts to a distant anchor; changing that count alone left the signal
unchanged. Inspected boundary169 still shows much of the anchor table/objects;
boundary303 has substantial shared content but the current-grid coverage triggers.
These examples motivate a different overlap measurement, not proof that their
cuts cause the observed trajectory error. Numerical results remain in FINDINGS.

`segment-detector flow` runs ORB25 and one new causal detector on identical pixels.
The new detector seeds up to20 Shi–Tomasi corners in each4x4 anchor cell, then
tracks their original identities through consecutive images with pyramidal
Lucas–Kanade flow (21px window,3 pyramid levels). Forward/backward status,1px
round-trip consistency and image bounds reject tracks. No new points replenish
the anchor support. Low support is surviving fraction<25% OR surviving anchor-cell
fraction<50%; normalize to actual seeded features/occupied cells. Preserve the
50-frame minimum age and five-frame persistence; cut at the current frame and
reseed. Fewer than40 initial corners is unknown texture, not a validated segment.
These are uncalibrated starting constants, not fitted to human boundaries or GT.
API: [OpenCV optical flow documentation](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html).

Hypothesis: temporal correspondence tolerates gradual appearance/viewpoint change
better than distant descriptor matching, with cheaper per-frame processing. Flow
can drift; blur/occlusion can irreversibly lose support and fragment the sequence.
It estimates surviving visible content, not guaranteed Pi3 accuracy. Runtime and
boundary quality are pending; do not call this an accuracy/speed improvement yet.

Contracts add gradual-motion tracking, prefix causality, no support replenishment,
minimum age/persistence and current-frame reset. Outputs include
`tum_<JOB>_segment_detector_inliers25.tar`, `tum_<JOB>_segment_detector_flow.tar`,
and context `segment_detector_flow_comparison.json`; full per-frame fractions,
anchor-cell coverage and boundary source pixels/matches are preserved.

From layer_good on head, after publication:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector flow"
O=../kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

Gate: contracts pass, both2585-frame passes complete, `SEGMENT FLOW PILOT COMPLETE`
and JOB OK; then review fragmentation, boundary pairs, lost-support intervals and
cost. No full tracking job is queued behind this unreviewed pilot. After review,
integrate the selected live detector into the existing matched reanchor comparison
and include the recorded human cuts0/275/575/700/975/1400 as a reference using the
same tracker, inputs and evaluation scope. Separate first1000 and full2585 metrics;
historical frozen-map/posthoc results are not matched live tracking baselines.
Retain the ORB25 reference and native. No further threshold sweep selected.

User-selected scope, 2026-10-03: the full office sequence (2585 frames), causal
decisions from current and past images only. First identify segmentation points;
The user subsequently authorized all five segmentation+reanchor tests in the
same batch (see `compare` below); the initial detector-only stages remain available.

## Mechanism

Keep ORB features from the current segment's first frame. Extract 1000 ORB
features per current image, match anchor to current descriptors with Hamming
distance and a 0.75 nearest/second-nearest ratio, then enforce unique current
features. Verify matches with both homography and fundamental-matrix RANSAC
(2-pixel threshold); use the model with more inliers. This accommodates planar
or rotation-dominated views and general epipolar motion within one detector.
Matching API: [OpenCV documentation](https://docs.opencv.org/4.12.0/d3/da1/classcv_1_1BFMatcher.html).

Low overlap means fewer than 20 verified inliers OR fewer than four occupied
cells of the current image's 4x4 grid. After the anchor is 50 frames old, five
consecutive low-overlap frames trigger a boundary at the **current** frame.
The triggering image becomes the next anchor. No backdating to the start of the
five-frame interval, future frames, manual boundaries or GT enters decisions.
Frames with fewer than 40 features in either image have unknown overlap and
clear the persistence count. A textureless anchor therefore cannot yield a
reliable boundary; its unknown interval must be reviewed, not silently treated
as successful detection.

All constants are initial engineering choices, not calibrated thresholds.
Image matches estimate visible overlap, not Pi3
reliability. Repeated textures, blur and geometric degeneracy can mislead the
detector; loss of overlap may also be too late for a reliable handoff. Reviewing
these limitations is part of the first gate. The support-loss interpretation
near 1850 is a hypothesis recorded in FINDINGS, not a training label.

## First experiment and evidence gate

Replay the archived exact model RGB frames **in chronological order** through
the online detector. Reading a stored video does not make the decisions offline:
`step` gets one current image, retains only anchor features and past counters,
and has no access to a future image. This diagnostic does not execute Pi3,
change its cache or score a trajectory. It measures candidate boundaries and
CPU detector cost, not tracking speed or accuracy.

Contracts on CAMP: identical-view support, unknown texture, five-frame
persistence with interruption, current-frame anchor reset, minimum segment age,
and identical prefix decisions when a future suffix is added. Full run: all
2585 pixels hash-match reviewed archive 26127; every decision's anchor precedes
or equals its current frame. OpenCV seed 0 and one thread are recorded.

Save every frame's timestamp, anchor, feature/match/inlier counts, cell coverage,
status, persistence and detector time in `scores.jsonl`. Every boundary preserves
both source PNGs and matched coordinates/inlier mask in NPZ. `summary.json`
records boundaries, unknown frames, median/p99 time and execution status.
The original hash-checked archive remains the exact source for all other frames.

Before integrating resets, transfer and review: candidate boundary image pairs,
score traces around transitions (including late-sequence failure and revisit
regions from FINDINGS), fragmentation, unknown intervals and cost. Manual
boundaries are historical comparison points, not ground-truth segmentation.
Do not declare detection quality from JOB OK or tune thresholds to manual cuts.
Then decide whether the signal is useful and sufficiently early for a handoff.
The eventual continuous target remains ATE, translation RPE and p99 within 5%
of matched full-sequence native; this diagnostic does not advance that gate.

## CAMP procedure

Requires `kvt_tum_out/tum_26127_oracle_maps_inputs.tar` with recorded SHA256;
no new downloads, checkpoints or container changes. Existing `kvt_tum.sbatch`
supplies the environment and archives source/revisions. Publish first and refresh
inside an allocation, never on head. Slurm snapshots the wrapper at submission:
its mode validation runs **before** its in-job pull. A stale wrapper therefore
cannot bootstrap a newly added mode (job 26156 exited with `Unknown mode`).
Use the established allocation-wrapped pull from [KVT_TUM.md](KVT_TUM.md)
(job 25660), which pulls and runs the refreshed wrapper in the same allocation.
This also avoids the documented interactive srun pthread_create failure. From
`/mnt/projects/gr/3DRecon/layer_good` on head:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector"
O=../kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

The wrapper's SBATCH directives are comments when executed by bash; the command
line above supplies the resource and log settings, as in the established procedure.

Expected: three contracts pass, zero or more `SEGMENT BOUNDARY` lines, then
`SEGMENT DETECTOR COMPLETE: 2585 frames` and `JOB OK`. Zero cuts still means
execution completed; detection quality must be reviewed. Output:
`kvt_tum_out/tum_<JOB>_segment_detector.tar`, plus context/all_runs tars and
`kvt_tum_slurm-<JOB>.log`. Stop at the first failure and return its context.
The initial detector-only stage does not queue a successor tracking job.

## User-selected four additional thresholds (2026-10-03)

Compare minimum verified inliers **10, 15, 20, 25, 30**: four new thresholds plus
the existing 20-match reference. Only the inlier threshold changes. The four-cell
coverage rule, minimum features, ORB/RANSAC settings, five-frame persistence and
minimum anchor age of 50 are fixed. This is an exploratory same-sequence comparison,
not held-out threshold validation. Higher thresholds require more matches to keep
a segment, but resulting anchors differ, so boundary counts need not be monotonic.

Every arm reruns the causal detector with an independent anchor/counter and RNG
seed 0. Do not apply four thresholds to the original run's scores: those scores
used different anchors after its first cut. The 20-match arm is rerun so it has
the same code/environment/provenance as the new arms; no baseline tracking rerun.

From layer_good on head, use the established wrapped pull/run:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector sweep"
O=../kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

Expected: four contracts pass, five `SEGMENT DETECTOR COMPLETE` lines, then
`SEGMENT DETECTOR SWEEP COMPLETE: 5 policies` and `JOB OK`. Save one archive per
arm: `tum_<JOB>_segment_detector_inliers<THRESHOLD>.tar`. Each summary also reports
segment lengths and cuts at the earliest possible age (54 frames); the context
contains `segment_detector_sweep.json` with all five summaries. Timing is measured
sequentially per arm and includes feature extraction, not IO or Pi3 inference.

Review execution, candidate boundary evidence, unknown intervals and cost before
tracking integration. Then test all five segmentations with the same reanchor
mechanism, same pixels/seed/checkpoint and one full-sequence alignment, reporting
ATE, translation RPE, translation p99, seams and total detector+tracking cost.
Preserve the continuous within-5%-native target for all three headline metrics.
If several arms pass, prefer the fastest measured end-to-end run; if none pass,
report tradeoffs instead of declaring the fewest-cuts arm the winner. A shorter
final segment may not reach the b+49 connection; that must be handled explicitly
before tracking integration, not silently dropped. The detector-only `sweep` stage
does not dispatch tracking; use the user-authorized `compare` stage below.

## Same batch: five segmentations + reanchor versus fresh native

User-authorized 2026-10-03. `segment-detector compare` replaces the prior
detector-only `sweep` submission for this experiment. No intermediate manual
boundary review is required between the five segmentation passes and tracking.
Contracts still run first; any execution/provenance/fidelity failure stops the job.

One allocation runs the five detector passes, then a fresh full-office native
MapHandoff run (existing `kvt_handoff_pilot.py --run` native path), then five fresh
processes with the same reanchor+refresh mechanism. All six tracking runs cover
2585 frames, resize308, seed0, same checkpoint, device and environment. Native
retains its original anchor and appends every50 frames to cap20. Its ATE/RPE/p99
must reproduce reviewed25680 to absolute1e-4m before the five reanchor runs start.
This native rerun makes timing a same-allocation observation rather than a
comparison between archived and current wall times.

Each reanchor arm runs its detector **live**, independently, and asserts every
non-timing decision against that threshold's first pass. Only live current/past
decisions enter the tracker; stored expected decisions are assertions. Old-map
images/KV are retired at each cut; a scheduled old-bank rebuild is skipped there
so transition camera and dense geometry use the same cache. Each new map retains
anchor+latest, refreshes every50 frames, restores first-rebuild scale and commits
the pose-anchored connection at local49.

At EOF only, an unfinished final segment rebuilds [anchor,last available frame]
and commits with its actual shorter delay. A one-frame tail uses [anchor,anchor].
No future image or backdated cut is used, and all saved local poses are kept.
CAMP contracts cover tail lengths1/30/49 and suppression of retiring-bank rebuilds.
The EOF pair supplies anchor geometry for the scale fit; it does not replace the
tail's previously emitted bootstrap-cache local poses.

Scoring uses one full-trajectory Sim(3), the same2583 GT-associated frames and2582
RPE pairs, with exact association/reference-array equality against fresh native.
The comparison records ATE, translation RPE/p99, seam errors, observed speedup,
cache/peak GPU memory, cut count and each ≤1.05×fresh-native verdict. Individual
reanchor summaries also retain the historical-native comparison; the combined
`comparison.json` is authoritative for the **fresh-native** verdict.

Tracking loop timing includes pixel loading/hash checks, live detector, geometry,
bridge fitting and logging; excludes model load, staging and evaluation. The
native helper uses the same instrumented loop timing scope but has its existing
logging/bookkeeping. No claim of statistical speedup from one run; report measured
FPS and scope. Bootstrap/query/rebuild/update timings overlap: never sum rebuild
and update_total as independent costs.

From layer_good on head, use the established allocation-wrapped pull/run:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector compare"
O=../kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

Requires both reviewed26127 input and context archives in kvt_tum_out; their hashes
and checkpoint identity are checked before native tracking. No new downloads or
container changes. The wrapper archives all new drivers/contracts with revisions.
Additional outputs: `tum_<JOB>_segment_native.tar`, five
`tum_<JOB>_segment_reanchor_inliers<THRESHOLD>.tar`, and
`tum_<JOB>_segment_comparison.tar`, plus detector/context/all_runs archives.
The comparison is checkpointed after each arm so partial failure retains evidence.
Expected final marker: `SEGMENT TRACKING COMPARISON COMPLETE: native + 5 policies,
2585 frames each`, then JOB OK. JOB OK requires all six tracking executions and
comparisons, not an accuracy pass. Fastest observed arm meeting all three5% gates
is recorded as best_passing_threshold; if none pass it is null and all tradeoffs
remain visible. Review artifacts after the batch before claiming success.


## Reviewed flow + matched human reanchoring comparison

After26160 artifact review, stage flow-compare tests three full2585-frame runs:
fresh native, existing human boundaries0/275/575/700/975/1400/end, and current
causal flow boundaries. Flow and human arms use exactly the same ReanchorMaps
refresh/bridge/retirement mechanism. Human boundaries are an offline reference,
not an online method. No new threshold grid or bridge-policy change. This isolates
boundary selection from reanchoring behavior; long segments remain a known risk.

The detector pass uses the existing flow diagnostic, including ORB25 for boundary
reference only. Tracking reruns flow live and asserts all non-timing decisions
against first-pass scores. No saved/future decision drives the tracker. Native
reproduction, source/checkpoint hashes, GT/reference associations and environment
gates match the earlier compare stage. Timing includes live detection for flow;
human arm has no detector cost. Report all metrics, seams, bridge validation,
within-segment errors, memory and FPS. A faster/less fragmented detector alone
does not establish better tracking.

From the CAMP checkout on head after publication:

```bash
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector flow-compare"
O=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

Expected SEGMENT COMPARE manual/flow and SEGMENT TRACKING COMPARISON COMPLETE:
native +2 policies,2585 frames each, then JOB OK. Verdict can still fail. Outputs
segment_native, segment_reanchor_manual, segment_reanchor_flow, segment_comparison
plus flow/ORB25 detector/context/all_runs archives. Inspect results before altering
flow thresholds or bridge acceptance. Independent query/LoopCD methods absent.


## Scope correction: human annotations stop at1000

User correction2026-10-03: only boundaries through frame1000 were identified by
human inspection. The legacy full list0/275/575/700/975/1400/2585 is not a valid
full-sequence human reference. Withdraw that interpretation of26173; preserve
its measured fixed-schedule execution as an artifact. A fair future human test
uses first1000 frames with cuts275/575/700/975 and end1000, matching native and
automatic evaluation on those same frames. No human arm enters the search below.

## Bounded automatic calibration:50% margin or30 trials

User-authorized2026-10-03. Stage optimize runs a fresh full2585-frame native
reference, then at most30 segmentation/reanchor candidates. It stops immediately
after the first arm with ATE, translation RPE and translation p99 each<=1.5 times
fresh native, or after30 candidate trials. This supersedes the5% requirement for
this calibration job only. Individual reanchor summaries still show historical
5% diagnostics; segment_search/comparison.json is authoritative for the requested
50% target. JOB OK also occurs after trial-limit exhaustion; inspect stop_reason.

Candidate schedule, fixed before running: profiles ORB minimum inliers25/30/35
and flow survivor fraction0.35/0.50/0.65, combined with maximum segment ages
none/100/150/75/200, in age-major order. Other detector settings and reanchor
bridge/refresh behavior remain fixed. The first candidate reproduces ORB25
without age cap, the best prior ORB arm. Ages>=75 preserve the existing50-frame
bootstrap/49-frame connection window. An age cap cuts at the current frame,
even when texture is unknown; overlap decisions retain their persistence rule.
Shorter age is motivated by26173 non-seam error and long-segment failures, not
by a claim that particular caps are optimal. No graph/LoopCD combination.

GT is used only by the offline evaluator to select/stop across completed trials.
The tracker sees only pixels, current frame index, fixed policy and past state.
This is same-sequence calibration, not held-out validation or deployment-ready
threshold evidence. No guaranteed pass: search reports best worst-metric ratio
when all30 fail. One ordered timing observation per arm, includes live detector
and tracking. Accuracy misses continue; execution/provenance/nonfinite errors
fail loudly rather than consuming a trial as a quality miss. No automatic reruns.

Existing detector/handoff/reanchor contracts run first, plus search-controller
contracts(first passing trial/all-three gate, bound exhaustion, exception
propagation) and new current-frame age-cap reset contract. Inputs/checkpoint
hashes, historical native reproduction and GT/reference associations are checked.
Each candidate is a fresh process. Persist each trial archive and checkpoint
summary after every completed candidate. Up to31 tracking runs including native.

From CAMP head:

```bash
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector optimize"
O=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

Expected SEGMENT SEARCH TRIAL/RESULT for each candidate, SEGMENT SEARCH COMPLETE
with target met true/false, then JOB OK. Outputs tum_<JOB>_search_native.tar,
_search_trial_<NN>.tar and _segment_search.tar plus context/all_runs. Source code
and tested policies recorded; best and passing trials remain independently
reviewable. Runtime verification pending. Read completed results before further
optimization. No automatic held-out or combination stage behind this search.
