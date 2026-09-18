# KV-Tracker: long TUM sequences and keyframe efficiency

Prepared 2026-09-18. **Locally syntax-checked only. Runtime gates have not run.**
Entry point: `tools/kvt_tum.sbatch`. Existing `localhost/kvt`, no installations,
downloads, model changes, or new weights. The user executes all cluster operations.

## Fixed exploratory protocol

All RGB frames in order, offset 0, scene/camera mode, `resize_dim=308`, seed 0.
The upstream TUM loader and resizing are retained, including its unused SAM model
initialization. The tracker, attention, reconstruction and cache rebuilds are unchanged.
Every scene patch participates in selection. Four semantic families, four thresholds
each, **16 semantic configurations per sequence**:

| Family | Feature | Score | Insertion thresholds |
|---|---|---|---|
| Semantic reference | Pi3 `decoder[0]` | `1 - max cosine` between mean-pooled frame features | 0.01, 0.025, 0.05, 0.10 |
| Encoder feature | Pi3 encoder `x_norm_patchtokens` | Same pooled cosine | 0.01, 0.025, 0.05, 0.10 |
| Patch coverage | Pi3 `decoder[0]` | Fraction of current patches whose best cached-patch cosine is <0.95 | 0.05, 0.10, 0.20, 0.40 |
| Patch-set Chamfer | Pi3 `decoder[0]` | Minimum over cached frames of symmetric mean squared L2 patch-set distance | 0.025, 0.05, 0.10, 0.20 |

Selection uses strict `score > threshold`. Patch descriptors are L2-normalized
float32; all patches are compared, in chunks without subsampling. Coverage matches
against the union of all retained patches. Chamfer matches one retained frame at a
time, averages both directed mean nearest-patch distances, then takes the minimum
over frames. For normalized descriptors squared L2 equals `2 - 2*cosine`; the
Chamfer variant changes the set-level comparison, not this underlying equivalence.

Both features are 1024-dimensional and frame-local. The first cross-frame decoder
block is `decoder[1]`; deeper/contextual features are not used. A read-only hook
checks the inspected Pi3 source SHA-256 (`cbcf68b3...2979`), excludes decoder special
tokens, and stores arrival-time features. Rebuilds never overwrite those features.
No extra model forward or encoder is added. Retained descriptor bytes and selector
time are accounted for separately from transformer K/V.

Original conditions:

1. Stock interval 50, cap 20.
2. Interval 50, the same higher cap K used by semantic conditions.
3. Sequence-spread periodic controls at adaptively chosen counts. After the semantic
   sweep, take its lowest ATE on that sequence as an **exploratory target**. Start
   at its keyframe count (minimum 2), double toward K until an ATE match or K, then
   try one lower/midpoint budget if matched. A match is ATE <= 1.05 times the target;
   RPE is reported separately, not used as an undisclosed second matching gate.

For N frames and target B, the periodic interval is `max(2, N // (B-1))`.
This distributes B insertions over the full clip when N is sufficiently large.
Known clip length configures this benchmark control; this is not a duration-agnostic
online selector. No future RGB, features, or GT pose enters an insertion decision.

**Counter phase:** upstream starts its internal counter at 2 for source frame 1.
Interval I inserts at source indices `I-1, 2I-1, ...`; source frame 0 is bootstrap.
Stock cap 20 therefore stops at frame **949**, not 950 or 1,000. Extended and spread
conditions preserve that phase. Stock condition uses the unchanged upstream boolean.
Only extended/spread selectors override the upstream hard cap through the existing
callback. For N=2,600, interval 50 without the cap consumes 53 unique frames; exact
counts come from the extracted RGB inventory, not estimated duration × frame rate.

Report the smallest **tested** matching count across all three original condition
types, semantic count, their ratio, ATE/RPE, actual time and memory. Accuracy is not
assumed monotonic in count. A missing match means none was found in tested settings
within the feasible cap. The best semantic setting is chosen on the same sequence:
this is exploratory oracle selection, not a held-out optimum or generalization claim.
Results for all thresholds remain in the report. No random controls or eviction.

## Sequences and ground truth

- `freiburg3_long_office_household`: first/main accuracy comparison, ~87 s,
  essentially complete GT.
- `freiburg2_large_with_loop`: ~173 s, only ~40.5 s GT; the middle is missing.
- `freiburg2_large_no_loop`: ~112 s, only ~21.4 s GT.

Official metadata: <https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download>.
The script inventories actual images, timestamps and GT coverage from the CAMP ZIPs.
All three are tracked over their **full** RGB streams. Accuracy associates the nearest
GT within **20 ms** only. It applies one Sim(3) over valid matched poses; it never
extends a boundary pose across missing GT. RPE uses consecutive valid source frames
with timestamp separation <=100 ms, never pairs spanning a GT hole. Rotation RPE
is geodesic degrees; translation RPE and ATE are RMSE metres. Every run saves its
matched indices, time differences, aligned/reference poses, and valid RPE pairs.
Partial-GT ATE is not full-route accuracy; the two industrial sequences chiefly
extend the saturation/resource evidence.

## Gates and memory

The batch stops at the first unexpected failure:

1. CPU contract tests in the existing container: timestamp gaps, counter phase,
   patch comparisons/chunking, caps, feature interfaces, threshold grid and prefix gate.
  A tiny synthetic visualization also checks PNG/PLY generation and MP4 encoding
   plus full decoding before the first model run. Feature tests check arrival-vector
   export, future-reference masking and reconstruction of native patch scores from
   the saved maps (including Chamfer's reverse contribution).
2. Five full original TUM runs at 308 reproduce the recorded ATEs within 0.002 m:
   xyz .021, rpy .045, desk2 .083, desk .059, room .361. This historical gate uses
   the original GT association; corrected timestamp-valid metrics are saved separately.
3. Unmodified original versus instrumented-original 128-frame xyz prefix: same
   poses (`atol=rtol=1e-4`) and identical keyframe indices.
4. Memory preflight: force insertion on each of the first K frames, then perform
   eight further queries with the full cache. Use coverage descriptors to include
   patch-bank storage and all actual tracker rebuilds. Try K=64, then 48 or 32 only
   if the previous choice explicitly OOMs or lacks 2 GiB estimated headroom. Test
   the accepted candidate on all three scenes. Fix **one common K** for the sweep.
   Any other exception stops immediately; three rejected caps stop the job.
5. Every experimental configuration has a separate 128-frame prefix. Full/prefix
   poses agree at 1e-4, decisions and cache IDs exactly, scores at atol 1e-5/rtol 1e-4.
   Every input's resized RGB hash matches the shared archive. Cache IDs must always
   precede the query, and accepted IDs must match `kf_idx.npy`.
   Before each scene's full experiments, one prefix from each of the four semantic
   families exercises the real feature hooks, export, PCA/similarity plots and
   patch overlays. Those prefixes are reused for the later causality comparisons.
   Prefix PCA bases stay separate from full-sequence bases.

Prefixes and memory probes deliberately do not run GT alignment: a short initial
segment can lack matched GT or enough motion for Sim(3), which is irrelevant to
their causality/memory checks. Full runs still require successful ATE/RPE evaluation.

The preflight estimates device peak as peak PyTorch reservation plus end-of-run
non-allocator use. It is not a guarantee against all later fragmentation. Allocated
and reserved peaks and 1-second `nvidia-smi` device samples are archived separately.
An unexpected full-run OOM stops and preserves partial results; no silent cap change.
The scene path does not accumulate per-query pointmaps. It does retain trajectories,
and writes them each frame; timings include that existing behaviour.
The visualization callback copies the latest keyframe reconstruction to CPU on
each insertion, overwriting its previous copy; it writes one `final_scene.npz`
when tracking returns or raises. Snapshot-copy time is reported separately as
`snapshot_copy_seconds` and is included in synchronous total time. Query/rebuild
model-call timing excludes this callback. Preflights include the same callback.

Natural saturation is insertion slowing/stopping below K, with a substantial
remaining tail. Hitting K is **cap-limited**, not natural saturation. Scores and
cap-blocked candidates continue to be recorded after K. Threshold-dependent lack
of insertions is not by itself proof that the scene is adequately represented.

## Visualizations saved by the batch

All visualization happens headlessly on CAMP after inference; matplotlib uses Agg.
`tools/kvt_tum_viz.py` reads saved artifacts and requires no model forward.

**Every full experimental run** gets `viz/` with:

- `diagnostics.png`: novelty and threshold (including cap-blocked candidates),
  retained-keyframe count and cap, per-frame translation error with GT holes blank,
  K/V and descriptor bytes, query/rebuild timings, and selector time. Insertions are
  marked against sequence time. Translation and rotation RPE panels show the
  per-frame jitter without bridging missing GT.
- `trajectory.png`: XY/XZ/YZ views of the complete estimated trajectory, valid GT,
  and selected keyframe locations. GT gaps remain disconnected.
- `keyframes_00.png`, etc.: every selected model-input frame in paginated contact
  sheets (16 per page), labelled with source index and timestamp, with aspect ratio
  preserved. No selected keyframes are omitted.
- `manifest.json`: artifact inventory and source-file references.

**Feature-space diagnostics for every full semantic run:**

- `feature_pca.png`: two views of the same 2D PCA projection, colored by sequence
  time and actual native novelty. Selected keyframes are red rings; cap-blocked
  candidates are orange crosses. The basis is fitted after inference on the first
  full condition for that sequence/layer and shared across its threshold/score
  plots. Axes report the reference explained variance. Encoder and decoder plots
  have separate bases; their coordinates are not directly comparable.
- `feature_similarity.png`: pooled cosine similarities between each arriving frame
  and selected keyframes, in the original descriptor space, above the actual
  selection-score trace. A cell is grey whenever the reference frame is not
  strictly earlier than the query. For cosine selection these are the same pooled
  descriptors used by the selector (CPU recomputation for plotting); for coverage
  and Chamfer this is explicitly a **pooled-feature proxy**, not their decision metric.
- `patch_novelty.png` for coverage/Chamfer: input RGB beside a patch-grid heat overlay.
  A deterministic set of up to 12 examples includes high-scoring selections,
  rejections near the threshold, cap-blocked candidates and evenly spaced times.
  Coverage overlays `1 - max cached-patch cosine`, with the maximum taken over the
  complete retained patch bank. Chamfer overlays the current-to-reference squared
  distances for the actual chosen reference frame; the score also includes its
  reverse-direction mean. Reference IDs and reverse contributions are logged.
  Rendering verifies that saved patch maps reproduce the native scores.

PCA is an **offline explanatory projection**, never used for selection. Apparent
distances/clusters in two dimensions do not establish the 1024-D novelty, object
identity, or a semantic class. The native score remains visible. Patch heatmap
display scales are labelled per-run 99th percentiles; clipping affects display
only. Raw maps and display limits are saved.

Data retained for these views:

- `frame_features.npz`: each arriving frame's L2-normalized mean of raw patch tokens,
  frame IDs and patch-grid shape. It is the actual selection vector for cosine;
  the patch-set policies export it only as a display diagnostic.
- `patch_novelty.npz`: every post-bootstrap query's patch map at its decision time.
  Coverage stores the maximum similarity itself (so the 0.95 boundary is preserved);
  Chamfer stores forward squared distances. These maps are reused from selection,
  not recomputed against the final cache or future keyframes.
- `viz/feature_projection.npz`: coordinates, mean, PCA components, reference explained
  variance, fitting-condition name, masked similarity matrix and native scores.
  Each run carries the shared basis for independent re-rendering.
- `viz/feature_manifest.json`, `viz/patch_overlays.json`: interpretation, sources,
  sampled source indices and display settings.

No additional encoder pass or full stream of patch descriptors is stored. At 5,000
frames, pooled float32 descriptors cost about 20 MiB per run; patch maps cost
`4 * (N-1) * patch_count` bytes. These diagnostic arrays live on CPU, outside the
inference cache. `diagnostic_cpu_feature_bytes` and query `feature_export_seconds`
are reported; copy time is also included in selector/total wall time. Capture is
observational: thresholds, retained descriptors and insertion decisions are unchanged.

After the semantic and periodic comparison for each sequence, the batch additionally
renders the **stock original, best semantic condition, and smallest tested matching
original** (deduplicated; two videos if the match is already the stock original):

- `viz/tracking.mp4`: 1280x800, model RGB on the left; one-pixel, depth-sorted
  **frozen final scene cloud** and growing camera trails on the right. Green is
  timestamp-valid GT, red is the estimate. The same verified full-valid-trajectory
  Sim(3) used for ATE transforms the cloud and trajectory; missing GT is explicitly
  labelled and never connected across a gap. Camera trails are diagram overlays.
  The video states that the cloud is frozen final geometry, not an online map.
- Playback is timestamp-resampled to 15 fps; `video_frames.json` records every
  displayed source index, fixed view, native confidence threshold, retained point
  count, video SHA-256, and verified decoded-frame count. This is playback speed,
  not inference FPS. Beginning/middle/end PNG previews are saved.
- `viz/scene_gt_aligned.ply`: native-confidence-filtered final cloud in the same
  metric GT alignment, for inspection in MeshLab.

`visualizations/<scene>/accuracy_cost.png` compares **all** tested settings by
keyframes, total time and peak reserved memory versus ATE, with threshold labels.
`saturation.png` compares all four thresholds per feature/score family against
stock and extended originals. The plots and all configuration results are retained,
not just the selected video winners. Summary plots are separately archived after
each scene and also included in the context archive.

**Necessary data is saved for every condition**, including conditions without an
automatic MP4: `final_scene.npz` retains the latest full-resolution keyframe XYZ,
confidence, poses, frame IDs and native display threshold. Color comes from the
exact archived `model_rgb` frames; all scene masks are true. `evaluation.npz` now
also retains the exact alignment and complete aligned trajectory. Bootstrap's
duplicate frame is deduplicated for display. Confidence thresholds can be inspected
afterward from the saved values without inference. Per-query pointmaps and every
historical reconstruction are not produced/stored by this camera-only protocol.

Re-render any archived condition after restoring its result and shared input
directories on a Linux allocation, using the existing container environment:

```bash
python /mnt/projects/gr/3DRecon/layer_good/tools/kvt_tum_viz.py run --inputs /tmp/restored/inputs/freiburg3_long_office_household --result /tmp/restored/runs/freiburg3_long_office_household/decoder0_cosine_0.05 --video
```

Rendering failures stop the job; completed inference artifacts remain archived.
`JOB_OK` is written only after per-run visuals, selected videos and scene comparisons
have all succeeded. A job that already started under the earlier code will not gain
this capture retroactively; use this revision for the next submission.

## Execution and archives

`kvt_tum.sbatch` requests one GPU on `24g/muenchen`, account `students`, QoS
`students_normal`, using the previously recorded allocation. It records the current
Slurm job and node inventory; it does not claim current queue availability.
Pulls happen inside the allocation, with HTTPS for the tracker fetch. Parent and
tracker pins, source snapshots, image identity/checksum and test logs are archived.
HF is offline: missing checkpoints fail rather than trigger a download.

The 16 semantic configurations on three sequences mean 48 full semantic runs,
plus prefixes, originals, five fidelity sequences and adaptive periodic runs.
No duration estimate is claimed before measurement. One container load per job;
each run starts a fresh Python process so CUDA state does not leak between policies.

Persistent destination: `/mnt/projects/gr/3DRecon/kvt_tum_out/`. Job-local RGB,
resized model inputs and results live in `/tmp/tum_<jobid>/`, never in the checkout.
Archives:

- `tum_<jobid>_inputs_<scene>.tar`: exact original RGB, resized RGB, GT/rgb metadata,
  source ZIP checksum and per-frame manifest. Scene masks are all true by definition.
- `tum_<jobid>_<scene>_<condition>.tar`: config, trajectory, keyframes/poses,
  decisions with score/cap/bytes/cost, inference and frame timings, evaluation,
  final reconstruction, peak memory, environment, process status and log. Full
  experimental runs include their `viz/` artifacts. Prefixes have separate archives.
- `tum_<jobid>_visualizations_<scene>.tar`: all-variant accuracy/cost and saturation
  plots, available immediately after that scene completes.
- `tum_<jobid>_context.tar`: protocol, inventory, chosen cap and preflight results,
  CSV/JSON summaries, per-scene periodic-match reports, source/container provenance,
  GPU samples, controller log and job exit status.
- `tum_<jobid>_all_runs.tar`: all run directories, including partial failures, staged
  by the EXIT trap. It deliberately duplicates completed result archives for recovery,
  but does not duplicate input RGB. No files are deleted from /tmp during the run.

Only `JOB_OK` plus context exit status 0 means the complete sweep passed.
Prefix/probe runs are gate artifacts and are excluded from the performance table.
Inference hooks synchronize CUDA: timing is instrumented wall time, **not** the
paper's asynchronous FPS. Query/rebuild/selector cost and descriptor storage are
reported, along with overall synchronous throughput.

After publication and once the batch file is on the cluster:

```bash
sbatch /mnt/projects/gr/3DRecon/layer_good/tools/kvt_tum.sbatch
```

For the initial deployment when the new file is absent, submit a wrapper that
pulls inside the same allocation and executes it (no pull on head):

```bash
sbatch --job-name=kvt-tum --account=students --qos=students_normal --partition=24g --nodelist=muenchen --gres=gpu:1 --nodes=1 --propagate=NONE --output=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log --wrap='git -C /mnt/projects/gr/3DRecon/layer_good -c fetch.recurseSubmodules=false pull --ff-only && bash /mnt/projects/gr/3DRecon/layer_good/tools/kvt_tum.sbatch'
```

After completion, on the Mac from the repo root (replace JOBID with the actual ID):

```bash
mkdir -p cluster_results/kvt_tum
rsync -av chunquancheng@131.159.11.60:'/mnt/projects/gr/3DRecon/kvt_tum_out/tum_JOBID_*.tar' cluster_results/kvt_tum/
```

Extract each archive into its own subdirectory. Read the context first, then all
configuration metrics; do not report just the best threshold. Results belong in
`tools/FINDINGS.md` only after the archives and gates have been inspected.
