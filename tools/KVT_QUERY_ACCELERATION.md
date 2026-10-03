# Independent query acceleration alongside the flow segmentation pilot

User-selected2026-10-03: prepare the methods in one overnight allocation, test
them independently first, combine only after review. The third method is an
independent query acceleration, not the combined segmentation/speed arm.

## Methods and scope

1. Segmentation: existing `segment-detector flow` pilot, ORB25 versus sequential
   anchor tracks on all2585 exact archived office frames. No graph/token reduction
   enters this diagnostic. Review boundary pixels, fragmentation and cost before
   live reanchor integration and the matched human-boundary comparison.
2. Speed: CUDA Graph replay of dense camera queries, using the existing reviewed
   GraphQueries/current-stream RoPE adapter (FINDINGS26041/26045). Camera head/SVD
   remains eager. Capture is invalidated before every dense cache reconstruction.
   This is a2x-FPS candidate, not a promised2x result. Earlier bounded real-scene
   pilots showed a smaller gain; full native-bank office behavior is untested.
3. Query compute: retain the fixed checkerboard half of the camera query's spatial
   patches before the encoder and decoder, preserving original patch positions and
   all register tokens. Rebuilds, dense geometry, native keyframes and cached KV
   remain dense and unchanged. No learned selector, extra teacher or threshold grid.
   Accuracy may degrade; report all results, not only a fast arm.

The independent query experiment uses MapHandoff native mode, same input pixels,
checkpoint, seed0, resize308, BF16/TF32 settings,50-frame schedule and cap20.
Four fresh processes each track2585 frames: native, dense graph checked, dense
graph timed, half-query eager. The checked graph compares every query against both
its kept-path eager reference and native Pi3; its doubled-forward runtime is
diagnostic only. The timed graph must reproduce native and checked trajectories
at rtol=atol=1e-4. Each arm has identical GT associations, keyframe IDs and maximum
cache bytes. Native must reproduce25680 headline metrics to absolute1e-4m.

GPU contracts first verify native dense cache/query interoperability, changing
input images, recapture after rebuild, cache value preservation, and half-query
graph/eager fidelity. The previously reviewed graph pilots support this direct
full-office execution experiment; this batch does not queue an automatic full
segmentation/reanchor evaluation behind the unreviewed flow detector pilot.

Both query techniques accept the same optional MapHandoff executor. Reanchor
passes it to each new local map, releases captured cache references before old
KV retirement, and forces the boundary camera read dense to agree with the dense
shared pointmap. `half_graph` is implemented for later compatibility tests but
is not dispatched by this batch. No segmentation+graph, segmentation+half or
graph+half tracking combination is launched until independent results are reviewed.

## Timing, results and completion

The tracking timer includes pixel IO/hash, upload, selection, graph capture/setup
and first-query validation, camera inference, native rebuilds and logging.
It excludes model/backend load, one-time extension compilation and evaluation;
backend load/setup and build provenance are retained separately. Query timing
includes the graph's eager camera head. Do not sum rebuild and update_total costs.
Native versus timed graph/half use the same allocation but one ordered run each;
no repeatability/statistical speed or generic scene claim follows from this run.

Report full-loop FPS and speedup, ATE/RPE/p99, original three-metric5% verdict,
cache/allocated/reserved memory, query/rebuild sums and capture overhead.
`doubled_fps` means full-loop speedup>=2 relative to the fresh native run, with
its accuracy verdict separately visible. JOB OK means execution/fidelity/archive
completion; it does not mean doubled FPS or acceptable accuracy.

Outputs in `/mnt/projects/gr/3DRecon/kvt_tum_out/`:
`tum_<JOB>_query_native.tar`, `query_graph_checked.tar`, `query_graph.tar`,
`query_half.tar`, `query_comparison.tar` (each with the same `tum_<JOB>_` prefix),
plus the ORB25/flow detector, context and all_runs archives. query_execution.json
retains per-rebuild captures, backend load and query counts. Exact inputs/checkpoint
are hash-verified against reviewed26127. Build lives in job-local/tmp; no image
change or installation on the Mac. The wrapper compiles for muenchen A5000sm86.

## Submission

Use the established allocation-wrapped pull/run; all code arrives by git inside
the allocation. From `/mnt/projects/gr/3DRecon/layer_good` on head:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector overnight"
O=../kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

Expected: detector contracts and ORB25/flow complete; stream/kernel, handoff,
reanchor and query contracts pass; four QUERY COMPARE lines; QUERY COMPARISON
COMPLETE; OVERNIGHT INDEPENDENT EXPERIMENTS COMPLETE; JOB OK. Stop at first failure
and return context. Review archives before any combination or successor run.

## 26160 fidelity repair and targeted retry

26160 stopped at the synthetic query contract before query timing. Dense graph
queries now use original Pi3 encoder/decode/camera-decoder computation, rather
than forward_kept with an all-true mask. The camera head remains outside capture.
Both native-versus-graph and eager-versus-replay tolerances remain1e-4; RoPE
backend substitution remains exact. Half-query and legacy kept-graph paths are
unchanged. This removes the alternative tensor assembly from the dense control;
GPU equivalence remains pending. Do not infer the original discrepancy cause
solely from this implementation change.

Use segment-detector query-contract for the repair gate. It skips the detector
pass and full query evaluation, runs stream/handoff/reanchor/query contracts,
archives via the existing EXIT trap, and prints QUERY CONTRACT RETRY COMPLETE
and JOB OK on success. Existing general setup/import checks still run.

From the CAMP checkout on head, after publication:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector query-contract"
O=../kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

Review the contract log before full evaluation. After it passes, query-run is the
same query-only stage with four full-sequence arms; it skips repeating the flow
pilot. The original overnight stage remains available. LoopCD26161 isolated
source and branches are unaffected.


26162 passed native eager/backend checks but capture failed in Pi3.decode at its
CPU-created special-position zeros copied to CUDA. The captured dense path now
prepares fixed positions outside capture and mirrors native single-query decode
with unchanged register repeat/concatenation, block calls, final pair and camera
decoder. Native Pi3 remains the independent eager reference. Existing warmup and
replay checks keep1e-4 tolerances. Repeat query-contract; GPU verification pending.


26167 completes capture but checked replay produces NaN poses after eager and
warmup fidelity gates pass. Capture now disables autocast weight caching, recording
weight conversions inside the graph instead of reusing casts from the surrounding
eager context. Cached-cast lifetime is a hypothesis, not a confirmed root cause.
Checked dense replay also asserts finite encoder, decoder and camera-decoder
stages to locate any remaining failure. Repeat query-contract only; tolerances
unchanged and full evaluation remains gated. LoopCD26168 is independent.
