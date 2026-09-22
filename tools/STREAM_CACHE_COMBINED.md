# Single combined cache policy

Updated 2026-09-22. User scope: **one combination**, confidence–novelty half
patches plus redundancy eviction, on StreamVGGT, LongStream and KV-Tracker across
office / with-loop / no-loop. Nine full evaluation runs, not a control grid.
Compare against the existing baseline/component results. Correctness gates are
short implementation checks, not additional scientific sweep conditions.

## Fixed policy and interpretation

Keep the initial anchor complete and all special tokens. Rank ordinary patches
by equal-weight confidence rank plus novelty rank; retain 50%. Evict the most
feature-redundant eligible frame, protecting the anchor and newly inserted frame.
Novelty is measured against retained selected patch descriptors before eviction.
No admission thresholds or extra learned components are introduced.

This is exploratory reuse of three TUM sequences, not held-out validation. Report
ATE, translation/rotation RPE, fitted scale, actual cache payload, peak allocated
and reserved memory, sampled device memory and measured latency separately. GT
holes on industrial sequences remain excluded. No object identity/mask signal
is used, and no object-pose or semantic-understanding claim follows.

## Streaming hosts

`stream_cache_combined.json` now has exactly one condition, budget eight, all-frame
admission, encoder features, unchanged native heads. LongStream retains native
refresh; StreamVGGT's camera-head cache continues growing. The long wrapper selects
this JSON explicitly with `combined-pilot` (256 frames) or `combined` (full scene).
Both stages accept only StreamVGGT and LongStream, and both are submitted with
`--array=0`; the wrapper no longer selects the scene from the array index. Host is
one name or several comma separated, `combined` runs office, with-loop and no-loop
sequentially for each host in the same job, and `combined-pilot` runs the office
pilot for each host. Slurm's MaxSubmitPU counts submissions, not loops, so the six
full streaming runs fit in two jobs. Each run still prepares, executes and archives
separately. Existing full/calibration modes retain their previous configurations
and their array-selected scene.

After publication and a pull inside an allocation, the pilot interface for both
hosts in one job is:

```bash
sbatch --array=0 tools/stream_cache_long.sbatch longstream,streamvggt combined-pilot
```

Review exit status, native fidelity, complete events, bounded retention, half-patch
selection and memory before the `combined` full runs, which are one job per host.
The pilot includes LongStream refresh; it does not establish full-run memory
safety for StreamVGGT's growing native camera-head cache.

## KV-Tracker adaptation

The exact Pi3 source is preserved in the job-25680 context archive; its model and
attention hashes are checked by the run adapter. Pi3 stores normalized keys after
RoPE in odd decoder layers, so gathering K/V together preserves key positions.
Its existing tokens_mask changes the dense reconstruction and heads too; this
experiment instead gathers only persistent query memory after a native dense
rebuild. No Pi3 source or dependency changes are required.

Model changes belong to the KV-Tracker fork: `kv_tracker/combined_cache.py` and
an optional `keyframe_cache` argument in `main.py`. The editing branch is
`codex/combined-cache` in the isolated `/private/tmp/kvt-combined-worktree` worktree.
The published fork revision is `e1930019f3bf751164396304a03786b2edb5cf30`;
the superproject pins it. Do not run the new tools against the previous tracker pin.

- Budget 20, interval-50 insertion opportunities, matching the original TUM
  baseline's cap and cadence. Continue replacing after the cap instead of stopping.
- Preserve the duplicated-frame bootstrap and collapse it on first real insertion.
- Evict matching RGB/mask/frame-ID rows before the dense rebuild. Frame zero remains
  first, preserving the existing first-frame gauge. Restrict the new path to the
  evaluated camera-only scene mode, without optional Sim(3), crops or manual input.
  The old paths are unchanged. Sim(3) support is not silently assumed.
- Encoder descriptors are captured at candidate arrival before cross-frame decoding.
  Half-patch confidence comes from the candidate's subsequent dense rebuild, which
  already computes confidence. Query-only inference stays camera-only. This host
  distinction is recorded, not claimed to be identical to streaming confidence.
- Existing frame patch choices persist until eviction, even though native rebuilds
  recompute their K/V. Removed patches are discarded again after every rebuild.
- Every retained frame keeps five register tokens; frame-zero patches stay complete.
- Native dense reconstruction geometry, origin normalization and pose heads remain.
  Sparse query memory does not imply lower dense-rebuild peak memory.
- `inserted_kf_idx.npy` records cumulative insertion history. `kf_idx.npy` and final
  scene frame IDs identify rows of the current reconstruction. Cache events record
  evictions, selected patch IDs, actual dense/query bytes and feature payload.

`tools/kvt_tum_combined.py` connects the policy to the existing TUM timing/evaluation
runner. Native inference timing excludes post-forward pruning; synchronous total
run time includes it. Query and rebuild timing remain separately reported.

## Remote gates and execution

Local verification is AST/JSON/shell syntax and diff checks only. CPU and GPU tests
must run in the existing cluster image; nothing here asserts runtime success.

`test_kvt_combined.py` checks independent masked-attention equivalence for sparse
cache gathering, full-retention identity, known rank-fusion choices, interval phase,
repeated eviction and preservation of anchor/latest and historical patch IDs.

`kvt_combined_run.py` takes one scene or several comma separated, stages each in
turn, and runs fresh subprocesses per scene:

1. A 128-frame periodic/native reference and full-retention adapter fidelity gate.
2. A 1,100-frame combined prefix (past the first evictions).
3. The same combined policy for 1,150 frames in pilot mode, or the complete sequence
   in full mode. Verify prefix poses and cache choices, repeated evictions, actual
   pruning, finite trajectory, complete insertion schedule and
   final-scene/retained-ID correspondence.

Only the final full combined run is an evaluation condition. Every gate failure
stops execution and the outer wrapper archives the failure context. No full job
should be chained behind a pilot before its outputs have been inspected.

After publishing the fork and superproject revisions and pulling inside an
allocation, the interface from head is:

```bash
sbatch tools/kvt_tum.sbatch combined freiburg3_long_office_household pilot
```

`full` replaces `pilot` only after successful pilot review. Several sequences run in
one job when the scene argument is a comma-separated list, each with its own inputs,
gates and archives, and the gates repeat per scene:

```bash
sbatch tools/kvt_tum.sbatch combined freiburg2_large_with_loop,freiburg2_large_no_loop full
```

The existing broad `kvt_tum.sbatch` invocation without arguments is unchanged.

All cluster execution/transfers remain user-operated. Preserve exact prepared
pixels, configuration, source/checkpoint identities, events, poses and final dense
geometry in job archives. Keep intermediates in job-local /tmp and only archives
on project storage. Extract each returned archive into its own directory.
