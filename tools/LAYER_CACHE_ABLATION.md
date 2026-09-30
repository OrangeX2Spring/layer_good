# Historical KV by depth: bounded live pilot

User authorized implementation after job26085 summaries,2026-10-01. All model
layers, weights, current tokens, normalization/RoPE, projections, residuals and
FFNs remain active. Only historical KV data changes. No module training.

## Why this experiment

26085's pasted summaries favor probe balancing over equal-byte uniform subsets
and moments over centroids at higher bytes. They do not demonstrate query speed:
even the explicit exact reader is about7x native fused SDPA in that sampled
operator screen. This pilot therefore uses the original native SDPA path and
asks whether allocating history across fewer depths improves the cache budget.
See FINDINGS for numerical evidence and limits; archive review remains pending.

[Does the VGGT Family Need All Its Layers?](https://arxiv.org/html/2609.36842v1)
supplies a depth-redundancy hypothesis, not proven omit-layer indices. Whole-block
pruning is a different intervention from current-only attention in active blocks.
Previous25939/25940 precision protection did not test loss of an entire history.
No hardcoded early/late regions or architecture pruning are used here.

## Fixed design

Calibration: Freiburg1 desk frames0–7 are one joint native bank; frames8–15
are independent cached queries. For each cache-bearing odd decoder block,
physically remove only its historical K/V and run the actual model through the
camera head. Score squared translation deviation from native, normalized by
the bank's RMS translation radius, plus rotation-matrix Frobenius deviation
squared/6. This is a native-pose fidelity proxy, not GT task optimization.
Save all native/ablated poses and per-layer components. Verify that every original
decoder block executes on every calibration query. Freeze the lowest-scoring
one third of cache-bearing layers, with index order breaking ties. On the pinned
36-block Pi3 this means6 of18 historical caches. No budget/interval sweep.
Individual-layer scores do not assume joint effects are additive; evaluation
measures the combined intervention.

Evaluation: desk frames128–255 and textureless-far frames0–127,128 frames each.
The prepared calibration archive includes desk0–127, but only0–15 enters fitting.
These are previously explored engineering sequences, not independent benchmark
datasets. Each evaluation uses the existing live KV-Tracker driver with fixed
admission at31,63,95; no adaptive admission or Sim(3) feedback. Dense bootstrap
and rebuilds remain native. All query rows/heads execute, not sampled traces.
Each arm runs in a fresh process; query results propagate through all subsequent
blocks, but fixed admission/native rebuilds prevent a fully adaptive closed loop.
Calibration's8-frame bank differs from evaluation's2–5 physical bank slots
(bootstrap duplicates frame0); transferring layer choices across these sizes is
part of the hypothesis and may fail.

Three arms only:

* native: full history at every depth, compacted without changing any value;
* omit: empty history at the selected depths, full history elsewhere;
* uniform: evenly spaced flattened historical tokens at every depth, with
  integer rounding distributed across layers to exactly match omit's total bytes.

Uniform includes special and patch tokens in its flattened sampling; it does not
guarantee retention of special tokens or anchors. This baseline tests uniform
data allocation, not a best-of-policies competitor. No extra control grid.

Pinned Pi3 cached attention concatenates cached K/V and freshly computed current
K/V. Empty stored arrays therefore produce current-only attention while the
whole block still executes. Empty arrays own zero storage; no mask or zero-value
placeholder is used. Retained arrays are compact clones or index-selected
arrays. Native V can be a view into full QKV storage, so all three arms compact
retained arrays identically. This prevents hidden backing storage from breaking
the equal-byte comparison. Omit/uniform retain exactly two thirds of native
compact KV bytes at each rebuild; Python bookkeeping is excluded. Transformation
cost occurs after native reconstruction and is included in total/frame wall time.

The calibration alone temporarily retains a dense teacher. Evaluation has no
dense shadow. All arms still build dense history before compaction: bootstrap/
rebuild peaks need not fall. Query attention's current tokens, encoder and FFNs
remain a compute floor. No production speed or whole-process memory claim.

## Review and contracts

Prepared4 GPU contracts: zero-sized storage and unaffected retained values,
equal-byte rounding, actual native attention with empty history versus current
only, and compacting strided views without changing data. CAMP only.
Permitted local checks: Python AST, Bash syntax, git whitespace; no project run.

Summary records per-scene ATE, translation p99, translation/rotation RPE,
query p50/p95 and total seconds, peak allocated/reserved memory, native-dtype
query cache bytes and actual compact persistent bytes. Inference's max_cache_bytes
also sees pre-transform dense rebuilds; use max_query_cache_bytes and transform
events to assess deployed history. Original timestamp-valid GT and per-arm Sim(3)
evaluation remain unchanged. Save pixels, manifests, configs, trajectories,
evaluation arrays, source/model/container identities and partial failure logs.
Omit/uniform event histories and persistent bytes must match. Dense reconstructed
keyframe poses must match native within numerical tolerance; query trajectories
at arrival frames may differ before rebuild. Runtime completion is not a quality
pass. Review each scene's quality/tails against native and equal-byte uniform,
and measure the query-time/byte trade-off before promoting a candidate.

## CAMP entry

From head, after published revisions; updates happen inside the allocation:

```bash
R=/mnt/projects/gr/3DRecon/layer_good
Q="-A students --qos=students_normal -p 24g -w stuttgart"
W="git -C $R -c fetch.recurseSubmodules=false pull --ff-only"
sbatch $Q --gres=gpu:1 --time=01:00:00 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/layer_cache_slurm-%j.log \
  --wrap="$W && bash $R/tools/layer_cache.sbatch"
```

Existing kvt.tar, Pi3 source/checkpoint cache, SAM source/checkpoint and TUM ZIPs
are borrowed offline. No installs, downloads, image mutation or second-host job.
Wrapper archives failures and completion to
/mnt/projects/gr/3DRecon/layer_cache_out/layer_cache_<job>.tar.
Completion line: LAYER CACHE JOB COMPLETE: inspect summary; no automatic next job.
