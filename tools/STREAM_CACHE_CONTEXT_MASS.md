# Context group-mass diagnostic

Frozen 2026-09-28 after job 25990 failed its attention-score screen. User authorized
this bounded follow-up. This is a baseline mechanism check, not a novel selector.

## Scope

One masked ARCTIC espresso clip, 128 frames, width 518, frozen StreamVGGT FP32,
eight retained frames with the existing protected anchor. Four isolated conditions:
`dense8`, `dense8_mass` (unit-mass implementation control), `spatial64`, and
`spatial64_mass`. No observer, appearance arm, threshold sweep or held-out claim.

Spatial arms retain every foreground patch and exactly 64 background groups when
enough background exists. Both use the same encoder features, spatial partition,
means, representative integer positions, reconstruction and retention. Each token's
mass is its original patch count; specials and foreground have mass one. Corrected
attention adds log(mass) to key logits in every frame block and global block,
including current and historical keys. Historical mass travels with its frame and
is evicted with K/V. Camera/depth/point heads remain unchanged. The uncorrected arm
stores the same masses but does not apply them, matching persistent storage.

This exactly restores multiplicity only when merged keys and values would be
identical at identical positions. It does not restore within-group feature/RoPE
variation, nonlinear evolution, or dense trajectories. Improvement therefore
supports this correction in the tested pipeline, not a unique causal account of
the prior probe mismatch. A negative outcome does not reject context compression.

## Gates and interpretation

Require remote unit tests (including duplicate-key equivalence with history,
frame attention, unit-mass identity and eviction), native fidelity, each research
condition, object evaluation, CONTEXT MASS CONTRACTS OK, cross-condition gate,
SWEEP OK and archived exit 0. Dense all-frame poses and final exported prediction
fields must match the unit-mass control within existing fidelity tolerances.
Spatial grouping, positions, token masses, retained IDs, per-frame token counts,
state and camera bytes must match exactly across arms. Group masses must sum to
the original token count. Archive source/config, input pixels/masks/GT, events,
poses, object metrics, timing and peak/persistent memory through the existing driver.

Primary descriptive scientific gate: corrected spatial64 has lower full-clip
adjacent translation RPE RMS AND p99 than uncorrected spatial64. Report both
against dense8; also inspect each temporal half, rotation RPE and aligned ATE.
Mixed outcomes are inconclusive, not a reason to change the gate after the run.
Report actual latency and peak memory: additive masks may change SDPA backend,
and correction overhead is included. One run is not a robust speed benchmark.
No full evaluation or selector submission until archive review. If the gate
passes, next design must retain mass correction as a baseline and validate on
held-out sequences; otherwise inspect the failure before choosing another run.

## User-run CAMP submission

Submit from head; both pulls and execution occur inside the allocation. Existing
stuttgart/24g target is retained; current availability is unverified.

```bash
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$W && bash tools/stream_cache_research.sbatch context_mass"
sbatch -A students --qos=students_normal -p 24g -w stuttgart \
  --gres=gpu:1 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/cache_research-%j.log --wrap="$W"
```

Return the job ID. Do not queue the main experiment behind this diagnostic.
Expected archive: `/mnt/projects/gr/3DRecon/stream_cache_out/cache_streamvggt_<job>_*.tar`.
