# Object-mode token dropping (KV-Tracker, ARCTIC S01)

Selected 2026-09-23. Semantic compression by the SAM mask, applied **before** the
forward passes rather than by pruning a finished cache.

## Why

Job 25833 pruned half the patches from a cache that had already been built
densely. Peak memory and per-frame query time did not move (query p50 0.041 s for
20 dense, 48 dense and 20 pruned keyframes; see FINDINGS, "where KV-Tracker's time
and memory actually go"). The query frame's own tokens and the dense rebuild set
the cost. In object mode KV-Tracker already zeroes every pixel outside the SAM
mask, yet computes every patch: the object touches about 30% / 6% / 14% of patches
on box / ketchup / espresso.

## What changes

`--token_drop` (fork `kv_tracker/token_drop.py`, object mode only) computes only
patches touching the Pi3-resolution SAM mask, in the query and in every keyframe
rebuild. Kept patches keep their DINOv2 position embedding and decoder RoPE
position; each frame keeps its five register tokens; the KV cache holds only kept
tokens. The camera head averages over kept tokens instead of all tokens, which is
the one intended behavioural change. Dropped pixels get points 0 and confidence 0.
Nothing else changes: same keyframe policy (upstream original, uncapped), same SAM
propagation, resolution 518, same evaluator. With every patch kept the path is the
upstream computation split per frame.

## Success criterion (agreed 2026-09-23)

Against the original tracker on each of the three objects: ATE, RPE_t and RPE_rot
each no more than 5% worse, **and** lower peak allocated memory **and** lower
tracking time. Timing compares the two runs from the same allocation.

## Job and gates

`tools/kvt_token_drop.sbatch`, one allocation:

1. `token-drop-check` (`tools/test_kvt_token_drop.py`): patch mapping; single-frame
   all-kept output identical to upstream `Pi3.forward`; three-keyframe rebuild plus
   cached query with all patches kept within 1e-2 on poses (bf16 batch-shape noise,
   printed as `ALL-KEPT`); partial mask caches exactly the kept tokens and predicts
   nothing outside; an 8-frame rebuild uses less memory (printed `8-FRAME REBUILD`).
2. Original run: `--expect-baseline` must reproduce the recorded r518 ATE (0.5 mm).
3. Token-drop run: `--check-masks-from` the original run, every SAM mask byte-equal.

Each run archives to `kvt_arctic_out/arctic_token_drop_<job>_{original,dropped}_*.tar`;
`metrics.json` carries ATE/RPE, seconds, peak allocated/reserved and
`token_drop` counts (patches computed / patches in frame, all model calls).

## Submit (CAMP head)

The pull happens inside the allocation, so the script's `#SBATCH` lines are
given on the command line:

```bash
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$W && bash tools/kvt_token_drop.sbatch"
Q="-A students --qos=students_normal -p 24g -w muenchen"
LOG=/mnt/projects/gr/3DRecon/kvt_token_drop-%j.log
sbatch $Q --gres=gpu:1 --propagate=NONE -o $LOG --wrap="$W"
```

Expected evidence in `/mnt/projects/gr/3DRecon/kvt_token_drop-<job>.log`: the
`COMMIT` line, test `OK` with the `ALL-KEPT` and `8-FRAME REBUILD` lines,
`BASELINE GATE OK`, three `MASK GATE OK`, and `TOKEN DROP JOB OK`.

## Limits

Object mode only: TUM scene mode has no mask to drop by. The dropped path cannot
see background even through attention, but object mode already blanked it, so the
information removed is zeros plus their positional context. Timing is the
synchronous harness, not the paper's FPS benchmark. One run per condition.
