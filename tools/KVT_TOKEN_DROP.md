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


## A2: kept background, mass correction and attention probe (mode `a2`)

Selected 2026-09-26; the user asked for the full plan in one job rather than a
diagnostic first. Job 25837 dropped every background patch and failed the 5%
accuracy gate (ATE +8–10%) while time fell 26–51%. A2 asks whether the background
acted as attention sinks or context, and whether a few background tokens with
the right attention mass recover accuracy at nearly the same cost.

Changes (fork `kv_tracker/token_drop.py`, flags through `main.py` and
`kvt_arctic_run.py`, all with `--token-drop`):

- `--keep-background K` keeps K background patches per frame, evenly spaced in
  raster order over that frame's non-object patches; `-1` keeps all of them,
  the dense computation via the token-drop path. The selection rule is fixed a
  priori; the probe reports whether mass sits where it keeps patches.
- `--background-mass` adds log(n_background / K) to those keys' attention
  logits in every decoder attention: frame, global (including cached keys; the
  per-key bias is stored with the cache), camera, point and confidence decoders.
  A representative then carries the softmax mass of the background it stands for
  (Co-Me). The DINOv2 encoder is unchanged. The survey wrote log(n_dropped / k);
  this uses the whole background count, log(1 + n_dropped / k), which is the
  exact duplicate-key identity tested below.
- Camera-head pooling stays a uniform mean over kept tokens, now including the K
  background tokens, in both arms, so mass bias versus none isolates attention.
- `--attention-probe` writes `attention_probe.jsonl`: every rebuild and the first
  query after it, each global layer. Per query class (object patches, registers):
  mass share on register / object / background keys; for background keys the
  top-4 and top-16 share, effective fraction (exp entropy / count; 1 = uniform,
  small = sinks), mass by Chebyshev patch distance to the object, and the
  distances of the top-16; hidden-state norm quantiles per token class. The
  probe only reads, but slows the run: probe runs are never timed.

Job (`bash tools/kvt_token_drop.sbatch a2`, one allocation, tag `token_mass_<job>`):
tests; original (`--expect-baseline`); k=0 token drop; timed arms bg4, bg4mass,
bg16, bg16mass; then probe runs dense (-1), dropped, bg4, bg4mass, bg16,
bg16mass. Every run after original has `--check-masks-from` the original.
Twelve run archives of roughly 0.55 GB each (25837 sizes), ~6.6 GB in
`kvt_arctic_out/`.

Contracts (`test_kvt_token_drop.py`, CPU and GPU): background selection;
a representative with bias log 7 through the real `BlockRope` equals seven
identical keys (1e-5); all background kept matches upstream within 1e-2 with and
without the bias path; K=4 caches exactly object + 4 per frame + registers with
the stored bias equal to the log weight; the probe records all 18 global layers
for a rebuild and its query with shares summing to 1.

Criterion: unchanged from 2026-09-23, each timed arm against the same-allocation
original on each object: ATE, RPE_t and RPE_rot each at most 5% worse, lower peak
allocated and lower tracking time. The k=0 arm reproduces the 25837 condition in
this allocation. The dense probe answers sink versus context; arm probes show
whether the bias restores the dense background mass share. Low-confidence pose
holds (`low conf detected` in the log) are reported per arm, since they explained
most of box's 25837 gap. No threshold or K sweep beyond {4, 16}.

Submit on CAMP head after publishing; check quota first (~6.6 GB added):

```bash
getquota
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$W && bash tools/kvt_token_drop.sbatch a2"
Q="-A students --qos=students_normal -p 24g -w muenchen"
LOG=/mnt/projects/gr/3DRecon/kvt_token_drop-%j.log
sbatch $Q --gres=gpu:1 --propagate=NONE -o $LOG --wrap="$W"
```

Expected evidence: `COMMIT`, test `OK` with `ALL-BACKGROUND`, `MASS k=4` and
`PROBE` lines, `BASELINE GATE OK`, `MASK GATE OK` for every later run, per-run
ATE tables, `PROBE <scene>` summaries for six probe runs, `TOKEN MASS JOB OK`.
Limits: one run per condition, three objects, synchronous harness timing.
