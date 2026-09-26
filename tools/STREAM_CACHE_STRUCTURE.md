# B1–B4 and overnight pilot jobs

Selected 2026-09-26: implement all B experiments and run pilots/diagnostics only.
Full evaluations still require human review. B3 full evaluation additionally
requires the A3 scientific gate in `STREAM_CACHE_PRECISION.md`.

## Fixed StreamVGGT office pilot

`stream_cache_structure.json` runs 200 consecutive office frames, width 308,
native inference precision and complete native camera-head history. Separate
workers run the following conditions in order:

| Condition | Intervention |
|---|---|
| b_profile | Anchor + recent cache, budget 32; B1 attention map and B3 channel spectra/bases |
| b_recent8 | Fresh native anchor + recent frame control, budget 8 |
| b1_heads | Same budget 8; local heads see anchor + previous four + own frame |
| b2_layers | Same budget 8; layers with every head local see own frame only |
| b3_rank | Same budget 8; half-rank per-head K/V channel projection on cache entry |
| b4_registers | Previous four frames retain patches; every older frame retains camera/register tokens |

The user has access only to 24 GB GPUs and authorized adjustment: the original
full-200-cache diagnostic is replaced with an explicit 32-frame retained budget.
At prediction time it has up to 32 past frames plus the arriving frame. Older
mass is conditional on this retained history; it cannot reveal attention to
evicted distant frames. Native precision and the 200-frame clip are unchanged.

B1 measures post-normalization/post-RoPE attention on frames 9,19,...,199.
Queries include all special tokens and up to 32 evenly spaced patch tokens.
The four disjoint bins are own, anchor, previous four (excluding anchor), older.
Mean mass is equally weighted across sampled queries and frames. A local head
has mean older mass below 0.05. Save all 20 samples, means and masks; a layer
with no qualifying heads is a legitimate negative result, not a forced selection.
B2 implements the layer-local option; no additional staggered-cache ladder.
B1/B2 mask attention only: they do not reduce actual cache storage.

B3 fits an uncentred channel basis separately for K/V, layer and head using up
to 2048 evenly spaced tokens from the final bounded cache. Eigenvalues of the channel
Gram matrix are squared singular values. Store the full spectra and bases in
`b_profile/profile.pt`. Keep the leading half of each head's channels, project
and reconstruct only the newly arriving raw K/V after prediction. Older entries
are not reprojected. This is an exploratory in-sample reconstruction pilot, not
xKV reproduction or held-out evidence. Actual native storage, simulation time,
and analytical coefficient-plus-FP32-basis bytes are distinct. No packed kernel
or speed claim. Fitting/profile overhead is outside per-frame timing summaries.

B4 removes patch rows and matching RoPE positions together. After each frame,
four frames retain all tokens and every earlier frame retains its special tokens,
including the anchor. At prediction time the current frame also attends the
previous four complete frames. Native camera caches remain complete. Its cache
grows with special-token history; it is not constant memory.

## Completion and evidence

Remote test discovery includes `test_stream_cache_structure.py`: disjoint bins,
mask consumption through SDPA, layer-local masks, spectral reconstruction,
new-entry-only projection, special-token/position alignment and head preservation.
The existing native fidelity test runs first. Require six RUN OK records,
CAMERA EVALUATION OK, STRUCTURE CONTRACT GATE OK, JOB OK and archived exit 0.
The gate verifies profile finiteness/bin normalization, event coverage and exact
B4 retention counts. Inspect `structure_gate.json`, `profile.pt`, configs, input
hashes, camera metrics (including translation-RPE p99), peaks and provenance.
Compare masks/projection to b_recent8 and register retention also to b_profile;
report unequal bytes. In-sample selection cannot establish generalization.

## Two user-operated overnight jobs

`stream_cache_overnight.sbatch a3` runs A3's 256-frame office pilots sequentially
on StreamVGGT and LongStream. `stream_cache_overnight.sbatch b` runs the B suite.
No full stage is chained. Both entry points pull only inside an allocation and
use existing read-only `optpose.tar` and checkpoints. Existing traps archive
inputs, source, logs, exit status, predictions and diagnostics separately under
`/mnt/projects/gr/3DRecon/stream_cache_out/cache_<host>_<job>_*.tar`.
No image, dataset or model download is introduced by this change.

Publish the task files before submission. Reuse the existing 24g/stuttgart target;
current availability is unverified and jobs may queue. The bounded profile reduces
memory, but actual peak and overnight completion time remain unmeasured. There
are two jobs, not an array of full evaluations. On CAMP head, after publication:

```bash
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$W && bash tools/stream_cache_overnight.sbatch a3"
sbatch -A students --qos=students_normal -p 24g -w stuttgart \
  --gres=gpu:1 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/cache_pilots-%j.log --wrap="$W"
```

For the second job replace `a3` with `b`, using the same resource options.
Follow each log with `tail -n 100 -f <log>`;
Ctrl-C stops watching. Transfer through the field-notes rsync route and extract
each archive separately. Review both A3 host archives and the B archive before
choosing any full evaluation. Local verification is syntax-only; runtime pending.
