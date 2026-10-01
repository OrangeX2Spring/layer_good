# Whole-keyframe office pilot

Selected 2026-10-01: Pi3 office only, three arms, one bank size. This replaces
the discussion-only next action; it does not resume layer allocation or training.

Question: can removing complete historical images before joint reconstruction
preserve tracking while reducing its cost, and does feature relevance improve
on recency at the same bank size? No post-rebuild patch or layer pruning.

## Fixed experiment

- Office frames 0–949 inclusive, original order, resize 308, seed 0, same pixels.
  This includes the mixed view at 499 and the later monitor by 699, with queries
  after the 749 transition. It is a partial-scene pilot, not a full-run result.
- Native: original interval-50 admission and cap 20; all historical images kept.
  This window ends before native admission would stop. All arms must insert
  exactly `[0,49,99,...,949]`; admission differences are a failed gate.
- Recent: eight complete images, frame 0 plus the seven latest insertions.
- Relevance: eight complete images. Protect frame 0, the arriving keyframe and
  its immediate predecessor. Rank other retained frames by the maximum cosine
  similarity of their normalized mean encoder patch descriptor to either of the
  two newest descriptors. Keep the five highest scores; prefer newer on ties.
  Preserve chronological bank order and discard evicted descriptors and images.
  Arrival descriptors precede joint decoding. No GT, future frames, extra encoder
  pass, offloaded history tier, candidate sweep or resurrection of evicted frames.

Both bounded arms collect identical features and scores so their comparison
includes matched instrumentation. Native has no descriptor overhead. CPU feature
payload is recorded separately from GPU KV. The bootstrap duplicates frame 0,
as in native, and collapses on first insertion. K/V remain fully dense after each
rebuild; existing main.py removes image/mask/ID rows before Pi3 is called.

This is an appearance-relevance proxy, not a validated semantic or geometric
utility score. It does not explicitly optimize complementary bridge coverage.
It can retain stale but visually similar frames or lose a useful future revisit.
The predecessor protection addresses the old redundancy policy's specific failure;
it does not establish that two local views are sufficient. No tuning on pilot GT.

## Evidence and gates

Nine fresh processes: three 128-frame fidelity runs, then an 850-frame prefix
and 950-frame main run for each arm. Prefixes extend past bank saturation and
the content transition. Remote unit contracts test different stale/recent choices,
ties, predecessor protection and bit-identical untouched dense K/V.

Stop on fidelity failure (raw poses atol 1e-5/rtol 1e-4), nonfinite outputs,
insertion mismatch, bank overflow, missing protected IDs, final-scene ID mismatch,
or noncausal prefix choices. Prefix poses use the existing 1e-4 tolerances.
Bank events record exact retained/evicted IDs, eligibility scores, protected IDs,
rebuild frame counts and dense/query bytes; retained IDs and final reconstruction
must agree. No full evaluation is queued behind this pilot.

Review `bank_summary.json`, every main `evaluation.npz`, cache events, inference
timings, raw trajectories and input images. Plot errors around evictions and the
transition; inspect retained views rather than treating feature score as utility.
GT-valid coverage and missing intervals must accompany metrics. Never align a
post-transition window independently to hide a gauge jump.

Report ATE, translation RPE RMS/p99 and rotation RPE. The inherited accuracy gate
requires each of ATE, translation RPE and translation p99 <= 1.05 times native.
For relevance evidence, require at least one different retention decision, no
regression on any of those three versus recent, and improvement on at least one.
One scene/seed only suggests a follow-up; it cannot establish general benefit.

Report synchronized total wall time (including model load, source loading,
selection and snapshots), query/rebuild/selection time separately, GPU peaks,
sampled device use, KV bytes and descriptor bytes. Snapshot overhead is recorded;
these are instrumented comparisons, not deployment FPS. Smaller rebuild inputs
are a structural result; lower measured time/memory must be observed independently.

## CAMP execution and artifacts

Entry: `tools/kvt_tum.sbatch bank-pilot`. The existing wrapper pulls code inside
the fixed `/mnt/projects/gr/3DRecon/layer_good` checkout; bank-pilot must not
derive that path from BASH_SOURCE because sbatch executes a spool copy.
Job26109 failed at the initial git pull for this reason, before model/tests.
Refresh the wrapper inside an allocation before submitting a replacement.
The wrapper runs inside
the allocation, uses the read-only borrowed kvt image without rebuilding it,
records parent/fork source, image hash, Pi3 source and package inventory, and
archives failures. Publish the fork and parent gitlink before submitting.
No cluster operation is performed by the editing agent.

After publication, from the CAMP checkout, submit only this pilot:

```bash
sbatch --time=04:00:00 tools/kvt_tum.sbatch bank-pilot
```

Expect `BANK PILOT COMPLETE: 3 fidelity + 3 prefixes + 3 main runs`, JOB_OK and
context exit_status 0. These establish implementation completion, not scientific
acceptance. Inspect the first failure before retrying.

Outputs: `/mnt/projects/gr/3DRecon/kvt_tum_out/tum_<JOB>_bank_inputs.tar`,
`tum_<JOB>_bank_<ARM>.tar` (including fidelity/prefix arms),
`tum_<JOB>_context.tar`, `tum_<JOB>_all_runs.tar`; log
`/mnt/projects/gr/3DRecon/kvt_tum_slurm-<JOB>.log`.
Inputs include exact model pixels, timestamps, GT associations, pixel hashes and
source ZIP hash. Completed/failed subprocess outputs are archived immediately;
final archives also include prefix evidence. No new container or checkpoint.
Transfer using the field notes, extract archives separately, and review before
selecting any longer run, new scene, budget, anchor policy or coverage extension.
