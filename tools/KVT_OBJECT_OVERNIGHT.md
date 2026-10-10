# Object history diagnostics: pilot and overnight repeats

2026-10-09. User selected object tracking, handoff/history diagnosis and a bounded
comparison, with **eight hours on one GPU**. Implemented locally; CAMP contracts
and pilot remain pending. Evidence: FINDINGS "Artifact audit26458" and "Object
detector audit26205". This protocol supersedes the initial proposed matrix.
Do not use `segment-detector overnight`: that runs older scene/query experiments.

## Fixed scope

One previously inspected sequence, full ketchup_grab_01, exact saved26458 model
pixels and masks. No live SAM rerun, downloads, training, semantic labels, threshold
grid or other-object expansion. The model fork remains a6ae705. The new behavior
is orchestration-only: `kvt_reanchor_maps.py --disable-retirement` logs the same
live detector but never adds its proposed cuts to the map's boundaries.

| Arm | Behavior | Question |
|---|---|---|
| N: native | Original object admission, saved-mask replay | Matched reference and fidelity |
| R: reanchor | Existing anchor-support detector and immediate retirement | Reproduce26458 control |
| L: no_retirement | Same two-image local map and refresh algorithm, first anchor never retired | Does the small map fail without repeated resets/connections? |

L also changes anchor age and available baseline; it is not a pure history-capacity
ablation. It has no inter-map bridge. Its detector remains active for matched
logging cost, and its scores must match26458 apart from timing. The original
reanchor control must reproduce26458 at rtol=atol=1e-4. Native replay has the same
fidelity tolerance against its saved reference. No online policy sees GT.

## Sequence

1. **Pilot, separate allocation, two-hour ceiling:** TUM/detector/handoff/reanchor
   contracts plus the new no-retirement and paired-error grouping contracts.
   Hash-check four source archives; verify prepared data and checkpoint contents,
   every original source-image and SAM-mask hash, decoded RGB/model-mask hashes,
   and native/reference trajectory identity. Save the audit before inference.
2. Diagnose the saved control's paired squared translation errors, excess error
   relative to native, event groups and segment interiors. Count foreground versus
   background support in its original fit, validation and robust-fit masks.
   These are descriptive diagnostics, not a refit or proof of a failure cause.
3. Run N, R, L once in fresh processes. Require finite full trajectories, matching
   evaluation pairs, control fidelity, identical detector decisions, bounded
   two-image banks, and no retirement events in L. Archive each completed arm.
4. **Human review gate:** inspect audit, fidelity, output completeness, quality,
   time/memory and any failure. Obtain the pilot comparison archive's SHA256.
   Do not schedule a successor behind an unreviewed pilot.
5. **Overnight, separate allocation, eight-hour ceiling:** repeat the same audit
   and contracts, then nine fresh-process runs in three fixed rotations:
   N/R/L, R/L/N, L/N/R. The runner requires the reviewed pilot tag and archive hash,
   and checks the same source archives, tracker revision and implementation hashes.
6. Stop when the manifest finishes. Do not fill unused time with extra sweeps.
   There is no claim these nine runs will consume eight hours. Repeats estimate
   execution variation on one sequence; they do not add independent scenes.

The implementation does not promote quality winners or change the matrix based
on earlier results. A complete but poor-quality arm is a valid result; other
selected independent arms continue. Stop on contracts, provenance, fidelity,
nonfinite/missing outputs, pair mismatch, OOM or unexpected exceptions. No retries.

## Runtime and artifacts

`kvt_tum.sbatch object-history pilot` uses a two-hour internal ceiling;
`object-history overnight tum_PILOT SHA256` uses eight hours. Both are clipped to
Slurm's actual end time and reserve30 minutes for packaging. Before every arm,
require at least30 minutes or twice the longest observed arm wall-time, whichever
is larger, plus packaging reserve. Each subprocess also has a hard timeout;
the outer container command is bounded before packaging. Time exhaustion returns
nonzero and leaves comparison complete=false, with completed/active runs recorded.
Unexpected failures are retained by the wrapper's context/all_runs EXIT archives.
Signal/timeout behavior is statically checked only until CAMP execution.

Source archives are fixed by SHA256 in `kvt_object_history.py`. They remain under
kvt_tum_out and kvt_arctic_out; extraction is job-local under /tmp/tum_JOB/inputs.
The reanchor driver's existing tum_26127_oracle_maps_inputs.tar and
tum_26127_context.tar prerequisites are hash-checked before any inference too.
The prepared dataset and existing kvt.tar are borrowed read-only. No environment
build or checkpoint download. Standard dataset staging occurs inside the allocation.

Outputs under `/mnt/projects/gr/3DRecon/kvt_tum_out/`:

- `tum_JOB_object_history_comparison.tar`: object_history/comparison.json,
  policy.json and native_metrics.json. Includes audit, implementation/input hashes,
  completed arms, quality ratios, timing scopes and per-arm archive hashes.
- `tum_JOB_object_history_rR_ARM.tar`: each arm's trajectory, evaluation, config
  and summary; reanchor arms also include local trajectories, inference/detector
  logs and bridge evidence. Native retains the existing replay output format.
- `tum_JOB_context.tar` and `tum_JOB_all_runs.tar`: wrapper provenance, source,
  contract logs, GPU samples, exit status and completed/partial outputs.

Inputs are referenced by immutable source archive hashes rather than duplicated
in new output archives. Preserve26458 originals. No new output exists until a job
actually runs; no artifact transfer or remote allocation has occurred in preparation.

Quality remains ATE, translation RPE and translation p99 each <=1.05 times native.
Report raw timing and allocated/reserved peak memory, cache bytes where available,
and scopes. Native/reanchor timing excludes SAM/model load; their included work
still differs. Event groups use cut, connection, refresh, other priority; segment
interiors exclude incoming cut pairs. No independent per-segment Sim(3) alignment
is computed. Foreground bridge support counts do not test a foreground-only policy.

## Execution readiness and next command

Job26460 stopped after the native arm with `KeyError: 'detector'` before
reanchor tracking. The saved26458 segmentation policy contains thresholds but
omits the CLI detector identifier. The history runner now explicitly restores
`detector='orb'` for this fixed ORB control, preserving all saved policy values.
The corrected pilot still requires CAMP verification; do not advance overnight.
User sacct confirms FAILED1:0; ls confirms final context (38M) and all_runs
(105M) archives. Contents/integrity remain unaudited. Preserve both; a fresh
pilot must use the published fix before any overnight review gate can pass.

Local checks: Python3.10 AST, Bash syntax, git diff --check. New runtime tests have
not run on the Mac, per execution boundaries. Publication and pilot status are
recorded in docs/STATUS.md; no model changes or gitlink update. Task files:
KVT_OBJECT_OVERNIGHT.md, kvt_object_history.py, test_kvt_object_history.py,
kvt_reanchor_maps.py and kvt_tum.sbatch. Preserve unrelated working-tree changes.

After publication and checking current queue/node availability, the pilot uses
allocation-wrapped pull/run from `/mnt/projects/gr/3DRecon/layer_good` on head.
For the existing muenchen/24g configuration, if still available:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch object-history pilot"
O=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE \
  -t 02:00:00 -o "$O" --wrap="$W"
```

Expected evidence: contracts pass, three OBJECT HISTORY ARM lines, OBJECT HISTORY
COMPLETE pilot 3, OBJECT HISTORY JOB OK, Slurm COMPLETED0:0 and final archives.
JOB OK establishes execution only. Transfer/review pilot evidence before supplying
the overnight command, which must use its real tag/hash and an08:00:00 allocation.
All cluster operations are user-run. The exact pending gate lives in docs/STATUS.md.
