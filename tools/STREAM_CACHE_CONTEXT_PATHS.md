# Final context mass-path test and stopping rule

Authorized 2026-09-28. Maximum remaining experiment budget: one four-condition
espresso mechanism job and, only after a passing archive review, one six-condition
validation job across box128 and ketchup128. No layer/threshold/token-budget
sweeps, substitute probe scores, or further scientific retries. The 25990 probe
score is retired. An execution failure permits a repair only for an identified
implementation problem; a scientific failure closes this mass-correction branch.

## Test 1: espresso128

Run `context_paths` on the same masked espressomachine_grab_01 clip, width 518,
FP32, eight-frame retention, foreground preservation and 64 background groups.
Four isolated arms: spatial64 (off), spatial64_frame, spatial64_global, and
spatial64_mass (both). Single-path arms change only where log(group mass) is
applied. Global correction includes both current and historical keys. Native
fidelity and existing unit tests precede runs. Grouping, representative positions,
mass, retained IDs, patch counts and cache/camera bytes must match across all arms.
Archive configs/sources, exact pixels/masks/GT, predictions, metrics and costs.

Each single-path candidate must meet ALL criteria against same-job spatial64:

- Translation adjacent-pair RPE RMS <= 0.95 times reference.
- Translation RPE p99 <= 0.95 times reference.
- RMS in EACH half <= 1.05 times its reference half. Pair end frames 1–63 and
  64–127 define the halves, consistent with the earlier diagnostic.
- Total measured inference seconds <= 1.10 times reference, identical cache occupancy.

These are practical descriptive thresholds, not significance tests. Total seconds
uses existing summary timing, including correction and dense heads but excluding
input loading/export; timing is a single-run screen. Report allocated/reserved
peaks, ATE and rotation residuals as secondary outcomes, without changing the gate.
Both-path arm is a reproduction control and is not eligible for selection.
Compare its/off outputs and metrics to 25991 at archive review; a material mismatch
requires an identified provenance/implementation explanation, not an extra tuning run.

`context_path_decision.json` records each criterion and the selection. If neither
qualifies: STOP mass correction. If both qualify: lower p99 wins, then lower RMS;
an exact tie uses frame-only (fixed order). Scientific failure still exits normally
and archives the valid negative result. Require tests, all runtime contracts,
CONTEXT MASS CONTRACTS OK, cross-condition gate, decision report, SWEEP OK/JOB OK,
exit 0 and source provenance before accepting any decision.

## Test 2: conditional held-out validation, then end

Only after reviewing Test 1, freeze its winner unchanged. One validation allocation
runs three arms (dense8, spatial64 off, winner) on EACH of box_grab_01 and
ketchup_grab_01, 128 frames/width 518 with the same masking/retention configuration.
No tuning on these clips. Apply the same four acceptance criteria separately on
each clip. Failure on either closes the branch. Passing both accepts a useful
context-compression baseline and ENDS diagnostics; it establishes neither novelty
nor the abandoned probe score. Validation configuration/submission is prepared
only after the winner and Test 1 archive pass review. Do not scheduler-chain it.

## Submit Test 1

On CAMP head; pull and execution are inside the allocation. Existing stuttgart/24g
target retained, current availability unverified.

```bash
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=false pull --ff-only'
W="$W && bash tools/stream_cache_research.sbatch context_paths"
sbatch -A students --qos=students_normal -p 24g -w stuttgart \
  --gres=gpu:1 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/cache_research-%j.log --wrap="$W"
```

Return job ID. Expected archive is
`/mnt/projects/gr/3DRecon/stream_cache_out/cache_streamvggt_<job>_*.tar`.
