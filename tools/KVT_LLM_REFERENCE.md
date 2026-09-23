# Subscription-first LLM keyframe reference — TUM

## Selected scope

2026-09-23: user selected whole-clip offline selection, subscription first, then
switched from ARCTIC to TUM. Earlier ARCTIC packet/upload prerequisites are
superseded for this experiment. API integration waits for useful measured results.
This is an offline visual reference with future information, not an online or
autonomous policy and not a real-time claim.

Start with `freiburg3_long_office_household`, RGB-only scene/camera mode, offset 0,
resolution 308. Budget: 20 unique keyframes including bootstrap frame 0, matching
the stock interval-50/cap-20 TUM baseline. No eviction, token dropping or budget
sweep. Review office before extending to the industrial sequences; those have
partial GT and cannot establish full-route accuracy.

The [official TUM page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license)
states CC BY 4.0 for benchmark data unless otherwise stated (checked 2026-09-23).
Packet attribution cites Sturm et al., IROS 2012, links the source/license and
records resizing and contact-sheet generation.

## Input packet and subscription session

`kvt_tum_llm_packet.py` uses existing TUM preparation and the read-only
`/mnt/datasets/tum-rgbd/rgbd_dataset_freiburg3_long_office_household.zip` on CAMP.
It preserves every frame as a lossless model-input PNG, verifies pixel hashes,
and creates chronological 12-frame contact sheets at the same pixel dimensions.
Every tracking index appears once, including bootstrap; no temporal subsampling.
Individual PNGs remain available for close inspection. The manifest records
source/model hashes, source ZIP hash, timestamps and sheet membership.

Only the packet goes to the selector: RGB, sanitized manifest, attribution,
filled prompt and `PACKET_OK.json`. No depth, GT, trajectories, scores or baseline
IDs. The separate context archive is for provenance, not the selector.

Use a fresh subscription session outside the repository/prior conversation. The
coordinating session has read results and must not choose experimental frames.
Use the packet's `PROMPT.md` (source: `KVT_TUM_LLM_PROMPT.md`). Inspect every sheet
and record coverage before returning 19 sorted unique insertion indices in
`selection.json`, with reasons and limitations. Preserve transcript, displayed
model, date, human interventions, inspection coverage and selection elapsed time.

Whole-clip review requires many batches; subscription capacity is not verified.
If limits interrupt review, save partial observations and report incomplete
coverage; do not silently switch to sparse candidates or claim every frame was
seen. Freeze one valid selection before scoring; only format/count corrections
are allowed beforehand, retaining the original response.

## Preparation job

After task files are published, from any directory on CAMP head. The bootstrap
pull runs inside the allocation, before invoking the new mode (the old checkout
does not yet recognize it):

```bash
sbatch -A students --qos=students_normal -p 24g -w muenchen \
  --chdir=/mnt/projects/gr/3DRecon/layer_good \
  --gres=gpu:1 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log \
  --wrap='git -c fetch.recurseSubmodules=false pull --ff-only &&
bash tools/kvt_tum.sbatch llm-packet'
```

Job 25850 failed before preparation because the original bootstrap inherited the
submission directory and `git pull` reported "not a git repository". The explicit
`--chdir` above fixes that dependency; no packet was produced by that invocation.

The wrapper pulls inside its allocation, borrows `localhost/kvt` read-only and
uses the existing GPU memory grant. This mode only prepares inputs: no tracker
inference or evaluation sweep, dependency installs or model downloads.
Work stays in `/tmp/tum_JOBID`; packet output is
`/mnt/projects/gr/3DRecon/kvt_tum_out/tum_JOBID_llm_packet.tar` plus separate
`tum_JOBID_context.tar`. Require `LLM PACKET OK` with frame/sheet counts,
`LLM PACKET JOB OK`, context exit status 0 and complete coverage after transfer.
Use canonical rsync; extract to its own local directory. Review before selection.

## Replay and evaluation gate

TUM uses `TumSelector`; ARCTIC's `--keyframes-from` CLI does not apply. TUM configs now accept `policy: fixed` and sorted `insertion_indices` excluding
bootstrap, with exact count/range validation and unchanged timestamp-valid evaluation.
Insert selected frames only when tracking reaches them; never preload future RGB.
Verify exact requested/actual insertion IDs and full trajectory length.

Compare stock original, replay of its exact IDs (fidelity gate), LLM selection,
full-clip uniform and random seeds 0/1/2. All use 20 unique frames, identical
inputs, preprocessing, seeds, model revision and container. Stock's counter
inserts at 49, 99, ... and stops at 949, not 50, 100. Full-clip controls prevent
attributing simple temporal spreading to reasoning. Verify compatibility before
reusing historical runs.

Report Sim(3)-aligned ATE, translation/rotation RPE, failures, tracking time,
peak allocated memory and actual IDs/count. Keep selection time separate; equal
final counts do not imply equal cache occupancy/rebuild cost. Retain 20 ms GT
association and gap-aware RPE; see `KVT_TUM.md`.

Proposed API-investment gate, to freeze before scoring: office ATE below stock
and all four uniform/random controls, with both RPE measures at most 5% worse
than stock, followed by an independent subscription selection with the same
frozen prompt. This is exploratory evidence on one scene, not significance,
generalization or proof that causal selection works.

## Completion state

2026-09-24: packet job 25851 completed and archives were transferred and reviewed.
The local packet at
`cluster_results/kvt_tum/tum_25851_llm_packet/llm_packet/` now contains
`selection.json` (19 insertion indices plus implicit bootstrap 0) and
`inspection_log.md` recording all 216 sheets and individual selected-frame review.
The log reports format, count, manifest and coverage checks; coordinating-session
review confirmed the artifacts exist and read their contents, without repeating
the visual inspection. User reports `gpt-6-astra low`; originals record
`GPT-6 (Codex)`. Preserve originals and record this distinction in provenance.
The saved selection is frozen before scoring; no LLM tracking result exists.

Fixed-index replay and the `replay-fidelity` job are implemented. The job runs
full-office stock (no selector) and fixed replay of its actual saved IDs in fresh
processes. It requires 20 identical IDs, full trajectories and elementwise pose
agreement (`atol=rtol=1e-4`, matching the existing hook-fidelity tolerance).
`replay_verified.json` records requested/actual IDs; `fidelity.json` records the
maximum pose difference and both runs' metrics. Unit contracts cover arrival-time
selection, ignoring stock candidates, and malformed indices; they run on CAMP.
Local Python AST, bash syntax and whitespace checks passed; runtime is pending.

From CAMP head, after publication:

```bash
sbatch -A students --qos=students_normal -p 24g -w muenchen \
  --chdir=/mnt/projects/gr/3DRecon/layer_good \
  --gres=gpu:1 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log \
  --wrap='git -c fetch.recurseSubmodules=false pull --ff-only &&
bash tools/kvt_tum.sbatch replay-fidelity'
```

Require tests passing, `STOCK-ID FIDELITY GATE OK`, final `JOB OK`, and context
exit status 0. The context archive contains `fidelity.json`, protocol and source;
individual `tum_JOBID_freiburg3_long_office_household_{stock,stock_replay}.tar`
archives contain configs, IDs, trajectories, metrics and logs. Inputs are archived
as `tum_JOBID_inputs_freiburg3_long_office_household.tar`. These paths are planned,
not yet observed. Review the gate evidence before the LLM/uniform/random comparison;
that comparison driver and frozen-selection packaging remain subsequent work.
First fidelity attempt (job ID not supplied) stopped in preflight: 17 tests,
one error because the encoder/decoder fixture lacked `policy`. That fixture now
explicitly sets `policy='semantic'`; AST and whitespace checks pass. Rerun the
command above; no fidelity tracking ran in the failed attempt.
No model-fork change or selection transfer has occurred. Do not repeat
packet preparation/selection or submit the historical sweep. API remains deferred.


Job **25855** completed and its context archive was transferred and reviewed:
exit status 0, 17 tests OK, parent `ba0def8`, tracker `3330f54`, clean recorded
diffs and matching container archive SHA with packet job 25851. Stock-ID fidelity
passed; see FINDINGS for results and limitations. This supersedes the retry and
pending-fidelity instructions above. Next is comparison-driver implementation
and frozen-selection packaging; do not rerun fidelity or packet preparation.
