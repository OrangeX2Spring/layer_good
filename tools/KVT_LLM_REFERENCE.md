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

After task files are published, from the CAMP checkout on head. The bootstrap
pull runs inside the allocation, before invoking the new mode (the old checkout
does not yet recognize it):

```bash
sbatch -A students --qos=students_normal -p 24g -w muenchen \
  --gres=gpu:1 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log \
  --wrap='git -c fetch.recurseSubmodules=false pull --ff-only &&
bash tools/kvt_tum.sbatch llm-packet'
```

The wrapper pulls inside its allocation, borrows `localhost/kvt` read-only and
uses the existing GPU memory grant. This mode only prepares inputs: no tracker
inference or evaluation sweep, dependency installs or model downloads.
Work stays in `/tmp/tum_JOBID`; packet output is
`/mnt/projects/gr/3DRecon/kvt_tum_out/tum_JOBID_llm_packet.tar` plus separate
`tum_JOBID_context.tar`. Require `LLM PACKET OK` with frame/sheet counts,
`LLM PACKET JOB OK`, context exit status 0 and complete coverage after transfer.
Use canonical rsync; extract to its own local directory. Review before selection.

## Replay and evaluation gate (next implementation)

TUM uses `TumSelector`; ARCTIC's `--keyframes-from` CLI does not apply. Add fixed
selection to TUM after the packet gate, preserving timestamp-valid evaluation.
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

Packet implementation and prompt prepared; static checks only. No job submitted,
packet created, images inspected by a selector or LLM tracking run completed.
Next gate: preparation job and archive review. Replay integration and remote
fidelity remain pending; do not submit the historical sweep.
