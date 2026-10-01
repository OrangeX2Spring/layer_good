# Full office: independent two-keyframe segments

User selected boundaries275,575,700,975,1400 and independent evaluation.
One experiment, six fresh subprocesses; no policy/budget sweep or additional
baseline inference. Full sequence0–2584, original timestamps/pixels, resize308,
seed0, full patches, fixed insertion of local frame49 after bootstrap.

| Global segment (inclusive) | Fixed global keyframes | Frames |
|---|---|---:|
| 0–274 | 0,49 | 275 |
| 275–574 | 275,324 | 300 |
| 575–699 | 575,624 | 125 |
| 700–974 | 700,749 | 275 |
| 975–1399 | 975,1024 | 425 |
| 1400–2584 | 1400,1449 | 1185 |

The first image is duplicated for the native bootstrap. Until local49 arrives,
there is one unique keyframe; subsequently exactly two. Queries up to and
including49 use the bootstrap cache; its rebuild serves queries from50 onward.
No future images are used. No further replacement/admission within a segment,
including the long final segment. Every segment has its own model/process,
image directory and output directory; frame0 from the full sequence is absent
from every later segment. There is no hidden old-frame/reference tier.

This deliberately tests whether two local views support tracking within manually
selected content regions. Boundaries were chosen from the full office image
viewer, so this is a diagnostic, not an online boundary detector or held-out test.
It does not test uninterrupted tracking across resets. Evaluate each segment with
its own timestamp-valid Sim(3) fit and report its scale, ATE, translation RPE,
translation p99, rotation RPE and GT coverage. No cross-boundary RPE or aggregate
stitched ATE; do not compare independently fitted segment metrics directly with
the prior whole-window pilot as an accuracy improvement claim. No new native run
or5% native-relative gate is included in this single experiment.

## Verification and artifacts

Existing remote TUM contract tests run first. Each segment must have finite poses
of its exact length, keyframes[0,49] in local indexing, exactly one bootstrap and
one rebuild, two actual Pi3 input images for each of those calls, and one image
for each query. Every decision must reference only the segment's local keyframes;
final reconstruction IDs must match. Save `frame_mapping.json`, source/checkpoint
hashes, exact model pixels, decisions, input image counts, trajectories, evaluation
arrays and final reconstruction. `segment_summary.json` preserves metrics per
segment; `segment_checks.json` records each segment's gates. Results/inputs are
archived per segment, including partial output on subprocess failure.

The TUM staging helper now permits sequences shorter than128 frames (minimum2)
so the user-selected125-frame segment is supported. Actual pilot segments all
exceed49 frames and have the fixed second keyframe. No model-fork change.

Local checks are syntax, static schedule/call-path inspection and whitespace only.
Runtime success requires the CAMP run and archive review. Stop on the first
failure; no automatic rerun or full successor.

## User-operated CAMP execution

Publish changes and refresh the parent checkout inside an allocation first.
Then, from `/mnt/projects/gr/3DRecon/layer_good`, submit:

```bash
sbatch tools/kvt_tum.sbatch office-segments
```

Use the scheduler default time limit, per user preference. The wrapper uses the
fixed checkout path (safe under Slurm spool execution) and existing kvt image.
Expected marker: `OFFICE SEGMENTS COMPLETE: 6 independent segments, 2585 frames`,
then `JOB OK` and archived exit0. These are execution gates, not accuracy claims.

Under `/mnt/projects/gr/3DRecon/kvt_tum_out/`, outputs are
`tum_<JOB>_office_<START>_<LAST>.tar` and matching `_inputs.tar` for all six
segments, plus existing context/all_runs archives. The context includes summary,
protocol and archive hashes. Log: `/mnt/projects/gr/3DRecon/kvt_tum_slurm-<JOB>.log`.
Transfer with the field-notes rsync route; preserve each archive separately.
