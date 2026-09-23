# Offline TUM keyframe selection

Use only the files in this packet. Do not read the repository, prior conversations,
ground truth, depth, estimated trajectories, tracking scores or other run outputs.
Do not run inference, install software or call a paid API. This is an interactive
subscription-based visual reference experiment.

Select keyframes for RGB-only camera tracking and static-scene reconstruction in
TUM `freiburg3_long_office_household`. You may inspect the whole clip, including
future frames, and revisit images before choosing. There are {{N}} tracked frames,
indexed from 0. Frame 0 is permanently included. Choose exactly 19 additional
distinct indices, excluding 0: 20 total keyframes.

First read `manifest.json` and `ATTRIBUTION.txt`. Inspect every numbered contact
sheet in chronological order, using the individual PNGs to resolve unclear views.
The manifest maps each index to a timestamp, image and sheet. Maintain a coverage
record of inspected sheets. Do not finalize until every sheet has been inspected;
if a session or image limit prevents this, save partial observations and report
incomplete coverage. Never infer that an image was inspected from its filename.

Prefer clear, complementary views of static scene structure with useful visual
overlap, temporal coverage and views useful when revisiting places. Consider blur,
texture, occlusion, repeated-looking areas, viewpoint change and redundant views.
Explain uncertainty without inventing camera poses or geometry. This is scene
tracking: no single foreground object is the target.

Return `selection.json` containing `scene`, `selected_indices` (19 sorted unique
integers), `reasons` (one index and brief visual reason per selection),
`inspected_sheets` (all inspected sheet paths), and `limitations`. Do not request
evaluation scores or revise the choices in response to scores. Also preserve a
brief inspection log, model label, date and any human interventions. Do not claim
that visually plausible selections establish better tracking; evaluation follows.
