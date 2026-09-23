# Long-sequence correspondence KV comparison

**2026-09-23 full-run scope supersedes the four/three-condition full stages below.**
The user selected geometric and semantic correspondence only for all three
ARCTIC objects, and geometric correspondence only for all three TUM sequences.
SAM3/TUM semantics are deferred. Full mode no longer runs dense FIFO or uniform
controls. ARCTIC checks semantic-run masks against the geometric run; gate mode
keeps its original controls and identity/prefix checks. Original implementations,
not these custom controls, are the requested baselines; reuse compatible original
artifacts from jobs 25632/25680 after a provenance/evaluation check. New method
outputs alone do not establish improvement over original. See
`STREAM_CACHE_CORRESPONDENCE.md` for the single all-four-model launch and its
evidence limits. The older scene-by-scene full-run instructions are superseded by
the user's explicit all-data authorization.

Updated 2026-09-22 after the user's scope correction: test **complete sequences**
on both the object and scene tasks. The previously prepared 160-frame ARCTIC pilot
did not exercise the cache budget and is superseded; it was never submitted. Do
not use `a64f3ec`'s wrapper. No CoMe-style token merging is involved. Novelty and
accuracy remain unverified.

## Object task: complete ARCTIC sequences

Use KV-Tracker at 518 on the three prepared S01 object sequences, 730 / 652 / 661
tracked frames (box / ketchup / espresso). Inputs and ground truth follow
`KVT_ARCTIC.md`. The reviewed initial mask and native online SAM mask propagation
remain. All variants insert at frames 0, 30, 60, ... into a budget of eight
simultaneously retained keyframes. The anchor stays; on overflow, evict the oldest
non-anchor keyframe and continue inserting. On box, the first eviction is frame
240, so the 241-frame gate includes one replacement and the full run covers many. This fixed
policy isolates patch selection from keyframe admission and eviction.

Four conditions share the schedule: dense K/V, spatial uniform half patches,
geometric correspondence half patches, and target-aware correspondence half
patches. The anchor and five special tokens per frame stay complete. Every other
frame retains exactly ceil(N/2) ordinary patches. Previously removed patches do
not reappear after a dense rebuild. Dense eight-frame FIFO is the matched accuracy
baseline; the earlier adaptive-keyframe ARCTIC results are contextual only.

The preferred first submission is the **combined gate**:
`sbatch tools/kvt_tum.sbatch correspondence-gates`. It starts the existing image
once and runs both gate protocols sequentially in one container. The ARCTIC and
TUM reports and run archives remain separate. `both_gate_processes.json` records
both stage return codes; a failure in ARCTIC does not skip the TUM gate. The job
does not start either full-sequence evaluation. The standalone ARCTIC gate,
`sbatch tools/kvt_correspondence.sbatch gate box_grab_01`, remains available for
an isolated rerun. `bash tools/kvt_tum.sbatch correspondence-gates arctic-only`
inside a Slurm allocation uses one container session for just the ARCTIC gate.

The ARCTIC gate runs remote CPU tests, a
200-frame native interval control, and all four conditions at 200 and 241 frames.
Dense must reproduce native poses exactly before the first eviction. Each longer
condition must reproduce its own first 200 poses and cache choices and show the
expected frame-240 eviction. The job stops after archiving its gate results. Review
the result before submitting `sbatch tools/kvt_correspondence.sbatch full
box_grab_01`. The full stage evaluates all four conditions on every frame of box;
after reviewing that run, use the same full command for ketchup and espresso.
The `full` stage does not chain from the gate inside one submission.

The object runner verifies insertion IDs, retained frame IDs, selected patch
counts, SAM mask identity, correspondence links, finite trajectories and artifact
archives. `gates.json` records whether geometric matches survive, whether matched
target patches survive in the semantic arm, and whether the two arms actually
make different choices. Inactive mechanisms invalidate the scientific comparison
even if all structural checks pass. `review.json` and pose-error NPZs report ATE,
native translation and rotation-part RPE, added angular rotation RPE in degrees,
99th-percentile pose errors, actual cache/metadata bytes, peak allocated/reserved
GPU memory, measured query-forward times and synchronous total tracking time.
The reported ATE/RPE use full-trajectory Sim(3) alignment; they do not establish
fixed-gauge metric object pose. Native `rpe_rot` is a dimensionless norm, while
`rpe_rotation_deg` is an angular metric.

The target-aware variant uses the existing SAM mask within an already object-
masked tracker. It tests target-guided cache retention, not segmentation or
multi-object identity. Non-target pixels in this mode are masked context; do not
describe them as unmasked background information.

## Scene task: complete long TUM sequences

Use the existing TUM input/evaluation path at 308, interval 50, budget 20,
anchor plus FIFO replacement. Scene masks are all true. Only dense, spatial
uniform half, and geometric correspondence half are meaningful here; no semantic
result can be claimed without object or entity labels. The scene task tests
transfer of the geometric selector, not the target-aware variant.

The combined gate runs the TUM native/dense 128-frame identity
check and 1100/1150-frame matched prefix checks across several replacements.
The isolated command is `sbatch tools/kvt_tum.sbatch correspondence
freiburg3_long_office_household gate`.
Review its complete evidence before `sbatch tools/kvt_tum.sbatch correspondence
freiburg3_long_office_household full`, which evaluates all three policies over all
2585 RGB frames. The two industrial sequences can be run later with the same
mode after reviewing office. GT is sparse on those scenes, so report coverage,
ATE and both local RPE measures separately; never pool all three into one score.
TUM runs use the existing `kvt_tum_out` archive, failure and GPU sampling workflow.

## Selector and measured mechanism

Capture arrival-time encoder descriptors only at fixed insertion events. Native
dense rebuilds provide pointmaps for all surviving frames in one current coordinate
system. Match the new frame to *surviving historical patches* by reciprocal cosine
nearest neighbour within twice the median positive adjacent-patch 3D spacing, at
a fixed cosine floor of 0.9. These are tentative model-derived correspondences,
not independently validated feature tracks. A zero geometric spacing yields no
matches. The scene and object implementations share this rule; TUM uses its native
phase (first insertion 49) while ARCTIC starts at frame 30.

Each frame gets 4x4 spatial quotas with exact largest-remainder allocation.
Matching distinct surviving tracks have priority inside a cell, with uniform fill
for the remaining quota. On ARCTIC, the target-aware variant also enforces
target/non-target label agreement and splits each cell's quota by majority-mask
label. It changes both matching and quota allocation, so any improvement needs a
later ablation to attribute those effects separately. K/V entries are gathered
after the existing dense rebuild; no feature averaging, extra model, quantization,
pose smoothing or token merging is introduced.

The model code is in the KV-Tracker fork's `kv_tracker/correspondence_cache.py`.
`main.py` passes the already computed pointmaps and masks to the policy. Existing
combined-cache calls accept these parameters without changing their selection.
The CPU contract suite checks geometry rejection, mask disambiguation, reciprocal
matches, exact spatial quotas, sparse K/V gather against dense masked attention,
retention through insertion, FIFO replacement and TUM's insertion phase. These
tests must run remotely in the existing image; Mac checks are static only.

The screening target for full sequences is no more than 5% degradation in ATE and
both RPE measures against matched dense, with no material increase in large-error
tails, plus lower *measured* memory and end-to-end cost. This is a proposed screen,
not proof of statistical equivalence. A shortened prefix cannot establish it.
KV-Tracker's native dense rebuild sets peak memory, so smaller persistent query
K/V may not lower peak GPU use or total time. Reports separate allocator peaks,
query-forward time, selector CPU time and total wall time. Dense-rebuild geometry
exports do not establish a sparse-cache geometry gain.

## Publication, execution and provenance

Publish the fork revision first, then the parent gitlink and tracked tools. Both
wrappers pull only within Slurm allocations. `kvt.tar` and existing model weights
are borrowed read-only. The user submits and transfers results; no cluster job is
submitted by the editing assistant. Use the field notes' exact rsync command from
the Mac repo root and extract every archive separately. Never advance to a full
evaluation behind an unreviewed gate.

ARCTIC writes per-condition archives and job-owned `inputs`, `review` TARs under
`/mnt/projects/gr/3DRecon/kvt_arctic_out/`. The combined job's shared context
TAR is under `kvt_tum_out/`; the standalone ARCTIC job writes its context under
`kvt_arctic_out/`. Context records source
revisions/diffs, Pi3 source, input hash and exit status; full result archives
contain masks, trajectories, cache events, final geometry and configs. TUM uses
the existing `kvt_tum_out` context/input/run archives, including exact source,
timestamp association and evaluation arrays. Failed stages preserve partial
results. These paths are expected outputs, not evidence of existence until the
user runs and transfers the jobs.
