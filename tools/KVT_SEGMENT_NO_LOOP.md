# Automatic segmentation and reanchor: no-loop transfer

Prepared 2026-10-04 at user request. Scope is the already-tested
`freiburg2_large_no_loop`, all 3359 RGB frames, two fresh processes:
native cap20/default admission and live ORB minimum25/max200 reanchor.
This is office search trial25 transferred unchanged, not a new threshold sweep.
No manual boundaries, query acceleration or keyframe-prefix policy is combined.

Both arms use identical resized pixels, seed, checkpoint and GT associations.
Reanchor retains anchor+latest, refreshes every50, deletes old history at cuts,
and uses the existing pose-anchored pointmap-scale connection with delayed output.
Detector cost is included in its synchronous tracking time. Existing detector,
handoff and reanchor contracts run before evaluation; no pilot auto-queues a
follow-on evaluation. Failed contracts or comparisons stop the wrapper.

Report native ratios for ATE/RPE/p99, FPS, phase costs, cache bytes, peak GPU
memory, live cuts, bridge evidence and raw trajectories. The inherited 5% checks
are descriptive; success of execution is distinct from an accuracy verdict.
Windows retain each arm's full-valid-trajectory Sim(3); absent GT gives null
metrics and zero counts. Contiguous GT spans are saved. Historical GT covers
only654 frames; the middle route is unscored. No whole-route accuracy claim.
Historical native metrics are in FINDINGS.md; its job/context transfer remains
pending. Fresh native supplies the matched control, not assumed historical
reproduction. Review historical context and trajectories before claiming exact
reproduction of that run. One ordered timing measurement per arm.

## CAMP procedure

Local preparation only; not published or runtime-verified yet. Publish the parent
revision first. From head in `/mnt/projects/gr/3DRecon/layer_good`, after publication:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector no-loop"
O=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

Pull and execution happen inside the allocation. Existing kvt.tar, reviewed
26127 input/context archives and Pi3 snapshot are verified and borrowed; the
no-loop ZIP is read from `/mnt/datasets/tum-rgbd`. No model/dependency downloads.
Outputs in `/mnt/projects/gr/3DRecon/kvt_tum_out/`:
`tum_JOB_segment_no_loop_inputs.tar`, `tum_JOB_no_loop_native.tar`,
`tum_JOB_no_loop_reanchor.tar`, `tum_JOB_segment_no_loop_comparison.tar`,
and shared context/all_runs archives. Inputs are archived separately.
Completion requires `SEGMENT NO LOOP COMPLETE` and wrapper `JOB OK`;
review comparison.json and saved artifacts before advancing any experiment.

## With-loop transfer (2026-10-04)

User selected the identical comparison on freiburg2_large_with_loop, all5182
RGB frames, after reviewing no-loop26198 timing/metrics. Stage `with-loop`
reuses the same driver with an explicit scene argument. Fresh native and fixed
ORB25/max200 reanchor; no retuning or additional arms. No model edits.
Existing contracts/checkpoint/pixel checks and matched GT comparisons apply.
Historical GT coverage1268/5182 is partial; actual spans/counts are saved and
missing windows remain null. One observation per arm cannot establish statistical
speed or loop causality across videos with different content and trajectories.

In the submission chunk above, replace its W extension with:

```bash
W="$W && bash tools/kvt_tum.sbatch segment-detector with-loop"
```

Outputs: tum_JOB_segment_with_loop_inputs.tar, tum_JOB_with_loop_native.tar,
tum_JOB_with_loop_reanchor.tar, tum_JOB_segment_with_loop_comparison.tar,
plus context/all_runs. Completion requires `SEGMENT WITH LOOP COMPLETE` and
`JOB OK`; accuracy/FPS/memory interpretation requires comparison/artifact review.
No-loop default and its output naming are preserved. Local syntax checks only;
with-loop staging/contracts/full evaluation remain pending user submission.
