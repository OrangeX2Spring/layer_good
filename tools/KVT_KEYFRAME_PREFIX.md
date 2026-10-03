# Frozen early banks and continued native admission

User-selected ablation of office job 25680's frozen bank. The observation that
native retains only frames through 949 does not establish that later keyframes
are unnecessary. This experiment tests early-bank sufficiency and the benefit
of continuing native admission; no segmentation, reanchoring, graph execution,
query token reduction, or GT-driven selection is enabled.

All seven arms process the same 2,585 frames, Pi3 checkpoint, exact model RGBs,
seed, native query/rebuild path, and 50-frame insertion schedule. Each arm is a
fresh process. Stop values below are exclusive source-frame indices, not a
resampling rate or a count of retained frames.

| Arm | Admission range | Retained keyframes | Last keyframe |
|---|---|---:|---:|
| native | Native cap of 20 | 20 | 949 |
| first200 | 0–199 | 5 | 199 |
| first500 | 0–499 | 11 | 499 |
| first750 | 0–749 | 16 | 749 |
| through1200 | 0–1199 | 25 | 1199 |
| through1600 | 0–1599 | 33 | 1599 |
| through2000 | 0–1999 | 41 | 1999 |

The larger arms retain every eligible native keyframe until their stop, without
the original 20-frame cap blocking admission. After the stop each bank freezes
for the rest of the sequence. The new model parameter defaults to 20; existing
experiments preserve their previous behavior. No old keyframes are pruned.

## Evidence and interpretation

Runtime gates assert exact admission/rebuild schedules, past-only cached IDs,
all 2,585 poses and 2,584 query calls, matched GT associations, identical environment,
and fresh-native reproduction of the original metrics. Prefix poses must match
native before the first changed cache can affect a query: first 200/500/750
frames respectively, and first 1,000 for extended banks. The CAMP handoff
contracts exercise the freeze/cap scheduling before these runs.

Report full-run ATE/RPE/p99, loop FPS, query/rebuild costs, peak allocated and
reserved GPU memory, and cached storage bytes. Window tables include 0–999,
1000–2499, 1025–1375 and 1025–2025, retaining each arm's one full-run Sim(3)
alignment. There is no independent window fit. Also save each fitted scale and
raw trajectory; global alignment is offline scoring, not tracker input.

This is one sequence with one ordered timing observation per arm. Keyframe
count, visual coverage and reconstruction gauge change together. Similar early
bank accuracy would demonstrate sufficiency here, not prove cached history is
unused. An improvement from continued admission would refute the claim that
later additions are unnecessary on this sequence. No automatic winner gate or
follow-on method is dispatched.

## Memory failures

Each arm's CUDA OOM prints its traceback, writes `oom.json` with peak allocated
and reserved bytes, and saves `partial_traj.npy` if tracking had started. Exit
75 is accepted only with that marker. The controller archives the arm, records
its OOM status, and continues after its process exits and releases GPU memory.
A native-baseline OOM stops the experiment. Any other process error or failed
contract stops loudly. A signal/Killed is not automatically labeled CUDA OOM;
inspect Slurm accounting if that occurs. No reduced bank or silent fallback is
substituted, and OOM arms receive no accuracy/FPS verdict.

## CAMP submission

After both fork and parent revisions are published, on head in
`/mnt/projects/gr/3DRecon/layer_good`:

```bash
W='git -c fetch.recurseSubmodules=0 pull --ff-only'
W="$W && bash tools/kvt_tum.sbatch segment-detector prefix-ablation"
O=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

Pull, checks and execution occur inside the allocation. Existing `kvt.tar` and
reviewed `tum_26127_oracle_maps_inputs.tar` / `tum_26127_context.tar` are borrowed
read-only; no dependency installs or downloads. Results persist in
`/mnt/projects/gr/3DRecon/kvt_tum_out/tum_JOB_keyframe_prefix_{ARM}.tar`, plus
`keyframe_prefix_comparison.tar`, `context.tar` and `all_runs.tar`. Each arm
contains its config, trace, raw trajectory and evaluation, or explicit failure
evidence. Completion marker: `KEYFRAME PREFIX ABLATION COMPLETE: 7 arms`,
followed by wrapper `JOB OK`; inspect comparison.json for any OOM.

Local preparation is syntax-checked only. Full runtime validation is pending.
