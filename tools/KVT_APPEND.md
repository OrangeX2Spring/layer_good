# A1: append-only keyframes in KV-Tracker

Prepared locally 2026-09-25, at the user's request. KV-Tracker only, TUM office,
scene/camera mode, resolution 308, seed 0, identical stock insertion IDs. This
implements the first A1 arm; no ARCTIC adaptation, partial recomputation, periodic
rebuild, eviction, token pruning or additional host is included.

## What changes

`kv_tracker/kv_tracker/append_cache.py` requests `ret_kv=True` on selected cached
query calls to Pi3's global decoder blocks, then restores the hidden-state-only
return expected by Pi3. Native `forward_w_cache` returns the concatenation of
old K/V and the query's own normalized, post-RoPE K/V. Clone only the new tail;
otherwise a slice would keep a second whole cache alive. Once the fixed selector
accepts the frame, append that tail to each persistent layer. There is no extra
encoder or decoder forward. Nonselected queries keep the native path.

The tracker records the arrival-time keyframe pose and skips the entire joint
rebuild branch. Existing K/V and keyframe poses stay unchanged. The implementation
lives in the KV-Tracker fork; Pi3's nested submodule/source need not change.
`tools/kvt_tum_run.py` wires the optional controller only for fixed-ID replay.
The default remains stock rebuilding. Both comparison arms disable dense snapshot
export; they produce trajectories and evaluation artifacts, not an updated dense map.

**Correction to the survey's mechanism claim:** `kvt_tum_run.py` does not pass
`--sim3`. The optional Umeyama refit in `main.py` therefore did not execute in the
TUM runs. Stock rebuilds change the predicted first-camera normalization
(`origin_offset`, in `pi3_utilts.pi3_inference`) and all cached K/V. Append freezes
that bootstrap transform and `scene_origin`; optional Sim(3) is disallowed for A1.
Offline evaluation still fits one full-trajectory Sim(3) in both arms. These are
different operations. The cause of the observed jumps is not established by code
inspection alone.

**Bootstrap accounting:** native initialization encodes `[0, 0]`. Preserve both
physical slots in the append arm to keep the pre-insertion trajectory identical
to stock and avoid deleting old K/V at the first insertion. Thus full append has
20 unique keyframes but 21 physical slots, versus stock's 20 after its first
rebuild. Keyframe ID/pose files contain unique frames; `append_events.json` records
physical slots. This is not a byte-matched cache comparison. The duplicate can
affect attention, so do not attribute every accuracy change solely to rebuild
removal. No duplicate-removal ablation is included.

## Gates and measurements

The default `append pilot` stage runs the first 256 office frames in fresh
processes: native stock, fixed-ID stock replay, and fixed-ID append. Insertion IDs
are 49, 99, 149, 199, 249, plus bootstrap 0. The native arm checks that replay is
still faithful on this fork; it is not an extra experimental policy.

Before the pilot, run the existing TUM dependency/contract tests and
`test_kvt_append.py`. The latter uses actual Pi3 blocks for capture fidelity,
tail identity and storage size, repeated prefix preservation, nonselected-query
immutability, incomplete-capture rejection and the insertion metric's index phase.
These tests execute on CAMP only.

Pilot correctness requires:

- Native stock and stock replay trajectories agree (`atol=rtol=1e-4`).
- Append agrees with stock through frame 49 inclusive, whose prediction precedes
  the first cache update. Every condition saves the same requested keyframe IDs.
- Every append preserves old K/V and old poses exactly. Final saved keyframe
  poses equal the corresponding arrival-time trajectory poses exactly.
- One bootstrap, one query per subsequent frame, zero append-arm rebuilds,
  expected append count, and identical bootstrap normalization in every event.

`pilot_gate.json` is a correctness gate, **not** an accuracy pass. Review it,
`comparison.json`, source/container provenance, logs and exit status before
deciding whether to run full office. No full evaluation is automatically queued.
The optional full stage requires `--reviewed-pilot` pointing to the successful
pilot context tar. It checks the gate, job success, source-file hashes, container
archive hash and staged input hashes. Source changes require a new pilot.

Full stage has two arms only: stock replay versus append, IDs `[0,49,...,949]`,
20 unique keyframes, every office RGB frame. Expensive prefix equality assertions
are pilot-only; metadata/gauge checks remain in both stages.

Report ATE and full t-/rotation-RPE from the existing evaluator, fitted scale,
translation-step RMS at insertion crossings, insertion t-RPE, t-RPE excluding
insertion crossings, per-insertion update and query-plus-update times, full-run
time, peak allocated/reserved memory, and cache bytes/physical slots. An insertion
at frame k changes the pair k→k+1, not k−1→k. Use timestamp-valid adjacent pairs
with the existing 0.1 s maximum gap. Step RMS includes real motion; it is not a
GT-subtracted jitter error, which is why insertion RPE is also reported.
Per-frame wall times include bookkeeping and exports; inference/commit timings
are instrumented substeps. Pilot equality checks add overhead. Do not promise a
speed or memory win before execution; bootstrap, model loading, pending K/V tails
and allocator effects remain in the measured peak.

## Publication and user-operated CAMP execution

Fork implementation `0eea94f` is published on `codex/token-drop` (base `3330f54`).
The parent commit carrying this protocol pins that fork revision and publishes the
A1 tools; see `docs/STATUS.md` for its revision. The separate combined-cache worktree,
unrelated working-tree edits and prior A0 files are untouched. Python AST, bash
syntax and whitespace checks passed; runtime verification is pending. No container
rebuild is needed.

After publication, from CAMP head, submit only the pilot; both git pulls execute
inside the allocation:

```bash
sbatch -A students --qos=students_normal -p 24g -w muenchen \
  --chdir=/mnt/projects/gr/3DRecon/layer_good \
  --gres=gpu:1 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log \
  --wrap='git -c fetch.recurseSubmodules=false pull --ff-only &&
bash tools/kvt_tum.sbatch append pilot'
```

Expected evidence: existing tests and all A1 contracts pass, three `RUN OK`
records, `APPEND PILOT CONTRACT GATE OK`, final `JOB OK`, and archived exit status
0. The next gate is human review of those artifacts and the accuracy/resource
trade-off. No submission has occurred and no runtime success is claimed.

Planned archives, outside the checkout under `/mnt/projects/gr/3DRecon/kvt_tum_out/`:
`tum_JOBID_context.tar`, `tum_JOBID_all_runs.tar`,
`tum_JOBID_inputs_freiburg3_long_office_household.tar`, and
`tum_JOBID_freiburg3_long_office_household_{stock_native,stock_replay,append_only}.tar`.
The native arm exists only in the pilot. Context includes the comparison, gate,
protocol, exact source archives, hashes and test logs. Failed/partial results are
preserved by the existing exit trap. These paths are planned, not verified outputs.
