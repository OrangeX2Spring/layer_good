# Streaming-host correspondence gates

The user selected the KV-Tracker correspondence comparison for STream3R,
StreamVGGT and LongStream on both long-sequence tasks. This is a structural
gate, not a full accuracy result. The ARCTIC box gate uses 121 consecutive
post-offset frames from `kvt_arctic_out/prepared.tar`, with the exact saved SAM
masks from `kvt_arctic_out/arctic_r518_20260915T152911Z.tar`. Its four matched
conditions are dense, spatial-uniform half, geometric correspondence half and
object-mask-aware correspondence half. Input RGB is object-masked before the
common width-308 resize, as in the KV-Tracker object comparison. The TUM
long-office gate uses the first 128 consecutive frames and three conditions:
dense, spatial-uniform half and geometric correspondence half. TUM has no
object masks, so it cannot establish a semantic result.

The user halved these gate lengths on 2026-09-23 to reduce waiting. Both still
cross FIFO eviction at budget eight and LongStream's 24-frame refresh multiple
times. Native fidelity prefixes and all gate assertions remain unchanged.

All conditions keep the first frame dense, retain at most eight frames with
the same FIFO schedule, and preserve special tokens and the hosts' native head
caches. The geometric selector matches normalized encoder patches with
reciprocal cosine similarity at least 0.9 and 3D distance within twice the
median adjacent-point spacing. It balances selected patches over a 4-by-4
image grid and favors distinct surviving correspondence tracks. The semantic
variant requires matching target/background labels and balances within each
spatial cell and label. It does not merge tokens or change model weights.

Unlike KV-Tracker's interval-30 object and interval-50 TUM insertion, these
models admit each streaming frame; the controls have the same schedule within
each host. LongStream's native 24-frame refresh resets correspondence history,
so its test is within each segment, not long-lived retention. StreamVGGT's
camera-head cache continues growing; smaller aggregator KV does not establish
bounded total memory. The ARCTIC gate records selection and memory but does
not yet score object pose. TUM camera ATE/RPE is scored where GT is valid.

`tools/stream_cache_long.sbatch` mode `correspondence-gates` runs the six
host/task pairs sequentially in one allocation, preserving one archive per
pairs under `stream_cache_out/`. All three hosts use the previously verified
`optpose.tar` base image. Each pair starts a fresh disposable `podman run`, and
`stream_cache_job.py` installs that host's missing inference wheels inside that
container without changing the saved image or another pair's environment.
The image is loaded once into the allocation's job-local Podman store.
A failure is recorded and does not skip later
pairs; the overall job exits nonzero if any gate fails. Each pair runs its native
fidelity oracle, contract tests and all conditions in isolated processes. The
gate checks fixed retention, half-patch counts including LongStream refresh,
FIFO activity, actual matches, distinct geometric choices, object matches and
semantic choice differences on ARCTIC, plus smaller aggregator payload. A gate
failure is diagnostic evidence, not a reason to auto-submit full runs.

After publishing the parent revision, submit from the CAMP head node. The
checkout pull occurs inside the allocation. Use `--array=0` to override the
wrapper's normal three-scene array. The single `correspondence-gates` positional
argument selects all three hosts internally; do not split the host list and
stage across shell lines. Job 25826's supplied log ran the wrapper's default
`full` stage, and `scontrol` showed the stage argument on a new line; its
2,585-frame `pooled_encoder_*` workers are not correspondence-gate evidence.

The wrapper now refuses a comma-separated host list without an explicit stage
and prints `WRAPPER stage=correspondence-gates hosts=...` before any preparation.
This prevents an omitted stage from silently starting the multi-host full sweep.

```bash
CORR_WRAP='git -c fetch.recurseSubmodules=false pull --ff-only'
CORR_WRAP="$CORR_WRAP && bash tools/stream_cache_long.sbatch correspondence-gates"
sbatch -A students --qos=students_normal -p 24g -w muenchen \
  --gres=gpu:1 --propagate=NONE --array=0 \
  --chdir=/mnt/projects/gr/3DRecon/layer_good \
  -o /mnt/projects/gr/3DRecon/stream_corr_gates-%j.log \
  --wrap="$CORR_WRAP"
```

Check `sacct` completion and the six archives' `exit_status.txt`, `run/fidelity.json`,
`run/correspondence_gate.json`, events, camera metrics and provenance before
selecting any complete-sequence run. Log viewing uses `tail -n 50 -f`.
