# KV-Tracker causal keyframe selection

Prepared 2026-09-17. **Locally syntax-checked only; cluster gates have not run.**
Entry point: `tools/kvt_arctic_online.sbatch`. Uses the existing `localhost/kvt`
image, ARCTIC inputs and reviewed masks. No new dependencies or weights.

## What changed

The replay batch (25627) selected schedules using a completed run. In particular,
arc25 used the final reconstructed centre, and controls used final baseline counts
and clip length. It is an offline diagnostic, not evidence of an online selector.

This batch calls an explicit `keyframe_selector` in `kv_tracker/main.py` at the
live insertion decision. The selector receives the current frame, current predicted
pose, current reconstructed object centre, and previously cached poses/frame IDs.
For the angular callback, the centre is computed from the current pointmap **after**
the same origin/Sim(3) transformations as the cached poses. Upstream computes its
own centre before the origin shift; its original decision is preserved for the
fidelity/capped-original conditions.
It never receives a sequence length, GT, saved trajectory, or final keyframe budget
from a different run. The original attention and cache-rebuild machinery is retained.
At each insertion, all observed keyframes are jointly re-encoded. This is causal
information use, not a change to triangular attention or token-level cache pruning.

## Fixed sweep

All three S01 sequences, offset 2, resolution 518, same reviewed SAM initialization.
All experimental policies have a **32 unique-keyframe cap including frame 0**.
At the cap they stop inserting; there is no eviction. This bounds the keyframe
cache, not all tracker memory (the original tracker still accumulates outputs).

| Policy | Parameters |
|---|---|
| Original online fidelity gate | Uncapped, unchanged geometric decision |
| Original online capped | Cap 32 |
| Live angular distance | 20, 25, 40 degrees |
| Fixed interval | Every 30 or 60 frames from frame 0 |
| Bernoulli random | Probability 1/30 on each arriving frame; seeds 0, 1, 2 |
| Semantic novelty | Thresholds 0.05, 0.10, 0.20 |

Thirteen full configurations, each paired with a 128-frame prefix run. These are
predeclared exploratory settings, shared across objects. No per-object tuning.
Random/interval policies use a fixed rate, **not exact matched final counts**.
Compare ATE together with actual count, cache bytes and measured cost; a common
maximum does not imply equal consumed budgets. Do not cherry-pick best thresholds
as a validated policy. Semantic selection runs immediately after the baseline so
interface faults stop early.

## Semantic feature contract

Inspected source: `Marwan99/Pi3_w_KV` at `27e96ce786447355bd7ad8b11df708a9deeafc35`,
`pi3/models/pi3.py`. SHA-256:
`cbcf68b3c05baab7680f6e24afda42501dc0ba799e85ca18668ba3e8a5812979`.
Runtime checks this source hash before attaching the hook.

`decoder[0]` returns `[B*N, 5 + H/14*W/14, 1024]` frame-local features before
`decoder[1]`, the first cross-frame/cached block. A read-only forward hook excludes
the five special tokens. Object patches have at least 50% mask coverage at the
actual model resolution. Masked mean pooling followed by L2 normalization produces
one vector per frame. Novelty is `1 - max(cosine(current, retained_keyframe))`.
Vectors for selected frames are retained from their arrival-time inference; they
are independent of cache contents. No additional encoder pass or model is used.

Bootstrap uses the first frame, which upstream duplicates internally. Later frames
with zero majority-object patches are logged and skipped by the semantic policy;
bootstrap requires at least one. This is an appearance-novelty proxy from pretrained
features, not a demonstrated semantic identity or re-identification mechanism.

## Gates and artifacts

The batch stops on the first failed command. First run the CPU selector contract
tests in the existing container, then check:

- Original full online ATE within 0.0005 m of 0.1825401 / 0.2943396 / 0.1502563.
- Every cache frame ID is earlier than the query; every decision agrees with saved
  `kf_idx.npy`. Logs have one decision per arriving frame after bootstrap.
- For each configuration, full-run prefix poses agree with its separate prefix run
  (`atol=rtol=1e-4`); selected/candidate/capped decisions and cache IDs agree exactly.
  Prefix masks and configuration agree exactly too. This is a suffix-availability
  regression gate, not proof by itself of every possible causal property.
- Full-run masks are byte-identical to the original online condition.

One TAR per prefix/full run under `kvt_arctic_out/arctic_online_<jobid>_*.tar`:
manifest/config, metrics, trajectory, saved keyframes, masks and per-frame
`decisions.jsonl` (angles/novelty/object patch counts, original decision, candidate,
cap status, selection, past frame IDs and actual cache bytes before the decision).
Metrics include actual final keyframe count, peak PyTorch allocated bytes, and
synchronous wall time including model setup/export. These are **not pure model FPS**;
allocator peak is **not total GPU/process memory**. ATE is the upstream full-trajectory
Sim(3)-aligned diagnostic, not metric online 6D pose error under a fixed initial gauge.
Prefix ATE is a gate artifact, not an accuracy result.

Shared exact source inputs: `online_<jobid>_inputs.tar` (copy of prepared.tar).
Source snapshot, input checksum and exit status: `online_<jobid>_context.tar`.
On failure, the EXIT trap also stages remaining partial run directories as one TAR.
The Slurm log is persistent. Finished run directories are removed from /tmp only
after their archives and gates pass; the baseline remains for mask comparisons.
Sync back the input/context archives along with results and the Slurm log.

## Deployment

Publish the tracker submodule change **first**, then commit/push the parent pin and
these scripts. The batch pulls the parent and updates only `kv_tracker` inside its
allocation, using HTTPS for the submodule (outbound SSH on CAMP is blocked).
It reuses the already initialized Pi3/SAM submodules. No edits are made remotely.

If the new batch file is already present: `sbatch tools/kvt_arctic_online.sbatch`.
Otherwise submit a wrapper with the same allocation options which pulls the parent
inside the allocation and then executes this script; avoid a fetch on head.
