# A trainable token policy for KV-Tracker

Research date: 2026-09-30. User selected implementation after this review. The
scene pilot is locally implemented below; CAMP runtime and scientific benefit
remain unverified. No cluster job has been submitted.

## Recommendation and success criterion

Train the existing small scorer against **the trajectory produced by actual hard
token selection**, using fixed-budget sampling and a self-critical policy gradient.
Keep Pi3 frozen. Use dense causal tracker outputs as training targets, preserving
the original no-GT training preference. Use held-out GT only for development and
final evaluation. Do not generate more gradient-importance labels.

Research completion means identifying a mathematically valid gradient estimator,
an integration point compatible with this fork, a bounded experiment and its
failure criteria. Scientific success would require a deployed deterministic
policy to improve held-out tracking at the fixed budget and meet the original
accuracy, time and memory goals. That remains untested.

This is an adaptation of established techniques, not a claim of a new algorithm
or a published KV-Tracker result. The main risk is high-variance learning with a
large discrete action space. No paper establishes that this proposal will work
on our sequences within our compute budget.

## What the existing result establishes

The independently reviewed 26072 result rejects the selected first-order
sensitivity teacher under its coverage-matched allocation. It does not search
all possible token subsets or optimize their downstream trajectory effects.
The earlier tiny ranking-training smoke also does not establish whether a
task-trained policy generalizes. See the canonical
[FINDINGS](FINDINGS.md#teacher26072--independent-archive-closure-review-2026-09-30)
and `cluster_results/atd_review_26072/raw_review.json` for evidence and limits.

A derivative of a dense, post-encoder gate is not the same intervention as
deleting input tokens before the encoder, changing attention normalization and
reusing a sparse cache over later frames. That mismatch is a reason to change
the objective; it is not proof that the mismatch caused every observed failure.

## Primary-source comparison

| Work | What it contributes | Why it is not a drop-in solution |
|---|---|---|
| [DORA, September 2026 preprint](https://arxiv.org/html/2609.34325v1) | A learned actor selects hard deletions with a frozen ViT, ordered Plackett–Luce sampling and downstream rewards. | Image classification, layer-wise pruning, PPO and a privileged critic; no persistent tracking cache. Very recent; no author implementation identified in this review. Borrow the discrete-policy formulation, not its whole system or performance claims. |
| [Self-critical sequence training, CVPR 2017](https://arxiv.org/abs/1612.00563) | Uses the deterministic deployment output as a reward baseline for training sampled sequences. | Demonstrated for captioning. Here the sequence is token-selection actions and the score is tracking fidelity. This adaptation avoids adding a critic but can still have high variance. |
| [Gumbel-Top-k, ICML 2019](https://proceedings.mlr.press/v97/kool19a.html) | Exact sampling without replacement from scored candidates. | Supplies the sampler, not the task objective. Retain sampled order for likelihood; sorting survivors into spatial order is a separate execution step. |
| [PaceVGGT, May 2026](https://arxiv.org/html/2605.08371v1) | Frozen geometry backbone with trained scoring, downstream task refinement, merging and restoration. | Selects after the full DINO encoder. Hard top-k alone does not transmit task gradients; score-weighted merging provides a differentiable path. Our original-token policy has no such merge path. |
| [Speed3R, March 2026](https://arxiv.org/html/2603.08055v1), [author code](https://github.com/Visual-AI/speed3r) | Trains sparse geometry models, including Pi3, with dense-model relative-pose and depth distillation. | Changes attention architecture and trains model weights. Its reported eight-GPU, approximately seven-day training is not a small frozen-backbone policy recipe. Borrow output supervision, not its architecture. |
| [DynamicViT, NeurIPS 2021](https://arxiv.org/abs/2106.02034), [author code](https://github.com/raoyongming/DynamicViT) | Task-trained token predictors with differentiable attention masking during training and hard removal at inference. | Joint backbone fine-tuning and training/inference operator differences. Retrofitting policy-aware masks throughout Pi3 is larger than training a scorer against the existing sparse operator. |
| [Differentiable patch selection, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Cordonnier_Differentiable_Patch_Selection_for_Image_Recognition_CVPR_2021_paper.html) | Differentiable selection through perturbed optimizers. | A useful gradient surrogate, but weighted training selections need not reproduce original discrete patch positions and sparse-cache behavior. Recognition results do not settle this geometry mismatch. |
| [Good Token Hunting, May 2026](https://arxiv.org/html/2605.23892v1), [author code](https://github.com/zsh2000/gotohunt) | Geometry-token reduction for VGGT/Pi3; highlights the importance of layer and token roles. | Training-free and uses multi-view information. Its whole-clip covisibility mechanism cannot be imported unchanged into a causal online tracker. |
| [StepPrune, June 2026](https://arxiv.org/html/2606.16067v1) | Models dependencies among selected tokens with a sequential selector. | MLLM method with a learned stop and a different training relaxation. It reinforces the concern that selecting many tokens independently is difficult; it does not validate our estimator's sample efficiency. |
| [Fin3R, November 2025](https://arxiv.org/html/2511.22429v1) | Geometry adaptation through encoder LoRA with a frozen decoder; feature renormalization addresses distribution shift. | Does not learn an original-token subset or preserve a completely frozen backbone. It is an alternative research direction, not evidence that our token policy already works. |

These results support the feasibility of learning under token reduction. Their
backbones, tasks, datasets and training budgets differ. No numerical speedup from
these papers is used to predict KV-Tracker performance.

## Exact proposed method

### Actor and action

Reuse candidate B's `PatchScorer` in
`kv_tracker/kv_tracker/atd_training.py`: 12 input channels, two 32-channel
convolutions, one score per patch; 12,769 trainable parameters by layer count.
Inputs are current RGB statistics/energy and five detached causal-map channels.
Initialize a new scorer; do not warm-start from the failed ranking checkpoint.
The inexpensive features may themselves limit performance; this pilot does not
assume they contain sufficient geometric information.

For scenes, retain the current budget `k = ceil(P/2)`. Sample `h = floor(k/2)`
distinct high-priority patches from the scorer logits using Gumbel-Top-k. Complete
the remaining budget with the existing spatial-spread filler. Preserve original
patch embeddings and RoPE indices; do not merge or regenerate tokens.

Deployment replaces sampling with stable top-h and calls the same filler:
the existing `scorer_keep` rule. Training and deployment use the same hard sparse
forward; only exploration differs. The fixed scene count supports the existing
graph query shape, but actor integration with graph capture is not yet verified.

The family extends to objects by protecting all SAM-object patches, sampling
half of the legal background budget, and spreading the remainder. Object data,
mask behavior and variable counts need separate validation. The first proposed
pilot is scene-only; it cannot establish object success.

### Output target and cost

Run the frozen dense **causal tracker**, once per training episode, to save
reference poses. This reference receives only the same current/past inputs; it
is not a future-conditioned multi-view reconstruction. Reference outputs are
training-only and never actor inputs.

For each sparse rollout, collect final tracker poses after its gauge transforms
and holds. Align its complete episode to the dense reference using one Sim(3),
then compute adjacent-frame relative pose errors. This follows the type of
alignment in `tools/kvt_tum_run.py:evaluate`, but uses dense outputs rather than
GT as the target. Whole-episode alignment is permitted in an offline reward;
it must never be used to construct online actor features or state. It removes
absolute scale/global gauge error, so this objective does not promise metric
scale recovery. Short degenerate-motion episodes must be excluded in the input
manifest rather than silently assigned a reward.

Define each pair cost as squared translation error divided by `z_ref^2`, plus
squared rotation angle in radians. `z_ref` is the median positive dense depth
from the episode's bootstrap, saved with the reference and held fixed across
all masks. Define episode cost C as mean pair cost plus the mean of its worst
10% of pairs. Pairs across RGB gaps over 0.1 s are excluded, as in the evaluator.
The tail term discourages an apparent average improvement that introduces
occasional large tracking errors. A motion-collapsed sparse trajectory whose
Sim(3) is undefined receives an explicit cost of 1e6; a degenerate dense reference
fails input eligibility. These are fixed choices, not tuned or verified
hyperparameters.

Dense-output agreement can inherit the dense model's errors. It is a training
surrogate; actual held-out GT quality remains mandatory. Matching the dense
reference more closely is not sufficient to call the method successful.

### Gradient through real deletion

Let `i_1,...,i_h` be the ordered sampled patches and `s` the actor logits.
The ordered probability is

```text
log p(i_1,...,i_h | state)
  = sum_j [s[i_j] - logsumexp(s[not previously sampled])].
```

Run two independent sparse trackers from the same initial episode state:
one sampled and one deterministic greedy, each with its own evolving cache.
With costs `C_sample` and `C_greedy`, minimize

```text
L_actor = stop_gradient(C_sample - C_greedy)
          * sum_over_decisions(log p(ordered_sample | causal_state)).
```

This is the self-critical score-function estimator. A better sampled trajectory
has negative cost advantage and its selections become more probable under
gradient descent. The greedy baseline is independent of the sampled actions;
detach it even though it uses the same actor. Multiple ordered samples can map
to the same final keep set; the estimator remains valid for this action process.

There is no backpropagation through Pi3, SVD, cache rebuilds or hard gather.
Scorer gradients come from its log probabilities, while the real tracker provides
the cost. This directly avoids the dense-gate/sparse-deletion surrogate mismatch.
It does not avoid the sample complexity of optimizing a discrete policy.

### Causal state and integration contracts

1. Retain first-arrival patch decisions by physical frame. Cache rebuilds reuse
   those decisions; the bootstrap duplicate is one decision, not two samples.
2. Construct each rollout's maps from its own past sparse cache and poses.
   Exclude dropped points explicitly: sparse point heads scatter zeros and
   invalid confidence, which must not become map evidence. Keep points and
   projection poses in the same gauge. Unknown support stays explicit.
3. Initially replay the native admission schedule to isolate token learning,
   retaining the student's own cache contents. A later full evaluation must
   use natural admissions; replay success does not establish deployment success.
4. `pi3_utilts.py` currently wraps policy execution in `no_grad`. Enable gradients
   locally for the scorer/log probabilities only; leave all Pi3 forwards and
   map construction detached. Store scorer graphs/log probabilities, not Pi3
   activation graphs or gradient-bearing KV tensors.
5. Charge feature extraction, map construction, scoring, selection and transfers
   to deployment timing. Compare graph sparse with graph all-kept, as well as
   native, so graph acceleration is not attributed to token learning.

Model changes belong in the tracker fork: sampler/log probability, online scorer
hook and live sparse-map state. Parent `tools/` owns episode manifests, the
training driver, limits, evaluation and provenance. The user subsequently selected
this proposal and the scene pilot is locally implemented in
`kv_tracker/kv_tracker/task_token_policy.py`, `tools/kvt_task_policy.py` and
`tools/test_kvt_task_policy.py`. Existing failed-teacher commands remain superseded.

## One bounded pilot, not another oracle sweep

Spending limit: **one GPU allocation capped at four hours**. The workload deadline
leaves ten minutes for archival, and training reserves measured development time
plus at least five minutes for final evaluation. This is a spending cap, not a
measured estimate of sufficient training. Do not queue a follow-on.

- Fix one scene budget, one scorer B, one objective and one training seed.
  Use unused static TUM training sequences from the existing dataset inventory.
  Archive contents and physical-scene grouping still need verification; the
  inventory only established file presence. Exclude the three final evaluation
  clips and any overlapping scene family. Freeze a separate development manifest.
- Use 128-frame episodes with the current admission schedule so training includes
  repeated cache refreshes. Each episode costs one cached dense reference plus
  sampled and greedy sparse rollouts. Cap at 256 episode updates as well as the
  wall deadline. Benchmark the first episodes and report actual throughput;
  do not convert diagnostic oracle timings into a training estimate.
- Use Adam at a fixed `1e-4` learning rate, actor gradient-norm clipping at 1,
  and one update after each paired episode. Divide the summed log probability
  by the episode's fixed decision count to set the loss scale. Detach the
  advantage; do not normalize it against action-dependent baselines. These are
  initial engineering choices, not paper-derived optimal hyperparameters.
- In the same allocation, run meaningful contracts before training: exact
  budgets/protected tokens; sampled-order likelihood and gradient checks against
  an enumerated tiny action space; no backbone gradients; decision reuse at
  rebuild; sparse-map validity; deterministic deployment causal prefixes.
  No separate cluster contract job is proposed.
- Freeze development windows before training. Compare deployed greedy initial,
  middle and final checkpoints with uniform at the same budget. Require improved
  dense-output fidelity **and** improved GT translation RMS without worse p99
  or rotation versus both initial policy and uniform, aggregated on development
  windows. Report each sequence and post-refresh segments to expose drift.
- Archive all checkpoints, masks, probabilities, actor inputs, dense references,
  trajectories, manifests, source/model hashes and cost components. Review the
  pilot before any long evaluation. Lower training loss alone never advances it.

If the cap yields no held-out improvement, stop this configuration. If it cannot
complete useful updates, report compute infeasibility at this budget. Neither
outcome proves trainable modules impossible, and neither triggers automatic
architecture changes, seed searches or more spending.

Only a reviewed pilot pass justifies the original full-length accuracy/time/
memory evaluation, with frozen controls, natural admission and a checkpoint
selected without the final evaluation clips. Existing partial-GT limitations
remain. Long banks, object generalization and end-to-end savings are unpassed gates.

## Decision

This is the most direct fit found for the user's fixed-backbone, original-token,
small-module requirements: train against the actual sparse tracking consequence
instead of trying to predict a rejected importance teacher. Its gradient is
well-defined and its execution path already exists in the fork. Its empirical
benefit and affordable convergence are hypotheses requiring the bounded pilot.

The selected scene pilot is locally implemented and statically checked; runtime
checks, training and deployment evidence remain pending. The next action is the
single user-operated submission below, after publication of the fork and parent.

## CAMP submission and evidence

From CAMP head, in Bash, one allocation only:

```bash
cd /mnt/projects/gr/3DRecon/layer_good
W='git -c fetch.recurseSubmodules=false pull --ff-only'
V='bash tools/kvt_tum.sbatch adaptive-tokens task-policy'
Q='-A students --qos=students_normal -p 24g -w stuttgart'
LOG=/mnt/projects/gr/3DRecon/kvt_tum_slurm-%j.log
sbatch $Q --gres=gpu:1 --time=04:00:00 --propagate=NONE -o "$LOG" --wrap="$W && $V"
```

Return the job ID. The wrapper runs existing baseline/import contracts and four
new mathematical/state tests in this allocation; it bypasses the closed
sensitivity-oracle test suite. The driver stages exactly twelve training windows
(four starts from each of `freiburg1_desk`, `freiburg1_desk2`, `freiburg1_room`) and
four development windows (two starts each from `freiburg3_structure_texture_far`
and `freiburg3_structure_notexture_far`). Contents and coverage are checked on CAMP.
Freiburg1 versus Freiburg3 structure is a scene-family split by dataset taxonomy,
not a claim of independently audited physical-site metadata. The texture-far
development source was previously used for the tiny engineering smoke, so it is
not a pristine final test set. Final evaluation sources are never training inputs.

Expect `TASK POLICY ROLLOUT OK`, two `TASK POLICY CAUSAL PREFIX OK` (initial/final),
development summaries for uniform/initial/final (middle after update128 if reached),
update records and `TASK POLICY PILOT DECISION`. `ATD TASK POLICY JOB FINISHED`
means a decision was recorded. Read `complete`, `pilot_pass`, actual update count
and clauses; a budget stop is incomplete execution. Initial/middle/final and the
latest completed-update checkpoint are archived; latest writes are atomic.
Final checkpoint reload is checked on real archived features. No speed, memory
or object success is inferred from this eager diagnostic.

The wrapper archives `${tag}_context.tar`, `${tag}_all_runs.tar`, and sixteen
`${tag}_task_inputs_<episode>.tar` files under
`/mnt/projects/gr/3DRecon/kvt_tum_out/`. Context holds source identities, driver,
tests, protocol, decision, update records and checkpoints. Runs hold exact masks,
ordered sampling likelihoods, logits, actor inputs, raw retained cache samples,
calibration, trajectories and GT development errors. Input archives hold exact
model RGB plus metadata/checksums; original source pixels are pinned to dataset
ZIP hashes. Review these before any next allocation. No bulk training, threshold
sweep or full evaluation is chained behind this job.

Local checks: affected Python AST parsing, wrapper Bash syntax and git whitespace.
No project modules, tests, dataset processing, CUDA, training or inference were
executed on the Mac. The implementation and new fixtures are runtime-pending.
