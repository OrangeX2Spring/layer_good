# Context attention diagnostic

User authorized diagnostic followed by a main experiment on 2026-09-28. Submit
`tools/stream_cache_research.sbatch context_diagnostic` inside the existing CAMP
workflow. Review this archive before implementing/submitting the main experiment;
do not use a scheduler dependency to skip the scientific review.

## Frozen diagnostic scope

One existing masked ARCTIC espresso clip, 128 frames, width 518, frozen StreamVGGT,
FP32 and the existing eight-frame retention policy. Five isolated conditions:
`dense8`, `dense8_probe`, `spread64`, `spatial64`, `appearance64`. The three sparse
controls retain all foreground patches and 64 background representatives. This
uses the original controls without attention log-mass correction; it cannot
separate lost attention mass from representative-feature error. No new selector,
trained model, threshold sweep, or Co-Me superiority claim is included.

The observer runs alongside the dense stream. Encoder features define the same
groups as the sparse controls. At global layers 0, 4, 11, 17, 23 it recomputes
candidate K/V from merged **dense frame-block outputs**, using native norm1,
QKV projection, K normalization and representative integer RoPE positions.
Camera and register queries are held at their dense values. Specials and
foreground are singletons. Drop uses representative features rather than means.

Separate shadow banks retain compressed K/V derived from previous dense layer
inputs, with exactly the dense stream's frame IDs and eviction. Compare attention
outputs for (1) compressed history plus current frame and (2) dense history plus
compressed current frame. This is a local counterfactual on the dense trajectory,
not a replay of the independently evolving sparse model. It does not change the
dense stream or claim fidelity to the full sparse trajectory.

## Contracts and interpretation

Remote tests cover identity compression, real Q/K normalization and 2D RoPE,
eviction, unchanged dense outputs/cache, materialized shadow storage, prefix
causality and correlation arithmetic. The run must additionally preserve all-frame
dense pose encodings, every final exported dense prediction field and dense cache
accounting against `dense8`, within the existing fidelity tolerances. Candidate
token counts must match same-job sparse controls. Shadow storage is separately
reported and included in total persistent bytes. Instrumented latency/peak memory
are diagnostic costs, not deployable speed measurements.

For each layer, relative squared attention-output error is squared difference
divided by dense squared magnitude (denominator floor 1e-12), separately for the
camera, registers and individual heads. The frozen primary frame score averages
camera/register error and then the five layers, using compressed history/current.
An adjacent-pair score is the maximum of its endpoint scores. The target is that
pair's sparse translation RPE minus its dense translation RPE from the existing
camera-in-object evaluator. Its full-prefix Sim(3) alignment is offline evaluation.

Compare Spearman correlation with normalized background encoder within-group
variance and frame index. Also report dense local error and negative dense
background depth confidence as offline controls; neither GT nor full-model
confidence is a deployable selection input. Background confidence is measured on
the masked input and is not calibrated merge confidence.

The exploratory flag requires positive probe correlation exceeding both variance
and frame index in **each temporal half**, for at least two of the three sparse
controls. Undefined correlations do not pass. This is a predeclared descriptive
screen, not a significance test: adjacent pairs are dependent and the clip was
already inspected. Raw paired values, per-head/layer scores and current-only
scores remain archived for review. A positive flag cannot establish causation,
cross-scene transfer, novelty, or a final pose-error bound.

## Next gate and artifacts

Require remote tests, fidelity, research contracts, object evaluation,
`CONTEXT DIAGNOSTIC CONTRACTS OK`, `SWEEP OK`, archive exit status zero and matching
published revisions. Inspect `context_diagnostic.json`,
`dense8_probe/context_probes.jsonl`, events, object metrics and source archive.
Contract success and exploratory signal support are separate outcomes.

After review, a supported signal permits the authorized main experiment to test a
causal fixed-budget refinement selector with measured selection overhead, matched
controls (including attention-mass correction), and held-out sequences. Freeze its
concrete protocol before submission. If unsupported, diagnose which assumption
failed before spending on the main experiment. Dense teacher/shadow computation
in this diagnostic must not become a hidden free dependency of that selector.
