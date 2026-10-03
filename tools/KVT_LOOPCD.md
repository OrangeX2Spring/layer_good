# LoopCD-inspired geometry: layer-depth contrast pilot

User requested a submit-ready experiment2026-10-03. Separate from26160 and all
segmentation/graph/token-reduction experiments. This is an exploratory untrained
analogue, not a reproduction of LoopCD on a trained recurrent transformer.

## Hypothesis and five controls

Apple [LoopCD](https://arxiv.org/html/2610.02185v1), Eq2, extrapolates a later
recurrent hidden state away from an earlier state before a shared readout.
Pi3 instead has36 distinct decoder blocks. Their early/late representations have
the same dimensions but are not known to be semantically aligned. Tonight tests
that prerequisite and whether contrast can compensate for reduced query depth;
it does not train a recurrent geometry model or claim guaranteed recovery.

Every arm uses the exact first256 reviewed office RGB frames, native keyframe
schedule/dense cache, unchanged encoder, all spatial patches and camera head.
Native rebuilds execute all36 blocks and store all KV layers. Only camera queries
change; each retained global block reads its own original cache layer.

| Arm | Query decoder blocks | Earlier reference | Guidance |
|---|---:|---:|---:|
| native |36, original Pi3 forward|none|none|
| full |36, kept-path control|none|0|
| full_guided |36|12|0.5|
| reduced |24|none|0|
| reduced_guided |24|12|0.5|

Pi3 normally concatenates its last two decoder outputs for the camera decoder.
The control uses the corresponding final two outputs at the chosen depth. The
weak state concatenates block11/12 outputs; the strong state concatenates35/36
or23/24 outputs. Before the unchanged camera decoder, apply in fp32:
`guided = strong + 0.5 * (strong - weak)`, then cast to the original dtype.
Guidance requires no additional encoder/decoder pass or extra camera-head call.
The earlier pair is retained during the same forward. No camera-matrix subtraction,
adaptive thresholds, coefficient/depth sweep, learned adapter or GT decisions.
The fixed depths/strength are initial engineering choices, not tuned winners.

Full-guided versus full tests guidance without truncation; reduced-guided versus
reduced isolates guidance at identical compute depth; reduced versus native tests
the cost/accuracy of truncation. Full must reproduce native trajectories at
rtol=atol=1e-4. The model architecture and checkpoints otherwise stay identical.

## Verification, timing and gate

CAMP contracts: zero-guidance full-depth pose/native equality; camera input
exactly equals the declared contrast of independently obtained weak/strong inputs;
only the declared number of decoder blocks execute; finite outputs; persistent
KV values and pointers unchanged. Each arm has a fresh256-frame process and a
fresh32-frame prefix process; prefix poses must agree at rtol=atol=1e-4. Inputs,
checkpoint, GT/pair associations, environment, keyframe IDs and cache bytes match.

Record ATE, translation RPE/p99, full-loop/query/rebuild time, FPS, peak allocated/
reserved memory and native ratios. Keep original three-metric5% verdict visible.
`reduced_guidance_improves_all` means all three metrics no worse than reduced,
with at least one strictly better. This is a pilot observation, not held-out
validation. One coefficient and one ordered timing observation per arm; no2x or
statistical speed claim. Eager execution deliberately keeps graphs separate.
Timing includes IO/hash, upload, feature retention/guidance, inference, native
rebuild and logging; excludes model load, evaluation and staging.

Success for preparation: published isolated experiment plus static checks. Runtime
success: contracts, baseline fidelity, prefixes and all five runs complete.
Scientific gate: review whether guidance improves its matched control, accuracy
relative to native, and actual compute savings before deciding on full2585-frame
evaluation or a trained alignment/recurrent adaptation. No full evaluation is
queued behind this unreviewed pilot; JOB OK is execution completion only.

## Isolated submission while26160 runs

Do not pull the shared layer_good checkout for this job. Clone the published
`codex/loopcd-pilot` parent branch inside a separate job-local
`/tmp/kvt_loopcd_<JOB>` checkout, following the
existing isolated graph pilot setup. The wrapper initializes its own fork and
third-party checkouts using read-only reference repositories in layer_good,
verifies nested revisions and links the existing SAM checkpoint. No dependency
install, model download, image change or shared source edit. The wrapper rejects
loopcd in the shared checkout. Parent/fork/input/container identities are archived.

From `/mnt/projects/gr/3DRecon/layer_good` on head:

```bash
W='D=/tmp/kvt_loopcd_$SLURM_JOB_ID'
R=https://github.com/OrangeX2Spring/layer_good
W="$W && git clone -q -b codex/loopcd-pilot $R \$D"
W="$W && bash \$D/tools/kvt_tum.sbatch loopcd"
O=../kvt_tum_slurm-%j.log
sbatch -p 24g -w muenchen --gres=gpu:1 --propagate=NONE -o "$O" --wrap="$W"
```

Expected: LOOPCD DEPTH CONTRAST CONTRACT OK, five LOOPCD COMPARE lines,
LOOPCD PILOT COMPLETE and JOB OK. Stop at first failure and return its context.
Outputs in kvt_tum_out: `tum_<JOB>_loopcd_<ARM>.tar`, five `_prefix.tar` archives,
`tum_<JOB>_loopcd_comparison.tar`, context and all_runs archives. Checkpoints and
original input archives are borrowed hash-checked from26127, never changed.
