# Online layer allocation: fixed-byte, causal pilot

User scope: Pi3 objects, Pi3 scenes and StreamVGGT scenes; at least two object
and three scene budgets, each compared with native. No offline layer list,
teacher sequence, labels, training or result-driven budget tuning.

Object caps:3 and5 dense-frame equivalents; scene caps:4,8,16. An equivalent
is all cache-bearing layers' K/V for one dense input frame. Convert once from
the first input's shape/dtype to fixed bytes; charge row IDs and, on StreamVGGT,
positions to the cap. Caps do not increase with sequence length. Exact byte
values are in each run's allocation/metrics. Protection: anchor and latest
admitted frame at every layer, both charged. Native is uncapped; uniform has
the identical absolute cap and protection. These are KV-state limits, not
whole-GPU limits: dense rebuild/query temporaries, camera history, model weights,
mapping, input frames and diagnostic logs are separately reported or preserved.
The established box native archive has only7 keyframes, so an8-frame object
budget would be non-binding;3/5 avoids that known design problem. New closed-loop
admission can still change counts; report whether each cap actually binds.

Online score: every16th query,8 uniformly sampled query rows per layer. Compare
local attention output using available history against anchor+latest+current
only. Relative squared output deviation is smoothed with EMA0.9. Allocate
remaining rows in proportion to sqrt(score), capacity-clipped largest-remainder
integer quotas. Initial scores are zero, so unobserved allocations are uniform.
Reserve up to8 optional rows per layer, charged before weighted allocation, to
avoid permanently starving a layer's importance observations. Uniform shares
the same reserve and uses equal weights. Within each layer, uniformly gather
eligible history rows; no K/V averaging, RoPE changes or model-block pruning.
This estimates value of the available optional history, not a dense teacher or
task oracle. Joint compression effects and long-term utility remain hypotheses.

Pi3 updates layer budgets after native dense rebuilds, using preceding/current
query scores; native reconstruction also refreshes discarded historical rows.
Native keyframe rules operate independently in every arm (closed loop); changes
to admissions are reported, not suppressed with a future baseline schedule.
No hidden dense shadow between rebuilds. StreamVGGT updates retention after
each frame, admits every frame like upstream inference, and cannot restore
evicted history. Its first/latest raw K/V and original positions remain intact.

Pi3: three ARCTIC S01 camera0 grab sequences, first min(1024,length) post-offset
frames,518resolution; three TUM office/loop/no-loop1024frame prefixes,308.
All object arms replay the SAME saved causal SAM masks from the established
ARCTIC native archive. Source manifest identity and recorded GT metrics verified
by stream_cache_arctic.py. Native Pi3 model/keyframe/mapping code is unchanged;
SAM compute is excluded equally. Shared source RGB/masks/GT preserved. Each
object gets5 conditions, each scene7:36 main runs. Fifteen online128frame prefix
runs check raw trajectory tolerance and exact discrete allocation decisions;
floating sensitivity scores must match at1e-6 absolute/1e-5 relative tolerance.
Model inputs are prepared/archived once outside timed runs and checked
pixel-identical in every arm. All conditions use the same saved causal SAM masks, including
native; no segmentation-time mismatch is counted as cache speed.

StreamVGGT: same three TUM sequence prefixes,128frames,308,FP32. Native uses
untouched upstream cached aggregator, all-frame admission and unchanged camera/
geometry heads, with no retention policy. Each scene7conditions:21main runs,
nine96frame online prefixes. Bounded native window limits unbounded FP32 cache
cost on24GB; longer operation is not claimed. Every arm uses the same suffix
64..end for timing; no calibration/refit. Prefix checks exclude timing fields.

Contracts run first: local sensitivity vs explicit intervention; Pi3 fixed byte
caps, original anchor/latest values and sensitivity-driven quota differences;
probe hook leaves query output bit-identical; StreamVGGT fixed caps and original
anchor K/positions through repeated admissions. CAMP only. Mac AST/Bash/diff
checks are static; no local inference or project test execution.

Pass: all planned runs and prefix checks complete, budgets bind (or explicitly
mark non-binding conditions), finite evaluated trajectories, exit0 and archive.
Inspect per-sequence ATE/RPE/p99, allocation/probe traces, peak allocation,
query/probe-inclusive time, retention time, rebuild calls/time, total tracking
time and keyframe schedule. A completed pilot does not prove quality superiority.
No automatic full-stage job, retry or training.

## Rebuild time

Pi3 native cache-byte samples from the existing instrumentation are logical
K/V sizes; final_physical_kv_bytes deduplicates actual backing storage, including
native V views into QKV projections. Native growing history's final physical
storage is its persistent maximum. Compact online/uniform actual storage and
row metadata are checked at every rebuild. Do not compare logical native bytes
to an actual-byte cap without identifying that distinction.

These pilots preserve Pi3's dense rebuilds. Measure their share before claiming
end-to-end speed. Reusing per-frame encoder features is a potential optimization
because the encoder processes each frame independently; it requires numerical
fidelity checks and explicit feature-memory accounting. Decoder historical KV
cannot generally be reused exactly after adding a frame: old states change under
joint attention. Append/partial refresh changes that computation; prior Pi3
continuity failures must not be revived as an established fix. StreamVGGT's
streaming path has no Pi3-style joint dense rebuild. No encoder reuse or append
variant implemented or silently included in this pilot.

## Submission

Publish both model forks and parent gitlinks first. Two independent user-run
jobs, serialized to protect the shared checkout. From CAMP head:

```bash
R=/mnt/projects/gr/3DRecon/layer_good
Q="-A students --qos=students_normal -p 24g -w stuttgart"
W="git -C $R -c fetch.recurseSubmodules=false pull --ff-only"
sbatch $Q --gres=gpu:1 --time=04:00:00 --propagate=NONE \
  -o /mnt/projects/gr/3DRecon/online_cache_slurm-%j.log \
  --wrap="$W && bash $R/tools/online_cache.sbatch pi3"
```

After receiving its job ID, submit streamvggt with afterany:JOB to serialize
checkout updates. This is an independent backbone pilot, not an unreviewed
pilot-to-full promotion. Do not duplicate active jobs. Expected output:
online_cache_out/online_cache_HOST_JOB.tar and online_cache_slurm-JOB.log.
Archive preserves failures/exit/source/container/input provenance plus results.
Existing native StreamVGGT regression contracts also run in its pilot. Native
baseline is the all-frame growing upstream cache, unlike the32-slot controlled
reference in26089/26090; do not describe those older references as uncapped native.
No container save, environment replacement or model download. StreamVGGT uses
the established constrained disposable-container dependency bootstrap.
