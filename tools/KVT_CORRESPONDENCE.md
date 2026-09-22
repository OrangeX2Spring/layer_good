# Object-first correspondence KV pilot

Prepared 2026-09-22. User selected both geometric and semantic variants, with
object-level evaluation the priority. This is a new exploratory pilot, not a
restart of the completed confidence/novelty plus redundancy sweep. No CoMe-style
merging is used. Method novelty and accuracy are unverified.

## Fixed scope

First host: KV-Tracker on ARCTIC `box_grab_01`, offset 2, resolution 518, the
reviewed initial SAM mask and existing online mask propagation. Both proposed
variants run on the same object task. Model input remains the native target-masked
RGB. Consequently this tests target-aware retention *within an already object-aware
tracker*, not segmentation, multi-object association, or novel semantic identity.
Non-target patches here are masked context, not full scene/background imagery.

All conditions insert at frame indices 0, 30, 60, ... and stop at 32 unique
keyframes; no eviction, redundancy score or pose-dependent insertion. Full anchor
and five special tokens remain. Every other frame keeps exactly ceil(N/2) ordinary
patches in each sparse condition. Persistent old selections never regain discarded
tokens after a rebuild. The bootstrap's duplicated frame is unchanged.

Four conditions: dense cache, spatial uniform half, geometric correspondence half,
and target-aware correspondence half. The first two are necessary matched controls;
the archived default/adaptive-keyframe runs do not share this schedule. There is
no threshold grid. Dense/native full retention is a separate correctness check.

## Proposed retention rule

- Capture frame-local encoder descriptors on insertion candidates; no extra forward.
- Use points at patch centres from the **same current dense rebuild** for all retained
  observations. This avoids comparing 3D positions from different predicted gauges.
- Match to surviving historical patches by reciprocal cosine nearest neighbours,
  restricted to a 3D distance within twice the median positive neighbouring patch
  spacing of the current pointmap. Fixed cosine floor 0.9; no tuning in this pilot.
  Zero geometric spacing yields no matches. These are tentative model-derived
  correspondences, not verified physical tracks.
- Divide the image into 4x4 cells and distribute the exact token budget proportional
  to cell population using largest-remainder rounding. Prefer matched distinct
  track IDs within each cell, then fill the remainder uniformly.
- The semantic variant additionally requires target/non-target label agreement and
  splits each cell's quota into those two strata. This preserves proportional target
  support without granting extra tokens. Target labels are majority SAM-mask patches.
  It changes both matching eligibility and quota allocation; a later ablation is
  needed to attribute an improvement to either mechanism independently.
- Gather original post-RoPE K/V pairs after the native dense rebuild. No feature
  averaging, merge predictor, new network, cache quantization or pose smoothing.

The selector lives in the model fork's `kv_tracker/correspondence_cache.py`.
`main.py` passes the already computed rebuild pointmaps/masks to the cache hook;
the old combined cache accepts but does not use them. Object mode is explicitly
enabled only for the new policy. Optional Sim(3), manual keyframes and crop mode
remain outside this pilot's contract.

## Pilot and gates

`kvt_correspondence.sbatch` runs remote CPU contract tests, then a 160-frame native
fixed-interval reference and dense adapter. Their poses must match exactly. Each
sparse condition gets a 128-frame prefix and a 160-frame run. Verify identical SAM
masks, insertion IDs, exact patch budgets, prefix poses and selections, and that
every match points to an older retained token. The CPU contracts cover known
correspondences, geometry rejection, label-disambiguated matches, reciprocity,
spatial/semantic quotas, exact gather/masked-attention equivalence, cap behaviour,
and persistence of old patch choices. They are **not executed on the Mac**.

`gates.json` distinguishes structural checks from measured mechanism activity:
whether generic matches survive, whether target matches survive in the semantic
arm, and whether the two proposed selectors make different choices. An inactive
mechanism is a review failure even if the executable exits successfully.

`review.json` reports ATE and native translation/rotation-part RPE, adds angular
rotation RPE in degrees and translation/rotation 99th percentiles, and reports
cache payload, selector metadata, peak allocated/reserved bytes, query-forward
median time and synchronous total tracking/export time. Saved error arrays support
inspection around insertion events. Metrics use the existing full-prefix Sim(3)
alignment: this is not fixed-gauge metric 6D pose accuracy. Native `rpe_rot` is a
dimensionless rotation-part norm; `rpe_rotation_deg` is the added angular metric.

For a later full-sequence screen, the proposed tolerance is at most 5% degradation
in ATE, translation RPE and angular rotation RPE relative to matched dense, with no
material worsening of error tails, plus actual cache and end-to-end savings. This
is a screening target, not statistical equivalence. The short box prefix cannot
establish it. Review all metrics, including selector overhead, before extending
to the three complete ARCTIC sequences or scene-mode streaming hosts.

KV-Tracker's native dense rebuild still sets peak memory. Half query memory need
not reduce peak GPU memory or wall time. `selector_host_seconds` includes CPU
selection and GPU gather launch, not synchronized GPU gather completion; total
tracking time includes completion, model load and snapshot export. Query-forward
time excludes mask generation, pruning and export. No geometry-quality claim is
made from the exported final **dense-rebuild** point cloud.

## Publication, execution and artifacts

Publish the tracker fork first, then update/publish the parent's gitlink and tools.
The wrapper pulls only inside its allocation, then checks out the published pin.
It reuses `kvt.tar` read-only through `kvt_run.sh`; no dependencies or weights change.
After those revisions are available on CAMP, the user submits from `layer_good`:

```bash
sbatch tools/kvt_correspondence.sbatch
```

This is **pilot only**; it never starts full evaluation. Expected evidence is remote
CPU tests passing, `FULL RETENTION GATE OK`, mask/prefix gates, `CORRESPONDENCE PILOT
GATES OK`, `JOB OK` and Slurm `COMPLETED 0:0`, plus review of `gates.json` activity
flags and `review.json`. Do not advance based on the log footer alone.

Under `/mnt/projects/gr/3DRecon/kvt_arctic_out/`, the job writes
`correspondence_<job>_{inputs,context,review}.tar` and per-run
`arctic_correspondence_<job>_<condition>[_prefix]_<timestamp>.tar`. Context records
source revisions/diffs, scripts, Pi3 model/attention sources, input hash and exit
status. Per-run archives preserve trajectories, selected indices, tentative links,
SAM masks, reviewed initialization, configs and final keyframe geometry. A failure
also archives partial result directories. Prepared inputs and transient runs stay
under allocation-local `/tmp`; no archives exist until the user runs the pilot.
