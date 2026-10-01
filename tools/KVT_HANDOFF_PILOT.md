# One office handoff: native, fixed anchor, dynamic anchor

Selected by user 2026-10-02: one handoff, three arms, office 0–974. Implements the
first bounded connection test from [DYNAMIC_ANCHOR_RESEARCH.md](DYNAMIC_ANCHOR_RESEARCH.md),
not automatic segmentation, multiple handoffs, a threshold sweep or a full run.
Model mechanism: KV-Tracker fork `kv_tracker/map_handoff.py`; parent driver
`kvt_handoff_pilot.py`. No Pi3 architecture/checkpoint change or training.

## Fixed arms and timing

| Arm | Bank before749 | At749 | After749 |
|---|---|---|---|
| native | Original interval50 growing bank, cap20 | Native reconstruction | Original schedule |
| fixed | Frame0 + latest admitted frame | Rebuild[0,749] | Frame0 + latest |
| handoff | Identical to fixed | Try new bank[699,749], shared view699 | If accepted:699 + latest; if rejected:0 + latest |

All arms use identical pixels, resolution308, seed0, initial duplicate[0,0],
queries and insertion opportunities49,99,...,949. Full patches; no KV pruning.
The handoff is attempted exactly once at749, after that frame's old-bank query.
Frame749's emitted pose is not rewritten. Accepted state first serves query750.
This tests a delayed handoff near the user's approximate700 boundary, not an
automatic detection result. Steady bounded state is two images; candidate and
old caches coexist briefly (four frame-equivalents of KV, three unique images).
Reconstruction inputs never exceed two images for the bounded arms. Activations,
temporary image batches, geometry/evidence buffers and allocator reservation are
additional and measured/reported separately; four-frame KV is not a GPU peak cap.

At749, old bank[0,699] already has a stored pointmap/confidence for699 from its
last rebuild. Candidate[699,749] recomputes699's pointmap under a new local anchor.
No extra old-bank reconstruction or pose-only scale inference is used. The old
predicted camera pose749 is independent of candidate reconstruction and supplies
a cross-view validation check. It is a consistency check, not ground truth.

After a successful switch, subsequent regular rebuilds preserve anchor699 and
replace only the second image. The global Sim(3) remains fixed across those
rebuilds; any reconstruction-induced local-scale/gauge drift remains observable.
One transition cannot establish robustness to repeated scale composition.

## Preregistered geometry connection

All fitting runs on CPU float64, sampled every4 pixels on the shared image.
Require finite3D points/confidence. Keep points at or above each map's median
confidence; this is a relative filter, not calibrated uncertainty or a static-object
mask. Moving objects/repeated texture can still invalidate the bridge.

Checkerboard sample cells split fit/validation points; require128 in each.
Fit proper Sim(3), mapping candidate-local points into old-local coordinates,
using SVD and three rounds of residual trimming. Each trim retains training
residual <= max(three times median residual,1% of old fit-point RMS extent).
Require128 surviving points and second/first covariance singular-value ratio
>=0.001. Planar rank2 geometry is allowed; collinear geometry is rejected.
Validation samples never enter the fit/trimming objective.

Acceptance requires ALL of:

- Positive fitted scale.
- Validation median residual <=2% and p90 <=5% of old fit-point RMS extent.
- At least80% of validation points within5% extent.
- Those inliers cover at least12 of16 image bins.
- Mapped candidate camera749 differs from old-bank query749 by <=5% extent
  in translation and <=5degrees in orientation.

These are fixed initial engineering gates, not tuned or validated confidence
bounds. No GT, future query, GT scale or per-segment evaluation transform enters
selection. Same-image held-out points are correlated; camera749 supplies another
view but can share model bias. A passing connection can still be wrong.
Evidence includes sample pixels, source/target points, masks, residuals, old/new
camera poses, estimated scale/rotation/translation and every gate outcome.

Compose candidate-to-old and old-to-global similarities. Camera centres transform
with scale, rotation and translation; orientation with rotation only. The initial
global translation preserves native scene-centering; local `origin_offset`
normalization remains separate. Do not multiply a scale into a camera rotation.

On acceptance, commit new origin/global transform, new images and latest geometry.
Check new cache storage is disjoint from old while both are alive, then remove old
references and assert old KV tensor weakrefs are dead. Keep no online old-image
descriptor/retrieval tier. Saved evidence on disk is a diagnostic output, not an
active bank. CUDA allocator reservation may remain after tensor release; that is
not live historical KV. No loop closure or later retrieval of retired imagery.

On rejection, restore the old cache, then perform the normal[0,749] fixed-anchor
rebuild. Log rejection and charge candidate+fallback work. No retry at a later
boundary and no silent acceptance. Execution may finish with a rejected bridge;
that is an informative failed connection, not a successful dynamic anchor.

## Runtime checks and comparisons

Ten fresh processes total, with only three scientific arms:

1. Two128-frame reference runs through existing tracker: native and cap2/full-
   patch CombinedCache (only victim is previous non-anchor, equivalent fixed bank).
2. Three128-frame new-harness runs: native/fixed/handoff. Compare to corresponding
   existing-tracker reference at atol1e-5/rtol1e-4; fixed/handoff must match exactly.
3. Three975-frame main arms.
4. Fixed/handoff800-frame causal prefixes extending beyond switch; raw poses and
   deterministic decision records must match full-run prefixes exactly.

Before inference, existing TUM contracts plus handoff contracts run remotely.
New contracts cover similarity recovery, scale-free camera orientation, transform
composition, held-out fit failure, degeneracy and commit/rejection transactions.
Local work is AST/shell/whitespace and source inspection only; no local torch/model
execution is authorized.

Main-arm checks: finite trajectories, expected query/insertion counts, image-bank
caps, actual cache row counts, fixed/handoff raw equality through749, exactly one
attempt, required post-switch anchor and dual-cache arithmetic. Rejection must
leave the complete trajectory identical to fixed control. Accepted old KV
retirement is checked independently of accuracy. Fidelity or prefix failures
stop the pilot and preserve failure outputs; do not interpret their metrics.

Quality uses one full-window Sim(3) fit, never per-map GT fits. Report ATE,
translation RPE/p99 and rotation, and inspect raw transition-window errors around
749/750 plus subsequent rebuilds. `transition_errors.json` records each valid
adjacent pair starting within five frames of 749, 799, 849, 899 and 949, using
the same full-window alignment, with translation RMSE/max and per-pair rotation.
The driver asserts identical GT associations and adjacent pairs across all arms.
Keep native and equal-two-frame fixed control
comparisons separate. The inherited accuracy criterion is each of ATE, translation
RPE and p99 <=1.05x native. Bridge acceptance is separate from this accuracy gate;
do not attribute benefit to adaptive segmentation or content-only eviction.

All three main arms share the same harness/geometry-copy overhead. Total tracking
time excludes model load/staging, includes image decoding/hash checks, CPU geometry
copies, fitting, evidence writing and instrumentation. Report candidate, query,
rebuild and update time; `update_total` INCLUDES candidate/rebuild work, so do not
double count. Report allocated/reserved GPU peaks, physical KV storage, current
image/geometry bytes and transient dual-cache/candidate geometry. No deployment
FPS claim or comparison to old jobs' model-load-inclusive totals.

## User-operated CAMP procedure

Publish fork and parent gitlink, then refresh the checkout inside an allocation.
Submit from `/mnt/projects/gr/3DRecon/layer_good` using the scheduler default limit:

```bash
sbatch tools/kvt_tum.sbatch handoff-pilot
```

Wrapper fixes checkout location independently of Slurm spool paths, records source
and image identities, uses existing kvt image, and saves failure context. Driver
hashes the locally cached Pi3 snapshot and TUM source archive; no downloads.
Expected `HANDOFF NATIVE FIDELITY OK`, `HANDOFF PILOT COMPLETE: 3 arms; bridge
accepted=True/False`, `JOB OK`, and archived exit0. Scientific review remains
pending even if all markers appear.

Outputs in `/mnt/projects/gr/3DRecon/kvt_tum_out/`:
`tum_<JOB>_handoff_inputs.tar`, `tum_<JOB>_handoff_<RUN>.tar`, plus context/all_runs
archives. Context includes `handoff_summary.json`, `handoff_gates.json` and
`handoff_archives.json`; candidate results include `bridge_evidence.npz`,
`bridge_attempt.json` (fit-time decision), final `handoff_events.json` (including
commit/rejection and retirement), and `final_transform.json`.
Log: `/mnt/projects/gr/3DRecon/kvt_tum_slurm-<JOB>.log`.

Review first failure or returned archives before any retry/full-sequence extension.
No next job, automatic segmentation, larger bank or threshold sweep is authorized
by a successful pilot alone.
