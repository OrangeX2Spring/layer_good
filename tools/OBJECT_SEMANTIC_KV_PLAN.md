# Persistent-object reanchoring and semantic KV

2026-10-09. User selected **implement detector; plan semantic KV**. Current
implementation and CAMP gates: [KVT_SEGMENT_ARCTIC.md](KVT_SEGMENT_ARCTIC.md).
Loop research is deferred. This document specifies follow-on experiments; it
does not authorize a new model download, semantic selector implementation or run.

## Question and current evidence

For one persistent target, a new appearance may be the object's other side,
occlusion, articulation or lighting. It is not evidence that the object identity
changed or that all history should be deleted. Keep three decisions distinct:
whether an anchor has sufficient correspondence support, whether a new local
reference can connect reliably, and which historical observations deserve memory.
The26205 audit in FINDINGS demonstrates limitations of the image-grid trigger,
but all actual cuts also have weak match counts. Bridge diagnostics fail and
errors are not confined to exact seam frames. No single cause is established.

## Stage1: object-relative support, implemented

Compare the original coverage rule with anchor-support-normalized coverage.
Preserve ORB descriptors, inlier threshold, persistence/age, bank size and handoff.
Use fresh native object tracking for masks and a fidelity-gated native replay for
the matched SAM-free timing scope. Ketchup first; review evidence before any
other-object confirmation. Success is trajectory quality plus measured resources,
not the number of segments. Detailed contract/pilot gates are in the protocol.

## Stage2: handoff and persistent view memory, conditional

If stage1 does not preserve quality, inspect error evolution within segments and
the reliability of saved connection geometry before another trigger change.
Potential separate intervention: retain a small overlap bank until the new map
has a valid object-supported connection. Explicitly restrict fitting support to
the object; account for articulation, mask errors and insufficient geometry.
This changes the original immediate-deletion policy and adds memory/latency.
Thresholds, finite overlap budget and behavior at an unsuccessful transition must
be agreed before implementation; do not silently turn diagnostic checks into
gates that can retain unbounded history or prevent a complete trajectory.
Evaluate it with a fixed trigger so improved results cannot be attributed to two
simultaneous changes. This is planned, not implemented in stage1.

## Stage3: semantic KV means named support, not completed geometry

The tutor's examples describe two related but distinct kinds of information:

| Information | Meaning | Requirement |
|---|---|---|
| cap, handle, body | Semantic component or part | Part mask/label associated with observed pixels; a detached cap may have independent motion |
| front, back, top | Surface/view in an object coordinate convention | Defined object axes and trustworthy pose/association; image top is not object top |
| unseen back side | Unobserved geometry | Reconstruction/completion would infer new geometry; KV retention cannot create an observation |

A semantic cache can preserve original historical tokens that cover distinct
parts and views. It does not require generating a complete mesh, and labels do
not make tokens at different decoder depths interchangeable. Keep original
layer slots, RoPE positions and K/V values; use metadata to select their rows.
Do not average all "cap" tokens into a synthetic key or treat an appearance
cluster as a verified semantic part.

Feasibility context: [SAMPart3D](https://arxiv.org/abs/2411.07184) demonstrates
semantic part segmentation using a3D representation and multi-view renderings
for naming. Its [official implementation](https://github.com/Pointcept/SAMPart3D)
is an offline3D-object pipeline, not evidence of a ready causal RGB tracker.
It supports the plausibility of part labels, not the proposed KV policy's accuracy,
latency or availability in our CAMP image. No new semantic dependency selected.

### Minimal future experiment

1. Start with one object and one observable distinction, such as cap versus body.
   Establish a causal label source: reviewed first-visible-frame part prompts and
   temporal propagation, or a separately validated per-frame part model. Never
   derive deployed labels from future frames, GT poses or a full-sequence mesh.
   A manually labelled diagnostic is an explicit upper-bound/annotation-assisted
   condition and must report that supervision separately.
2. Before cache intervention, validate label consistency, occlusion/unknown
   handling and alignment to Pi3's actual resized patch grid. Record part IDs,
   source frame/patch, label provenance/confidence and observation time. Maintain
   unknown/unlabelled support; never guess "front" from the image's orientation.
   For view labels, first define object axes from an explicit annotation or
   canonicalization method. Symmetric or ambiguous objects can remain unknown;
   initial-camera-relative view bins may be used but must not be called semantic
   front/top. Articulated components require separate association assumptions.
3. Hold native admission, queries and rebuilds fixed. At a single agreed KV byte
   budget, compare an existing geometry/appearance selector with a part-coverage
   selector using the same metadata/accounting overhead where applicable. Protect
   existing special/anchor requirements and retain exact original K/V rows.
   Reserve coverage across reliably visible parts rather than discarding an old
   useful cap view merely because the current frame shows the body. Rebuilds must
   regenerate selected historical layers from their proper context; labels alone
   do not authorize transplanting a stale cache into a new map.
4. Assess ATE/RPE/p99, actual persistent/peak bytes, query/rebuild/labeler time,
   rare-part retention and reacquisition after occlusion. Charge label generation,
   propagation and metadata storage to the method. Require the same5% tracking
   gate before claiming efficiency, then test a different object/view sequence.

The semantic allocation rule, exact budget, labeler and coordinate convention are
pending choices. Do not implement a speculative general ontology or run a grid
of models/budgets. Existing [semantic/geometric patch experiments](SEMANTIC_KV_OVERNIGHT_PLAN.md)
include appearance-feature grouping; their results
must not be relabelled as cap/front/top understanding.

## Stop and advance rules

Stage1 local implementation is complete; contracts and tracking remain pending
on CAMP. Review each gate before moving on. Failure does not automatically select
stage2, a semantic model, retraining or another threshold. Semantic KV remains a
design until a useful, causal part-label source and fixed comparison are chosen.
