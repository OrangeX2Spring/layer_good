"""Causal anchor-overlap boundary diagnostic; CAMP only, no Pi3 or GT."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np

from kvt_tum_run import write_json
from kvt_tum_sweep import archive_directory


SOURCE = 'tum_26127_oracle_maps_inputs.tar'
SOURCE_SHA256 = '9ecd7546a56f2135bfbfc2d33a1be4de45c1670cb8cf56e1bedf2639a2933b33'
POLICY = dict(features=1000, ratio=.75, ransac_pixels=2., minimum_features=40,
              minimum_inliers=20, grid=4, minimum_cells=4,
              persistence=5, minimum_segment_frames=50)
INLIER_THRESHOLDS = (10, 15, 20, 25, 30)
FLOW_POLICY = dict(corners_per_cell=20, quality=.01, minimum_distance=5,
                   grid=4, window=21, pyramid_levels=3,
                   forward_backward_pixels=1., surviving_fraction=.25,
                   surviving_cell_fraction=.5, minimum_features=40,
                   persistence=5, minimum_segment_frames=50)


class FlowSegmentDetector:
    """Retain original anchor feature identities through consecutive frames."""
    def __init__(self, surviving_fraction=.25, maximum_segment_frames=None):
        assert 0 < surviving_fraction <= 1
        assert maximum_segment_frames is None or maximum_segment_frames >= 50
        self.surviving_fraction = surviving_fraction
        self.maximum_segment_frames = maximum_segment_frames
        self.previous = None
        self.last_frame = -1
        self.low_run = 0

    def seed(self, gray, frame):
        h, w = gray.shape
        points = []
        grid = FLOW_POLICY['grid']
        for y in range(grid):
            for x in range(grid):
                mask = np.zeros_like(gray)
                mask[y * h // grid:(y + 1) * h // grid,
                     x * w // grid:(x + 1) * w // grid] = 255
                corners = cv2.goodFeaturesToTrack(gray, FLOW_POLICY['corners_per_cell'],
                    FLOW_POLICY['quality'], FLOW_POLICY['minimum_distance'], mask=mask)
                if corners is not None:
                    points.append(corners.reshape(-1, 2))
        self.points = np.concatenate(points) if points else np.empty((0, 2), np.float32)
        self.anchor_points = self.points.copy()
        self.anchor_frame = frame
        self.initial_features = len(self.points)
        self.initial_cells = len(np.unique(np.floor(
            self.points * [grid / w, grid / h]).astype(int), axis=0))
        self.previous = gray
        self.low_run = 0

    def step(self, rgb, frame):
        assert frame == self.last_frame + 1
        assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
        self.last_frame = frame
        started = time.perf_counter()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.previous is None:
            self.seed(gray, frame)
            status = 'bootstrap'
        else:
            if len(self.points):
                current, forward, _ = cv2.calcOpticalFlowPyrLK(
                    self.previous, gray, self.points, None,
                    winSize=(FLOW_POLICY['window'],) * 2, maxLevel=FLOW_POLICY['pyramid_levels'])
                back, backward, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.previous, current, None,
                    winSize=(FLOW_POLICY['window'],) * 2, maxLevel=FLOW_POLICY['pyramid_levels'])
                h, w = gray.shape
                keep = (forward.ravel().astype(bool) & backward.ravel().astype(bool)
                        & (np.linalg.norm(back - self.points, axis=1)
                           <= FLOW_POLICY['forward_backward_pixels'])
                        & (current[:, 0] >= 0) & (current[:, 0] < w)
                        & (current[:, 1] >= 0) & (current[:, 1] < h))
                self.points = current[keep]
                self.anchor_points = self.anchor_points[keep]
            self.previous = gray
            status = 'supported'
        h, w = gray.shape
        cells = len(np.unique(np.floor(
            self.anchor_points * [FLOW_POLICY['grid'] / w,
                                  FLOW_POLICY['grid'] / h]).astype(int), axis=0))
        fraction = len(self.points) / self.initial_features if self.initial_features else 0.
        coverage = cells / self.initial_cells if self.initial_cells else 0.
        if self.initial_features < FLOW_POLICY['minimum_features']:
            status = 'unknown_texture'
            self.low_run = 0
        elif status != 'bootstrap':
            low = (fraction < self.surviving_fraction
                   or coverage < FLOW_POLICY['surviving_cell_fraction'])
            status = 'low_overlap' if low else 'supported'
            eligible = frame - self.anchor_frame >= FLOW_POLICY['minimum_segment_frames']
            self.low_run = self.low_run + 1 if low and eligible else 0
        boundary = self.low_run >= FLOW_POLICY['persistence']
        if self.maximum_segment_frames is not None and frame - self.anchor_frame >= self.maximum_segment_frames:
            boundary = True
            status = 'maximum_age'
        row = dict(frame=frame, anchor_frame=self.anchor_frame,
                   features=len(self.points), anchor_features=self.initial_features,
                   ratio_matches=len(self.points), inliers=len(self.points), cells=cells,
                   surviving_fraction=fraction, surviving_cell_fraction=coverage,
                   status=status, boundary=boundary, low_run=self.low_run)
        evidence = dict(anchor_points=self.anchor_points.copy(),
                        current_points=self.points.copy(),
                        inliers=np.ones(len(self.points), bool))
        if boundary:
            self.seed(gray, frame)
        row['seconds'] = time.perf_counter() - started
        return row, evidence


class SegmentDetector:
    def __init__(self, minimum_inliers=20, maximum_segment_frames=None,
                 coverage_mode='image', minimum_anchor_coverage=.5):
        assert isinstance(minimum_inliers, int) and minimum_inliers > 0
        assert maximum_segment_frames is None or maximum_segment_frames >= 50
        assert coverage_mode in ('image', 'anchor_support')
        assert 0 < minimum_anchor_coverage <= 1
        self.coverage_mode = coverage_mode
        self.minimum_anchor_coverage = minimum_anchor_coverage
        self.maximum_segment_frames = maximum_segment_frames
        self.minimum_inliers = minimum_inliers
        self.orb = cv2.ORB_create(nfeatures=POLICY['features'])
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.anchor = None
        self.last_frame = -1
        self.low_run = 0

    def step(self, rgb, frame):
        assert frame == self.last_frame + 1
        assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
        self.last_frame = frame
        started = time.perf_counter()
        keypoints, descriptors = self.orb.detectAndCompute(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
        points = np.array([p.pt for p in keypoints], dtype=np.float32).reshape(-1, 2)
        evidence = dict(anchor_points=np.empty((0, 2), np.float32),
                        current_points=np.empty((0, 2), np.float32),
                        inliers=np.empty(0, bool))
        row = dict(frame=frame, anchor_frame=self.anchor[0] if self.anchor else frame,
                   features=len(points), ratio_matches=0, inliers=0, cells=0,
                   status='bootstrap', boundary=False, low_run=0)
        if self.anchor is None:
            self.anchor = (frame, points, descriptors)
        else:
            anchor_frame, anchor_points, anchor_descriptors = self.anchor
            row['anchor_features'] = len(anchor_points)
            if min(len(points), len(anchor_points)) < POLICY['minimum_features']:
                # Insufficient texture is unknown overlap, not evidence of a cut.
                row['status'] = 'unknown_texture'
                self.low_run = 0
            else:
                pairs = self.matcher.knnMatch(anchor_descriptors, descriptors, k=2)
                matches = [a for a, b in pairs if a.distance < POLICY['ratio'] * b.distance]
                # Each current feature contributes at most once to support.
                matches.sort(key=lambda match: match.distance)
                unique = {}
                for match in matches:
                    unique.setdefault(match.trainIdx, match)
                matches = list(unique.values())
                source = np.array([anchor_points[m.queryIdx] for m in matches],
                                  dtype=np.float32).reshape(-1, 2)
                target = np.array([points[m.trainIdx] for m in matches],
                                  dtype=np.float32).reshape(-1, 2)
                inliers = np.zeros(len(matches), dtype=bool)
                # Explicit two-model verification: planar/rotation and general
                # epipolar motion. Take the model with the most inliers.
                if len(matches) >= 4:
                    _, mask = cv2.findHomography(source, target, cv2.RANSAC,
                                                POLICY['ransac_pixels'])
                    if mask is not None:
                        inliers = mask.ravel().astype(bool)
                if len(matches) >= 8:
                    _, mask = cv2.findFundamentalMat(source, target, cv2.FM_RANSAC,
                                                    POLICY['ransac_pixels'], .99)
                    if mask is not None and int(mask.sum()) > int(inliers.sum()):
                        inliers = mask.ravel().astype(bool)
                h, w = rgb.shape[:2]
                grid = POLICY['grid']
                cells = np.floor(target[inliers] * [grid / w, grid / h]).astype(int)
                occupied = len(np.unique(cells, axis=0))
                low_coverage = occupied < POLICY['minimum_cells']
                if self.coverage_mode == 'anchor_support':
                    # Same features/matches as the image-grid control. Measure
                    # surviving anchor support in its own feature extent, so
                    # object size/position in the image cannot impose four cells.
                    origin = anchor_points.min(axis=0)
                    extent = np.maximum(anchor_points.max(axis=0) - origin, 1.)
                    anchor_cells = np.clip(np.floor(
                        (anchor_points - origin) * grid / extent), 0, grid - 1).astype(int)
                    supported_cells = np.clip(np.floor(
                        (source[inliers] - origin) * grid / extent), 0, grid - 1).astype(int)
                    available = len(np.unique(anchor_cells, axis=0))
                    supported = len(np.unique(supported_cells, axis=0))
                    coverage = supported / available
                    low_coverage = coverage < self.minimum_anchor_coverage
                    row.update(anchor_cells=available, supported_anchor_cells=supported,
                               anchor_coverage=coverage,
                               low_inliers=int(inliers.sum()) < self.minimum_inliers,
                               low_coverage=low_coverage)
                low = int(inliers.sum()) < self.minimum_inliers or low_coverage
                eligible = frame - anchor_frame >= POLICY['minimum_segment_frames']
                self.low_run = self.low_run + 1 if low and eligible else 0
                boundary = self.low_run >= POLICY['persistence']
                row.update(ratio_matches=len(matches), inliers=int(inliers.sum()),
                           cells=occupied, status='low_overlap' if low else 'supported',
                           boundary=boundary, low_run=self.low_run)
                evidence = dict(anchor_points=source, current_points=target, inliers=inliers)
                if boundary:
                    # Commit at the current frame, never backdate using future evidence.
                    self.anchor = (frame, points, descriptors)
                    self.low_run = 0
        if self.maximum_segment_frames is not None and frame - row['anchor_frame'] >= self.maximum_segment_frames and not row['boundary']:
            row.update(boundary=True, status='maximum_age')
            self.anchor = (frame, points, descriptors)
            self.low_run = 0
        row['seconds'] = time.perf_counter() - started
        return row, evidence


def run_policy(args, minimum_inliers, flow=False):
    # Independent RNG and anchor history per arm; thresholds change future anchors.
    cv2.setRNGSeed(0)
    name = 'segment_detector_flow' if flow else (
        f'segment_detector_inliers{minimum_inliers}' if args.sweep or args.flow else 'segment_detector')
    result = args.work / 'runs' / name
    result.mkdir(parents=True)
    write_json(result / 'config.json', dict(policy=FLOW_POLICY if flow else dict(POLICY, minimum_inliers=minimum_inliers), source=SOURCE,
        source_sha256=SOURCE_SHA256, opencv=cv2.__version__, seed=0, threads=1,
        causal=True, gt_used=False, tracking=False, thresholds_calibrated=False))
    detector = FlowSegmentDetector() if flow else SegmentDetector(minimum_inliers)
    boundaries = [0]
    rows = []
    with tarfile.open(args.out / SOURCE) as packed, (result / 'scores.jsonl').open('w') as log:
        manifest = json.load(packed.extractfile('office/manifest.json'))['inputs']
        assert len(manifest) == 2585
        for frame, entry in enumerate(manifest):
            data = packed.extractfile('office/model_rgb/' + Path(entry['file']).name).read()
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            assert bgr is not None, frame
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == entry['model_rgb_sha256'], frame
            row, evidence = detector.step(rgb, frame)
            row['timestamp'] = entry['timestamp']
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            if row['boundary']:
                boundaries.append(frame)
                anchor = row['anchor_frame']
                # Save the source pair and verified matches for every decision.
                anchor_data = packed.extractfile('office/model_rgb/' +
                    Path(manifest[anchor]['file']).name).read()
                (result / f'boundary_{frame:04d}_anchor.png').write_bytes(anchor_data)
                (result / f'boundary_{frame:04d}_current.png').write_bytes(data)
                np.savez(result / f'boundary_{frame:04d}.npz', **evidence)
                print('SEGMENT BOUNDARY', frame, 'threshold', minimum_inliers,
                      'anchor', anchor, 'inliers', row['inliers'],
                      'cells', row['cells'], flush=True)
    assert len(rows) == 2585
    assert all(r['anchor_frame'] <= r['frame'] for r in rows)
    seconds = np.array([r['seconds'] for r in rows])
    summary = dict(detector='flow' if flow else 'orb', minimum_inliers=None if flow else minimum_inliers,
        frames=len(rows), boundaries=boundaries,
        end_exclusive=len(rows), unknown_texture_frames=sum(
            r['status'] == 'unknown_texture' for r in rows),
        detector_seconds=float(seconds.sum()), median_ms=float(np.median(seconds) * 1000),
        p99_ms=float(np.quantile(seconds, .99) * 1000),
        segment_lengths=np.diff(boundaries + [len(rows)]).tolist(),
        earliest_possible_cuts=sum(b - a == 54 for a, b in zip(boundaries[:-1], boundaries[1:])),
        execution_passed=True, boundary_quality_verified=False)
    write_json(result / 'summary.json', summary)
    archive_directory(result, args.out / f'{args.tag}_{name}.tar')
    print('SEGMENT DETECTOR COMPLETE:', len(rows), 'frames; threshold', minimum_inliers,
          'boundaries', boundaries, flush=True)
    return summary


def main(args):
    cv2.setNumThreads(1)
    digest = hashlib.sha256()
    with (args.out / SOURCE).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8388608), b''):
            digest.update(chunk)
    assert digest.hexdigest() == SOURCE_SHA256, 'Input archive differs from reviewed 26127'
    thresholds = INLIER_THRESHOLDS if args.sweep else ((25,) if args.flow else (20,))
    summaries = [run_policy(args, threshold) for threshold in thresholds]
    if args.flow:
        summaries.append(run_policy(args, None, flow=True))
        write_json(args.work / 'segment_detector_flow_comparison.json', dict(
            summaries=summaries, tracking_test_pending=True,
            caveat='one uncalibrated flow policy versus ORB25; review before tracking'))
        print('SEGMENT FLOW PILOT COMPLETE: ORB25 + flow, tracking pending', flush=True)
    if args.sweep:
        write_json(args.work / 'segment_detector_sweep.json', dict(
            varied_parameter='minimum_inliers', thresholds=thresholds, summaries=summaries,
            tracking_test_pending=True, best_segmentation=None))
        print('SEGMENT DETECTOR SWEEP COMPLETE:', len(summaries), 'policies', flush=True)
    (args.work / 'JOB_OK').write_text('Detector executed; boundary quality review pending\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--sweep', action='store_true',
                        help='Compare 10/15/20/25/30 inliers, other settings fixed')
    modes.add_argument('--flow', action='store_true',
                       help='Review sequential anchor-track overlap versus ORB25 before tracking')
    main(parser.parse_args())
