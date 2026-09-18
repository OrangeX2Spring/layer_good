"""Headless TUM plots and RGB/cloud/trail videos from saved artifacts, no inference."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from kvt_artifacts import write_ply
from kvt_confidence_videos import render


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save_figure(figure, path):
    figure.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(figure)
    assert cv2.imread(str(path)) is not None, path


def letterbox(image, width, height):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)))
    canvas = np.full((height, width, 3), 24, np.uint8)
    y, x = (height - len(resized)) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + len(resized), x:x + resized.shape[1]] = resized
    return canvas


def feature_views(inputs, result, manifest, config, decisions, keyframes, times):
    """Offline pooled-feature projection, causal-reference similarities and patch maps."""
    output = result / 'viz'
    with np.load(result / 'frame_features.npz') as saved:
        vectors = saved['descriptors']
        frame_ids = saved['frame_ids']
    n, dimension = vectors.shape
    assert n == len(times) and np.array_equal(frame_ids, np.arange(n))
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)
    # A single basis per sequence/layer keeps threshold plots comparable. Each
    # condition also saves that basis, so its archive is independently renderable.
    basis_path = result.parent / f'pca_{config["layer"]}.npz'
    own_projection = output / 'feature_projection.npz'
    if own_projection.exists():
        with np.load(own_projection) as saved:
            mean, components, variance_ratio = saved['mean'], saved['components'], saved['variance_ratio']
            basis_condition = str(saved['basis_condition'])
    elif basis_path.exists():
        with np.load(basis_path) as saved:
            mean, components, variance_ratio = saved['mean'], saved['components'], saved['variance_ratio']
            basis_condition = str(saved['basis_condition'])
    else:
        mean = vectors.mean(0, dtype=np.float64)
        centered = vectors.astype(np.float64) - mean
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
        total = float(np.maximum(eigenvalues, 0).sum())
        assert total > 0, 'All pooled frame features are identical; no PCA variance'
        components = eigenvectors[:, -2:][:, ::-1].T
        # Stable sign convention, without changing the subspace.
        for component in components:
            if component[np.argmax(np.abs(component))] < 0:
                component *= -1
        variance_ratio = np.maximum(eigenvalues[-2:][::-1], 0) / total
        basis_condition = config['name']
        np.savez(basis_path, mean=mean, components=components, variance_ratio=variance_ratio,
                 basis_condition=basis_condition)
    assert components.shape == (2, dimension)
    xy = (vectors - mean) @ components.T
    native_score = np.r_[np.nan, [r['score'] for r in decisions]]
    cap_blocked = np.array([r['frame'] for r in decisions if r['cap_blocked']], dtype=int)
    similarities = np.clip(vectors @ vectors[keyframes].T, -1, 1)
    available = frame_ids[:, None] > keyframes[None, :]
    similarities[~available] = np.nan  # Includes self: decisions precede insertion.
    np.savez(own_projection, xy=xy, mean=mean, components=components,
             variance_ratio=variance_ratio, native_score=native_score, frame_ids=frame_ids,
             similarity=similarities, keyframe_ids=keyframes, basis_condition=basis_condition)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, color, label in zip(axes, (times, native_score), ('Sequence time (s)', 'Native novelty score')):
        axis.plot(xy[:, 0], xy[:, 1], color='grey', alpha=.2, lw=.5)
        dots = axis.scatter(xy[:, 0], xy[:, 1], c=color, cmap='viridis', s=8)
        axis.scatter(xy[keyframes, 0], xy[keyframes, 1], facecolors='none', edgecolors='red',
                     s=65, linewidths=1, label='selected keyframes (incl. bootstrap)')
        axis.scatter(xy[cap_blocked, 0], xy[cap_blocked, 1], marker='x', color='orange',
                     s=12, label='candidate blocked by cap')
        axis.set(xlabel=f'PC1 ({variance_ratio[0]:.1%} reference variance)',
                 ylabel=f'PC2 ({variance_ratio[1]:.1%} reference variance)')
        axis.legend(fontsize=7)
        figure.colorbar(dots, ax=axis, label=label)
    figure.suptitle(f"{config['name']} | pooled {config['layer']} features\n"
                   'Offline PCA; shared basis per layer/sequence; projected distances are not selection scores')
    save_figure(figure, output / 'feature_pca.png')

    figure, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True,
                                gridspec_kw={'height_ratios': [3, 1]})
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#dddddd')
    matrix = axes[0].imshow(similarities.T, origin='lower', aspect='auto', cmap=cmap,
        extent=(-.5, n-.5, -.5, len(keyframes)-.5),
        vmin=float(np.nanmin(similarities)), vmax=1.)
    slots = np.unique(np.linspace(0, len(keyframes)-1, min(12, len(keyframes))).astype(int))
    axes[0].set_yticks(slots, [str(keyframes[i]) for i in slots])
    axes[0].set(xlabel='Source frame', ylabel='Cached keyframe source ID')
    figure.colorbar(matrix, ax=axes[0], label=f'Pooled cosine similarity (original {dimension}-D space)')
    axes[1].plot(frame_ids, native_score, label=f"native {config['score']} score", lw=.8)
    axes[1].axhline(config['threshold'], color='red', label='insertion threshold')
    axes[1].scatter(keyframes[1:], native_score[keyframes[1:]], color='red', s=12, label='inserted')
    axes[1].set(xlabel='Source frame', ylabel='Native novelty')
    axes[1].legend(fontsize=8)
    caveat = 'same pooled feature as selector' if config['score'] == 'cosine' else (
        'pooled-feature proxy only; patch-set selector uses a different score')
    figure.suptitle(f"{config['name']} | {caveat}\nGrey = keyframe not yet available; no future references")
    save_figure(figure, output / 'feature_similarity.png')

    if config['score'] in ('coverage', 'chamfer'):
        patch_overlays(inputs, result, manifest, config, decisions, times)
    (output / 'feature_manifest.json').write_text(json.dumps(dict(
        layer=config['layer'], frame_count=n, descriptor_dimension=dimension,
        pooling='L2-normalized mean of raw frame-local patch tokens',
        projection='offline PCA; full sequence fit of first condition per layer; never feeds selection',
        reference_variance_ratio=variance_ratio.tolist(),
        basis_condition=basis_condition,
        pooled_similarity_is_native_metric=config['score'] == 'cosine',
        unavailable_reference_cells='NaN where keyframe ID >= query ID',
        sources=['frame_features.npz', 'decisions.jsonl', 'kf_idx.npy', 'patch_novelty.npz (patch policies)']),
        indent=2) + '\n')


def patch_overlays(inputs, result, manifest, config, decisions, times):
    with np.load(result / 'patch_novelty.npz') as saved:
        maps, indices, grid = saved['maps'], saved['frame_ids'], saved['patch_grid']
    assert np.array_equal(indices, np.arange(1, len(times)))
    assert maps.shape == (len(indices), int(np.prod(grid)))
    # Coverage stores actual max cosine; Chamfer stores current-to-reference d².
    novelty = 1 - maps if config['score'] == 'coverage' else maps
    if config['score'] == 'coverage':
        recomputed = (maps < config['similarity_floor']).mean(1)
    else:
        recomputed = (maps.mean(1) + np.array([r['chamfer_reverse_mean'] for r in decisions])) / 2
    np.testing.assert_allclose(recomputed, [r['score'] for r in decisions], atol=1e-6, rtol=1e-5)
    choices = {}
    categories = [('high selected', [r for r in decisions if r['selected']], lambda r: -r['score']),
                  ('near rejected', [r for r in decisions if not r['candidate']],
                   lambda r: abs(r['score'] - config['threshold'])),
                  ('cap blocked', [r for r in decisions if r['cap_blocked']], lambda r: -r['score'])]
    for label, rows, ordering in categories:
        for row in sorted(rows, key=ordering)[:3]:
            choices.setdefault(row['frame'], label)
    for index in np.linspace(1, len(times)-1, 4).astype(int):
        choices.setdefault(int(index), 'timeline')
    choices = list(choices.items())[:12]
    maximum = max(float(np.quantile(novelty, .99)), 1e-6)
    figure, axes = plt.subplots(len(choices), 2, figsize=(10, 2.5*len(choices)),
                                squeeze=False, constrained_layout=True)
    for (index, reason), (left, right) in zip(choices, axes):
        path = inputs / 'model_rgb' / Path(manifest['inputs'][index]['file']).name
        bgr = cv2.imread(str(path))
        assert bgr is not None, path
        rgb = bgr[:, :, ::-1]
        row = decisions[index-1]
        patch_map = novelty[index-1].reshape(tuple(grid))
        pixels = cv2.resize(patch_map, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        left.imshow(rgb)
        left.set_title(f"frame {index}, {times[index]:.2f}s | {reason}", fontsize=9)
        right.imshow(rgb)
        overlay = right.imshow(pixels, cmap='magma', vmin=0, vmax=maximum, alpha=.65)
        reference = f" | ref {row['chamfer_reference_frame']}" if config['score'] == 'chamfer' else ''
        right.set_title(f"score {row['score']:.4f} / threshold {config['threshold']:g}{reference}", fontsize=9)
        for axis in (left, right):
            axis.axis('off')
    label = '1 - best cached-patch cosine' if config['score'] == 'coverage' else (
        'current-to-chosen-reference squared distance; score also includes reverse direction')
    figure.colorbar(overlay, ax=axes[:, 1].tolist(), label=label, shrink=.6)
    figure.suptitle(f"{config['name']} | actual pre-insertion patch novelty\n"
                   f'Per-run display scale clips above 99th percentile ({maximum:.4g}); stored maps are unclipped')
    save_figure(figure, result / 'viz' / 'patch_novelty.png')
    (result / 'viz' / 'patch_overlays.json').write_text(json.dumps(dict(
        source_frames=[dict(frame=i, reason=reason) for i, reason in choices],
        display_maximum=maximum, score_reconstruction_verified=True, label=label), indent=2) + '\n')


def visualize_run(inputs, result, video=False):
    manifest = json.loads((inputs / 'manifest.json').read_text())
    config = json.loads((result / 'config.json').read_text())
    metrics = json.loads((result / 'metrics.json').read_text())
    decisions = read_rows(result / 'decisions.jsonl')
    inference = read_rows(result / 'inference.jsonl')
    keyframes = np.load(result / 'kf_idx.npy')
    with np.load(result / 'evaluation.npz') as packed:
        evaluation = {key: packed[key] for key in packed.files}
    n = metrics['frames']
    times = np.array([r['timestamp'] for r in manifest['inputs'][:n]])
    times -= times[0]
    output = result / 'viz'
    output.mkdir(exist_ok=True)
    error = np.full(n, np.nan)
    error[evaluation['rgb_indices']] = evaluation['ate_per_frame_m']
    ground_truth = np.full((n, 3), np.nan)
    ground_truth[evaluation['rgb_indices']] = evaluation['reference'][:, :3, 3]
    estimate = evaluation['full_aligned_positions']
    np.testing.assert_allclose(np.sqrt(np.mean(evaluation['ate_per_frame_m'] ** 2)), metrics['ate_m'])

    figure, axes = plt.subplots(4, 2, figsize=(13, 13), constrained_layout=True)
    query_ids = np.array([r['frame'] for r in decisions])
    if config['policy'] == 'semantic':
        axes[0, 0].plot(times[query_ids], [r['score'] for r in decisions], lw=.7)
        axes[0, 0].axhline(config['threshold'], color='red', label='threshold')
        blocked = [r['frame'] for r in decisions if r['cap_blocked']]
        axes[0, 0].scatter(times[blocked], [decisions[i-1]['score'] for i in blocked],
                           s=5, color='orange', label='cap-blocked')
        axes[0, 0].legend(fontsize=8)
    else:
        axes[0, 0].text(.1, .5, f"Periodic interval {config['interval']}", transform=axes[0, 0].transAxes)
    axes[0, 0].set_ylabel('Novelty score')
    axes[0, 1].step(times, np.searchsorted(keyframes, np.arange(n), side='right'), where='post')
    axes[0, 1].axhline(config['cap'], color='red', ls='--', label='cap')
    axes[0, 1].set_ylabel('Retained keyframes')
    axes[0, 1].legend()
    axes[1, 0].plot(times, error, lw=.8)
    axes[1, 0].set_ylabel('Translation error (m); GT gaps blank')
    axes[1, 1].plot(times[query_ids], np.array([r['cache_bytes_before'] for r in decisions]) / 1024**2,
                    label='transformer K/V')
    axes[1, 1].plot(times[query_ids], np.array([r['feature_bytes_before'] for r in decisions]) / 1024**2,
                    label='selector descriptors')
    axes[1, 1].set_ylabel('Stored payload (MiB)')
    axes[1, 1].legend()
    for kind in ('query', 'rebuild'):
        rows = [r for r in inference if r['kind'] == kind]
        axes[2, 0].scatter([times[r['frame']] for r in rows],
                           [r['seconds'] * 1000 for r in rows], s=5, label=kind)
    axes[2, 0].set_ylabel('Model call (ms, synchronized)')
    axes[2, 0].legend()
    axes[2, 1].plot(times[query_ids], [r['selector_seconds'] * 1000 for r in decisions], lw=.7)
    axes[2, 1].set_ylabel('Selector (ms)')
    for axis, field, label in zip(axes[3],
            ('rpe_translation_per_pair_m', 'rpe_rotation_per_pair_deg'),
            ('Adjacent-frame translation RPE (m)', 'Adjacent-frame rotation RPE (deg)')):
        values = np.full(n, np.nan)
        values[evaluation['rpe_pair_start_indices'] + 1] = evaluation[field]
        axis.plot(times, values, lw=.8)
        axis.set_ylabel(label)
    for axis in axes.flat:
        axis.set_xlabel('Sequence time (s)')
        axis.grid(alpha=.2)
        for index in keyframes:
            axis.axvline(times[index], color='green', alpha=.08, lw=.6)
    figure.suptitle(f"{config['scene']} / {config['name']}\n"
                   f"ATE {metrics['ate_m']:.4f} m | GT {metrics['evaluated_fraction']:.1%} | "
                   f"{len(keyframes)} keys | cap reached: {metrics['cap_reached']}")
    save_figure(figure, output / 'diagnostics.png')

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (a, b) in zip(axes, ((0, 1), (0, 2), (1, 2))):
        axis.plot(estimate[:, a], estimate[:, b], color='red', lw=.8, label='estimate, one Sim(3)')
        axis.plot(ground_truth[:, a], ground_truth[:, b], color='green', lw=1, label='GT, gaps blank')
        axis.scatter(estimate[keyframes, a], estimate[keyframes, b], s=8, color='blue', label='keyframes')
        axis.set(xlabel=f'{"XYZ"[a]} (m)', ylabel=f'{"XYZ"[b]} (m)')
        axis.axis('equal')
        axis.legend(fontsize=7)
    figure.suptitle(f"{config['name']} | full-valid-trajectory alignment, {metrics['evaluated_fraction']:.1%} GT")
    save_figure(figure, output / 'trajectory.png')

    for page, start in enumerate(range(0, len(keyframes), 16)):
        sheet = np.full((4 * 230, 4 * 320, 3), 24, np.uint8)
        for slot, index in enumerate(keyframes[start:start+16]):
            path = inputs / 'model_rgb' / Path(manifest['inputs'][index]['file']).name
            bgr = cv2.imread(str(path))
            assert bgr is not None, path
            tile = letterbox(bgr, 320, 200)
            y, x = slot // 4 * 230, slot % 4 * 320
            sheet[y:y+200, x:x+320] = tile
            cv2.putText(sheet, f'frame {index} | {times[index]:.2f}s', (x+8, y+222),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
        assert cv2.imwrite(str(output / f'keyframes_{page:02d}.png'), sheet)

    if config['policy'] == 'semantic':
        feature_views(inputs, result, manifest, config, decisions, keyframes, times)
    if video:
        tracking_video(inputs, result, manifest, config, metrics, evaluation, times, error, keyframes)
    artifacts = sorted(p for p in output.iterdir() if p.suffix in ('.png', '.mp4', '.ply'))
    record = dict(condition=config['name'], scene=config['scene'], video=video,
                  alignment='same full-valid-trajectory Sim(3) as reported ATE',
                  frozen_final_cloud=True, source_files=['final_scene.npz', 'evaluation.npz',
                      'config.json', 'metrics.json', 'kf_idx.npy', 'decisions.jsonl',
                      'inference.jsonl', 'shared inputs/manifest.json and model_rgb/'],
                  artifacts=[dict(file=p.name, bytes=p.stat().st_size) for p in artifacts])
    (output / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print('VIZ OK', config['scene'], config['name'], flush=True)


def tracking_video(inputs, result, manifest, config, metrics, evaluation, times, error, keyframes):
    output = result / 'viz'
    with np.load(result / 'final_scene.npz') as scene:
        _, unique = np.unique(scene['frame_ids'], return_index=True)
        ids = scene['frame_ids'][unique]
        xyz, confidence = scene['xyz'][unique], scene['confidence'][unique]
        threshold = float(scene['threshold'])
    rgb = []
    for index in ids:
        path = inputs / 'model_rgb' / Path(manifest['inputs'][index]['file']).name
        bgr = cv2.imread(str(path))
        assert bgr is not None, path
        rgb.append(bgr[:, :, ::-1])
    rgb = np.stack(rgb)
    assert rgb.shape == xyz.shape and confidence.shape == xyz.shape[:-1]
    keep = confidence > threshold
    assert keep.any(), 'No scene points survive the native visualization threshold'
    points = (float(evaluation['alignment_scale']) * xyz[keep] @ evaluation['alignment_rotation'].T
              + evaluation['alignment_translation'])
    colors = rgb[keep]
    assert np.isfinite(points).all()
    write_ply(output / 'scene_gt_aligned.ply', points, colors)
    estimated = evaluation['full_aligned_positions']
    gt = np.full_like(estimated, np.nan)
    gt[evaluation['rgb_indices']] = evaluation['reference'][:, :3, 3]
    low, high = np.percentile(points, [1, 99], axis=0)
    bounds = np.vstack([low, high, estimated, evaluation['reference'][:, :3, 3]])
    center = (bounds.min(0) + bounds.max(0)) / 2
    span = float(np.linalg.norm(bounds.max(0) - bounds.min(0)) * 1.1)
    assert span > 0
    angle = np.deg2rad(20)
    rotation = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                         [0., np.sin(angle), np.cos(angle)]])
    cloud = np.ascontiguousarray(render(points, colors, np.ones(len(points)),
                                        center, rotation, span, 640)[..., ::-1])
    estimated_pixels = np.floor((((estimated - center) @ rotation)[:, :2] / span + .5) * 640).astype(int)
    valid = np.isfinite(gt).all(1)
    gt_pixels = np.zeros((len(gt), 2), dtype=int)
    gt_pixels[valid] = np.floor((((gt[valid] - center) @ rotation)[:, :2] / span + .5) * 640).astype(int)
    fps = 15
    source_ids = np.searchsorted(times, np.arange(0, times[-1], 1 / fps), side='right') - 1
    source_ids = np.r_[source_ids, len(times) - 1]
    path = output / 'tracking.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 800))
    assert writer.isOpened(), f'Cannot open video encoder: {path}'
    trail = cloud.copy()
    previous = 0
    for video_index, index in enumerate(source_ids):
        for i in range(previous + 1, index + 1):
            cv2.line(trail, tuple(estimated_pixels[i-1]), tuple(estimated_pixels[i]), (70, 70, 240), 1)
            if valid[i-1] and valid[i] and times[i] - times[i-1] <= .1:
                cv2.line(trail, tuple(gt_pixels[i-1]), tuple(gt_pixels[i]), (80, 220, 80), 1)
        previous = index
        image_path = inputs / 'model_rgb' / Path(manifest['inputs'][index]['file']).name
        bgr = cv2.imread(str(image_path))
        assert bgr is not None, image_path
        canvas = np.full((800, 1280, 3), 24, np.uint8)
        canvas[:640, :640] = letterbox(bgr, 640, 640)
        canvas[:640, 640:] = trail
        cv2.circle(canvas, (int(estimated_pixels[index, 0]) + 640, int(estimated_pixels[index, 1])),
                   4, (70, 70, 240), -1)
        per_frame = f'error {error[index]:.3f} m' if valid[index] else 'GT unavailable here'
        lines = [f"{config['scene']} | {config['name']}",
                 f"frame {index}/{len(times)-1} | {times[index]:.2f}s | "
                 f"keys {np.searchsorted(keyframes, index, side='right')}/{config['cap']} | {per_frame}",
                 f"ATE {metrics['ate_m']:.4f} m on {metrics['evaluated_fraction']:.1%} GT | "
                 'green: GT, red: estimate (one Sim(3)); gaps not connected',
                 'FROZEN FINAL CLOUD; growing trails | 15 fps timestamp playback | RGB input, no object mask']
        for row, text in enumerate(lines):
            cv2.putText(canvas, text, (12, 670 + row*35), cv2.FONT_HERSHEY_SIMPLEX,
                        .53, (240, 240, 240), 1, cv2.LINE_AA)
        writer.write(canvas)
        if video_index in (0, len(source_ids)//2, len(source_ids)-1):
            assert cv2.imwrite(str(output / f'preview_{video_index:05d}.png'), canvas)
    writer.release()
    # Decode every frame: an encoder opening successfully does not prove a valid MP4.
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape == (800, 1280, 3)
        decoded += 1
    capture.release()
    assert decoded == len(source_ids), (decoded, len(source_ids), path)
    (output / 'video_frames.json').write_text(json.dumps(dict(fps=fps,
        source_indices=source_ids.tolist(), view_center=center.tolist(), view_span=span,
        view_rotation=rotation.tolist(), confidence_threshold=threshold,
        points=int(keep.sum()), decoded_frames=decoded,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()), indent=2) + '\n')


def visualize_summary(work, scene):
    records = [r for r in json.loads((work / 'summary.json').read_text()) if r['scene'] == scene]
    output = work / 'visualizations' / scene
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    families = [('original', 'black', 's'), ('periodic', 'grey', 's'),
                ('decoder0_cosine', 'tab:blue', 'o'), ('encoder_cosine', 'tab:orange', 'o'),
                ('decoder0_coverage', 'tab:green', '^'), ('decoder0_chamfer', 'tab:purple', 'D')]
    for family, color, marker in families:
        selected = [r for r in records if (r['policy'] if r['policy'] != 'semantic'
                     else f"{r['layer']}_{r['score']}") == family]
        for axis, field, factor, label in zip(axes,
                ('keyframes', 'seconds', 'peak_reserved_bytes'), (1, 1, 1/1024**3),
                ('Actual keyframes', 'Synchronous run seconds (includes export)', 'Peak reserved GiB')):
            axis.scatter([r[field]*factor for r in selected], [r['ate_m'] for r in selected],
                         color=color, marker=marker, label=family)
            for r in selected:
                text = f"{r['threshold']:g}" if r['policy'] == 'semantic' else str(r['keyframes'])
                axis.annotate(text, (r[field]*factor, r['ate_m']), fontsize=7, xytext=(3, 3),
                              textcoords='offset points')
            axis.set(xlabel=label, ylabel='ATE RMSE (m)')
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    figure.suptitle(f'{scene} | all tested settings; threshold labels | partial GT stays partial')
    save_figure(figure, output / 'accuracy_cost.png')

    manifest = json.loads((work / 'inputs' / scene / 'manifest.json').read_text())
    times = np.array([r['timestamp'] for r in manifest['inputs']])
    times -= times[0]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, family in zip(axes.flat, [f[0] for f in families[2:]]):
        for r in records:
            if r['policy'] == 'semantic' and f"{r['layer']}_{r['score']}" == family:
                ids = np.load(work / 'runs' / scene / r['name'] / 'kf_idx.npy')
                axis.step(times, np.searchsorted(ids, np.arange(len(times)), side='right'),
                          where='post', label=f"threshold {r['threshold']:g}")
        for name, linestyle in (('original', '--'), ('original_extended', ':')):
            ids = np.load(work / 'runs' / scene / name / 'kf_idx.npy')
            axis.step(times, np.searchsorted(ids, np.arange(len(times)), side='right'),
                      where='post', color='black', ls=linestyle, label=name)
        axis.set(title=family, xlabel='Sequence time (s)', ylabel='Retained keyframes')
        axis.legend(fontsize=8)
    save_figure(figure, output / 'saturation.png')
    print('SUMMARY VIZ OK', scene, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='mode', required=True)
    one = subparsers.add_parser('run')
    one.add_argument('--inputs', type=Path, required=True)
    one.add_argument('--result', type=Path, required=True)
    one.add_argument('--video', action='store_true')
    summary = subparsers.add_parser('summary')
    summary.add_argument('--work', type=Path, required=True)
    summary.add_argument('--scene', required=True)
    args = parser.parse_args()
    if args.mode == 'run':
        visualize_run(args.inputs, args.result, args.video)
    else:
        visualize_summary(args.work, args.scene)
