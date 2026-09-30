"""CAMP-only capture/replay of independent matrix cache approximations.

Dense model trajectories supply immutable teacher traces. Replay is an operator
ablation, NOT a closed-loop tracking evaluation or deployable kernel speed test.
External traces in the same manifest format support other frozen backbones.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'kv_tracker'))
from kv_tracker.math_cache import MathCache, dense_response


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class TraceCapture:
    """Observe the actual SDPA inputs; never recompute norm/RoPE independently."""
    def __init__(self, attentions, out, config, host):
        self.attentions, self.out, self.config, self.host = attentions, out, config, host
        self.frame, self.active = -1, None
        self.records, self.prefix = [], {}

    def before(self, layer, module, args, kwargs):
        assert self.active is None
        self.active = layer

    def after(self, module, args, kwargs, output):
        self.active = None

    def sdpa(self, q, k, v, *args, **kwargs):
        output = self.original_sdpa(q, k, v, *args, **kwargs)
        if self.active is None or self.frame < self.config['bank_frames']:
            return output
        assert q.ndim == k.ndim == v.ndim == 4 and q.shape[0] == 1
        assert not args, 'Capture contract requires keyword SDPA options'
        assert kwargs.get('attn_mask') is None and not kwargs.get('is_causal', False)
        assert kwargs.get('dropout_p', 0.) == 0. and kwargs.get('scale') is None
        assert not kwargs.get('enable_gqa', False)
        assert q.shape[1] == k.shape[1] == v.shape[1]
        layer = self.active
        if layer not in self.prefix:
            self.prefix[layer] = k.shape[2] - q.shape[2]
        prefix = self.prefix[layer]
        assert prefix > 0 and k.shape[2] > prefix
        rows = min(self.config['query_rows'], q.shape[2])
        special = min(5, rows)
        selected = torch.arange(special, device=q.device)
        if rows > special:
            selected = torch.cat((selected, torch.linspace(special, q.shape[2] - 1,
                rows - special, device=q.device).round().long()))
        for head in sorted({0, q.shape[1] - 1}):
            name = f'layer{layer:02d}_head{head:02d}_frame{self.frame:03d}.pt'
            trace = dict(q=q[0, head, selected].detach().cpu().clone(),
                k=k[0, head, :prefix].detach().cpu().clone(),
                v=v[0, head, :prefix].detach().cpu().clone(),
                current_k=k[0, head, prefix:].detach().cpu().clone(),
                current_v=v[0, head, prefix:].detach().cpu().clone(),
                native=output[0, head, selected].detach().cpu().clone())
            torch.save(trace, self.out / name)
            self.records.append(dict(file=name, frame=self.frame, layer=layer, head=head,
                query_indices=selected.cpu().tolist(),
                role='calibration' if self.frame == self.config['bank_frames'] else 'evaluation',
                cache_semantics='post-norm/post-position; fixed past prefix; exact later/current suffix',
                sha256=sha256(self.out / name)))
        return output

    @contextmanager
    def installed(self):
        from functools import partial
        self.original_sdpa = F.scaled_dot_product_attention
        hooks, aliases = [], []
        for layer, attention, scope in self.attentions:
            # Pi3 calls attention.forward_w_cache directly, bypassing attention
            # module hooks. Scope the enclosing block, which is called normally.
            hooks.append(scope.register_forward_pre_hook(partial(self.before, layer), with_kwargs=True))
            hooks.append(scope.register_forward_hook(self.after, with_kwargs=True))
            namespace = sys.modules[attention.__class__.__module__]
            if hasattr(namespace, 'scaled_dot_product_attention') and namespace not in [x[0] for x in aliases]:
                aliases.append((namespace, namespace.scaled_dot_product_attention))
                namespace.scaled_dot_product_attention = self.sdpa
        F.scaled_dot_product_attention = self.sdpa
        try:
            yield self
        finally:
            F.scaled_dot_product_attention = self.original_sdpa
            for namespace, original in aliases:
                namespace.scaled_dot_product_attention = original
            for hook in hooks:
                hook.remove()
            self.active = None


def capture(args, config):
    import cv2
    import numpy as np
    from kvt_tum_run import prepare
    assert args.zip.is_file()
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = args.out / 'inputs'
    prepare(args.zip, inputs, 308, .02, start=args.start, count=128)
    manifest = json.loads((inputs / 'manifest.json').read_text())
    count = config['bank_frames'] + config['query_frames']
    assert 2 <= config['bank_frames'] and 2 <= config['query_frames'] and count <= 128
    images = []
    for record in manifest['inputs'][:count]:
        bgr = cv2.imread(str(inputs / 'model_rgb' / Path(record['file']).name))
        assert bgr is not None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == record['model_rgb_sha256']
        images.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.)
    if args.host == 'pi3':
        from kv_tracker.pi3_utilts import load_pi3_from_pretrained, move_pi3_mlps_to_bfloat32
        model = move_pi3_mlps_to_bfloat32(load_pi3_from_pretrained()).eval()
        layers = list(range(1, len(model.decoder), 2))
        chosen = sorted({layers[0], layers[len(layers) // 2], layers[-1]})
        attentions = [(i, model.decoder[i].attn, model.decoder[i]) for i in chosen]
        adapter = None
    else:
        from types import SimpleNamespace
        from stream_cache_sweep import load_model
        from stream_cache_adapters import StreamAdapter
        assert args.checkpoint is not None and args.checkpoint.is_file()
        model = load_model(SimpleNamespace(host='streamvggt', checkpoint=args.checkpoint))
        adapter = StreamAdapter(model, 'streamvggt')
        blocks = model.aggregator.global_blocks
        chosen = sorted({0, len(blocks) // 2, len(blocks) - 1})
        attentions = [(i, blocks[i].attn, blocks[i]) for i in chosen]
    identity = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        identity.update(name.encode())
        identity.update(str((value.dtype, tuple(value.shape))).encode())
        identity.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    write(args.out / 'model_identity.json', dict(host=args.host,
        loaded_state_sha256=identity.hexdigest(),
        digest_scope='sorted tensor names/dtypes/shapes and loaded raw bytes; includes inference dtype conversion'))
    traces = args.out / 'traces'
    traces.mkdir()
    recorder = TraceCapture(attentions, traces, config, args.host)
    try:
        with recorder.installed(), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if args.host == 'pi3':
                bank = torch.stack(images[:config['bank_frames']])[None].cuda()
                model(bank, cam_only=True, store_cache=True)
                for frame in range(config['bank_frames'], count):
                    recorder.frame = frame
                    model(images[frame][None, None].cuda(), cam_only=True, use_cache=True)
            else:
                for frame in range(count):
                    recorder.frame = frame
                    adapter.forward(images[frame][None, None].cuda(), frame)
    finally:
        if adapter is not None:
            adapter.close()
    expected = len(chosen) * 2 * config['query_frames']
    assert len(recorder.records) == expected, (len(recorder.records), expected)
    write(traces / 'manifest.json', dict(host=args.host, source_zip_sha256=sha256(args.zip),
        source_start=args.start, config=config, records=recorder.records,
        evidence='dense teacher replay only; no closed-loop approximation execution'))
    print('MATH CACHE TRACE CAPTURE OK', args.host, len(recorder.records), flush=True)


def load_trace(root, record):
    path = root / record['file']
    assert path.resolve().is_relative_to(root.resolve()) and sha256(path) == record['sha256']
    trace = torch.load(path, map_location='cuda', weights_only=True)
    assert set(trace) == {'q', 'k', 'v', 'current_k', 'current_v', 'native'}
    assert all(torch.isfinite(x).all() for x in trace.values())
    return trace


def replay(args, config):
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    groups = {}
    for record in manifest['records']:
        groups.setdefault((record['layer'], record['head']), []).append(record)
    reports = []
    for (layer, head), records in sorted(groups.items()):
        records.sort(key=lambda r: r['frame'])
        calibration = [r for r in records if r['role'] == 'calibration']
        evaluation = [r for r in records if r['role'] == 'evaluation']
        assert len(calibration) == 1 and evaluation
        assert all(r['frame'] > calibration[0]['frame'] for r in evaluation)
        base = load_trace(root, calibration[0])
        dense_bytes = sum(x.numel() * x.element_size() for x in (base['k'], base['v']))
        for case in config['cases']:
            cache = MathCache(dict(case, seed=config['seed']))
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            cache.fit(base['k'], base['v'], base['q'])
            torch.cuda.synchronize()
            construction_seconds = time.perf_counter() - started
            construction_peak = torch.cuda.max_memory_allocated() - before
            rows = []
            for record in evaluation:
                trace = load_trace(root, record)
                assert torch.equal(trace['k'], base['k']) and torch.equal(trace['v'], base['v'])
                q, ck, cv = trace['q'], trace['current_k'], trace['current_v']
                reference, _ = dense_response(q, torch.cat((trace['k'], ck)), torch.cat((trace['v'], cv)))
                native_error = (trace['native'].float() - reference).norm() / reference.norm().clamp_min(1e-12)
                # BF16 fused output and FP32 explicit softmax need not be bit exact.
                assert float(native_error) <= .05, ('native capture fidelity', float(native_error))
                estimate = cache.read(q, ck, cv)
                finite = bool(torch.isfinite(estimate).all())
                torch.save(dict(reference=reference.cpu(), estimate=estimate.cpu()),
                    args.out / f'{case["name"]}_l{layer}_h{head}_f{record["frame"]}.pt')
                row = dict(frame=record['frame'], finite=finite, native_relative_error=float(native_error))
                if finite:
                    difference = estimate - reference
                    per_query = difference.norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-12)
                    row.update(relative_error=float(difference.norm() / reference.norm().clamp_min(1e-12)),
                        query_error_p99=float(torch.quantile(per_query, .99)),
                        max_abs=float(difference.abs().max()))
                    timings = []
                    full_k = torch.cat((trace['k'], ck))[None, None]
                    full_v = torch.cat((trace['v'], cv))[None, None]
                    native_times = []
                    for _ in range(2):
                        F.scaled_dot_product_attention(q[None, None], full_k, full_v)
                    torch.cuda.synchronize()
                    for _ in range(config['timing_repeats']):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        native_value = F.scaled_dot_product_attention(q[None, None], full_k, full_v)
                        torch.cuda.synchronize()
                        native_times.append(time.perf_counter() - started)
                        del native_value
                    # Teacher inputs stay resident as benchmark fixtures, but are
                    # excluded from incremental reader peaks and method state.
                    for _ in range(2):
                        cache.read(q, ck, cv)
                    torch.cuda.synchronize()
                    resident = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    for _ in range(config['timing_repeats']):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        value = cache.read(q, ck, cv)
                        torch.cuda.synchronize()
                        timings.append(time.perf_counter() - started)
                        del value
                    row.update(read_median_seconds=statistics.median(timings),
                        native_sdpa_median_seconds=statistics.median(native_times),
                        read_workspace_peak_bytes=max(0, torch.cuda.max_memory_allocated() - resident))
                rows.append(row)
            report = dict(case=case, layer=layer, head=head, history_tokens=len(base['k']),
                dense_history_bytes=dense_bytes, persistent_tensor_bytes=cache.persistent_bytes(),
                construction_seconds=construction_seconds, construction_peak_extra_bytes=construction_peak,
                finite_pass=all(row['finite'] for row in rows), rows=rows,
                memory_scope='tensor storage; excludes Python objects and shared teacher fixtures',
                timing_scope='FP32 eager reader including decoding; not a fused production kernel')
            if case['method'] == 'exact':
                assert report['finite_pass'] and max(row['max_abs'] for row in rows) < 2e-5
            reports.append(report)
            write(args.out / 'report.json', dict(host=manifest['host'], config=config, reports=reports,
                complete=False, next='manual operator/resource review; no automatic tracking evaluation'))
            print('MATH CACHE CASE', case['name'], layer, head,
                  report['persistent_tensor_bytes'], report['finite_pass'], flush=True)
            del cache
    write(args.out / 'report.json', dict(host=manifest['host'], config=config, reports=reports,
        complete=True, next='manual operator/resource review; no automatic tracking evaluation'))
    summary = []
    for case in config['cases']:
        selected = [r for r in reports if r['case']['name'] == case['name']]
        valid = all(r['finite_pass'] for r in selected)
        row = dict(name=case['name'], finite_pass=valid,
            max_persistent_to_dense_ratio=max(r['persistent_tensor_bytes'] / r['dense_history_bytes'] for r in selected),
            construction_seconds=sum(r['construction_seconds'] for r in selected))
        if valid:
            samples = [sample for r in selected for sample in r['rows']]
            row.update(mean_relative_error=statistics.mean(s['relative_error'] for s in samples),
                worst_query_error_p99=max(s['query_error_p99'] for s in samples),
                median_reader_to_native_time_ratio=statistics.median(
                    s['read_median_seconds'] / s['native_sdpa_median_seconds'] for s in samples))
        summary.append(row)
    write(args.out / 'summary.json', dict(host=manifest['host'], cases=summary,
        evidence='operator screen only; different actual byte budgets; no tracking or production speed claim'))
    print('MATH CACHE SUMMARY', json.dumps(summary, allow_nan=False), flush=True)
    print('MATH CACHE ABLATION COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'replay'))
    parser.add_argument('--config', type=Path, default=REPO / 'tools/math_cache_ablation.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--host', choices=('pi3', 'streamvggt'))
    parser.add_argument('--zip', type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    assert sys.platform == 'linux' and torch.cuda.is_available(), 'Execute in a CAMP GPU allocation'
    config = json.loads(args.config.read_text())
    assert len({c['name'] for c in config['cases']}) == len(config['cases'])
    assert all(Path(c['name']).name == c['name'] and c['name'] not in ('.', '..')
               for c in config['cases']), 'Case names must be single safe path components'
    assert config['timing_repeats'] > 0 and config['query_rows'] > 0
    torch.manual_seed(config['seed'])
    if args.mode == 'capture':
        assert args.host is not None and args.zip is not None
        capture(args, config)
    else:
        assert args.manifest is not None
        replay(args, config)


if __name__ == '__main__':
    main()
