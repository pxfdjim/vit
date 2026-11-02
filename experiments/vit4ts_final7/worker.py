"""Fixed 2x2 ViT4TS sweep, with the old source stitch kept as a separate baseline."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/pengxf/jim/VLM4TS')
OLD = ROOT / 'data/runs/20261016_vit4ts_diagnostics'
OUT = ROOT / 'data/runs/20261016_vit4ts_final7'
sys.path.insert(0, str(ROOT / 'experiments/vit4ts_diagnostics'))
import experiment as e

DATASETS = ('artificialWithAnomaly', 'realAWSCloudwatch', 'realAdExchange',
            'realTraffic', 'realTweets', 'MSL', 'SMAP')
CONFIGS = ('top_240', 'top_224', 'q25_240', 'q25_224')


def score_file(path, **items):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **items)
    temp.replace(path)


def q25_from_maps(maps, window, length):
    """Map each image onto its raw window, overlap 2-D maps, then row Q25."""
    count, height, width = maps.shape
    stride = window // 4
    covered = stride * (count - 1) + window
    sums = np.zeros((height, covered), dtype=np.float32)
    counts = np.zeros(covered, dtype=np.int32)
    if window != width:
        x = np.linspace(0, width - 1, window)
        left = np.floor(x).astype(int)
        right = np.minimum(left + 1, width - 1)
        weight = (x - left).astype(np.float32)
    for i, anomaly_map in enumerate(maps):
        mapped = (anomaly_map if window == width else
                  anomaly_map[:, left] * (1 - weight) + anomaly_map[:, right] * weight)
        start = i * stride
        sums[:, start:start + window] += mapped
        counts[start:start + window] += 1
    q = np.quantile(sums / np.maximum(counts, 1)[None, :], 0.25, axis=0)
    return e.align_anomaly_vector(q, length, window, stride, count)


def infer(model, frame, window):
    model.window_size = window
    capture = {}
    original = e.vit.stitch_anomaly_maps

    def stitch(maps, ratio, percent):
        capture['vectors'] = np.stack([e.aggregate_anomaly_map(m, percent) for m in maps])
        capture['q25'] = q25_from_maps(maps, window, len(frame))
        return original(maps, ratio, percent)

    e.vit.stitch_anomaly_maps = stitch
    try:
        official, timestamps = model.predict_scores(frame)
    finally:
        e.vit.stitch_anomaly_maps = original
    top = e.mapped_score(capture['vectors'], window, window // 4, len(frame))
    return official, top, capture['q25'], timestamps, capture['vectors']


def load_old(ds, signal, window, top, q25, official):
    source_name = 'official_240' if window == 240 else 'paper_window_224'
    old_file = OLD / 'results' / ds / signal / (source_name + '.json')
    if old_file.exists():
        record = json.loads(old_file.read_text())
        top['metrics'] = record['corrected_mapping']
        top['reused_from'] = str(old_file)
        official['metrics'] = record['official_mapping']
        official['reused_from'] = str(old_file)
    if window == 224:
        q_file = OLD / 'quantile_diagnostic/results' / ds / (signal + '.json')
        if q_file.exists():
            record = json.loads(q_file.read_text())
            if record['source_score_max_abs_diff'] != 0:
                raise RuntimeError('Old q25 source mismatch: ' + str(q_file))
            q25['metrics'] = record['row_quantile_metrics']
            q25['reused_from'] = str(q_file)


def process(model, ds, signal, labels):
    target = OUT / 'results' / ds / (signal + '.json')
    if target.exists():
        record = json.loads(target.read_text())
        if all(record['configs'][c].get('metrics') for c in CONFIGS) and record['official_baseline'].get('metrics'):
            return
    frame = pd.read_csv(ROOT / 'data' / ds / (signal + '.csv'))
    frame = frame.sort_values('timestamp')
    timestamps = frame.timestamp.to_numpy()
    audit = e.audit(ds, signal, frame, labels)
    configs = {c: {'window': 240 if c.endswith('240') else 224,
                    'stride': 60 if c.endswith('240') else 56,
                    'aggregation': 'top_25_percent_mean' if c.startswith('top') else 'row_q25_after_2d_overlap',
                    'time_mapping': 'image_columns_to_raw_window_then_overlap'} for c in CONFIGS}
    baseline = {'window': 240, 'stride': 60,
                'aggregation': 'top_25_percent_mean', 'time_mapping': 'official_source'}
    for window in (240, 224):
        top = configs[f'top_{window}']
        q25 = configs[f'q25_{window}']
        load_old(ds, signal, window, top, q25, baseline if window == 240 else {})
        if window == 224 and top.get('metrics') and q25.get('metrics'):
            continue
        # 240 always needs 2-D maps for Q25. 224 only runs for missing signals.
        official, top_score, q_score, observed, vectors = infer(model, frame, window)
        if not np.array_equal(observed, timestamps):
            raise RuntimeError(f'Timestamp mismatch: {ds}/{signal}')
        if window == 240:
            old_cache = OLD / 'cache' / ds / signal / 'official_240.npz'
            if old_cache.exists():
                with np.load(old_cache) as cache:
                    diff = float(np.max(np.abs(vectors - cache['vectors'])))
                if diff > 1e-6:
                    raise RuntimeError(f'240 cache mismatch: {ds}/{signal}, diff={diff}')
                top['cached_vector_max_abs_diff'] = diff
            if 'metrics' not in baseline:
                baseline['metrics'] = e.evaluate(model, official, timestamps, labels)
            if 'metrics' not in top:
                top['metrics'] = e.evaluate(model, top_score, timestamps, labels)
            score_file(OUT / 'scores' / ds / signal / 'window_240.npz',
                       official=official, top=top_score, q25=q_score, timestamps=timestamps)
        else:
            if 'metrics' not in top:
                top['metrics'] = e.evaluate(model, top_score, timestamps, labels)
            score_file(OUT / 'scores' / ds / signal / 'window_224.npz',
                       official=official, top=top_score, q25=q_score, timestamps=timestamps)
        q25['metrics'] = e.evaluate(model, q_score, timestamps, labels)
    if not all(configs[c].get('metrics') for c in CONFIGS) or not baseline.get('metrics'):
        raise RuntimeError('Incomplete metrics: ' + ds + '/' + signal)
    e.save_json(target, {'dataset': ds, 'signal': signal, 'audit': audit,
                         'clip': 'OpenAI CLIP ViT-B/16', 'preprocess': 'detrend_minmax',
                         'image_size': [224, 224], 'alphas': list(e.ALPHAS),
                         'configs': configs, 'official_baseline': baseline})
    print('DONE', ds, signal, 'length', len(frame), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    if os.environ.get('CLIP_PRETRAINED') != 'openai':
        raise RuntimeError('CLIP_PRETRAINED must equal openai')
    index, labels = e.index_and_labels()
    tasks = [(ds, name) for ds in DATASETS for name in index[ds]]
    assert len(tasks) == 125, len(tasks)
    assigned = [task for i, task in enumerate(tasks) if i % args.workers == args.worker]
    model = e.vit.ViT4TS(window_size=240, window_step_ratio=4.0,
                         image_size=(224, 224), verbose=False)
    assert model.model.pretrained_tag == 'openai'
    print('START', args.worker, 'gpu', os.environ.get('CUDA_VISIBLE_DEVICES'),
          'assigned', len(assigned), flush=True)
    for ds, signal in assigned:
        process(model, ds, signal, labels[signal])
    print('WORKER_DONE', args.worker, flush=True)


if __name__ == '__main__':
    main()
