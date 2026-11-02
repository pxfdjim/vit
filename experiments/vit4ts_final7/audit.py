"""Verify run completeness, input provenance, and cache identity."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/home/pengxf/jim/VLM4TS')
OLD = ROOT / 'data/runs/20261016_vit4ts_diagnostics'
OUT = ROOT / 'data/runs/20261016_vit4ts_final7'
CODE = ROOT / 'experiments/vit4ts_final7'
sys.path.insert(0, str(CODE))
from worker import DATASETS, CONFIGS
sys.path.insert(0, str(ROOT / 'experiments/vit4ts_diagnostics'))
import experiment as e

PAPER_TOTAL = {'artificialWithAnomaly': 24192, 'realAWSCloudwatch': 67644,
               'realAdExchange': 7965, 'realTraffic': 15662,
               'realTweets': 158511, 'MSL': 132046, 'SMAP': 562800}
PAPER_INTERVALS = {'artificialWithAnomaly': 6, 'realAWSCloudwatch': 30,
                   'realAdExchange': 11, 'realTraffic': 14,
                   'realTweets': 33, 'MSL': 36, 'SMAP': 67}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_manifest = json.loads((OLD / 'manifest.json').read_text())
    hashes = {name: sha(ROOT / name) for name in old_manifest['source_sha256']}
    changed = {name: [old_manifest['source_sha256'][name], hashes[name]]
               for name in hashes if hashes[name] != old_manifest['source_sha256'][name]}
    if changed:
        raise RuntimeError('Source or data changed since cached runs: ' + json.dumps(changed))
    index, _ = e.index_and_labels()
    dataset_audit = {}
    reused = {config: 0 for config in CONFIGS}
    score_files = 0
    old_vector_checks = []
    for ds in DATASETS:
        records = []
        for signal in index[ds]:
            record = json.loads((OUT / 'results' / ds / (signal + '.json')).read_text())
            records.append(record)
            if set(record['configs']) != set(CONFIGS):
                raise RuntimeError('Config mismatch ' + ds + '/' + signal)
            for config in CONFIGS:
                metrics = record['configs'][config]['metrics']
                if [row['alpha'] for row in metrics] != list(e.ALPHAS):
                    raise RuntimeError('Alpha mismatch ' + ds + '/' + signal)
                if 'reused_from' in record['configs'][config]:
                    reused[config] += 1
            if 'cached_vector_max_abs_diff' in record['configs']['top_240']:
                old_vector_checks.append(record['configs']['top_240']['cached_vector_max_abs_diff'])
            score_files += sum((OUT / 'scores' / ds / signal / f'window_{window}.npz').exists()
                               for window in (224, 240))
        raw_points = sum(r['audit']['length'] for r in records)
        duplicates = sum(r['audit']['timestamp_duplicates'] for r in records)
        intervals = sum(r['audit']['ground_truth_count'] for r in records)
        dataset_audit[ds] = {'signals': len(records), 'raw_points': raw_points,
                             'unique_timestamp_points': raw_points - duplicates,
                             'duplicate_timestamps': duplicates,
                             'ground_truth_intervals': intervals,
                             'paper_points': PAPER_TOTAL[ds],
                             'paper_intervals': PAPER_INTERVALS[ds],
                             'paper_point_delta_raw': raw_points - PAPER_TOTAL[ds]}
        if intervals != PAPER_INTERVALS[ds]:
            raise RuntimeError('Paper anomaly count mismatch: ' + ds)
    manifest = {'experiment': 'ViT4TS 7 datasets 2x2, fixed OpenAI CLIP',
                'paper': 'https://arxiv.org/pdf/2506.06836',
                'official_code': old_manifest['official_repository'],
                'official_commit': old_manifest['official_commit'],
                'clip_checkpoint_sha256': old_manifest['clip_checkpoint_sha256'],
                'source_sha256': hashes,
                'experiment_sha256': {p.name: sha(p) for p in CODE.glob('*.py')},
                'datasets': dataset_audit, 'reused_metrics': reused,
                'existing_240_vector_identity_checks': {'count': len(old_vector_checks),
                                                        'max_abs_diff': max(old_vector_checks)},
                'new_score_file_count': score_files,
                'new_result_count': sum(x['signals'] for x in dataset_audit.values()),
                'gpu_workers': [0, 1, 3, 4, 5, 7],
                'run_seconds': json.loads((OUT / 'finished.json').read_text())['wall_seconds']}
    if manifest['new_result_count'] != 125 or len(old_vector_checks) != 75:
        raise RuntimeError('Unexpected result or cache identity count')
    e.save_json(OUT / 'audit_manifest.json', manifest)
    print(json.dumps({'datasets': dataset_audit, 'reused_metrics': reused,
                      'existing_240_vector_identity_checks': manifest['existing_240_vector_identity_checks'],
                      'new_score_file_count': score_files}, indent=2))


if __name__ == '__main__':
    main()
