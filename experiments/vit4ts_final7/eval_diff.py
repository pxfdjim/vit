"""Compare paper-consistent project TP counts against author repository counts."""
import importlib.util
import csv
import json
from pathlib import Path
from statistics import mean

from worker import ROOT, OUT, DATASETS, CONFIGS, e

path = ROOT / 'data/runs/official_reference/src/evaluation/evaluate.py'
spec = importlib.util.spec_from_file_location('author_evaluate', path)
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)
index, labels = e.index_and_labels()
checked = 0
differences = []
author_records = {}
for ds in DATASETS:
    author_records[ds] = {config: [] for config in (*CONFIGS, 'official_baseline')}
    for signal in index[ds]:
        record = json.loads((OUT / 'results' / ds / (signal + '.json')).read_text())
        for config in (*CONFIGS, 'official_baseline'):
            config_record = (record['official_baseline'] if config == 'official_baseline'
                             else record['configs'][config])
            signal_rows = []
            for item in config_record['metrics']:
                official = author.window_wise_metrics(labels[signal], item['predicted_intervals'])
                actual = {key: item[key] for key in ('TP', 'FP', 'FN')}
                official_metrics = author.compute_precision_recall_f1(official)
                signal_rows.append({'alpha': item['alpha'], 'precision': official_metrics['precision'],
                                    'recall': official_metrics['recall'], 'f1': official_metrics['F1']})
                checked += 1
                if official != actual:
                    differences.append({'dataset': ds, 'signal': signal, 'config': config,
                                        'alpha': item['alpha'], 'project': actual, 'author': official})
            author_records[ds][config].append(signal_rows)
result = {'checked': checked, 'different': len(differences), 'differences': differences}
e.save_json(OUT / 'evaluation_implementation_diff.json', result)
rows = []
for ds in DATASETS:
    for config in (*CONFIGS, 'official_baseline'):
        by_alpha = []
        for alpha in e.ALPHAS:
            sample = [next(item for item in signal if item['alpha'] == alpha)
                      for signal in author_records[ds][config]]
            by_alpha.append({'alpha': alpha, **{metric: mean(item[metric] for item in sample)
                                                for metric in ('precision', 'recall', 'f1')}})
        chosen = max(by_alpha, key=lambda item: item['f1'])
        rows.append({'dataset': ds, 'config': config, 'signals': len(author_records[ds][config]),
                     'best_alpha': chosen['alpha'], 'precision': chosen['precision'],
                     'recall': chosen['recall'], 'f1max_fixed_alpha': chosen['f1'],
                     'f1max_per_signal': mean(max(item['f1'] for item in signal)
                                              for signal in author_records[ds][config])})
with (OUT / 'author_evaluation_comparison.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({'checked': checked, 'different': len(differences),
                  'affected_datasets': sorted(set(x['dataset'] for x in differences)),
                  'first_differences': differences[:3],
                  'official_source_macro_f1': mean(row['f1max_fixed_alpha'] for row in rows
                                                   if row['config'] == 'official_baseline')}, indent=2))
