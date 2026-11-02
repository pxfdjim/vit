"""Summarize all 125 signals under one fixed evaluation protocol."""
import csv
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path('/home/pengxf/jim/VLM4TS')
OUT = ROOT / 'data/runs/20261016_vit4ts_final7'
sys.path.insert(0, str(ROOT / 'experiments/vit4ts_diagnostics'))
import experiment as e
from worker import DATASETS, CONFIGS

PAPER_F1 = {
    'artificialWithAnomaly': (0.260, 0.545, 0.286),
    'realAWSCloudwatch': (0.366, 0.368, 0.400),
    'realAdExchange': (0.400, 0.615, 0.353),
    'realTraffic': (0.500, 0.486, 0.615),
    'realTweets': (0.395, 0.549, 0.597),
    'MSL': (0.425, 0.543, 0.500),
    'SMAP': (0.484, 0.677, 0.726),
}
PAPER_PR = {
    'artificialWithAnomaly': {'precision': (0.152, 0.750, 1.000), 'recall': (0.909, 0.429, 0.167)},
    'realAWSCloudwatch': {'precision': (0.233, 0.286, 0.643), 'recall': (0.857, 0.516, 0.290)},
    'realAdExchange': {'precision': (0.276, 0.571, 0.500), 'recall': (0.727, 0.667, 0.273)},
    'realTraffic': {'precision': (0.353, 0.391, 0.727), 'recall': (0.857, 0.643, 0.533)},
    'realTweets': {'precision': (0.252, 0.413, 0.590), 'recall': (0.921, 0.816, 0.605)},
    'MSL': {'precision': (0.271, 0.472, 0.593), 'recall': (0.978, 0.641, 0.432)},
    'SMAP': {'precision': (0.326, 0.598, 0.831), 'recall': (0.940, 0.779, 0.645)},
}


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records, key, ds, config):
    by_alpha = []
    for alpha in e.ALPHAS:
        rows = [next(row for row in record[key]['metrics'] if row['alpha'] == alpha)
                for record in records]
        by_alpha.append({'alpha': alpha, **{metric: mean(row[metric] for row in rows)
                                            for metric in ('precision', 'recall', 'f1')},
                         **{metric: sum(row[metric] for row in rows)
                            for metric in ('TP', 'FP', 'FN')}})
    chosen = max(by_alpha, key=lambda row: row['f1'])
    source_style = mean(max(row['f1'] for row in record[key]['metrics']) for record in records)
    paper = PAPER_F1[ds]
    alpha_index = e.ALPHAS.index(chosen['alpha'])
    paper_precision = PAPER_PR[ds]['precision'][alpha_index]
    paper_recall = PAPER_PR[ds]['recall'][alpha_index]
    summary = {'dataset': ds, 'config': config, 'signals': len(records),
               'best_alpha': chosen['alpha'], 'precision': chosen['precision'],
               'recall': chosen['recall'], 'f1max_fixed_alpha': chosen['f1'],
               'f1max_per_signal': source_style, 'paper_f1max': max(paper),
               'gap_to_paper': chosen['f1'] - max(paper),
               'paper_precision_at_alpha': paper_precision,
               'paper_recall_at_alpha': paper_recall,
               'precision_gap_at_alpha': chosen['precision'] - paper_precision,
               'recall_gap_at_alpha': chosen['recall'] - paper_recall}
    detail = {**summary, 'per_alpha': by_alpha,
              'paper_per_alpha': dict(zip([str(a) for a in e.ALPHAS], paper)),
              'paper_precision_per_alpha': dict(zip([str(a) for a in e.ALPHAS], PAPER_PR[ds]['precision'])),
              'paper_recall_per_alpha': dict(zip([str(a) for a in e.ALPHAS], PAPER_PR[ds]['recall'])),
              'per_alpha_gap': {str(a): by_alpha[i]['f1'] - paper[i]
                                for i, a in enumerate(e.ALPHAS)}}
    return summary, detail


def main():
    index, _ = e.index_and_labels()
    rows, details, baseline = [], [], []
    per_signal = []
    for ds in DATASETS:
        records = []
        for name in index[ds]:
            path = OUT / 'results' / ds / (name + '.json')
            if not path.exists():
                raise RuntimeError('Missing result ' + str(path))
            record = json.loads(path.read_text())
            if record['signal'] != name or record['dataset'] != ds:
                raise RuntimeError('Identity mismatch ' + str(path))
            records.append(record)
            for config in CONFIGS:
                for metric in record['configs'][config]['metrics']:
                    per_signal.append({'dataset': ds, 'signal': name, 'config': config,
                                       'alpha': metric['alpha'],
                                       **{k: metric[k] for k in ('precision', 'recall', 'f1', 'TP', 'FP', 'FN')}})
        for config in CONFIGS:
            selected = [{config: record['configs'][config]} for record in records]
            summary, detail = aggregate(selected, config, ds, config)
            rows.append(summary)
            details.append(detail)
        selected = [{'official_baseline': record['official_baseline']} for record in records]
        source_summary, source_detail = aggregate(selected, 'official_baseline', ds, 'official_source_240')
        baseline.append(source_summary)
        details.append(source_detail)
    write_csv(OUT / 'comparison_7x4.csv', rows)
    write_csv(OUT / 'official_baseline.csv', baseline)
    write_csv(OUT / 'per_signal_metrics.csv', per_signal)
    by_config = {config: [row for row in rows if row['config'] == config] for config in CONFIGS}
    macro = {config: {metric: mean(row[metric] for row in group)
                      for metric in ('precision', 'recall', 'f1max_fixed_alpha',
                                     'f1max_per_signal', 'gap_to_paper')}
             for config, group in by_config.items()}
    official_macro = {metric: mean(row[metric] for row in baseline)
                      for metric in ('precision', 'recall', 'f1max_fixed_alpha',
                                     'f1max_per_signal', 'gap_to_paper')}
    paper_macro = mean(max(PAPER_F1[ds]) for ds in DATASETS)
    effect_rows = []
    for ds in DATASETS:
        f = {row['config']: row['f1max_fixed_alpha'] for row in rows if row['dataset'] == ds}
        effect_rows.append({'dataset': ds, 'window_effect_top': f['top_224'] - f['top_240'],
                            'window_effect_q25': f['q25_224'] - f['q25_240'],
                            'aggregation_effect_240': f['q25_240'] - f['top_240'],
                            'aggregation_effect_224': f['q25_224'] - f['top_224'],
                            'interaction': f['q25_224'] - f['top_224'] - f['q25_240'] + f['top_240']})
    write_csv(OUT / 'factor_effects.csv', effect_rows)
    result = {'expected_signals': 125, 'datasets': {ds: len(index[ds]) for ds in DATASETS},
              'paper_macro_f1max': paper_macro,
              'macro_by_config': macro, 'official_source_macro': official_macro,
              'macro_effects': {key: mean(row[key] for row in effect_rows)
                                for key in effect_rows[0] if key != 'dataset'},
              'paper_reference': 'https://arxiv.org/pdf/2506.06836, F1 Tables 13/14 and P/R Tables 7/8',
              'rows': details}
    e.save_json(OUT / 'summary.json', result)
    lines = [
        '# ViT4TS seven-dataset fixed 2×2 verification', '',
        'OpenAI CLIP ViT-B/16; official detrend→Min-Max preprocessing; 224×224 images; '
        'α={0.1,0.01,0.001}; 125 signals. Each configuration is evaluated on every dataset.', '',
        'F1-max here means: first average F1 across all signals at each fixed α, then take the '
        'largest of the three means. The per-signal-best column follows the public code aggregation '
        'and is not used for comparison with the paper. P and R are signal means at the α selected '
        'for the corresponding F1-max. Δ is experiment minus the paper ViT4TS F1-max.', '',
        '| Dataset | Config | n | α | P | R | F1-max | Per-signal best | Paper | Δ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append('| {dataset} | {config} | {signals} | {best_alpha:g} | '
                     '{precision:.3f} | {recall:.3f} | {f1max_fixed_alpha:.3f} | '
                     '{f1max_per_signal:.3f} | {paper_f1max:.3f} | {gap_to_paper:+.3f} |'.format(**row))
    lines += ['', '## Macro average across seven datasets', '',
              '| Config | P | R | F1-max | Per-signal best | Δ vs paper |',
              '|---|---:|---:|---:|---:|---:|']
    for config in CONFIGS:
        m = macro[config]
        lines.append(f'| {config} | {m["precision"]:.3f} | {m["recall"]:.3f} | '
                     f'{m["f1max_fixed_alpha"]:.3f} | {m["f1max_per_signal"]:.3f} | '
                     f'{m["gap_to_paper"]:+.3f} |')
    lines += ['', f'Paper macro F1-max: {paper_macro:.3f}.', '',
              '## Official source time mapping baseline', '',
              '| Dataset | α | P | R | F1-max | Per-signal best | Paper | Δ |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in baseline:
        lines.append('| {dataset} | {best_alpha:g} | {precision:.3f} | {recall:.3f} | '
                     '{f1max_fixed_alpha:.3f} | {f1max_per_signal:.3f} | '
                     '{paper_f1max:.3f} | {gap_to_paper:+.3f} |'.format(**row))
    author_path = OUT / 'author_evaluation_comparison.csv'
    if author_path.exists():
        with author_path.open(newline='') as file:
            author_rows = {row['dataset']: row for row in csv.DictReader(file)
                           if row['config'] == 'official_baseline'}
        author_macro = mean(float(author_rows[ds]['f1max_fixed_alpha']) for ds in DATASETS)
        lines += ['', 'The table above uses the paper TP definition (one overlapping predicted interval '
                  'counts once). The author repository counts each ground-truth interval overlapped by '
                  'a prediction. Under that source evaluator, the official-source baseline macro F1-max '
                  f'is {author_macro:.3f}; see author_evaluation_comparison.csv and '
                  'evaluation_implementation_diff.json. The two evaluators differ in 24/1875 '
                  'signal/config/α rows, affecting realTraffic, realTweets and MSL.', '']
    lines += ['', '## Main factor effects in F1-max', '',
              'Window effects are 224/56 minus 240/60. Aggregation effects are Q25 minus Top-25% Mean.', '',
              '| Dataset | Window at Top | Window at Q25 | Q25 at 240 | Q25 at 224 | Interaction |',
              '|---|---:|---:|---:|---:|---:|']
    for row in effect_rows:
        lines.append('| {dataset} | {window_effect_top:+.3f} | {window_effect_q25:+.3f} | '
                     '{aggregation_effect_240:+.3f} | {aggregation_effect_224:+.3f} | '
                     '{interaction:+.3f} |'.format(**row))
    m = result['macro_effects']
    lines.append('| Macro | {window_effect_top:+.3f} | {window_effect_q25:+.3f} | '
                 '{aggregation_effect_240:+.3f} | {aggregation_effect_224:+.3f} | '
                 '{interaction:+.3f} |'.format(**m))
    lines += ['', '## Methods and provenance', '',
              '- Four factorial cells use the same corrected mapping: interpolate 224 image columns '
              'to 224/240 raw points per window, then overlap at stride 56/60.',
              '- Q25 overlaps the full 2-D maps before taking the 25th percentile across rows. '
              'The separate official baseline retains the public stitch implementation.',
              '- Existing 75-signal Top scores and 224/56 Q25 scores are referenced from '
              '`data/runs/20261016_vit4ts_diagnostics`; their original files are preserved.',
              '- New per-signal results, score vectors, all three α metrics and factor effects are '
              'in this run directory. No dataset-specific configuration selection was performed.',
              '- Paper F1 numbers: https://arxiv.org/pdf/2506.06836, Tables 13 and 14; '
              'matched-alpha paper P/R and differences are in comparison_7x4.csv (Tables 7 and 8). '
              'Its primary weight is LAION-400M, whereas this requested experiment fixes OpenAI CLIP.', '',
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print('COMPLETE', len(per_signal), 'per-signal alpha rows', flush=True)
    for config in CONFIGS:
        print(config, 'macro F1', round(macro[config]['f1max_fixed_alpha'], 4), flush=True)


if __name__ == '__main__':
    main()
