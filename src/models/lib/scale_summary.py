"""Summarize saved scale diagnostic scores; no model inference."""
import ast
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from models.model_utils import compute_detection_intervals

OUT=ROOT/'data/runs/vit4ts_challenges/scale'
CASES=('aws','ad','traffic','artificial','tweets','msl','smap')
ORDER=('baseline','3_official','10_official','10_official_fixedref',
       '10_official_purenormalref','10_fixed_scale_fixedref',
       '10_fixed_trend_scale','10_fixed_trend_scale_fixedref')

def save_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fields);writer.writeheader();writer.writerows(rows)

def main():
    summaries=[];alpha_rows=[];selection=[]
    label_map={}
    with (ROOT/'data/anomalies.csv').open() as handle:
        next(handle)
        for line in handle:
            name,intervals=line.strip().split(',',1)
            label_map[name]=ast.literal_eval(intervals.strip('"'))
    for case in CASES:
        result=json.loads((OUT/f'{case}.json').read_text())
        frame=pd.read_csv(ROOT/'data'/result['dataset']/(result['signal']+'.csv')).sort_values('timestamp')
        raw=frame.value.to_numpy(float);times=frame.timestamp.to_numpy()
        first,last=result['target_timestamp_interval']
        start=int(np.searchsorted(times,first,'left'))
        end=int(np.searchsorted(times,last,'right')-1)
        nearby=np.r_[np.arange(max(0,start-100),start),
                     np.arange(end+1,min(len(raw),end+101))]
        clean=np.ones(len(nearby),dtype=bool)
        for a,b in label_map[result['signal']]:
            clean &= ~((times[nearby]>=a)&(times[nearby]<=b))
        neighbor=raw[nearby[clean]];event=raw[start:end+1]
        neighbor_std=float(np.std(neighbor))
        selection.append({'case':case,'dataset':result['dataset'],'signal':result['signal'],
            'target_full_index_start':start,'target_full_index_end':end,
            'target_point_count':len(event),'normal_flank_point_count':len(neighbor),
            'normal_flank_definition':'100 points before and after target, excluding all original labels',
            'target_raw_median':float(np.median(event)),
            'normal_flank_raw_median':float(np.median(neighbor)),
            'normal_flank_raw_std':neighbor_std,
            'median_shift_over_flank_std':float(abs(np.median(event)-np.median(neighbor))/neighbor_std) if neighbor_std else '',
            'target_nonzero_fraction':float(np.mean(event!=0)),
            'normal_flank_nonzero_fraction':float(np.mean(neighbor!=0)),
            'selection_before_clip_inference':True})
        if set(ORDER)-set(result['records']):raise RuntimeError(f'Incomplete {case}: {set(ORDER)-set(result["records"])}')
        baseline=result['records']['baseline']
        target=result['target_index_interval']
        threshold_by_alpha={float(m['alpha']):m['threshold'] for m in baseline['metrics']}
        for condition in ORDER:
            record=result['records'][condition]
            with np.load(ROOT/record['score_file']) as saved:scores=saved['scores']
            tests=[]
            for metric in record['metrics']:
                alpha=float(metric['alpha'])
                _,_,smoothed=compute_detection_intervals(scores,alpha)
                fixed_hit=bool(np.any(smoothed[target[0]:target[1]+1]>threshold_by_alpha[alpha]))
                tests.append(fixed_hit)
                alpha_rows.append({'case':case,'dataset':result['dataset'],'signal':result['signal'],
                    'condition':condition,'alpha':alpha,'target_detected':metric['target_detected'],
                    'peak_detected':metric['peak_detected'],
                    'target_hit_with_fixed_baseline_threshold':fixed_hit,
                    'threshold':metric['threshold'],'baseline_threshold':threshold_by_alpha[alpha],
                    'contextual_f1_peak_true':metric['contextual_f1_with_peak_label'],
                    'contextual_precision_peak_true':metric['contextual_precision_with_peak_label'],
                    'contextual_recall_peak_true':metric['contextual_recall_with_peak_label'],
                    'contextual_tp_peak_true':metric['contextual_counts_with_peak_label']['TP'],
                    'contextual_fp_peak_true':metric['contextual_counts_with_peak_label']['FP'],
                    'contextual_fn_peak_true':metric['contextual_counts_with_peak_label']['FN'],
                    'original_events_only_f1':metric['f1'],'original_events_only_fp':metric['FP'],
                    'original_events_only_fn':metric['FN']})
            best=max(record['metrics'],key=lambda m:m['contextual_f1_with_peak_label'])
            summaries.append({'case':case,'dataset':result['dataset'],'signal':result['signal'],
                'condition':condition,'target_start':target[0],'target_end':target[1],
                'spike_index':result['spike_index'],'target_peak_separation':result['spike_distance_from_target'],
                'normal_reference_windows':record.get('normal_reference_window_count',''),
                'target_score_mean':record['target_score_mean'],
                'score_ratio_to_baseline':record['target_score_mean']/baseline['target_score_mean'],
                'target_score_max':record['target_score_max'],
                'target_interval_rank':record['target_interval_rank'],
                'rank_denominator':record['rank_denominator'],
                'normalized_target_range':record['normalized_target_range'],
                'normalized_range_ratio_to_baseline':record['normalized_target_range']/baseline['normalized_target_range'] if baseline['normalized_target_range'] else '',
                'target_window_pixel_mae_vs_baseline':record['target_window_pixel_mae_vs_baseline'],
                'target_detected_alpha_0_1':record['metrics'][0]['target_detected'],
                'target_detected_alpha_0_01':record['metrics'][1]['target_detected'],
                'target_hit_fixed_threshold_alpha_0_1':tests[0],
                'contextual_f1max_peak_true':best['contextual_f1_with_peak_label'],
                'best_alpha_diagnostic_only':best['alpha']})
    save_csv(OUT/'summary.csv',summaries);save_csv(OUT/'alpha_metrics.csv',alpha_rows)
    save_csv(OUT/'case_selection.csv',selection)
    by={(r['case'],r['condition']):r for r in summaries}
    fig,axes=plt.subplots(3,1,figsize=(12,12),constrained_layout=True)
    x=np.arange(len(CASES));width=.18
    for j,c in enumerate(('baseline','3_official','10_official','10_fixed_trend_scale')):
        axes[0].bar(x+(j-1.5)*width,[by[(case,c)]['score_ratio_to_baseline'] for case in CASES],width,label=c)
    axes[0].axhline(1,color='0.3',lw=.8);axes[0].set_ylabel('target mean score / baseline');axes[0].legend(ncol=4,fontsize=8)
    controls=('10_official','10_official_fixedref','10_official_purenormalref','10_fixed_trend_scale','10_fixed_trend_scale_fixedref')
    width=.15
    for j,c in enumerate(controls):
        axes[1].bar(x+(j-2)*width,[by[(case,c)]['score_ratio_to_baseline'] for case in CASES],width,label=c)
    axes[1].axhline(1,color='0.3',lw=.8);axes[1].set_ylabel('10x score / baseline');axes[1].legend(ncol=3,fontsize=8)
    heat=np.array([[int(by[(case,c)]['target_detected_alpha_0_1']) for c in controls] for case in CASES])
    axes[2].imshow(heat,cmap='RdYlGn',vmin=0,vmax=1,aspect='auto')
    axes[2].set_yticks(range(len(CASES)),CASES);axes[2].set_xticks(range(len(controls)),controls,rotation=25,ha='right',fontsize=8)
    axes[2].set_title('Target detected under original alpha=0.1 threshold protocol (green=yes)')
    for ax in axes[:2]:ax.set_xticks(x,CASES)
    fig.savefig(OUT/'cross_case_comparison.png',dpi=150);plt.close(fig)
    dynamic=[by[(case,'10_official')] for case in CASES]
    base=[by[(case,'baseline')] for case in CASES]
    fixedref=[by[(case,'10_official_fixedref')] for case in CASES]
    pure=[by[(case,'10_official_purenormalref')] for case in CASES]
    frozen=[by[(case,'10_fixed_trend_scale')] for case in CASES]
    lost=[case for case in CASES if by[(case,'baseline')]['target_detected_alpha_0_1'] and not by[(case,'10_official')]['target_detected_alpha_0_1']]
    f1_dropped=sum(v['contextual_f1max_peak_true']<b['contextual_f1max_peak_true'] for v,b in zip(dynamic,base))
    f1_rose=sum(v['contextual_f1max_peak_true']>b['contextual_f1max_peak_true'] for v,b in zip(dynamic,base))
    lines=['# Challenge 1: global-scale diagnostic','',
      'Seven labelled dataset excerpts, one preselected target event per dataset; an excerpt may contain additional original labels. The targets were chosen before CLIP inference from labels and relatively small local median shifts. The raw amplitude selection audit uses the absolute target-vs-neighbor median difference divided by the standard deviation of up to 100 unlabelled flank points on each side; see `case_selection.csv`. The artificial series is a subtle spike-density pattern rather than a small point-amplitude event. This is a controlled case study, not an estimate over every signal in each dataset. Seed: 20261027. No training, VLM, or LLM.','',
      'ViT4TS first-stage source path: whole-series linear detrend → Min–Max → 224×224 line images with fixed [0,1] y-axis at 224-point windows and 56-point stride → OpenAI CLIP ViT-B/16 multi-scale tokens → per-token coordinate-wise median over all test windows → patch dissimilarity and overlap score → Gaussian alpha thresholds with EWMA smoothing. Core scoring code was reused without modification.','',
      'Each experiment uses an 896-point excerpt; original target values and labels are unchanged. A single 3× or 10× peak (multiples of the baseline detrended full range) is inserted more than 224 points from the target. This guarantees no target-covering image window contains the peak. The baseline fixed reference includes the original labelled event; the pure-normal fixed reference excludes every baseline 224-point window that overlaps any labelled event.','',
      'The main target detection uses the original alpha thresholds 0.1, 0.01, 0.001 and interval overlap. The contextual F1 columns label the injected peak as a true anomaly. Original-event-only FP/FN are retained solely as a diagnostic. F1-max in this file is the maximum over those three alpha values for one excerpt, not a seven-dataset paper benchmark.','',
      '| Dataset | target | raw median shift / flank SD | baseline score | 3× score | 10× score | 10× target image range ratio | baseline→10× α=.1 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for case in CASES:
        b=by[(case,'baseline')];v3=by[(case,'3_official')];v10=by[(case,'10_official')]
        weak=next(s for s in selection if s['case']==case)
        lines.append(f"| {b['dataset']} | {b['target_start']}–{b['target_end']} | {weak['median_shift_over_flank_std']:.3f} | {b['target_score_mean']:.4f} | {v3['target_score_mean']:.4f} | {v10['target_score_mean']:.4f} | {v10['normalized_range_ratio_to_baseline']:.3f} | {int(b['target_detected_alpha_0_1'])}→{int(v10['target_detected_alpha_0_1'])} |")
    lines += ['',f'At 10×, the target mean score fell in {sum(v["target_score_mean"]<b["target_score_mean"] for v,b in zip(dynamic,base))}/7 cases; α=.1 target recall was lost in {len(lost)} of {sum(b["target_detected_alpha_0_1"] for b in base)} baseline-detected cases ({", ".join(lost)}).',
      f'The target range in the normalized image fell in {sum(v["normalized_range_ratio_to_baseline"]<1 for v in dynamic)}/7 cases. Fixed original reference gave a higher 10× target score than baseline in {sum(v["target_score_mean"]>b["target_score_mean"] for v,b in zip(fixedref,base))}/7 cases; pure-normal fixed reference did so in {sum(v["target_score_mean"]>b["target_score_mean"] for v,b in zip(pure,base))}/7. Frozen detrend and scale with self reference kept the target score within 5% of baseline in {sum(abs(v["score_ratio_to_baseline"]-1)<.05 for v in frozen)}/7 cases.',
      f'Contextual F1-max with the added peak correctly labelled fell in {f1_dropped}/7 cases and rose in {f1_rose}/7; target-event recall is the more stable diagnostic here. At 10× with the baseline all-window reference held, the target exceeds the baseline α=.1 threshold in {sum(v["target_hit_fixed_threshold_alpha_0_1"] for v in fixedref)}/7 cases, but the original adaptive threshold actually detects it in only {sum(v["target_detected_alpha_0_1"] for v in fixedref)}/7. This separates score changes from threshold competition.',
      '', 'Interpretation: the global normalization visibly compresses unchanged target images, and the coupled production pipeline shows score and recall losses. Yet with the baseline reference frozen, compressed target images often score *higher*. This rules out a simple claim that compression alone lowers CLIP dissimilarity. The changing median reference and, independently, the new peak competing in ranking and adaptive thresholding contribute. The all-window fixed reference is contaminated by pre-existing labels; the pure-normal fixed reference gives a cleaner reference control but may contain only a few windows for long events. These results support investigating scale/reference coupling as a research motivation, with causal wording narrower than “scale compression alone causes all misses.”',
      '', 'Artifacts: `case_selection.csv` (raw-data selection audit), `summary.csv` (one row per condition), `alpha_metrics.csv` (threshold/confusion counts), `cross_case_comparison.png`, per-case `*_series_scores.png`, `*_target_images.png`, raw score `.npz`, and JSON provenance.']
    (OUT/'SCALE_NOTES.md').write_text('\n'.join(lines)+'\n')
    print('Wrote',len(summaries),'condition rows and',len(alpha_rows),'alpha rows')

if __name__=='__main__':main()
