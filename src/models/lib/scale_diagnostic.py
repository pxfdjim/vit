"""Controlled first-stage ViT4TS global-scale diagnostic on labelled dataset excerpts.

Runs the unchanged ViT4TS image/CLIP/score path, intercepting only its preprocessing
input and optional reference builder to perform diagnostic counterfactuals.
"""
import argparse
import ast
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from scipy.signal import detrend

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.evaluate import evaluate_intervals, window_wise_metrics
from models.model_utils import compute_detection_intervals
from preprocessing.preprocess import draw_image, draw_windowed_images
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
import models.vit4ts as vit

OUT = ROOT / 'data/runs/vit4ts_challenges/scale'
CASES = {
    'aws': ('realAWSCloudwatch', 'iio_us-east-1_i-a2eb1cd9_NetworkIn', 308, 370),
    'ad': ('realAdExchange', 'exchange-4_cpm_results', 729, 769),
    'traffic': ('realTraffic', 'speed_7578', 740, 768),
    'artificial': ('artificialWithAnomaly', 'art_increase_spike_density', 1805, 2207),
    'tweets': ('realTweets', 'Twitter_volume_UPS', 1027, 1343),
    'msl': ('MSL', 'M-3', 3287, 3537),
    'smap': ('SMAP', 'T-1', 9425, 9460),
}
ALPHAS = (0.1, 0.01, 0.001)
AMPLITUDES = (0.0, 3.0, 10.0)
SEED = 20261027


def labels_for(signal):
    with (ROOT / 'data/anomalies.csv').open() as handle:
        next(handle)
        for line in handle:
            name, intervals = line.strip().split(',', 1)
            if name == signal:
                return ast.literal_eval(intervals.strip('"'))
    raise KeyError(signal)


def case_data(case):
    ds, name, original_start, original_end = CASES[case]
    frame = pd.read_csv(ROOT / 'data' / ds / (name + '.csv')).sort_values('timestamp').reset_index(drop=True)
    # Fixed 896-point excerpt; the short traffic series needs an earlier peak.
    start = max(0, original_start - 300)
    if start + 896 > len(frame):
        start = len(frame) - 896
    frame = frame.iloc[start:start+896].reset_index(drop=True)
    target = (original_start-start, original_end-start)
    spike = 200 if case == 'traffic' else (50 if target[1] >= 550 else 720)
    separation = (spike-target[1]) if spike>target[1] else (target[0]-spike)
    if separation <= 224:
        raise ValueError('Intervention overlaps a target window')
    ts = frame.timestamp.to_numpy()
    labels=[]
    for a,b in labels_for(name):
        aa,bb=np.searchsorted(ts,[a,b],side='left')
        if aa < len(ts) and bb >= 0 and ts[max(0,aa)] <= b and ts[min(len(ts)-1,bb)] >= a:
            aa=max(0,int(np.searchsorted(ts,a,side='left')))
            bb=min(len(ts)-1,int(np.searchsorted(ts,b,side='right')-1))
            if aa<=bb:labels.append((aa,bb))
    if target not in labels:
        raise ValueError((target,labels))
    if any(a<=spike<=b for a,b in labels):
        raise ValueError('Injected peak overlaps an original label')
    return frame, target, spike, labels, ds, name


def process(raw, mode, base_trend, base_min, base_span):
    if mode == 'official':
        return vit_original_preprocess(raw)
    if mode == 'fixed_scale':
        return (detrend(raw)-base_min)/base_span
    if mode == 'fixed_trend_scale':
        return (raw-base_trend-base_min)/base_span
    raise ValueError(mode)


def evaluate(scores, timestamps, labels, target, spike=None):
    target_scores = scores[target[0]:target[1]+1]
    width=len(target_scores)
    means=np.convolve(scores,np.ones(width)/width,mode='valid')
    rank=int(1+np.count_nonzero(means>target_scores.mean()+1e-12))
    intervals=[]
    truth=[(timestamps[a],timestamps[b]) for a,b in labels]
    truth_with_peak=truth+([] if spike is None else [(timestamps[spike],timestamps[spike])])
    for alpha in ALPHAS:
        found, threshold, smoothed = compute_detection_intervals(scores,alpha)
        predicted=[(timestamps[a],timestamps[b]) for a,b in found]
        result=evaluate_intervals(truth,predicted)
        conf=window_wise_metrics(truth,predicted)
        augmented=evaluate_intervals(truth_with_peak,predicted)
        augmented_conf=window_wise_metrics(truth_with_peak,predicted)
        intervals.append({'alpha':alpha,'f1':result['F1'],'precision':result['precision'],
                          'recall':result['recall'],**conf,'target_detected':any(a<=target[1] and b>=target[0] for a,b in found),
                          'peak_detected':None if spike is None else any(a<=spike<=b for a,b in found),
                          'contextual_f1_with_peak_label':augmented['F1'],
                          'contextual_precision_with_peak_label':augmented['precision'],
                          'contextual_recall_with_peak_label':augmented['recall'],
                          'contextual_counts_with_peak_label':augmented_conf,
                          'threshold':float(threshold), 'detected_index_intervals':found})
    return {'target_score_mean':float(np.mean(target_scores)), 'target_score_max':float(np.max(target_scores)),
            'target_interval_rank':rank,'rank_denominator':len(means),'metrics':intervals}


def save_plot(case, raw, target, spike, records, images):
    fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
    x=np.arange(len(raw));axes[0].plot(x,raw,color='0.2',lw=.8)
    axes[0].axvspan(*target,color='orange',alpha=.28,label='original weak anomaly')
    axes[0].axvline(spike,color='crimson',label='new peak');axes[0].legend(loc='upper right')
    separation=(spike-target[1]) if spike>target[1] else (target[0]-spike)
    axes[0].set_title(f'{case}: original excerpt; injected peak is {separation} points away')
    for key,r in records.items():
        if r.get('scores') is not None:
            axes[1].plot(r['scores'],label=key,lw=1)
    axes[1].axvspan(*target,color='orange',alpha=.25);axes[1].set_ylabel('raw anomaly score')
    axes[1].legend(fontsize=7,ncol=2)
    colors={'baseline':'black','3_official':'tab:blue','10_official':'tab:red',
            '10_fixed_scale':'tab:green','10_fixed_trend_scale':'tab:purple'}
    for key,r in records.items():
        if key in colors:
            axes[2].plot(x,r['processed'],color=colors[key],label=key,lw=.9)
    axes[2].set_xlim(target[0]-35,target[1]+35);axes[2].set_ylim(0,1)
    axes[2].set_ylabel('normalized amplitude');axes[2].set_xlabel('point in 896-point excerpt')
    axes[2].legend(fontsize=7,ncol=2)
    fig.savefig(OUT / f'{case}_series_scores.png',dpi=150);plt.close(fig)
    if images:
        fig,axes=plt.subplots(1,len(images),figsize=(3*len(images),3),constrained_layout=True)
        for ax,(key,img) in zip(np.atleast_1d(axes),images.items()):
            ax.imshow(np.moveaxis(img,0,-1));ax.set_title(key);ax.axis('off')
        fig.savefig(OUT/f'{case}_target_images.png',dpi=150);plt.close(fig)


def run(case):
    if os.getenv('CLIP_PRETRAINED')!='openai':raise RuntimeError('Set CLIP_PRETRAINED=openai')
    np.random.seed(SEED)
    frame,target,spike,labels,ds,name=case_data(case)
    raw=frame.value.to_numpy(float)
    baseline_detrended=detrend(raw)
    base_min=float(baseline_detrended.min());base_span=float(np.ptp(baseline_detrended))
    base_trend=raw-baseline_detrended
    if base_span<=0:raise ValueError('Constant excerpt')
    model=vit.ViT4TS(window_size=224,window_step_ratio=4.0,image_size=(224,224),batch_size=8,verbose=False)
    assert model.model.pretrained_tag=='openai'
    original_preprocess=vit.preprocess_time_series
    original_build=vit.build_memory
    global vit_original_preprocess
    vit_original_preprocess=original_preprocess
    captured={};records={};images={}
    step=56
    target_window_start=(target[0]//step)*step
    # Distinct normal references: the baseline series is used only when requested.
    for amp, mode, ref in [(0,'official','self'),(3,'official','self'),(10,'official','self'),
                           (10,'official','baseline'),(10,'fixed_scale','baseline'),
                           (10,'fixed_trend_scale','self'),
                           (10,'fixed_trend_scale','baseline')]:
        key='baseline' if amp==0 else f'{amp}_{mode}' + ('_fixedref' if ref=='baseline' else '')
        run_file=OUT/f'{case}_{key}.npz'
        values=raw.copy();values[spike]+=amp*base_span
        proc=process(values,mode,base_trend,base_min,base_span)
        if not np.allclose(values[target[0]:target[1]+1],raw[target[0]:target[1]+1]):raise AssertionError('Target changed')
        if ref=='baseline' and 'memory' not in captured:raise AssertionError('No baseline memory')
        vit.preprocess_time_series=lambda _raw, p=proc:p
        if ref=='baseline':
            vit.build_memory=lambda *args,**kwargs:captured['memory']
        else:
            def capture_memory(*args,**kwargs):
                memory=original_build(*args,**kwargs)
                if amp==0:captured['memory']=memory
                return memory
            vit.build_memory=capture_memory
        try:
            score,ts=model.predict_scores(frame)
        finally:
            vit.preprocess_time_series=original_preprocess
            vit.build_memory=original_build
        np.savez_compressed(run_file,scores=score,processed=proc,raw=values,
                            target=np.array(target),spike=spike,timestamps=ts)
        stats=evaluate(score,ts,labels,target,spike if amp else None)
        records[key]={**stats,'amplitude_in_baseline_ranges':amp,'preprocess':mode,'reference':ref,
                      'normalized_target_range':float(np.ptp(proc[target[0]:target[1]+1])),
                      'normalized_target_mean':float(np.mean(proc[target[0]:target[1]+1])),
                      'processed_outside_0_1':int(np.count_nonzero((proc<0)|(proc>1))),
                      'score_file':str(run_file.relative_to(ROOT)),'scores':score,'processed':proc}
        img=draw_image(key, str(OUT/'temp_images'),proc[target_window_start:target_window_start+224],
                       np.arange(target_window_start,target_window_start+224),image_size=(224,224),
                       plot_params=('-',1,'*',0.1,'black',(0,1)))
        images[key]=img
        records[key]['target_window_pixel_mae_vs_baseline']=float(np.mean(np.abs(img-images['baseline'])))
        print(case,key,'score',round(stats['target_score_mean'],6),'rank',stats['target_interval_rank'],
              'recall',[(m['alpha'],m['target_detected']) for m in stats['metrics']],flush=True)
    save_plot(case,raw,target,spike,records,images)
    for r in records.values():r.pop('scores');r.pop('processed')
    summary={'dataset':ds,'signal':name,'seed':SEED,'clip':'OpenAI CLIP ViT-B/16',
             'window':224,'stride':56,'excerpt_length':len(frame),'target_index_interval':target,
             'target_timestamp_interval':[int(frame.timestamp.iloc[target[0]]),int(frame.timestamp.iloc[target[1]])],
             'spike_index':spike,'spike_distance_from_target':
             (spike-target[1]) if spike>target[1] else (target[0]-spike),
             'original_labels_index_intervals':labels,'peak_amplitude_unit':'baseline detrended full range',
             'baseline_detrended_span':base_span,'records':records}
    (OUT/f'{case}.json').write_text(json.dumps(summary,indent=2)+'\n')

def refresh_metrics(case):
    """Re-evaluate saved scores without rerunning CLIP (also upgrades old records)."""
    path=OUT/f'{case}.json'
    result=json.loads(path.read_text())
    for key,record in result['records'].items():
        with np.load(ROOT/record['score_file']) as saved:
            scores=saved['scores'];ts=saved['timestamps']
        target=tuple(result['target_index_interval'])
        spike=None if key=='baseline' else result['spike_index']
        stats=evaluate(scores,ts,result['original_labels_index_intervals'],target,spike)
        record.update(stats)
    path.write_text(json.dumps(result,indent=2)+'\n')

def add_frozen_self(case):
    """Append the frozen-image, changing-reference contrast to an older run."""
    if os.getenv('CLIP_PRETRAINED')!='openai':raise RuntimeError('Set CLIP_PRETRAINED=openai')
    path=OUT/f'{case}.json';result=json.loads(path.read_text())
    key='10_fixed_trend_scale'
    if key in result['records']:return
    frame,target,spike,labels,ds,name=case_data(case)
    raw=frame.value.to_numpy(float);base_detrended=detrend(raw)
    base_trend=raw-base_detrended;span=float(np.ptp(base_detrended));minimum=float(base_detrended.min())
    values=raw.copy();values[spike]+=10*span
    proc=(values-base_trend-minimum)/span
    model=vit.ViT4TS(window_size=224,window_step_ratio=4.0,image_size=(224,224),batch_size=8,verbose=False)
    assert model.model.pretrained_tag=='openai'
    original=vit.preprocess_time_series
    vit.preprocess_time_series=lambda _raw:proc
    try:score,ts=model.predict_scores(frame)
    finally:vit.preprocess_time_series=original
    run_file=OUT/f'{case}_{key}.npz'
    np.savez_compressed(run_file,scores=score,processed=proc,raw=values,target=np.array(target),spike=spike,timestamps=ts)
    window=(target[0]//56)*56
    image=draw_image(key,str(OUT/'temp_images'),proc[window:window+224],np.arange(window,window+224),
                     image_size=(224,224),plot_params=('-',1,'*',0.1,'black',(0,1)))
    base_proc=(base_detrended-minimum)/span
    base_image=draw_image('baseline',str(OUT/'temp_images'),base_proc[window:window+224],
                          np.arange(window,window+224),image_size=(224,224),
                          plot_params=('-',1,'*',0.1,'black',(0,1)))
    result['records'][key]={**evaluate(score,ts,labels,target,spike),'amplitude_in_baseline_ranges':10,
        'preprocess':'fixed_trend_scale','reference':'self',
        'normalized_target_range':float(np.ptp(proc[target[0]:target[1]+1])),
        'normalized_target_mean':float(np.mean(proc[target[0]:target[1]+1])),
        'processed_outside_0_1':int(np.count_nonzero((proc<0)|(proc>1))),
        'score_file':str(run_file.relative_to(ROOT)),
        'target_window_pixel_mae_vs_baseline':float(np.mean(np.abs(image-base_image)))}
    path.write_text(json.dumps(result,indent=2)+'\n')
    records={};images={}
    for item,record in result['records'].items():
        with np.load(ROOT/record['score_file']) as saved:
            records[item]={'scores':saved['scores'],'processed':saved['processed']}
        images[item]=draw_image(item,str(OUT/'temp_images'),
             records[item]['processed'][window:window+224],np.arange(window,window+224),
             image_size=(224,224),plot_params=('-',1,'*',0.1,'black',(0,1)))
    save_plot(case,raw,target,spike,records,images)
    print(case,key,'score',result['records'][key]['target_score_mean'],flush=True)

def add_normal_ref(case):
    """Score the 10x official image against baseline windows clear of all labels."""
    if os.getenv('CLIP_PRETRAINED')!='openai':raise RuntimeError('Set CLIP_PRETRAINED=openai')
    path=OUT/f'{case}.json';result=json.loads(path.read_text())
    key='10_official_purenormalref'
    if key in result['records']:return
    frame,target,spike,labels,ds,name=case_data(case)
    safe=[i for i in range(13) if all(i*56+223<a or i*56>b for a,b in labels)]
    if not safe:raise ValueError('No pure normal windows')
    raw=frame.value.to_numpy(float);base_detrended=detrend(raw)
    span=float(np.ptp(base_detrended));values=raw.copy();values[spike]+=10*span
    baseline_proc=vit.preprocess_time_series(raw)
    model=vit.ViT4TS(window_size=224,window_step_ratio=4.0,image_size=(224,224),batch_size=8,verbose=False)
    assert model.model.pretrained_tag=='openai'
    with tempfile.TemporaryDirectory() as tmp:
        draw_windowed_images('series',tmp,baseline_proc,np.arange(len(raw)),window_size=224,step_size=56,
                             image_size=(224,224),plot_params=('-',1,'*',0.1,'black',(0,1)))
        dataset=CLIPTimeSeriesDataset(tmp,'series',plot_type='line')
        loader=DataLoader(Subset(dataset,safe),batch_size=8,shuffle=False)
        normal_memory=vit.build_memory(model.model,loader,model.patch_size,model.device)
    original_build=vit.build_memory
    vit.build_memory=lambda *args,**kwargs:normal_memory
    try:score,ts=model.predict_scores(frame.assign(value=values))
    finally:vit.build_memory=original_build
    proc=vit.preprocess_time_series(values)
    run_file=OUT/f'{case}_{key}.npz'
    np.savez_compressed(run_file,scores=score,processed=proc,raw=values,target=np.array(target),spike=spike,timestamps=ts)
    baseline_pixel_mae=result['records']['10_official']['target_window_pixel_mae_vs_baseline']
    result['records'][key]={**evaluate(score,ts,labels,target,spike),'amplitude_in_baseline_ranges':10,
        'preprocess':'official','reference':'pure_normal_baseline',
        'normal_reference_window_count':len(safe),'normal_reference_window_indices':safe,
        'normalized_target_range':float(np.ptp(proc[target[0]:target[1]+1])),
        'normalized_target_mean':float(np.mean(proc[target[0]:target[1]+1])),
        'processed_outside_0_1':int(np.count_nonzero((proc<0)|(proc>1))),
        'score_file':str(run_file.relative_to(ROOT)),
        'target_window_pixel_mae_vs_baseline':baseline_pixel_mae}
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(case,key,'normal_windows',len(safe),'score',result['records'][key]['target_score_mean'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--add-frozen-self',action='store_true')
    parser.add_argument('--add-normal-ref',action='store_true')
    args=parser.parse_args()
    if args.refresh:refresh_metrics(args.case)
    elif args.add_frozen_self:add_frozen_self(args.case)
    elif args.add_normal_ref:add_normal_ref(args.case)
    else:run(args.case)
