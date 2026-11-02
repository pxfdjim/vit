"""Independent ViT4TS diagnostic sweep; does not instantiate VLM4TS."""
import argparse, ast, csv, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import detrend

ROOT=Path('/home/pengxf/jim/VLM4TS')
RUN=ROOT/'data/runs/20261016_vit4ts_diagnostics'
sys.path.insert(0,str(ROOT/'src'))
from evaluation.evaluate import evaluate_intervals, window_wise_metrics
from models.model_utils import aggregate_anomaly_map, align_anomaly_vector
import models.vit4ts as vit

DATASETS=('realAdExchange','realAWSCloudwatch','SMAP')
ALPHAS=(0.1,0.01,0.001)
CONFIGS={'official_240':(240,'detrend_minmax'),
         'paper_window_224':(224,'detrend_minmax'),
         'paper_preprocess_240':(240,'minmax_detrend')}
ORIGINAL_STITCH=vit.stitch_anomaly_maps
ORIGINAL_PREPROCESS=vit.preprocess_time_series

def save_json(path,item):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(item,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
    tmp.replace(path)

def minmax_detrend(x):
    x=np.asarray(x,dtype=float)
    span=x.max()-x.min()
    return detrend((x-x.min())/span if span>0 else np.zeros_like(x))

def overlap_add(vectors,stride):
    n,width=vectors.shape
    sums=np.zeros(stride*(n-1)+width)
    counts=np.zeros(len(sums),dtype=int)
    for i,row in enumerate(vectors):
        start=i*stride
        sums[start:start+width]+=row
        counts[start:start+width]+=1
    return sums/np.maximum(counts,1)

def mapped_score(vectors,window,stride,length):
    width=vectors.shape[1]
    new_x=np.linspace(0,width-1,window)
    expanded=np.stack([np.interp(new_x,np.arange(width),row) for row in vectors])
    stitched=overlap_add(expanded,stride)
    return align_anomaly_vector(stitched,length,window,stride,len(vectors))

def index_and_labels():
    with (ROOT/'data/datasets.csv').open(newline='') as f:
        index={name:sorted(ast.literal_eval(names)) for name,names in csv.reader(f)}
    labels={}
    with (ROOT/'data/anomalies.csv').open() as f:
        next(f)
        for line in f:
            name,intervals=line.strip().split(',',1)
            labels[name]=ast.literal_eval(intervals.strip('"'))
    return index,labels

def audit(ds,signal,frame,labels):
    ts=frame.timestamp.to_numpy()
    values=frame.value.to_numpy(dtype=float)
    item={'dataset':ds,'signal':signal,'length':len(frame),
          'columns':list(frame.columns),
          'timestamps_monotonic':bool(pd.Index(ts).is_monotonic_increasing),
          'timestamps_unique':bool(pd.Index(ts).is_unique),'timestamp_duplicates':int(pd.Index(ts).duplicated().sum()),
          'values_finite':bool(np.isfinite(values).all()),
          'timestamp_min':str(ts.min()),'timestamp_max':str(ts.max()),
          'ground_truth_intervals':labels,'ground_truth_count':len(labels),
          'ground_truth_within_range':all(ts.min()<=a<=b<=ts.max() for a,b in labels),
          'raw_min':float(values.min()),'raw_max':float(values.max())}
    if not item['values_finite'] or not item['ground_truth_within_range']:
        raise ValueError('Data audit failed: '+json.dumps(item))
    return item

def evaluate(model,score,timestamps,labels):
    rows=[]
    for alpha in ALPHAS:
        intervals=model.get_intervals(score,timestamps,alpha)[['start','end']].values.tolist()
        metric=evaluate_intervals(labels,intervals)
        rows.append({'alpha':alpha,**window_wise_metrics(labels,intervals),
                     'precision':metric['precision'],'recall':metric['recall'],'f1':metric['F1'],
                     'predicted_intervals':intervals})
    return rows

def process(model,ds,signal,labels):
    frame=pd.read_csv(ROOT/'data'/ds/(signal+'.csv'))
    audit_path=RUN/'audit'/ds/(signal+'.json')
    if not audit_path.exists():
        save_json(audit_path,audit(ds,signal,frame,labels))
    ordered=frame.sort_values('timestamp')
    timestamps=ordered.timestamp.to_numpy()
    raw=ordered.value.to_numpy(dtype=float)
    for config,(window,preprocessing) in CONFIGS.items():
        result=RUN/'results'/ds/signal/(config+'.json')
        if result.exists():
            continue
        cache=RUN/'cache'/ds/signal/(config+'.npz')
        model.window_size=window
        model.window_step_ratio=4.0
        vit.preprocess_time_series=ORIGINAL_PREPROCESS if preprocessing=='detrend_minmax' else minmax_detrend
        if cache.exists():
            with np.load(cache) as saved:
                vectors=saved['vectors']
            official=align_anomaly_vector(overlap_add(vectors,56),len(frame),window,window//4,len(vectors))
        else:
            capture={}
            def capture_stitch(maps,ratio,percent):
                capture['vectors']=np.stack([aggregate_anomaly_map(m,percent) for m in maps])
                return ORIGINAL_STITCH(maps,ratio,percent)
            vit.stitch_anomaly_maps=capture_stitch
            official,observed_timestamps=model.predict_scores(frame)
            assert np.array_equal(observed_timestamps,timestamps)
            vectors=capture['vectors']
            cache.parent.mkdir(parents=True,exist_ok=True)
            tmp=cache.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,vectors=vectors)
            tmp.replace(cache)
            vit.stitch_anomaly_maps=ORIGINAL_STITCH
        corrected=mapped_score(vectors,window,window//4,len(frame))
        score_path=RUN/'scores'/ds/signal/(config+'.npz')
        score_path.parent.mkdir(parents=True,exist_ok=True)
        tmp=score_path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,official=official,corrected=corrected,timestamps=timestamps)
        tmp.replace(score_path)
        processed=ORIGINAL_PREPROCESS(raw) if preprocessing=='detrend_minmax' else minmax_detrend(raw)
        save_json(result,{'dataset':ds,'signal':signal,'config':config,
                          'window':window,'stride':window//4,'image_size':[224,224],
                          'preprocess':preprocessing,'plot_ylim':[0,1],'raw_length':len(frame),
                          'n_windows':len(vectors),'processed_min':float(processed.min()),
                          'processed_max':float(processed.max()),
                          'processed_outside_plot_range':int(np.count_nonzero((processed<0)|(processed>1))),
                          'score_mae_mapping':float(np.mean(np.abs(official-corrected))),
                          'official_mapping':evaluate(model,official,timestamps,labels),
                          'corrected_mapping':evaluate(model,corrected,timestamps,labels)})
        print('DONE',ds,signal,config,'windows',len(vectors),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    assert os.getenv('CLIP_PRETRAINED')=='openai'
    index,labels=index_and_labels()
    tasks=[(ds,signal) for ds in DATASETS for signal in index[ds]]
    assigned=[task for i,task in enumerate(tasks) if i%args.workers==args.worker]
    model=vit.ViT4TS(window_size=240,window_step_ratio=4.0,image_size=(224,224),verbose=False)
    assert model.model.pretrained_tag=='openai'
    print('START',args.worker,'gpu',os.getenv('CUDA_VISIBLE_DEVICES'),'assigned',len(assigned),flush=True)
    for ds,signal in assigned:
        start=time.monotonic()
        process(model,ds,signal,labels[signal])
        print('SIGNAL_DONE',ds,signal,round(time.monotonic()-start,2),flush=True)
    print('WORKER_DONE',args.worker,flush=True)

if __name__=='__main__':
    main()
