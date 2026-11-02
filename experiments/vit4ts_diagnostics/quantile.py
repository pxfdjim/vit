"""Diagnostic of paper-described q=0.25 row quantile using unchanged 224-window CLIP maps."""
import argparse,json,os,time
from pathlib import Path
import numpy as np,pandas as pd
import experiment as e

OUT=e.RUN/'quantile_diagnostic'
ORIGINAL=e.vit.stitch_anomaly_maps

def row_quantile(maps,stride=56):
    n,h,w=maps.shape
    covered=stride*(n-1)+w
    sums=np.zeros((h,covered),dtype=np.float32)
    counts=np.zeros(covered,dtype=np.int32)
    for i,m in enumerate(maps):
        start=i*stride
        sums[:,start:start+w]+=m
        counts[start:start+w]+=1
    return np.quantile(sums/np.maximum(counts,1)[None,:],0.25,axis=0)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    assert os.getenv('CLIP_PRETRAINED')=='openai'
    index,labels=e.index_and_labels()
    tasks=[(ds,s) for ds in e.DATASETS for s in index[ds]]
    assigned=[x for i,x in enumerate(tasks) if i%args.workers==args.worker]
    model=e.vit.ViT4TS(window_size=224,window_step_ratio=4,image_size=(224,224),verbose=False)
    assert model.model.pretrained_tag=='openai'
    print('START',args.worker,'assigned',len(assigned),flush=True)
    for ds,signal in assigned:
        path=OUT/'results'/ds/(signal+'.json')
        if path.exists():continue
        frame=pd.read_csv(e.ROOT/'data'/ds/(signal+'.csv'))
        capture={}
        def stitch(maps,ratio,percent):
            capture['quantile']=row_quantile(maps)
            capture['n_windows']=len(maps)
            return ORIGINAL(maps,ratio,percent)
        e.vit.stitch_anomaly_maps=stitch
        official,timestamps=model.predict_scores(frame)
        score=e.align_anomaly_vector(capture['quantile'],len(frame),224,56,capture['n_windows'])
        with np.load(e.RUN/'scores'/ds/signal/'paper_window_224.npz') as old:
            match=np.max(np.abs(official-old['official']))
        result={'dataset':ds,'signal':signal,'window':224,'stride':56,
                'preprocess':'detrend_minmax','clip':'openai','q':0.25,
                'source_score_max_abs_diff':float(match),
                'source_metrics':e.evaluate(model,official,timestamps,labels[signal]),
                'row_quantile_metrics':e.evaluate(model,score,timestamps,labels[signal])}
        e.save_json(path,result)
        scores=OUT/'scores'/ds/(signal+'.npz')
        scores.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(scores,source=official,quantile=score)
        print('DONE',ds,signal,'source_diff',match,flush=True)
    print('WORKER_DONE',args.worker,flush=True)

if __name__=='__main__':main()
