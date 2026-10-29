#!/usr/bin/env python3
"""Sample project signals and reuse immutable Block-6 features for training."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, torch
from scipy.signal import detrend

from experiments.temporal_matching.runner import discover_tasks, load_json, load_signal
from src.models.curve_aware_clip import input_digest,intermediate_cache_path,expected_window_count,validate_intermediate_cache

def fold_id(dataset,signal):
 return int(hashlib.sha256(f'{dataset}/{signal}'.encode()).hexdigest()[:8],16)%3

def dev_id(dataset,signal):
 return int(hashlib.sha256(f'dev/{dataset}/{signal}'.encode()).hexdigest()[:8],16)%5==0

def standardized(values):
 y=np.asarray(detrend(np.asarray(values,np.float64)),np.float64); med=np.median(y)
 mad=1.4826*np.median(np.abs(y-med)); q1,q3=np.quantile(y,[.25,.75]); scale=max(mad,(q3-q1)/1.349,np.std(y)*.1,1e-8)
 return torch.from_numpy(((y-med)/scale).astype(np.float32))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True)
 p.add_argument('--block-cache-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
 p.add_argument('--windows-per-signal',type=int,default=12); a=p.parse_args(); config=load_json(a.config)
 raws=[]; visuals=[]; records=[]
 for ordinal,(dataset,signal,path) in enumerate(discover_tasks(config,a.data_root),1):
  frame=load_signal(path,dataset); values=frame.value.to_numpy(np.float64); digest=input_digest(values); count=expected_window_count(len(values))
  cache_path=intermediate_cache_path(a.block_cache_root,{'dataset':dataset,'signal':signal},digest)
  cached=torch.load(cache_path,map_location='cpu',weights_only=True); validate_intermediate_cache(cached,digest=digest,expected_windows=count,path=cache_path)
  chosen=np.unique(np.linspace(0,count-1,min(count,a.windows_per_signal),dtype=int)); sw=standardized(values).unfold(0,224,56).contiguous()
  raws.append(sw[torch.as_tensor(chosen)]); visuals.append(cached['intermediate_patch'][torch.as_tensor(chosen)].half())
  records.extend({'dataset':dataset,'signal':signal,'window_index':int(i),'fold':fold_id(dataset,signal),'dev':dev_id(dataset,signal)} for i in chosen)
  if ordinal%50==0: print(ordinal,len(records),flush=True)
 payload={'raw':torch.cat(raws),'visual':torch.cat(visuals),'records_json':json.dumps(records,separators=(',',':')),
          'signals':492,'windows_per_signal_cap':a.windows_per_signal,'labels_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.output.exists(): raise FileExistsError(a.output)
 torch.save(payload,a.output); print(a.output,payload['raw'].shape,payload['visual'].shape)
if __name__=='__main__': main()
