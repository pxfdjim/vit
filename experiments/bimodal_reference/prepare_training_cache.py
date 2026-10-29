#!/usr/bin/env python3
"""Build an independent UCR-normal-prefix CLIP/number training cache."""
from __future__ import annotations
import argparse, hashlib, re, tempfile, time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import detrend

from src.models.curve_aware_clip import create_openai_block6_encoder, encode_block6
from experiments.curve_aware.extract_visual_cache import render_images

PATTERN = re.compile(r"^(\d+)_UCR_Anomaly_.+_(\d+)_(\d+)_(\d+)\.txt$")

def standardize(values):
    values = np.asarray(detrend(values), np.float64)
    med = np.median(values); mad = 1.4826 * np.median(np.abs(values-med))
    q1,q3=np.quantile(values,[.25,.75]); scale=max(mad,(q3-q1)/1.349,np.std(values)*.1,1e-8)
    return ((values-med)/scale).astype(np.float32)

def visual_values(values):
    y=np.asarray(detrend(values),np.float64); span=np.ptp(y)
    return ((y-y.min())/span if span>0 else np.zeros_like(y)).astype(np.float64)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--ucr-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--windows-per-series',type=int,default=8)
    a=p.parse_args(); files=[]
    for path in sorted(a.ucr_root.glob('*.txt')):
        m=PATTERN.match(path.name)
        if m: files.append((path,int(m.group(2))))
    if len(files)!=250: raise RuntimeError(f'expected 250 UCR series, found {len(files)}')
    encoder=create_openai_block6_encoder(a.checkpoint,a.device)
    raws=[]; tokens=[]; series=[]; starts=[]; split=[]; started=time.perf_counter()
    for idx,(path,end) in enumerate(files):
        values=np.loadtxt(path,dtype=np.float64)[:end]
        if len(values)<224: continue
        all_starts=np.arange(0,len(values)-223,56,dtype=int)
        chosen=np.unique(np.linspace(0,len(all_starts)-1,min(a.windows_per_series,len(all_starts)),dtype=int))
        offsets=all_starts[chosen]
        std=standardize(values); vis=visual_values(values)
        raw_batch=np.stack([std[s:s+224] for s in offsets])
        # Renderer gets the whole sequence, then retain deterministic selected windows.
        images=render_images(vis)[torch.as_tensor(chosen,dtype=torch.long)]
        block=encode_block6(encoder,images,device=a.device,batch_size=32).half()
        raws.append(torch.from_numpy(raw_batch)); tokens.append(block)
        series.extend([path.name]*len(offsets)); starts.extend(offsets.tolist())
        digest=int(hashlib.sha256(path.name.encode()).hexdigest()[:8],16)
        split.extend(['dev' if digest%5==0 else 'train']*len(offsets))
        if (idx+1)%25==0: print(idx+1,len(files),sum(x.shape[0] for x in raws),flush=True)
    payload={'raw':torch.cat(raws),'visual':torch.cat(tokens),'series':series,'starts':starts,
             'split':split,'source_count':len(files),'normal_prefix_only':True,
             'elapsed_seconds':time.perf_counter()-started}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists(): raise FileExistsError(a.output)
    torch.save(payload,a.output)
    print(a.output,payload['raw'].shape,payload['visual'].shape,payload['elapsed_seconds'])
if __name__=='__main__': main()

