#!/usr/bin/env python3
"""Label-free Patch alignment diagnostics on each held-out signal fold."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.models.bimodal_reference import load_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--checkpoint-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:5');a=p.parse_args()
 data=torch.load(a.cache,map_location='cpu',weights_only=False); rec=json.loads(data['records_json']); folds=np.asarray([x['fold'] for x in rec]); rows=[]
 for fold in range(3):
  idx=np.flatnonzero(folds==fold); idx=idx[np.linspace(0,len(idx)-1,min(512,len(idx)),dtype=int)]
  raw=data['raw'][idx].to(a.device); visual=data['visual'][idx].float().to(a.device)
  for kind in ('full','no_reference'):
   model,_=load_model(a.checkpoint_dir/f'fold{fold}_{kind}.pt',a.device)
   with torch.inference_mode(): out=model(raw,visual); n,v,f=out['numeric'],out['visual'],out['fused']; sim=torch.einsum('bid,bjd->bij',n,v)
   eye=torch.eye(14,dtype=torch.bool,device=a.device)[None]; same=sim[eye.expand_as(sim)]; off=sim[~eye.expand_as(sim)]
   ng=torch.nn.functional.normalize(n.mean(1),dim=-1);vg=torch.nn.functional.normalize(v.mean(1),dim=-1)
   rows.append({'fold':fold,'checkpoint':kind,'windows':len(idx),'same_column_cosine':float(same.mean()),'off_diagonal_cosine':float(off.mean()),
    'time_correspondence_margin':float(same.mean()-off.mean()),'reference_geometry_mae':float(((ng@ng.T)-(vg@vg.T)).abs().mean()),
    'numeric_adjacent_cosine':float((n[:,1:]*n[:,:-1]).sum(-1).mean()),'visual_adjacent_cosine':float((v[:,1:]*v[:,:-1]).sum(-1).mean()),
    'fused_adjacent_cosine':float((f[:,1:]*f[:,:-1]).sum(-1).mean())})
 frame=pd.DataFrame(rows);a.output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(a.output,index=False);print(frame.to_string(index=False))
if __name__=='__main__':main()
