#!/usr/bin/env python3
"""Fixed self-supervised training on independent UCR normal prefixes."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.bimodal_reference import AlignedMultimodalReference, ModelConfig, patch_alignment_loss

def main():
 p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=20261029)
 p.add_argument('--epochs',type=int,default=16); p.add_argument('--batch-size',type=int,default=64)
 p.add_argument('--no-reference-loss',action='store_true'); a=p.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
 data=torch.load(a.cache,map_location='cpu',weights_only=False); split=np.asarray(data['split'])
 train_idx=torch.from_numpy(np.flatnonzero(split=='train')); dev_idx=torch.from_numpy(np.flatnonzero(split=='dev'))
 train=TensorDataset(data['raw'][train_idx],data['visual'][train_idx].float())
 dev=TensorDataset(data['raw'][dev_idx],data['visual'][dev_idx].float())
 generator=torch.Generator().manual_seed(a.seed)
 loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=generator,drop_last=True)
 model=AlignedMultimodalReference().to(a.device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
 history=[]; started=time.perf_counter(); peak=0
 for epoch in range(a.epochs):
  model.train(); sums={}; count=0
  for raw,visual in loader:
   raw,visual=raw.to(a.device),visual.to(a.device)
   # semantic-preserving raw perturbation; visual form remains equivalent after plot normalization.
   raw=raw*(.9+.2*torch.rand(raw.shape[0],1,device=a.device))+(.01*torch.randn_like(raw))
   out=model(raw,visual); loss,parts=patch_alignment_loss(out,not a.no_reference_loss)
   opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
   for k,v in {'total':loss,**parts}.items(): sums[k]=sums.get(k,0.)+float(v.detach())
   count+=1
  model.eval(); dev_losses=[]
  with torch.inference_mode():
   for s in range(0,len(dev),a.batch_size):
    raw,visual=dev.tensors[0][s:s+a.batch_size].to(a.device),dev.tensors[1][s:s+a.batch_size].to(a.device)
    dev_losses.append(float(patch_alignment_loss(model(raw,visual),not a.no_reference_loss)[0]))
  row={'epoch':epoch+1,**{k:v/count for k,v in sums.items()},'dev_total':float(np.mean(dev_losses))}
  history.append(row); print(json.dumps(row),flush=True)
  if torch.cuda.is_available(): peak=max(peak,torch.cuda.max_memory_allocated(a.device))
 payload={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
          'model_config':ModelConfig().__dict__,'training':vars(a)|{'cache':str(a.cache),'output':str(a.output)},
          'history':history,'train_windows':len(train),'dev_windows':len(dev),
          'elapsed_seconds':time.perf_counter()-started,'peak_cuda_memory_bytes':peak}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.output.exists(): raise FileExistsError(a.output)
 torch.save(payload,a.output); print(a.output,payload['elapsed_seconds'])
if __name__=='__main__': main()
