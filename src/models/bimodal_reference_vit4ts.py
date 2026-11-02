"""Trainable visual/numeric reference routing around immutable T3 scoring."""
from __future__ import annotations
import hashlib, time
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from scipy.signal import detrend

from src.models.bimodal_reference import load_model
from src.models.curve_aware_clip import expected_window_count,input_digest,intermediate_cache_path,load_final_tokens,validate_intermediate_cache
from src.models.numeric_temporal_matching import match_paths, mix_costs, score_costs_on_paths
from src.models.temporal_matching import column_cost_matrix
from src.models.temporal_vit4ts import _column_map,_diagnostic_numpy,_pool_digest
from src.models.vfg_paired_vit4ts import GRID_SIZE,MAX_COMMON_WINDOWS,OVERLAP_RADIUS,_patch_tokens,point_scores_from_maps,select_common_visual_windows,stable_scale_map

EPS=1e-8

def _fold_id(dataset,signal):
 return int(hashlib.sha256(f'{dataset}/{signal}'.encode()).hexdigest()[:8],16)%3

def _standardized_windows(raw):
 y=np.asarray(detrend(np.asarray(raw,np.float64)),np.float64); med=np.median(y)
 mad=1.4826*np.median(np.abs(y-med)); q1,q3=np.quantile(y,[.25,.75]); scale=max(mad,(q3-q1)/1.349,np.std(y)*.1,EPS)
 x=torch.from_numpy(((y-med)/scale).astype(np.float32))
 return x.unfold(0,224,56).contiguous()

def _positive_median(costs):
 x=torch.cat([c.reshape(-1).cpu() for c in costs]); x=x[torch.isfinite(x)&(x>EPS)]
 return float(x.median()) if x.numel() else 1.0

def _token_costs(tokens,pool):
 ids_all=torch.as_tensor(pool.window_ids,dtype=torch.long,device=tokens.device)
 valid_all=torch.as_tensor(pool.valid,dtype=torch.bool,device=tokens.device)
 out=[]
 for q in range(tokens.shape[0]):
  ids=ids_all[q,valid_all[q]]; qt=tokens[q]
  rt=tokens[ids]
  out.append((1-torch.einsum('id,rjd->rij',qt,rt)).clamp(0,2).cpu())
 return out

def _visual_costs(grids,pool):
 ids_all=torch.as_tensor(pool.window_ids,dtype=torch.long,device=grids.device)
 valid_all=torch.as_tensor(pool.valid,dtype=torch.bool,device=grids.device); out=[]
 for q in range(grids.shape[0]):
  ids=ids_all[q,valid_all[q]]; out.append(column_cost_matrix(grids[q],grids[ids]).cpu())
 return out

def _calibrate(reference,other):
 ratio=float(np.clip(_positive_median(reference)/max(_positive_median(other),EPS),1e-4,1e4))
 return [c*ratio for c in other],ratio

def _learned_cost_kind(selector):
 """Map each learned selector to its intended independent representation.

 Late fusion combines the immutable visual cost with the numeric-branch cost.
 Cross-modal AMR combines it with the bidirectional cross-attention fused cost.
 Keeping this mapping explicit prevents both ablations from silently sharing
 the same fused-token cost.
 """
 selector=str(selector).upper()
 if selector in {'TS_ONLY','LATE_JOINT'}: return 'numeric'
 if selector=='CROSS_MODAL': return 'fused'
 raise ValueError(f'unsupported learned selector: {selector}')

class BimodalReferenceBackend:
 def __init__(self,config:Mapping[str,Any],device,cache_dir):
  self.config=dict(config); self.model_config=dict(config.get('model',{})); self.cache_config=dict(config.get('cache',{}))
  self.device=torch.device('cuda' if str(device)=='auto' and torch.cuda.is_available() else device)
  self.cache_dir=Path(cache_dir); roots=self.cache_config.get('final_cache_roots',[])
  self.final_cache_roots=[Path(x) for x in ([roots] if isinstance(roots,(str,Path)) else roots)]
  checkpoints=self.model_config.get('checkpoints',{})
  self.models={name:load_model(path,self.device)[0] for name,path in checkpoints.items()}
  expected={f'fold{fold}_{kind}' for fold in range(3) for kind in ('full','no_reference')}
  if set(self.models)!=expected: raise ValueError(f'expected cross-fit checkpoints: {expected}')
 def _clock(self):
  if self.device.type=='cuda': torch.cuda.synchronize(self.device)
  return time.perf_counter()
 def _load(self,raw,context):
  digest=input_digest(raw); windows=expected_window_count(len(raw)); _,fp=load_final_tokens(self.final_cache_roots,context,digest,windows)
  final=torch.load(fp,map_location='cpu',weights_only=True); bp=intermediate_cache_path(self.cache_dir,context,digest)
  block=torch.load(bp,map_location='cpu',weights_only=True); validate_intermediate_cache(block,digest=digest,expected_windows=windows,path=bp)
  return final,block,digest,fp,bp
 @torch.inference_mode()
 def predict_variants(self,data,variants,context):
  raw=np.asarray(data['value'],np.float64); timestamps=np.asarray(data['timestamp']); final,block,digest,fp,bp=self._load(raw,context)
  final_patch=_patch_tokens(final['patch']).to(self.device); patch=_patch_tokens(block['intermediate_patch']).to(self.device); nw=patch.shape[0]
  large=stable_scale_map(final['large'],final['large_mask'],device=self.device); mid=stable_scale_map(final['mid'],final['mid_mask'],device=self.device)
  pool=select_common_visual_windows(final_patch); pool_digest=_pool_digest(pool.window_ids,pool.valid)
  started=self._clock(); visual=_visual_costs(patch.reshape(nw,14,14,-1),pool)
  base=match_paths(visual,pool,band=2,penalty_scale=.1); baseline_scores=.5*base.scores.to(self.device)
  raw_windows=_standardized_windows(raw).to(self.device)
  fold=_fold_id(context['dataset'],context['signal'])
  required={}
  for spec in variants.values():
   kind=str(spec.get('selector','T3'))
   if kind=='T3': continue
   model_key=str(spec.get('checkpoint','full'))
   required.setdefault(model_key,set()).add(_learned_cost_kind(kind))
  encoded={}
  for key,kinds in required.items():
   model=self.models[f'fold{fold}_{key}']
   # Late fusion and TS-only depend only on the independent numeric branch.
   # Avoid evaluating visual projection and Cross-Attention when fused tokens
   # are not requested. This is score preserving and substantially reduces
   # the corrected Late-joint rerun cost.
   if 'fused' in kinds:
    output=model(raw_windows,patch)
    encoded[key]={'numeric':output['numeric'],'fused':output['fused']}
   else:
    encoded[key]={'numeric':model.numeric(raw_windows)}
  costsets={}
  scales={}
  for key,outputs in encoded.items():
   scales[key]={}
   for learned_kind,tokens in outputs.items():
    costs=_token_costs(tokens,pool); costs,scale=_calibrate(visual,costs)
    costsets[(key,learned_kind)]=costs; scales[key][learned_kind]=scale
  shared_seconds=self._clock()-started; matches={'T3':base}
  weight=.25
  for name,spec in variants.items():
   kind=str(spec.get('selector','T3')); model_key=str(spec.get('checkpoint','full'))
   if kind=='T3': continue
   learned_kind=_learned_cost_kind(kind)
   learned=costsets[(model_key,learned_kind)]
   chosen=learned if kind=='TS_ONLY' else mix_costs(visual,learned,weight)
   matches[name]=match_paths(chosen,pool,band=2,penalty_scale=.1)
  results={}
  for name,spec in variants.items():
   tick=self._clock(); kind=str(spec.get('selector','T3')); match=base if kind=='T3' else matches[name]
   local=baseline_scores if kind=='T3' else .5*score_costs_on_paths(visual,match).to(self.device)
   combined=(large+mid+_column_map(local).detach().cpu().double())/3
   scores=point_scores_from_maps(combined,full_length=len(raw),device=self.device); elapsed=self._clock()-tick
   ref_changed=match.reference_ids.cpu()!=base.reference_ids.cpu(); path_changed=(match.path_masks!=base.path_masks).reshape(nw,-1).any(1)
   results[name]={'scores':np.asarray(scores,np.float64),'timestamps':timestamps,
    'metadata':{'family':kind,'checkpoint':str(spec.get('checkpoint','full')),'numeric_weight':0 if kind in {'T3','TS_ONLY'} else weight,
     'learned_cost_kind':'none' if kind=='T3' else _learned_cost_kind(kind),
     'candidate_pool_digest':pool_digest,'reference_change_fraction':float(ref_changed.float().mean()),'path_change_fraction':float(path_changed.float().mean()),
     'learned_cost_scales':scales.get(str(spec.get('checkpoint','full')),{}),'shared_encoding_and_cost_seconds':shared_seconds,
     'matching_seconds':elapsed,'inference_seconds':elapsed,'n_windows':nw,'input_digest':digest[:20],'final_cache_path':str(fp),'block_cache_path':str(bp),
     'final_anomaly_evidence':'T3_visual_only','cross_fit_eval_fold':fold,'max_common_visual_windows':MAX_COMMON_WINDOWS,'overlap_radius':OVERLAP_RADIUS},
    'timing':{'shared_encoding_and_cost_seconds':shared_seconds,'matching_seconds':elapsed,'inference_seconds':elapsed},
    'diagnostics':{'reference_window_ids':_diagnostic_numpy(match.reference_ids),'reference_columns':_diagnostic_numpy(match.reference_columns),
     'visual_local_costs':_diagnostic_numpy(local),'t3_reference_window_ids':_diagnostic_numpy(base.reference_ids),
     'reference_changed':_diagnostic_numpy(ref_changed.byte()),'path_changed':_diagnostic_numpy(path_changed.byte())}}
  return results

def create_backend(config,device,cache_dir): return BimodalReferenceBackend(config,device,cache_dir)

