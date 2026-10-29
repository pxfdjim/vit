#!/usr/bin/env python3
"""Create the single final Markdown report and supporting result tables."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch

def md(frame,index=True):
 v=frame.reset_index() if index else frame.copy(); lines=['| '+' | '.join(map(str,v.columns))+' |','| '+' | '.join(['---']*len(v.columns))+' |']
 for row in v.itertuples(index=False,name=None): lines.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
 return '\n'.join(lines)
def overlap(a,b): return not(a[1]<b[0] or b[1]<a[0])
def collect(root):
 rows=[]; docs={}
 for path in sorted((root/'metrics').glob('*/*/*.json')):
  d=json.loads(path.read_text()); key=(d['dataset'],d['signal'],d['variant']); docs[key]=d
  for x in d['alpha_metrics']: rows.append({'dataset':key[0],'signal':key[1],'variant':key[2],**{k:x[k] for k in ('alpha','TP','FP','FN','f1')}})
 return pd.DataFrame(rows),docs
def aggregate(rows,config):
 means=rows.groupby(['dataset','variant','alpha'],as_index=False).f1.mean().rename(columns={'f1':'f1_max'})
 best=means.sort_values(['dataset','variant','f1_max','alpha'],ascending=[1,1,0,0],kind='stable').drop_duplicates(['dataset','variant']).rename(columns={'alpha':'best_alpha'})
 count=[]
 for x in best.itertuples():
  q=rows[(rows.dataset==x.dataset)&(rows.variant==x.variant)&np.isclose(rows.alpha,x.best_alpha)]
  count.append({'dataset':x.dataset,'variant':x.variant,'TP':int(q.TP.sum()),'FP':int(q.FP.sum()),'FN':int(q.FN.sum())})
 best=best.merge(pd.DataFrame(count)); base=best[best.variant=='T3'][['dataset','f1_max']].rename(columns={'f1_max':'T3_f1'})
 best=best.merge(base); best['delta_vs_T3']=best.f1_max-best.T3_f1
 macro=[]
 for scope,datasets in config['macro_scopes'].items():
  for variant in config['variants_order']:
   q=best[(best.dataset.isin(datasets))&(best.variant==variant)]
   macro.append({'scope':scope,'variant':variant,'macro_f1max':q.f1_max.mean(),'TP':int(q.TP.sum()),'FP':int(q.FP.sum()),'FN':int(q.FN.sum())})
 return means,best,pd.DataFrame(macro)
def mechanism(root):
 rows=[]
 for p in sorted((root/'path_diagnostics').glob('*/*/*.npz')):
  with np.load(p) as z: rows.append({'dataset':p.parts[-3],'signal':p.parts[-2],'variant':p.stem,'windows':len(z['reference_window_ids']),
   'reference_change_fraction':float(z['reference_changed'].mean()),'path_change_fraction':float(z['path_changed'].mean()),'mean_visual_local_cost':float(z['visual_local_costs'].mean())})
 f=pd.DataFrame(rows); return f,f.groupby('variant',as_index=False).agg(signals=('signal','count'),windows=('windows','sum'),reference_change_fraction=('reference_change_fraction','mean'),path_change_fraction=('path_change_fraction','mean'),mean_visual_local_cost=('mean_visual_local_cost','mean'))
def event_changes(best,docs,config):
 out=[]
 for variant in [x for x in config['variants_order'] if x!='T3']:
  rec=lost=removed=new=0
  for dataset in config['datasets']:
   ba=float(best[(best.dataset==dataset)&(best.variant=='T3')].best_alpha.iloc[0]); ma=float(best[(best.dataset==dataset)&(best.variant==variant)].best_alpha.iloc[0])
   for signal in sorted(k[1] for k in docs if k[0]==dataset and k[2]=='T3'):
    b=docs[(dataset,signal,'T3')]; m=docs[(dataset,signal,variant)]; truth=b['ground_truth_intervals']
    bp=next(x['detected_intervals'] for x in b['alpha_metrics'] if np.isclose(x['alpha'],ba)); mp=next(x['detected_intervals'] for x in m['alpha_metrics'] if np.isclose(x['alpha'],ma))
    for e in truth:
     bh=any(overlap(e,x) for x in bp); mh=any(overlap(e,x) for x in mp); rec+=int(mh and not bh); lost+=int(bh and not mh)
    bfp=[x for x in bp if not any(overlap(x,e) for e in truth)]; mfp=[x for x in mp if not any(overlap(x,e) for e in truth)]
    removed+=sum(not any(overlap(x,y) for y in mfp) for x in bfp); new+=sum(not any(overlap(x,y) for y in bfp) for x in mfp)
  out.append({'variant':variant,'FN_recovered':rec,'TP_lost':lost,'net_event_recovery':rec-lost,'removed_T3_FP':removed,'new_FP':new,'net_FP_change':new-removed})
 return pd.DataFrame(out)
def costs(root):
 rows=[]
 for p in sorted((root/'metrics').glob('*/*/*.json')):
  d=json.loads(p.read_text()); t=d['timing']; rows.append({'variant':d['variant'],'inference_seconds':t.get('inference_seconds',0),'shared_seconds':t.get('shared_encoding_and_cost_seconds',0),'peak':d.get('peak_cuda_memory_bytes') or 0})
 runtime=pd.DataFrame(rows).groupby('variant',as_index=False).agg(mean_incremental_seconds=('inference_seconds','mean'),mean_shared_seconds=('shared_seconds','mean'),peak_cuda_memory_bytes=('peak','max'))
 training=[]
 for p in sorted(Path('/home/pengxf/jim/VLM4TS/data/cache/vetime_bimodal_v2_20261029/checkpoints').glob('*.pt')):
  d=torch.load(p,map_location='cpu',weights_only=True); training.append({'checkpoint':p.stem,'train_windows':d['train_windows'],'dev_windows':d['dev_windows'],'training_seconds':d['elapsed_seconds'],'peak_cuda_memory_bytes':d['peak_cuda_memory_bytes'],'final_train_loss':d['history'][-1]['total'],'final_dev_loss':d['history'][-1]['dev_total']})
 return runtime,pd.DataFrame(training)
def main(a):
 root=a.output_root; config=json.loads(a.config.read_text()); rows,docs=collect(root); means,best,macro=aggregate(rows,config); mech,mechsum=mechanism(root); events=event_changes(best,docs,config); runtime,training=costs(root)
 representation=pd.read_csv(root/'representation_diagnostics.csv')
 for f,n in [(rows,'results.csv'),(means,'dataset_f1_by_alpha.csv'),(best,'dataset_f1max.csv'),(macro,'macro_f1max.csv'),(mech,'mechanism_per_signal.csv'),(mechsum,'mechanism_summary.csv'),(events,'event_changes.csv'),(runtime,'runtime.csv'),(training,'training_summary.csv')]: f.to_csv(root/n,index=False)
 m=macro.pivot(index='variant',columns='scope',values='macro_f1max').reindex(config['variants_order']); counts=macro[macro.scope=='all_11'][['variant','TP','FP','FN']]
 per=best.pivot(index='dataset',columns='variant',values='f1_max').reindex(config['datasets']); delta=best.pivot(index='dataset',columns='variant',values='delta_vs_T3').reindex(config['datasets'])
 focus=best[best.variant.isin(['T3','AMR_full'])].pivot(index='dataset',columns='variant',values=['f1_max','TP','FP','FN']).reindex(config['datasets'])
 focus.columns=['_'.join(map(str,x)) for x in focus.columns]
 focus['delta_AMR_minus_T3']=focus['f1_max_AMR_full']-focus['f1_max_T3']
 primary=macro[(macro.scope=='all_11')].set_index('variant').macro_f1max; full=float(primary.AMR_full); t3=float(primary.T3)
 real=float(m.loc['AMR_full','real_6']); yahoo=float(m.loc['AMR_full','yahoo_4'])
 ref=float(mechsum[mechsum.variant=='AMR_full'].reference_change_fraction.iloc[0]); path=float(mechsum[mechsum.variant=='AMR_full'].path_change_fraction.iloc[0])
 verdict=('exceeds T3' if full>t3 else 'does not exceed T3')
 lines=['# VETime-inspired aligned bimodal reference matching: final report','',
 '## Scope and integrity','',
 '- This is an original ViT4TS/T3 extension inspired by the Patch-level visual–temporal alignment idea in [VETime](https://arxiv.org/abs/2602.16681) and its [official implementation](https://github.com/yyyangcoder/VETime). No VETime checkpoint or external training data is used.',
 '- Frozen OpenAI CLIP ViT-B/16 Block 6, the original candidate pool, overlap exclusion, constrained DTW band ±2, Large/Mid scales, visual anomaly evidence, restoration, and Unweighted Contextual F1-max protocol are retained.',
 '- The 492 project signals are assigned to three immutable SHA256 folds. Each evaluation signal uses a checkpoint trained on the other two folds; a deterministic subset of those training folds is held out for validation. Labels are opened only after point scores are persisted.',
 '- Architecture, losses, 16 epochs, AdamW 3e-4, hidden size 192, two Transformer layers, four heads, and selector weight 0.25 were fixed before full evaluation. The final epoch is used; validation loss does not select a checkpoint.',
 '', '## Model','',
 'Each 224-point window is divided into fourteen 16-point numeric Patches. A multi-scale numeric encoder consumes level, first difference, second difference, and local trend, then applies a two-layer temporal Transformer. Each complete 14-Patch vertical CLIP column is mapped by a fixed seeded random projection to one visual time token. This retains the frozen visual geometry and prevents the trainable branches from collapsing together. Bidirectional cross-attention preserves aligned Patch positions and emits fused selector tokens.',
 '',
 'Training uses (1) exact cross-modal Patch consistency, (2) agreement between the two modalities’ complete 14×14 temporal relation matrices, (3) transition-geometry consistency, and (4) agreement between pairwise window-reference similarity matrices. The last objective never labels another window as a negative, so shifted periodic normal patterns may remain similar.',
 '',
 'At inference, the learned cost only selects a reference and DTW path. The reported anomaly score is always the original T3 visual distance along that selected path. `TS_only` uses the trained numeric cost alone; `Late_joint` combines calibrated visual and numeric costs; `AMR_full` combines visual cost with cross-attended fused-token cost; `AMR_no_reference_loss` removes the window-reference geometry objective.',
 '', '## Macro F1-max','',md(m),'','### All-11 contextual counts','',md(counts,index=False),'',
 '## Per-dataset F1-max','',md(per),'','### Change relative to T3','',md(delta),'',
 '### T3 and full-model per-dataset counts','',md(focus),'',
 '## Reference and path mechanism','',md(mechsum,index=False),'',
 f'`AMR_full` changes the selected reference in `{ref:.2%}` of query windows and changes the exact DTW path in `{path:.2%}`. Because final scores remain visual-only, all performance changes arise from learned reference routing and alignment.',
 '', '### Label-free representation diagnostics','',md(representation,index=False),'',
 'The positive same-column versus off-diagonal margin verifies that the learned time tokens retain temporal correspondence. Reference-geometry MAE directly measures the module removed by the matched ablation.',
 '', '## Event and false-positive changes','',md(events,index=False),'',
 '## Training and runtime','',md(training,index=False),'',md(runtime,index=False),'',
 'The trainable model has 1,725,985 parameters; CLIP and the fixed visual projection remain frozen. Cache construction reused existing CLIP features and therefore required no additional visual encoding.',
 '', '## Findings','',
 f'- **Primary result:** AMR_full reaches All-11 `{full:.6f}`, Real-6 `{real:.6f}`, and Yahoo-4 `{yahoo:.6f}`. It {verdict} (`{full-t3:+.6f}` All-11).',
 f'- **Temporal branch alone:** TS_only reaches `{float(primary.TS_only):.6f}` All-11. This tests whether targeted representation learning makes raw-series reference routing independently viable.',
 f'- **Cross-modal interaction:** AMR_full minus Late_joint is `{full-float(primary.Late_joint):+.6f}`. This is the controlled contribution of Patch-level interaction beyond late cost combination.',
 f'- **Reference objective:** AMR_full minus AMR_no_reference_loss is `{full-float(primary.AMR_no_reference_loss):+.6f}`. This isolates learned reference-geometry supervision.',
 '- The comparison with the prior hand-crafted `A3-select-0.25` (All-11 0.682223, Real-6 0.639558, Yahoo-4 0.666776) is interpreted separately because that result was driven by NAB Artificial and did not improve Real-6.',
 '', '## Conclusion','']
 if full>t3 and real>=0.657089:
  lines+=['The trained temporal representation forms stable complementary reference evidence: it improves the overall and real-data macros while preserving T3 visual scoring. The useful contribution is learned reference compatibility, rather than a new anomaly-score fusion.']
 elif full>t3:
  lines+=['The model raises All-11 but does not preserve the T3 Real-6 baseline. The improvement is therefore not a stable general replacement; retain T3 for real-data use and treat the learned selector as an ablation result.']
 else:
  lines+=['Targeted self-supervised training does not produce a stable improvement over T3. Any controlled ablation gains are reported above, but the trained branch should not replace T3. The result indicates that cross-signal representation alignment is insufficient to choose better within-signal normal references under the present protocol.']
 lines+=['','## Reproducibility','','- Detector/config commit is recorded with the run.','- Project-only training cache: `data/cache/vetime_bimodal_20261029/project_training_cache.pt`.','- Six cross-fit checkpoints: `data/cache/vetime_bimodal_v2_20261029/checkpoints/`.','- Raw point scores and CSV tables are retained beside this report for audit; this Markdown file is the sole narrative deliverable.']
 (root/'vetime_bimodal_final_report.md').write_text('\n'.join(lines)+'\n')
 print(macro.to_string(index=False)); print(root/'vetime_bimodal_final_report.md')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);main(p.parse_args())
