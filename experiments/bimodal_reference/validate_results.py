#!/usr/bin/env python3
"""Strict completeness, cross-fit, and immutable-T3 validation."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def fold(dataset,signal): return int(hashlib.sha256(f'{dataset}/{signal}'.encode()).hexdigest()[:8],16)%3
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--source-t3-root',type=Path,required=True);a=p.parse_args()
 config=json.loads((a.output_root/'config.json').read_text()); variants=config['variants_order']; signals=0; max_error=0.; failures=[]
 for dataset in config['datasets']:
  dirs=sorted((a.output_root/'point_scores'/dataset).iterdir()) if (a.output_root/'point_scores'/dataset).is_dir() else []
  if len(dirs)!=config['expected_signal_counts'][dataset]: failures.append(f'{dataset}: signal count {len(dirs)}')
  for d in dirs:
   signals+=1; digests=set()
   for variant in variants:
    sp=d/f'{variant}.npz'; mp=a.output_root/'metrics'/dataset/d.name/f'{variant}.json'; dp=a.output_root/'path_diagnostics'/dataset/d.name/f'{variant}.npz'
    if not all(x.is_file() for x in (sp,mp,dp)): failures.append(f'missing {dataset}/{d.name}/{variant}');continue
    with np.load(sp) as z:
     scores=z['scores']; meta=json.loads(str(z['metadata_json'][0]))
     if not np.isfinite(scores).all() or np.ptp(scores)<=0: failures.append(f'invalid score {dataset}/{d.name}/{variant}')
     if meta['cross_fit_eval_fold']!=fold(dataset,d.name): failures.append(f'fold mismatch {dataset}/{d.name}/{variant}')
     digests.add(meta['candidate_pool_digest'])
     if variant=='T3':
      candidates=[a.source_t3_root/'point_scores'/dataset/d.name/'T3_constrained_dtw.npz',a.source_t3_root/'point_scores'/dataset/d.name/'T3.npz']
      old=next((x for x in candidates if x.is_file()),None)
      if old is None:
       failures.append(f'missing source T3 {dataset}/{d.name}');continue
      with np.load(old) as q: max_error=max(max_error,float(np.max(np.abs(scores-q['scores']))))
   if len(digests)>1: failures.append(f'pool mismatch {dataset}/{d.name}')
 expected=sum(config['expected_signal_counts'].values())
 result={'status':'passed' if not failures and signals==expected and max_error<=1e-7 else 'failed','signals':signals,'expected_signals':expected,
  'variants':len(variants),'score_files_expected':expected*len(variants),'max_t3_point_score_abs_error':max_error,'failures':failures[:50]}
 (a.output_root/'validation.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); return 0 if result['status']=='passed' else 1
if __name__=='__main__': raise SystemExit(main())
