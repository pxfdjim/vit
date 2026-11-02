"""Summarize checkpointed ViT4TS diagnostics in both aggregation conventions."""
import csv, json, sys
from pathlib import Path
from statistics import mean
ROOT=Path('/home/pengxf/jim/VLM4TS')
RUN=ROOT/'data/runs/20261016_vit4ts_diagnostics'
sys.path.insert(0,str(RUN))
import experiment as e
PAPER={'realAdExchange':0.615,'realAWSCloudwatch':0.400,'SMAP':0.726}
index,_=e.index_and_labels()
rows=[]
for ds in e.DATASETS:
    for config in e.CONFIGS:
        records=[]
        for signal in index[ds]:
            path=RUN/'results'/ds/signal/(config+'.json')
            if path.exists():
                records.append(json.loads(path.read_text()))
        for mapping in ('official_mapping','corrected_mapping'):
            means={}
            for alpha in e.ALPHAS:
                metric_rows=[next(x for x in r[mapping] if x['alpha']==alpha) for r in records]
                means[str(alpha)]={key:mean(x[key] for x in metric_rows) if metric_rows else None
                                   for key in ('precision','recall','f1')}
                for key in ('TP','FP','FN'):
                    means[str(alpha)][key]=sum(x[key] for x in metric_rows)
            best=max(e.ALPHAS,key=lambda a:means[str(a)]['f1']) if records else None
            source=mean(max(x['f1'] for x in r[mapping]) for r in records) if records else None
            chosen=means[str(best)] if best is not None else {}
            rows.append({'dataset':ds,'config':config,'mapping':mapping,
                         'completed':len(records),'expected':len(index[ds]),
                         'best_alpha':best,'precision':chosen.get('precision'),
                         'recall':chosen.get('recall'),
                         'sum_TP':chosen.get('TP'),'sum_FP':chosen.get('FP'),'sum_FN':chosen.get('FN'),
                         'paper_style_f1max':chosen.get('f1'),
                         'source_style_f1max':source,'paper_vit4ts_f1max':PAPER[ds],
                         'gap_to_paper':chosen['f1']-PAPER[ds] if best is not None else None,
                         'per_alpha':means})
table=RUN/'comparison.csv'
with table.open('w',newline='') as f:
    cols=[x for x in rows[0] if x!='per_alpha']
    w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
    for row in rows: w.writerow({k:row[k] for k in cols})
(RUN/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print('SUMMARY',table)
for row in rows:
    print(row['dataset'],row['config'],row['mapping'],row['completed'],
          round(row['paper_style_f1max'],4) if row['paper_style_f1max'] is not None else None)
