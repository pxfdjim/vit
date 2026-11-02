"""Data-only diagnostic: deduplicate timestamps without modifying source CSV."""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import experiment as e

def main():
    assert os.getenv('CLIP_PRETRAINED')=='openai'
    index,labels=e.index_and_labels()
    model=e.vit.ViT4TS(window_size=240,window_step_ratio=4,image_size=(224,224),verbose=False)
    result_dir=e.RUN/'data_dedup'
    for ds in ('realAdExchange','realAWSCloudwatch'):
        for signal in index[ds]:
            original=pd.read_csv(e.ROOT/'data'/ds/(signal+'.csv'))
            if original.timestamp.is_unique:
                continue
            for policy in ('first','last'):
                out=result_dir/ds/signal/(policy+'.json')
                if out.exists(): continue
                frame=original.drop_duplicates('timestamp',keep=policy)
                score,ts=model.predict_scores(frame)
                e.save_json(out,{'dataset':ds,'signal':signal,'policy':policy,
                                 'original_length':len(original),'dedup_length':len(frame),
                                 'official_mapping':e.evaluate(model,score,ts,labels[signal])})
                print('DEDUP_DONE',ds,signal,policy,flush=True)
    rows=[]
    for ds in ('realAdExchange','realAWSCloudwatch'):
        for policy in ('first','last'):
            records=[]
            for signal in index[ds]:
                special=result_dir/ds/signal/(policy+'.json')
                path=special if special.exists() else e.RUN/'results'/ds/signal/'official_240.json'
                records.append(json.loads(path.read_text())['official_mapping'])
            by_alpha={}
            for a in e.ALPHAS:
                metrics=[next(x for x in rec if x['alpha']==a) for rec in records]
                by_alpha[str(a)]={k:sum(x[k] for x in metrics)/len(metrics) for k in ('precision','recall','f1')}
            best=max(e.ALPHAS,key=lambda a:by_alpha[str(a)]['f1'])
            rows.append({'dataset':ds,'policy':policy,'best_alpha':best,
                         'paper_style_f1max':by_alpha[str(best)]['f1'],'per_alpha':by_alpha})
    (result_dir/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('DEDUP_SUMMARY',json.dumps(rows))

if __name__=='__main__':main()
