"""Restartable five-GPU supervisor for the isolated first-stage sweep."""
import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
RUN=Path('/home/pengxf/jim/VLM4TS/data/runs/20261016_vit4ts_diagnostics')
ROOT=Path('/home/pengxf/jim/VLM4TS')
PYTHON='/home/pengxf/anaconda3/envs/vit/bin/python'
GPUS=(0,1,3,5,7)
(RUN/'supervisor.pid').write_text(str(os.getpid())+'\n')
started=time.time()
processes={}
attempts={i:0 for i in range(5)}
finished=set()
failed=set()

def launch(i):
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':str(GPUS[i]),'CLIP_PRETRAINED':'openai',
                'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
    with (RUN/f'worker_{i}.log').open('a') as out:
        process=subprocess.Popen(['nice','-n','10',PYTHON,'-u',str(RUN/'experiment.py'),
                                  '--worker',str(i),'--workers','5'],
                                 cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                 stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    processes[i]=process
    attempts[i]+=1
    (RUN/f'worker_{i}.pid').write_text(str(process.pid)+'\n')
    print('LAUNCHED',i,'gpu',GPUS[i],'pid',process.pid,'attempt',attempts[i],flush=True)

for i in range(5): launch(i)
while len(finished)+len(failed)<5:
    time.sleep(10)
    for i,process in list(processes.items()):
        code=process.poll()
        if code is None: continue
        del processes[i]
        if code==0:
            finished.add(i);print('WORKER_OK',i,flush=True)
        elif attempts[i]<3:
            print('RESTART',i,'exit',code,flush=True)
            time.sleep(5);launch(i)
        else:
            failed.add(i);print('WORKER_FAILED',i,'exit',code,flush=True)
    progress={}
    for ds,total in (('realAdExchange',5),('realAWSCloudwatch',17),('SMAP',53)):
        progress[ds]={name:len(list((RUN/'results'/ds).glob(f'*/{name}.json')))
                      for name in ('official_240','paper_window_224','paper_preprocess_240')}
        progress[ds]['expected']=total
    (RUN/'progress.json').write_text(json.dumps({'datasets':progress,'finished_workers':sorted(finished),
        'failed_workers':sorted(failed),'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')

code=subprocess.call([PYTHON,str(RUN/'summarize.py')],cwd=ROOT)
print('SUMMARY_EXIT',code,flush=True)
(RUN/'finished.json').write_text(json.dumps({'finished_workers':sorted(finished),
    'failed_workers':sorted(failed),'wall_seconds':round(time.time()-started,2),
    'completed_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
if failed or code:
    sys.exit(1)
