import json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
RUN=Path('/home/pengxf/jim/VLM4TS/data/runs/20261016_vit4ts_diagnostics')
OUT=RUN/'quantile_diagnostic'
OUT.mkdir(exist_ok=True)
(OUT/'supervisor.pid').write_text(str(os.getpid())+'\n')
PYTHON='/home/pengxf/anaconda3/envs/vit/bin/python'
GPUS=(0,1,3,5,7)
processes={};attempts={i:0 for i in range(5)};finished=set();failed=set()
started=time.time()
def launch(i):
 env=os.environ.copy();env.update({'CLIP_PRETRAINED':'openai','CUDA_VISIBLE_DEVICES':str(GPUS[i]),
        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
 with (OUT/f'worker_{i}.log').open('a') as stream:
  p=subprocess.Popen(['nice','-n','10',PYTHON,'-u',str(RUN/'quantile.py'),'--worker',str(i)],
                     cwd=RUN,env=env,stdin=subprocess.DEVNULL,stdout=stream,
                     stderr=subprocess.STDOUT,start_new_session=True)
 processes[i]=p;attempts[i]+=1
 print('LAUNCHED',i,'gpu',GPUS[i],'pid',p.pid,'attempt',attempts[i],flush=True)
for i in range(5):launch(i)
while len(finished)+len(failed)<5:
 time.sleep(10)
 for i,p in list(processes.items()):
  code=p.poll()
  if code is None:continue
  del processes[i]
  if code==0:finished.add(i);print('WORKER_OK',i,flush=True)
  elif attempts[i]<3:print('RESTART',i,code,flush=True);launch(i)
  else:failed.add(i);print('WORKER_FAILED',i,code,flush=True)
 count={ds:len(list((OUT/'results'/ds).glob('*.json'))) for ds in ('realAdExchange','realAWSCloudwatch','SMAP')}
 (OUT/'progress.json').write_text(json.dumps({'completed_by_dataset':count,
    'finished_workers':sorted(finished),'failed_workers':sorted(failed),
    'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
(OUT/'finished.json').write_text(json.dumps({'completed':count,'failed_workers':sorted(failed),
 'wall_seconds':round(time.time()-started,2),'completed_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
if failed:sys.exit(1)
