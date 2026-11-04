#!/usr/bin/env python3
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; WORKER=ROOT/"experiments/bimodal_reference/run_apra_loss_weight_inference.py"
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--gpus",type=int,nargs="+",required=True); a=p.parse_args(); jobs=[]; started=time.time()
 for i,g in enumerate(a.gpus):
  log=a.output_root/"inference_logs"/f"worker_{i:02d}.log"; log.parent.mkdir(parents=True,exist_ok=True); h=log.open("w"); proc=subprocess.Popen([sys.executable,str(WORKER),"--config",str(a.config),"--output-root",str(a.output_root),"--device",f"cuda:{g}","--worker-index",str(i),"--num-workers",str(len(a.gpus)),"--skip-complete"],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,text=True); jobs.append((proc,h,i,g,log)); print(json.dumps({"event":"started","worker":i,"gpu":g,"pid":proc.pid}),flush=True)
 records=[]
 while jobs:
  time.sleep(2)
  for job in list(jobs):
   proc,h,i,g,log=job; code=proc.poll()
   if code is None: continue
   h.close(); jobs.remove(job); records.append({"worker":i,"gpu":g,"returncode":code,"status":"passed" if code==0 else "failed","log":str(log)}); print(json.dumps(records[-1]),flush=True)
 status="passed" if all(x["status"]=="passed" for x in records) else "failed"; (a.output_root/"inference_workers.json").write_text(json.dumps({"status":status,"records":records,"elapsed_wall_seconds":time.time()-started},indent=2)+"\n"); return 0 if status=="passed" else 1
if __name__=="__main__": raise SystemExit(main())
