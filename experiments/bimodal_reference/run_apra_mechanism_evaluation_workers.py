#!/usr/bin/env python3
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; WORKER=ROOT/"experiments/bimodal_reference/evaluate_apra_mechanism.py"
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--workers",type=int,default=8); a=p.parse_args(); jobs=[]
 for i in range(a.workers):
  log=a.output_root/"evaluation_logs"/f"worker_{i:02d}.log"; log.parent.mkdir(parents=True,exist_ok=True); h=log.open("w"); proc=subprocess.Popen([sys.executable,str(WORKER),"--config",str(a.config),"--output-root",str(a.output_root),"--worker-index",str(i),"--num-workers",str(a.workers)],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,text=True); jobs.append((proc,h,i,log))
 records=[]
 while jobs:
  time.sleep(1)
  for job in list(jobs):
   proc,h,i,log=job; code=proc.poll()
   if code is None: continue
   h.close(); jobs.remove(job); records.append({"worker":i,"returncode":code,"status":"passed" if code==0 else "failed","log":str(log)})
 status="passed" if all(x["status"]=="passed" for x in records) else "failed"; (a.output_root/"evaluation_workers.json").write_text(json.dumps({"status":status,"records":records},indent=2)+"\n")
 if status!="passed": return 1
 return subprocess.call([sys.executable,str(WORKER),"--config",str(a.config),"--output-root",str(a.output_root),"--num-workers",str(a.workers),"--combine-shards"],cwd=ROOT)
if __name__=="__main__": raise SystemExit(main())
