#!/usr/bin/env python3
"""Schedule frozen-APRA TREM training jobs."""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TRAIN=ROOT/"experiments/bimodal_reference/train_apra_trem_independent.py"
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--gpus",type=int,nargs="+",required=True); p.add_argument("--python",default=sys.executable); a=p.parse_args(); cfg=json.loads(a.config.read_text()); cache=Path(cfg["rcca_training_cache"])
 jobs=[]
 for seed in cfg["seeds"]:
  variants=cfg["variants"] if seed==cfg["primary_seed"] else cfg["multiseed_variants"]
  jobs += [(variant,seed,dataset) for variant in variants for dataset in cfg["datasets"]]
 jobs.sort(key=lambda x:(cache/f"{x[2]}.pt").stat().st_size,reverse=True); active={}; records=[]; started=time.time()
 while jobs or active:
  for gpu in a.gpus:
   if gpu in active or not jobs: continue
   variant,seed,dataset=jobs.pop(0); output=a.output_root/"checkpoints"/variant/f"seed_{seed}"/f"{dataset}.pt"; log=a.output_root/"training_logs"/variant/f"seed_{seed}"/f"{dataset}.log"
   if output.is_file(): records.append({"variant":variant,"seed":seed,"dataset":dataset,"status":"existing"}); continue
   output.parent.mkdir(parents=True,exist_ok=True); log.parent.mkdir(parents=True,exist_ok=True); h=log.open("w")
   cmd=[a.python,str(TRAIN),"--config",str(a.config),"--variant",variant,"--seed",str(seed),"--cache",str(cache/f"{dataset}.pt"),"--output",str(output),"--device",f"cuda:{gpu}"]
   proc=subprocess.Popen(cmd,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,text=True); active[gpu]=(proc,h,variant,seed,dataset,output,log,time.time()); print(json.dumps({"event":"started","gpu":gpu,"variant":variant,"seed":seed,"dataset":dataset,"pid":proc.pid}),flush=True)
  time.sleep(1)
  for gpu,job in list(active.items()):
   proc,h,variant,seed,dataset,output,log,tick=job; code=proc.poll()
   if code is None: continue
   h.close(); del active[gpu]; record={"variant":variant,"seed":seed,"dataset":dataset,"status":"passed" if code==0 and output.is_file() else "failed","returncode":code,"gpu":gpu,"elapsed_wall_seconds":time.time()-tick,"output":str(output),"log":str(log)}; records.append(record); print(json.dumps({"event":"finished",**record}),flush=True)
   if record["status"]=="failed":
    for other in active.values(): other[0].terminate(); other[1].close()
    (a.output_root/"training_manifest.json").write_text(json.dumps({"status":"failed","records":records},indent=2)+"\n"); return 1
 (a.output_root/"training_manifest.json").write_text(json.dumps({"status":"passed","records":records,"elapsed_wall_seconds":time.time()-started,"labels_used":False},indent=2)+"\n"); return 0
if __name__=="__main__": raise SystemExit(main())
