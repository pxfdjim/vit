#!/usr/bin/env python3
"""Schedule APRA loss-ablation jobs; reuse the existing E0 primary seed."""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; TRAIN=ROOT/"experiments/bimodal_reference/train_apra_loss_weight.py"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True); p.add_argument("--gpus",type=int,nargs="+",required=True)
    p.add_argument("--datasets",nargs="+"); p.add_argument("--python",default=sys.executable); a=p.parse_args()
    cfg=json.loads(a.config.read_text()); datasets=a.datasets or cfg["datasets"]
    jobs=[(v,s,d) for v in cfg["variants"] for s in cfg["seeds"] for d in datasets
          if not (v=="E0_full" and s==cfg["primary_seed"])]; cache=Path(cfg["rcca_training_cache"])
    jobs.sort(key=lambda x:(cache/f"{x[2]}.pt").stat().st_size,reverse=True)
    active={}; records=[]; started=time.time()
    while jobs or active:
        for gpu in a.gpus:
            if gpu in active or not jobs: continue
            variant,seed,dataset=jobs.pop(0); out=a.output_root/"checkpoints"/variant/f"seed_{seed}"/f"{dataset}.pt"
            log=a.output_root/"training_logs"/variant/f"seed_{seed}"/f"{dataset}.log"
            if out.is_file(): records.append({"variant":variant,"seed":seed,"dataset":dataset,"status":"existing"}); continue
            out.parent.mkdir(parents=True,exist_ok=True); log.parent.mkdir(parents=True,exist_ok=True)
            handle=log.open("w"); command=[a.python,str(TRAIN),"--config",str(a.config),"--variant",variant,"--seed",str(seed),
              "--cache",str(cache/f"{dataset}.pt"),"--output",str(out),"--device",f"cuda:{gpu}"]
            process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,text=True)
            active[gpu]=(process,handle,variant,seed,dataset,out,log,time.time())
            print(json.dumps({"event":"started","gpu":gpu,"variant":variant,"seed":seed,"dataset":dataset,"pid":process.pid}),flush=True)
        time.sleep(1)
        for gpu,job in list(active.items()):
            process,handle,variant,seed,dataset,out,log,tick=job; code=process.poll()
            if code is None: continue
            handle.close(); del active[gpu]; record={"variant":variant,"seed":seed,"dataset":dataset,
              "status":"passed" if code==0 and out.is_file() else "failed","returncode":code,
              "gpu":gpu,"elapsed_wall_seconds":time.time()-tick,"output":str(out),"log":str(log)}
            records.append(record); print(json.dumps({"event":"finished",**record}),flush=True)
            if record["status"]=="failed":
                for other in active.values(): other[0].terminate(); other[1].close()
                (a.output_root/"training_manifest.json").write_text(json.dumps({"status":"failed","records":records},indent=2)); return 1
    (a.output_root/"training_manifest.json").write_text(json.dumps({"status":"passed","records":records,
      "elapsed_wall_seconds":time.time()-started,"labels_used":False},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
