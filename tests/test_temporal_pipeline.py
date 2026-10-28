from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import experiments.temporal_matching.runner as runner_module
from experiments.temporal_matching.runner import normalize_prediction, publish_bytes_no_replace
from experiments.temporal_matching.summarize import aggregate, comparisons, macro_table, path_diagnostics
from experiments.temporal_matching.validate_results import validate_bmid

ROOT=Path(__file__).parents[1]

def config(): return json.loads((ROOT/"experiments/temporal_matching/config.json").read_text())

def test_protocol_config_is_fixed_and_complete():
    c=config(); assert sum(c["expected_signal_counts"].values())==492 and len(c["datasets"])==11
    assert len(c["variants_order"])==10
    assert c["model"]["temporal_matching"]["shift_radius"]==2 and c["model"]["temporal_matching"]["dtw_band"]==2
    lambdas=[c["variants"][n].get("fusion_lambda") for n in c["variants_order"] if n.startswith(("T4_","T5_"))]
    assert lambdas==[.25,.5,.25,.5]
    assert set(c["macro_scopes"])=={"real_6","original_7","yahoo_4","all_11"}

def test_atomic_publication_and_diagnostic_normalization(tmp_path):
    target=tmp_path/"x"; assert publish_bytes_no_replace(target,b"one"); assert not publish_bytes_no_replace(target,b"two")
    raw={"scores":[.1,.2],"diagnostics":{"shift":np.array([-1,1]),"column_costs":np.ones((1,2))}}
    score,timestamps,meta,timing,diag=normalize_prediction(raw,np.array([1,2]))
    assert score.tolist()==[.1,.2] and timestamps.tolist()==[1,2] and set(diag)=={"shift","column_costs"}

def test_dataset_level_alpha_selection_and_fixed_lambda_comparisons():
    c={"datasets":["d1","d2"],"expected_signal_counts":{"d1":2,"d2":1},"evaluation":{"alphas":[.1,.01]},
       "macro_scopes":{"all_2":["d1","d2"]},"variants_order":["B","T4a","T4b"],"comparisons":[{"name":"a-B","left":"T4a","right":"B"},{"name":"b-B","left":"T4b","right":"B"}]}
    rows=[]
    for model in c["variants_order"]:
      for ds,signals in {"d1":["a","b"],"d2":["c"]}.items():
       for signal in signals:
        for alpha in c["evaluation"]["alphas"]:
         value=(.6 if alpha==.01 else (.9 if signal=="a" else .1)) + ({"B":0,"T4a":.1,"T4b":-.1}[model])
         rows.append({"dataset":ds,"signal":signal,"model":model,"alpha":alpha,"f1":value,"precision":value,"recall":value,"TP":1,"FP":0,"FN":0})
    _,best=aggregate(pd.DataFrame(rows),c); macro=macro_table(best,c); changes=comparisons(best,macro,c)
    assert len(changes[changes.scope=="all_2"])==2
    assert changes[(changes.scope=="all_2")&(changes.comparison=="a-B")].iloc[0].delta_f1max > 0

def test_path_diagnostics_summarizes_shift_and_dtw(tmp_path):
    root=tmp_path/"path_diagnostics"/"MSL"/"s"; root.mkdir(parents=True)
    np.savez(root/"T2_shift_consistent.npz",reference_window_ids=np.array([8,9]),shift=np.array([-2,1]),column_costs=np.ones((2,14)))
    np.savez(root/"T3_constrained_dtw.npz",reference_window_ids=np.array([8]),path_offsets=np.array([[0,1]*7]),reference_columns=np.arange(14)[None],local_costs=np.ones((1,14)),normalized_path_costs=np.array([.5]))
    signal,dataset=path_diagnostics(tmp_path)
    assert len(signal)==2 and len(dataset)==2
    assert signal[signal.path_kind=="shift"].iloc[0].fraction_band_edge==.5

def test_baseline_validator_checks_all_scores_and_macro(tmp_path):
    cfg={"datasets":["d1","d2"],"expected_signal_counts":{"d1":1,"d2":1},"evaluation":{"alphas":[.1,.01]}}
    old=tmp_path/"old"; out=tmp_path/"out"
    for dataset,signal,f1 in (("d1","s1",.6),("d2","s2",.8)):
      for base in (old,out):
       path=base/"point_scores"/dataset/signal; path.mkdir(parents=True)
       np.savez(path/"B_paired_2d_mid.npz",scores=np.array([.1,.2]),timestamps=np.array([1,2]))
      metric=out/"metrics"/dataset/signal; metric.mkdir(parents=True)
      (metric/"B_paired_2d_mid.json").write_text(json.dumps({"dataset":dataset,"signal":signal,"alpha_metrics":[{"alpha":.1,"f1":f1},{"alpha":.01,"f1":f1-.1}]}))
    result=validate_bmid(cfg,out,[old],.7)
    assert result["passed"] and result["maximum_absolute_score_difference"]==0

def test_cpu_synthetic_runner_writes_scores_metrics_and_paths(tmp_path, monkeypatch):
    data_root=tmp_path/"data"; (data_root/"toy").mkdir(parents=True)
    length=700; pd.DataFrame({"timestamp":np.arange(length),"value":np.sin(np.arange(length)/20)}).to_csv(data_root/"toy/s.csv",index=False)
    (data_root/"anomalies.csv").write_text('signal,events\ns,"[[100, 120]]"\n')
    cfg={"seed":1,"datasets":["toy"],"expected_signal_counts":{"toy":1},
         "evaluation":{"alphas":[.1,.01,.001],"threshold_method":"mean","smoothing":True,"sliding":False,"anomaly_padding":0},
         "model":{"window_size":224,"step_size":56},"cache":{},
         "variants":{"T2":{"family":"T2"}},"variants_order":["T2"]}
    cfg_path=tmp_path/"config.json"; cfg_path.write_text(json.dumps(cfg))
    class Backend:
      def predict_variants(self,data,variants,context):
        nwin=1+(len(data)-224)//56
        return {name:{"scores":np.linspace(0,1,len(data)),"timestamps":data.timestamp.to_numpy(),
                      "metadata":{"candidate_pool_digest":"same","candidate_pool_min_windows":1,"candidate_pool_max_windows":2},
                      "diagnostics":{"reference_window_ids":np.arange(nwin),"shift":np.zeros(nwin),"column_costs":np.ones((nwin,14))}}
                for name in variants}
    monkeypatch.setattr(runner_module,"load_backend",lambda name: SimpleNamespace(create_backend=lambda config,device,cache_dir:Backend()))
    args=SimpleNamespace(config=cfg_path,data_root=data_root,output_root=tmp_path/"out",cache_root=tmp_path/"cache",
                         final_cache_root=[tmp_path/"final"],backend_module="fake",variants=None,worker_id=0,num_workers=1,device="cpu",limit=None,dry_run=False)
    assert runner_module.run(args)==0
    assert (args.output_root/"point_scores/toy/s/T2.npz").is_file()
    assert (args.output_root/"metrics/toy/s/T2.json").is_file()
    assert (args.output_root/"path_diagnostics/toy/s/T2.npz").is_file()
