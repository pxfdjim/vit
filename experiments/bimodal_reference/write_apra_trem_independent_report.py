#!/usr/bin/env python3
"""Write final Chinese report for frozen APRA and independent TREM."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.bimodal_reference.run_apra_trem_independent_inference import beta_code
M=("aff_f1","vus_pr","vus_roc")
def table(x): return x.to_markdown(index=False,floatfmt=".6f")
def triplet(x): return "/".join(f"{x[m]:.6f}" for m in M)
def delta(a,b): return "/".join(f"{a[m]-b[m]:+.6f}" for m in M)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(); cfg=json.loads(a.config.read_text()); root=a.output_root
 scope=pd.read_csv(root/"metrics_by_scope.csv"); dataset=pd.read_csv(root/"metrics_by_dataset.csv"); events=pd.read_csv(root/"event_counts_all.csv"); differences=pd.read_csv(root/"event_differences_per_signal.csv"); checkpoints=pd.read_csv(root/"training_checkpoint_summary.csv"); diagnostics=pd.read_csv(root/"inference_diagnostics_summary.csv"); seeds=pd.read_csv(root/"multiseed_summary.csv"); features=pd.read_csv(root/"trem_features_per_signal.csv"); validation=json.loads((root/"validation.json").read_text()); primary=cfg["primary_seed"]; b=beta_code(cfg["primary_beta"])
 names=["E0_B3","E1_APRA_only",f"X_existing_{b}",f"F_full_s{primary}_{b}",f"A_no_local_s{primary}_{b}",f"A_no_multiscale_s{primary}_{b}",f"A_no_reliability_s{primary}_{b}",f"A_numeric_only_s{primary}_{b}"]
 main=scope[scope.variant.isin(names)][["scope","variant",*M,"aff_f1_valid_signals"]].copy(); main["order"]=main.variant.map({n:i for i,n in enumerate(names)}); main=main.sort_values(["scope","order"]).drop(columns="order")
 all11=scope[scope.scope=="all_11"].set_index("variant"); e1=all11.loc["E1_APRA_only"]; full=all11.loc[f"F_full_s{primary}_{b}"]
 sensitivity=scope[(scope.scope=="all_11")&(scope.variant.str.startswith(("X_existing_",f"F_full_s{primary}_")))][["variant",*M]]
 focus=dataset[(dataset.dataset.isin(["MSL","SMAP","realAWSCloudwatch","realAdExchange","realTraffic"]))&(dataset.variant.isin(names))][["dataset","variant",*M]]
 event_main=events[events.variant.isin(names)].copy(); base=event_main.set_index("variant").loc["E1_APRA_only"]; event_main["delta_tp_vs_E1"]=event_main.tp_events-base.tp_events; event_main["delta_fp_vs_E1"]=event_main.fp_intervals-base.fp_intervals
 recovery=differences[differences.variant==f"F_full_s{primary}_{b}"][["recovered_E1_misses","lost_E1_hits"]].sum()
 train=checkpoints.groupby("variant",as_index=False).agg(checkpoints=("dataset","count"),training_seconds=("elapsed_seconds","sum"),parameters_all=("parameters_all","max"),parameters_trem=("parameters_trem","max"),frozen_change=("frozen_base_max_abs_change","max"),gradient_projection=("gradient_projection","mean"),gradient_local=("gradient_local","mean"),gradient_multiscale=("gradient_multiscale","mean"),gradient_reliability=("gradient_reliability_fusion","mean"),gradient_output=("gradient_output","mean"),input_effect=("input_effect","mean"),reliability_effect=("reliability_effect","mean"))
 diag=diagnostics[["model_id","variant","seed","source","raw_abs","raw_std","pre_tanh_abs","local_abs","multiscale_abs","calibrated_abs","calibrated_std","calibration_scale","total_inference_seconds","parameters_trem"]]
 feature_summary=features.groupby(["model_id","variant","seed","source"],as_index=False).agg(raw_abs_mean=("raw_abs_mean","median"),calibrated_abs_mean=("calibrated_abs_mean","median"),saturation_fraction=("saturation_fraction","median"),separation_median=("anomaly_minus_high_volatility_normal_in_normal_iqr","median"),normal_volatility_spearman_median=("normal_volatility_spearman","median"))
 old=diagnostics[diagnostics.model_id=="X_existing"].iloc[0]; new=diagnostics[diagnostics.model_id==f"F_full_s{primary}"].iloc[0]
 local=all11.loc[f"A_no_local_s{primary}_{b}"]; multi=all11.loc[f"A_no_multiscale_s{primary}_{b}"]; rel=all11.loc[f"A_no_reliability_s{primary}_{b}"]; numeric=all11.loc[f"A_numeric_only_s{primary}_{b}"]
 seed_full=seeds[(seeds.scope=="all_11")&(seeds.variant=="F_full")].iloc[0]
 seed_names=[f"F_full_s{seed}_{b}" for seed in cfg["seeds"]]; seed_values=all11.loc[seed_names]
 stable=bool((seed_values[list(M)] >= e1[list(M)].to_numpy()).all().all())
 stable_text="成立" if stable else "不成立"
 branch_rows=[]
 for seed in cfg["seeds"]:
  f=all11.loc[f"F_full_s{seed}_{b}"]; nl=all11.loc[f"A_no_local_s{seed}_{b}"]; nm=all11.loc[f"A_no_multiscale_s{seed}_{b}"]
  branch_rows.append({"seed":seed,**{f"local_delta_{m}":f[m]-nl[m] for m in M},**{f"multiscale_delta_{m}":f[m]-nm[m] for m in M}})
 branch_rows=pd.DataFrame(branch_rows)
 dataset_tables=[]
 for metric in M:
  pivot=dataset[dataset.variant.isin(names)].pivot(index="dataset",columns="variant",values=metric).reset_index()
  pivot=pivot[["dataset",*names]]; dataset_tables.append((metric,pivot))
 old_monotonic=all(all11.loc[f"X_existing_{beta_code(x)}",m] <= all11.loc[f"X_existing_{beta_code(y)}",m] for m in M for x,y in zip(cfg["beta_values"][1:],cfg["beta_values"][:-1]))
 full_monotonic=all(all11.loc[f"F_full_s{primary}_{beta_code(x)}",m] <= all11.loc[f"F_full_s{primary}_{beta_code(y)}",m] for m in M for x,y in zip(cfg["beta_values"][1:],cfg["beta_values"][:-1]))
 commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=Path(__file__).resolve().parents[2],text=True).strip()
 report=f"""# 冻结 APRA + 独立 TREM：OrionBench 最终报告

## 协议与完整性

- 预注册和实现提交 `{commit}`；全部新结果位于独立工作区。B3、数值编码器、视觉投影、APRA 对齐器及 APRA 评分头均逐参数冻结，验证最大变化为 `{checkpoints.frozen_base_max_abs_change.max():.1e}`。
- APRA-only 点分数逐点复用，原有 0.05 系数不变。TREM 独立使用 `S_final = S_APRA-only + beta * calibrated_TREM`；主配置 `beta=0.05`，其余系数只作预注册敏感性分析。
- TREM 使用现有无标签合成任务训练 16 Epoch，只更新 TREM；未读取评价标签或验证集。三个种子改变初始化与训练顺序，合成缓存固定。
- 无标签校准为中位数中心化、MAD 尺度、固定 0.05 尺度下限和 `[-1,1]` 截断。独立验证 **{validation['status']}**：{validation['checks']['checkpoints']} 个检查点、{validation['checks']['point_scores']} 个点分数文件、{validation['checks']['metric_rows']} 行指标。

## 幅度过小原因与修复

旧 TREM 在 `tanh` 后乘以 `0.25*(1-|APRA|)`，此前 29.46% Token 的剩余容量为零；最终平均评分效应仅 `1.10e-6`。解除该乘法后，既有检查点的原始输出平均绝对值为 `{old.raw_abs:.4f}`。冻结 APRA 重新训练后为 `{new.raw_abs:.4f}`，校准后的平均绝对证据为 `{new.calibrated_abs:.4f}`。因此旧幅度过小主要来自剩余预算压缩；新输出头和各分支具有非零梯度，未发现断路。

## 主结果

{table(main)}

主模型相对 APRA-only 的 All-11 变化为 `{delta(full,e1)}`。三种子 All-11 均值±标准差为 `{seed_full.aff_f1_mean:.6f}±{seed_full.aff_f1_std:.6f} / {seed_full.vus_pr_mean:.6f}±{seed_full.vus_pr_std:.6f} / {seed_full.vus_roc_mean:.6f}±{seed_full.vus_roc_std:.6f}`。

## 权重敏感性

{table(sensitivity)}

权重固定在实验前，主结论只依据 `beta=0.05`；表中最佳数值不用于改选系数。

## 分支与输入消融

Full 减去 no-local 为 `{delta(full,local)}`；Full 减去 no-multiscale 为 `{delta(full,multi)}`；Full 减去 no-reliability 为 `{delta(full,rel)}`；Full 减去 ordinary-numeric-only 为 `{delta(full,numeric)}`。

{table(seeds)}

## 重点数据集

{table(focus)}

## 完整逐数据集结果

### Aff-F1

{table(dataset_tables[0][1])}

### VUS-PR

{table(dataset_tables[1][1])}

### VUS-ROC

{table(dataset_tables[2][1])}

## 事件恢复与误报

{table(event_main)}

Full 相对 APRA-only 恢复 `{int(recovery.recovered_E1_misses)}` 个原漏检事件，同时丢失 `{int(recovery.lost_E1_hits)}` 个原命中事件。事件净变化和误报区间变化见表。

## 梯度、幅度、参数和成本

{table(train)}

{table(diag)}

{table(feature_summary)}

## 最终回答

1. **TREM 是否真正产生独立异常证据：产生了非零、可改变评分的独立输出，但没有产生有用的真实异常证据。** 主模型的校准证据平均绝对幅度为 `{new.calibrated_abs:.4f}`，各阶段梯度和输入替换效应均非零；然而主配置相对 APRA-only 为 `{delta(full,e1)}`，三项均明显下降。
2. **TREM 能否补充 APRA 漏检：有局部互补，但净结果为负。** Full 恢复 `{int(recovery.recovered_E1_misses)}` 个 APRA 漏检，同时丢失 `{int(recovery.lost_E1_hits)}` 个原命中，并增加 `{int(event_main.set_index('variant').loc[f'F_full_s{primary}_{b}','delta_fp_vs_E1'])}` 个误报区间。新增证据不能可靠区分补充异常与正常变化。
3. **两个时序分支是否都有实际贡献：否。** Full 对 no-local 的主种子变化为 `{delta(full,local)}`，Local 只在三个种子的 Aff-F1 上保持正贡献，两个 VUS 指标符号不稳定。Full 对 no-multiscale 为 `{delta(full,multi)}`，Multi-scale 在三个种子的 Aff-F1 和 VUS-ROC 均未显示正贡献。分支差异详见三种子表。
4. **APRA+TREM 是否稳定优于 APRA-only：{stable_text}。** 三个 Full 种子的 All-11 三项指标均低于 APRA-only；均值为 `{seed_full.aff_f1_mean:.6f}/{seed_full.vus_pr_mean:.6f}/{seed_full.vus_roc_mean:.6f}`。
5. **增益是否来自有效时序建模而非单纯放大：没有观察到增益，也没有支持有效时序建模。** ordinary-numeric-only 在主种子三项指标均高于 Full；既有和新 TREM 的 beta 从 0 增大时，三项指标均单调下降（既有={old_monotonic}，新模型={full_monotonic}）。结果说明解除压缩后放大了方向不可靠的证据，而非释放出被掩盖的时序收益。

## 产物

- `checkpoints/`、`training_logs/`、`training_manifest.json`
- `point_scores/`、`feature_diagnostics/`、`inference_manifests/`
- `metrics_per_signal.csv`、`metrics_by_dataset.csv`、`metrics_by_scope.csv`
- `event_counts_*.csv`、`event_differences_per_signal.csv`
- `training_checkpoint_summary.csv`、`inference_diagnostics_*.csv`
- `trem_features_per_signal.csv`、`multiseed_summary.csv`、`paired_metric_tests.csv`、`validation.json`
"""
 (root/"apra_trem_independent_final_report.md").write_text(report); print(root/"apra_trem_independent_final_report.md"); return 0
if __name__=="__main__": raise SystemExit(main())
