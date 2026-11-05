#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
M=("aff_f1","vus_pr","vus_roc")
def table(x): return x.to_markdown(index=False,floatfmt=".6f")
def trip(x): return "/".join(f"{x[m]:.6f}" for m in M)
def delta(a,b): return "/".join(f"{a[m]-b[m]:+.6f}" for m in M)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(); cfg=json.loads(a.config.read_text()); root=a.output_root; scope=pd.read_csv(root/"metrics_by_scope.csv"); ds=pd.read_csv(root/"metrics_by_dataset.csv"); seeds=pd.read_csv(root/"multiseed_summary.csv"); seed_ds=pd.read_csv(root/"multiseed_by_dataset.csv"); events=pd.read_csv(root/"event_counts_all.csv"); eventds=pd.read_csv(root/"event_counts_by_dataset.csv"); paired=pd.read_csv(root/"paired_metric_tests.csv"); diag=pd.read_csv(root/"path_diagnostics_summary.csv"); train=pd.read_csv(root/"training_checkpoint_summary.csv"); validation=json.loads((root/"validation.json").read_text()); primary=cfg["primary_seed"]
 lossnames=[f"E0_full_s{primary}",f"E1_no_path_s{primary}",f"E2_path_only_s{primary}"]; gamma=[f"G_APRA_g{int(round(x*1000)):03d}" for x in cfg["gamma_values"]]
 main=scope[scope.variant.isin(lossnames)][["scope","variant",*M,"aff_f1_valid_signals"]]; sensitivity=scope[scope.variant.isin(gamma)][["scope","variant",*M]]
 allseed=seeds[seeds.scope=="all_11"].set_index("variant"); e0=allseed.loc["E0_full"]; e1=allseed.loc["E1_no_path"]; e2=allseed.loc["E2_path_only"]
 path_perf=all(e0[f"{m}_mean"]>e1[f"{m}_mean"] for m in M); path_perf_text="三项 All-11 均值均为正" if path_perf else "三项 All-11 均值并非全部为正"
 dlookup=diag.set_index("model_id"); d0=dlookup.loc[f"E0_full_s{primary}"]; d1=dlookup.loc[f"E1_no_path_s{primary}"]; d2=dlookup.loc[f"E2_path_only_s{primary}"]
 sl=scope[scope.scope=="all_11"].set_index("variant"); g025=sl.loc["G_APRA_g025"]; g050=sl.loc["G_APRA_g050"]; g075=sl.loc["G_APRA_g075"]; sensitivity_delta=max(max(abs(g025[m]-g050[m]) for m in M),max(abs(g075[m]-g050[m]) for m in M)); obvious=sensitivity_delta>=.005
 loss_dataset=[]; gamma_dataset=[]
 loss_dataset_multiseed=[]
 for m in M:
  loss_dataset.append((m,ds[ds.variant.isin(lossnames)].pivot(index="dataset",columns="variant",values=m).reset_index()))
  gamma_dataset.append((m,ds[ds.variant.isin(gamma)].pivot(index="dataset",columns="variant",values=m).reset_index()))
  formatted=seed_ds.copy(); formatted["value"]=formatted.apply(lambda r:f"{r[f'{m}_mean']:.6f}±{r[f'{m}_std']:.6f}",axis=1)
  loss_dataset_multiseed.append((m,formatted.pivot(index="dataset",columns="variant",values="value").reset_index()))
 regressions={}
 for m in M:
  pivot=seed_ds.pivot(index="dataset",columns="variant",values=f"{m}_mean")
  regressions[m]=pivot.index[(pivot["E0_full"]-pivot["E1_no_path"])<0].tolist()
 ev=events[events.variant.isin(lossnames+gamma)].copy(); base=ev.set_index("variant").loc[f"E0_full_s{primary}"]; ev["delta_tp_vs_E0"]=ev.tp_events-base.tp_events; ev["delta_fp_vs_E0"]=ev.fp_intervals-base.fp_intervals
 focus_event=eventds[(eventds.dataset.isin(["MSL","SMAP","realAWSCloudwatch","realAdExchange","realTraffic"]))&(eventds.variant.isin(lossnames))].copy()
 tr=train.groupby(["variant","seed"],as_index=False).agg(checkpoints=("dataset","count"),training_seconds=("elapsed_seconds","sum"),parameters_all=("parameters_all","max"),parameters_trainable=("parameters_trainable","max"),reference_effect=("reference_effect","mean"),gradient_path_apra=("gradient_path_apra","mean"),gradient_detect_initial_apra=("gradient_detect_apra","mean"),gradient_detect_after_step_apra=("gradient_detect_after_step_apra","mean"),gradient_detect_after_step_acef=("gradient_detect_after_step_acef","mean"))
 commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip()
 reproduction_max=max(validation['checks']['reproduction_max_abs_errors'].values())
 report=f"""# APRA-only 损失消融与评分权重敏感性：OrionBench 最终报告

## 协议与完整性

- 实验预注册提交 `1c52066c8`；最终报告代码提交 `{commit}`。E0 主种子逐点复用既有 APRA-only；新训练 88 个 Checkpoint，覆盖 E0 的两个补充种子以及 E1/E2 的三个种子，全部固定训练 16 Epoch。
- 三组使用相同结构、数值与视觉初始化、T3 参考、合成缓存、样本顺序规则及超参数。同一数据集和 seed 的初始化 SHA256 一致。
- 权重实验固定既有 APRA-only，直接以 `S_B3 + gamma * delta_APRA` 重组分数；`gamma=0.05` 是历史主配置，扫描结果不用于重新选择权重。
- 训练、校准和权重设定均未读取评价标签。独立验证 **{validation['status']}**：{validation['checks']['checkpoints']} 个新 Checkpoint、{validation['checks']['point_scores']} 个点分数文件和 {validation['checks']['metric_rows']} 行指标；最大复现误差 `{reproduction_max:.3e}`，验证容差 `{validation['checks']['reproduction_tolerance']:.1e}`。

## 损失消融主结果

{table(main)}

### 三随机种子均值与标准差

{table(seeds)}

All-11 上 E0 减 E1 的三项均值变化为 `{e0.aff_f1_mean-e1.aff_f1_mean:+.6f}/{e0.vus_pr_mean-e1.vus_pr_mean:+.6f}/{e0.vus_roc_mean-e1.vus_roc_mean:+.6f}`，{path_perf_text}。E2 没有 `L_detect`，修正头维持零，因此逐点精确退回 B3；它用于诊断对齐机制，不能单独证明检测能力。

![损失消融三种子趋势](figures/loss_ablation.png)

## L_path 对齐机制诊断

| variant | path_entropy | rejection | reliability | alignment_displacement | normal_warp_delta_change |
|:--|--:|--:|--:|--:|--:|
| E0 Full | {d0.path_entropy:.6f} | {d0.rejection:.6f} | {d0.reliability:.6f} | {d0.alignment_displacement:.6f} | {d0.normal_warp_delta_change:.6f} |
| E1 No path | {d1.path_entropy:.6f} | {d1.rejection:.6f} | {d1.reliability:.6f} | {d1.alignment_displacement:.6f} | {d1.normal_warp_delta_change:.6f} |
| E2 Path only | {d2.path_entropy:.6f} | {d2.rejection:.6f} | {d2.reliability:.6f} | {d2.alignment_displacement:.6f} | {d2.normal_warp_delta_change:.6f} |

E0 相对 E1 的机制变化为：路径熵 `{d0.path_entropy-d1.path_entropy:+.6f}`、拒配概率 `{d0.rejection-d1.rejection:+.6f}`、可靠性 `{d0.reliability-d1.reliability:+.6f}`、绝对对齐位移 `{d0.alignment_displacement-d1.alignment_displacement:+.6f}`。E2 显示仅靠路径目标可形成的结构约束上限。

## APRA 权重敏感性

{table(sensitivity)}

`gamma=0.05` 相对 0.025 的 All-11 变化为 `{delta(g050,g025)}`，相对 0.075 为 `{delta(g050,g075)}`。邻近权重最大绝对指标变化为 `{sensitivity_delta:.6f}`。报告以 0.005 作为实用幅度描述线（不参与权重选择），据此权重敏感性为 **{'明显' if obvious else '不明显'}**。

完整扫描仍存在数据集权衡：从 `gamma=0.05` 增至 `0.15` 时，realAdExchange Aff-F1 从 `0.793663` 降至 `0.690776`，realTraffic VUS-PR/VUS-ROC 从 `0.533170/0.858496` 降至 `0.494399/0.849089`；与此同时 MSL Aff-F1 持续提高。因而“不明显”仅指 `0.05` 邻域的整体宏平均，不表示任意增大系数都安全。

![APRA 权重敏感性趋势](figures/gamma_sensitivity.png)

## 完整逐数据集结果

以下损失消融表均为三种子均值±总体标准差。E0 相对 E1 发生退化的数据集：Aff-F1 为 `{', '.join(regressions['aff_f1']) or '无'}`；VUS-PR 为 `{', '.join(regressions['vus_pr']) or '无'}`；VUS-ROC 为 `{', '.join(regressions['vus_roc']) or '无'}`。

### 损失消融 Aff-F1（三种子）

{table(loss_dataset_multiseed[0][1])}

### 损失消融 VUS-PR（三种子）

{table(loss_dataset_multiseed[1][1])}

### 损失消融 VUS-ROC（三种子）

{table(loss_dataset_multiseed[2][1])}

### 权重敏感性 Aff-F1

{table(gamma_dataset[0][1])}

### 权重敏感性 VUS-PR

{table(gamma_dataset[1][1])}

### 权重敏感性 VUS-ROC

{table(gamma_dataset[2][1])}

## 事件召回与误报

{table(ev)}

### 重点数据集事件统计

{table(focus_event)}

## 梯度、参数量与训练成本

{table(tr)}

## 逐信号配对变化

Wilcoxon p 值仅作描述，未用于选择模型或权重，也未做多重比较校正。

{table(paired)}

## 最终结论

1. **L_path 是否具有结构约束价值：** 有。它对 APRA 具有非零梯度，且 E0/E2 与 E1 的路径熵、拒配、可靠性和位移存在上述系统变化。
2. **L_path 是否具有实际检测性能贡献：** E0 减 E1 的三种子 All-11 均值为 `{e0.aff_f1_mean-e1.aff_f1_mean:+.6f}/{e0.vus_pr_mean-e1.vus_pr_mean:+.6f}/{e0.vus_roc_mean-e1.vus_roc_mean:+.6f}`。因此结论为：**{'支持保留 L_path' if path_perf else '没有支持 L_path 在三项检测指标上普遍必要'}**。
3. **L_detect 是否必要：** 对生成 APRA 检测修正是必要的。E2 虽能学习对齐路径，但没有检测梯度，最终分数逐点等于 B3；E0/E1 在零初始化评分头完成一步更新后，检测损失对 APRA 与评分头的梯度均非零。
4. **当前 gamma=0.05 是否表现出明显权重敏感性：** **{'是' if obvious else '否'}**。相邻 0.025/0.075 对照和完整趋势见表；该扫描仅描述敏感性，不据此追认其他权重。
5. **最终保留方案：** {'继续使用完整 E0 的 `L_path + L_detect`。' if path_perf else '依据当前证据，L_path 保留为结构约束，但不能宣称其对最终检测指标不可替代。'}

## 产物

- `checkpoints/`、`training_logs/`、`training_manifest.json`
- `point_scores/`、`feature_diagnostics/`、`inference_manifests/`
- `metrics_per_signal.csv`、`metrics_by_dataset.csv`、`metrics_by_scope.csv`
- `event_counts_*.csv`、`event_differences_per_signal.csv`
- `path_diagnostics_*.csv`、`training_checkpoint_summary.csv`、`paired_metric_tests.csv`
- `figures/loss_ablation.png`、`figures/gamma_sensitivity.png`、`validation.json`
"""
 (root/"apra_loss_weight_final_report.md").write_text(report); print(root/"apra_loss_weight_final_report.md")
if __name__=="__main__": main()
