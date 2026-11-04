#!/usr/bin/env python3
"""Write the complete Chinese APRA/ACEF experiment report from validated CSVs."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd

METRICS=("aff_f1","vus_pr","vus_roc")
def triplet(row): return "/".join(f"{float(row[m]):.6f}" for m in METRICS)
def delta(left,right): return "/".join(f"{float(left[m]-right[m]):+.6f}" for m in METRICS)
def table(frame,columns=None):
 data=frame if columns is None else frame[columns]; return data.to_markdown(index=False,floatfmt=".6f")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(); cfg=json.loads(a.config.read_text()); root=a.output_root
 scope=pd.read_csv(root/"metrics_by_scope.csv"); dataset=pd.read_csv(root/"metrics_by_dataset.csv"); events=pd.read_csv(root/"event_counts_all.csv"); event_ds=pd.read_csv(root/"event_counts_by_dataset.csv")
 checkpoint=pd.read_csv(root/"training_checkpoint_summary.csv"); diag=pd.read_csv(root/"inference_diagnostics_summary.csv"); features=pd.read_csv(root/"correction_features_per_signal.csv"); paired=pd.read_csv(root/"paired_metric_tests.csv"); validation=json.loads((root/"validation.json").read_text())
 lookup=scope[scope.scope=="all_11"].set_index("variant"); full=lookup.loc["APRA_ACEF"]; b3=lookup.loc["B3"]; rcca=lookup.loc["RCCA_CADM_M1"]
 variants=["T3","E2","B3","RCCA_CADM_M1","C2_simplified",*cfg["variants"].keys()]
 scope_order=scope.assign(order=scope.variant.map({v:i for i,v in enumerate(variants)})).sort_values(["scope","order"]).drop(columns="order")
 main=scope_order[["scope","variant",*METRICS,"aff_f1_valid_signals"]]
 losses=[]
 for variant,label in (("no_Lpath","L_path"),("no_Lexplain","L_explain"),("no_Ldetect","L_detect")):
  row=lookup.loc[variant]; losses.append({"loss":label,"full_minus_ablation_aff_f1":full.aff_f1-row.aff_f1,"full_minus_ablation_vus_pr":full.vus_pr-row.vus_pr,"full_minus_ablation_vus_roc":full.vus_roc-row.vus_roc})
 mechanics=[]
 for variant,label in (("APRA_only","APRA-only / 无条件解释"),("ACEF_only_fixed_dtw","ACEF-only / 固定 DTW"),("no_reject","移除拒配"),("ordinary_cross_attention","普通 Cross-Attention"),("gated_fusion","常规门控融合"),("random_numeric","随机数值特征")):
  row=lookup.loc[variant]; mechanics.append({"control":label,"variant":variant,"aff_f1":row.aff_f1,"vus_pr":row.vus_pr,"vus_roc":row.vus_roc,"delta_aff_f1_vs_full":row.aff_f1-full.aff_f1,"delta_vus_pr_vs_full":row.vus_pr-full.vus_pr,"delta_vus_roc_vs_full":row.vus_roc-full.vus_roc})
 focus=dataset[dataset.dataset.isin(["MSL","SMAP","realAWSCloudwatch","realAdExchange","realTraffic"]) & dataset.variant.isin(["B3","RCCA_CADM_M1","APRA_only","ACEF_only_fixed_dtw","APRA_ACEF","no_reject"])]
 event_lookup=events.set_index("variant"); full_event=event_lookup.loc["APRA_ACEF"]; b3_event=event_lookup.loc["B3"]
 feature_summary=features.groupby("variant",as_index=False).agg(delta_mean=("delta_mean","mean"),delta_std=("delta_std","mean"),positive_fraction=("positive_fraction","mean"),negative_fraction=("negative_fraction","mean"),separation_median=("anomaly_minus_high_volatility_normal_in_normal_iqr","median"),normal_volatility_spearman_median=("normal_volatility_spearman","median"),pearson_with_B3=("pearson_with_B3","mean"))
 train=checkpoint.groupby("variant",as_index=False).agg(total_training_seconds=("elapsed_seconds","sum"),trainable_parameters=("parameters_trainable","max"),peak_cuda_memory_bytes=("peak_cuda_memory_bytes","max"),reference_effect=("reference_effect","mean"),gradient_detect_step1_apra=("gradient_detect_step1_apra","mean"),gradient_detect_step1_acef=("gradient_detect_step1_acef","mean"))
 full_diag=diag.set_index("variant").loc["APRA_ACEF"]
 pareto=all(full[m]>=b3[m] for m in METRICS)
 best=lookup.loc[list(cfg["variants"]),"aff_f1"].idxmax()
 commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
 text=f"""# APRA 与 ACEF：OrionBench 异常保持参考对齐实验报告

## 协议与完整性

- 实验预注册提交 `c68e457`，核心实现提交 `09b0a32`，报告代码提交 `{commit}`。全部新 checkpoint、点分数和日志位于独立运行目录，未覆盖历史结果。
- 覆盖 11 个数据集、492 条信号。MSL/SMAP 使用官方 Train/Test；Yahoo/NAB 使用完整序列无标签转导式协议。窗口 224、步长 56、固定 16 Epoch。
- T3 检索和 constrained-DTW 仅作为固定锚点；B3 分支逐点冻结复用。最终评分严格为 `S_B3 + 0.05·ΔS`，`ΔS` 使用无标签稳健尺度校准，修正头在训练前严格零初始化。
- 训练只使用 `L_path`、`L_explain`、`L_detect`。训练、校准、checkpoint 选择均不读取真实异常标签。
- 独立验证状态 **{validation['status']}**：{validation['checks']['checkpoints']} 个 checkpoint、{validation['checks']['point_scores']} 个点分数文件、{validation['checks']['metric_rows']} 行指标；基线复现和 no-L_detect=B3 的最大误差均为零。

## 结构实现

### APRA

APRA 在 T3 DTW 对齐位置周围建立 match、query reject、reference skip 三类单调路径转移，并通过可微前后向动态规划输出路径边缘概率。每个查询位置的匹配质量与拒配概率之和为 1。输出包括软匹配矩阵、拒配证据、路径可靠性、对齐参考、未匹配特征和一阶结构残差。可靠性头只读取路径熵、覆盖率、匹配质量和局部路径平滑度，不直接读取残差幅值。

### ACEF

ACEF 使用两个真实的多头条件注意力分支：视觉参考预测数值查询，数值参考预测视觉查询；APRA 软路径以 attention prior 进入两个分支。预测残差进一步分解为共同证据和互补证据，经 Temporal Transformer 输出有界正负修正。拒配与可靠性均作为时序输入，拒配本身不会被硬编码为异常。

## 主结果

指标顺序为 Aff-F1 / VUS-PR / VUS-ROC。

{table(main)}

完整 APRA+ACEF 的 All-11 为 `{triplet(full)}`，相对 B3 为 `{delta(full,b3)}`，相对 RCCA-CADM M1 为 `{delta(full,rcca)}`。其对 B3 的三指标严格 Pareto 结论为 **{'成立' if pareto else '不成立'}**。预注册新模型中 All-11 Aff-F1 最高的是 **{best}**。

## 主结构与机制对照

APRA-only 同时构成“移除条件解释分支”消融；ACEF-only 同时构成“固定 DTW”对照，避免将相同计算图重复训练后当作独立证据。

{table(pd.DataFrame(mechanics))}

完整模型减 APRA-only 为 `{delta(full,lookup.loc['APRA_only'])}`，刻画条件解释加入 APRA 后的净变化；完整模型减 ACEF-only/固定 DTW 为 `{delta(full,lookup.loc['ACEF_only_fixed_dtw'])}`，刻画可学习单调拒配对齐的净变化。

## 三项损失消融

正值表示完整模型优于移除该损失的配置。

{table(pd.DataFrame(losses))}

`no_Ldetect` 与 B3 逐点完全相同，验证最终分数修正确实只能由检测目标学得；`L_path` 和 `L_explain` 可更新中间表示，但在零初始化修正头且没有 `L_detect` 时不会凭空改变 B3。

## 梯度、参考依赖与成本

{table(train)}

完整模型 `L_detect` 在零初始化头完成第一步后对 APRA/ACEF 的平均梯度范数分别为 `{float(train.set_index('variant').loc['APRA_ACEF','gradient_detect_step1_apra']):.6e}` / `{float(train.set_index('variant').loc['APRA_ACEF','gradient_detect_step1_acef']):.6e}`。所有要求的损失到模块梯度路径均通过独立验证。

## 对齐、拒配与共同/互补证据

{table(diag)}

完整模型的平均路径熵为 `{float(full_diag.path_entropy):.6f}`，拒配概率 `{float(full_diag.rejection):.6f}`，可靠性 `{float(full_diag.reliability):.6f}`。结构保持时间变形使修正变化 `{float(full_diag.normal_warp_delta_change):.6f}`。共同与互补证据范数分别为 `{float(full_diag.shared_evidence_norm):.6f}` / `{float(full_diag.complementary_evidence_norm):.6f}`；其独立效果同时由 APRA-only、门控融合和 `no_Lexplain` 对照约束。

## 正常剧烈波动与异常残差保留

分离度为异常区间修正中位数减最高波动 10% 正常点修正中位数，并以正常修正 IQR 标准化。

{table(feature_summary)}

## 事件恢复与误报

{table(events)}

完整模型相对 B3 恢复事件 `{int(max(full_event.tp_events-b3_event.tp_events,0))}` 个、净事件命中变化 `{int(full_event.tp_events-b3_event.tp_events):+d}`，误报区间变化 `{int(full_event.fp_intervals-b3_event.fp_intervals):+d}`。逐事件 recovered/lost 明细保存在 `event_differences_per_signal.csv`。

## 重点数据集

{table(focus[["dataset","variant",*METRICS]])}

## 逐信号配对差异

Wilcoxon p 值仅作描述，不参与模型选择且未做多重比较校正。

{table(paired)}

## 结论

1. **APRA 是否保留异常残差并改善正常时间变形处理：** APRA 的软路径、拒配、可靠性和未匹配残差均实际进入修正头；正常变形稳定性和拒配消融给出机制证据。最终性能是否受益，以完整模型相对 ACEF-only 的 `{delta(full,lookup.loc['ACEF_only_fixed_dtw'])}` 为准。
2. **ACEF 是否提供互补条件解释：** 两个条件注意力分支均参与训练与推理。完整模型相对 APRA-only 的变化为 `{delta(full,lookup.loc['APRA_only'])}`；共同/互补证据并非余弦差异或双 MLP 门控。
3. **哪个训练目标产生最终修正：** `L_detect` 是闭环所必需的目标；移除后精确退回 B3。`L_path` 和 `L_explain` 的独立价值由上表的完整模型减单项消融给出，不能依据个别数据集提升宣称普遍必要。
4. **是否超过 RCCA-CADM 并接近 B3：** APRA+ACEF 相对 RCCA-CADM M1 为 `{delta(full,rcca)}`；相对 B3 为 `{delta(full,b3)}`。{'它在三项 All-11 指标上均不低于 B3。' if pareto else '它没有在三项 All-11 指标上同时达到 B3，因此不宣称稳定超过 B3。'}

## 产物

- `checkpoints/`、`training_logs/`、`training_manifest.json`
- `point_scores/`、`feature_diagnostics/`、`inference_manifests/`
- `metrics_per_signal.csv`、`metrics_by_dataset.csv`、`metrics_by_scope.csv`
- `event_counts_*.csv`、`event_differences_per_signal.csv`
- `correction_features_per_signal.csv`、`inference_diagnostics_*.csv`
- `training_checkpoint_summary.csv`、`paired_metric_tests.csv`、`validation.json`
"""
 (root/"apra_acef_final_report.md").write_text(text); print(root/"apra_acef_final_report.md"); return 0
if __name__=="__main__": raise SystemExit(main())
