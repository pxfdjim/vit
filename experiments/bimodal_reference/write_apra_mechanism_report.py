#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import pandas as pd

M=("aff_f1","vus_pr","vus_roc")
GROUPS=("E1_existing","E2_select","E3_no_reject","E4_no_warp","E5_fixed_dtw",
        "E6_detect_only","E7_no_query_reject","E8_no_reference_skip",
        "E9_no_structural_residual","E10_match_only")
PRIMARY_NAMES={"E1_existing":"E1_existing","E2_select":"E2_select","E3_no_reject":"E3_no_reject",
               "E4_no_warp":"E4_no_warp","E5_fixed_dtw":"E5_fixed_dtw","E6_detect_only":"E6_detect_only",
               "E7_no_query_reject":"E7_no_query_reject","E8_no_reference_skip":"E8_no_reference_skip",
               "E9_no_structural_residual":"E9_no_structural_residual","E10_match_only":"E10_match_only"}

def table(frame): return frame.to_markdown(index=False,floatfmt=".6f")
def trip(row,suffix=""): return "/".join(f"{row[f'{m}{suffix}']:.6f}" for m in M)
def delta(a,b,suffix=""): return "/".join(f"{a[f'{m}{suffix}']-b[f'{m}{suffix}']:+.6f}" for m in M)

def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args()
 cfg=json.loads(a.config.read_text()); r=a.output_root; seed=cfg["primary_seed"]
 scope=pd.read_csv(r/"metrics_by_scope.csv"); ds=pd.read_csv(r/"metrics_by_dataset.csv"); multi=pd.read_csv(r/"multiseed_summary.csv"); multids=pd.read_csv(r/"multiseed_by_dataset.csv")
 events=pd.read_csv(r/"event_counts_all.csv"); eventds=pd.read_csv(r/"event_counts_by_dataset.csv"); path=pd.read_csv(r/"path_diagnostics_summary.csv"); state=pd.read_csv(r/"real_state_separation_summary.csv")
 train=pd.read_csv(r/"training_checkpoint_summary.csv"); paired=pd.read_csv(r/"paired_metric_tests.csv"); validation=json.loads((r/"validation.json").read_text())
 grad=pd.read_csv(r/"gradient_path_summary.csv"); cosine=pd.read_csv(r/"gradient_direction_cosines.csv")
 primary=["B3"]+[f"{v}_s{seed}" for v in GROUPS]
 main=scope[scope.variant.isin(primary)][["scope","variant",*M,"aff_f1_valid_signals"]]
 means=multi[multi.variant.isin(GROUPS)]; allm=means[means.scope.eq("all_11")].set_index("variant")
 b3=scope[(scope.scope.eq("all_11"))&(scope.variant.eq("B3"))].iloc[0]
 comparisons=[]
 for left,right in (("E1_existing","E6_detect_only"),("E2_select","E6_detect_only"),("E6_detect_only","E7_no_query_reject"),("E6_detect_only","E8_no_reference_skip"),("E6_detect_only","E9_no_structural_residual"),("E6_detect_only","E10_match_only")):
  comparisons.append({"comparison":f"{left} - {right}",**{f"delta_{m}":allm.loc[left,f"{m}_mean"]-allm.loc[right,f"{m}_mean"] for m in M}})
 comparisons=pd.DataFrame(comparisons)
 event_summary=(events[events.variant.ne("B3")].assign(group=lambda x:x.variant.str.replace(r"_s\d+$","",regex=True)).groupby("group",as_index=False).agg(seeds=("variant","size"),tp_events_mean=("tp_events","mean"),tp_events_std=("tp_events","std"),fn_events_mean=("fn_events","mean"),fp_intervals_mean=("fp_intervals","mean"),fp_intervals_std=("fp_intervals","std")))
 diag=path[path.seed.eq(seed)].set_index("variant").loc[list(GROUPS)].reset_index()
 diag_cols=["variant","match_probability","rejection","reference_skip","path_entropy","alignment_displacement","structural_residual","inference_seconds"]
 real=state.set_index("variant").loc[list(GROUPS)].reset_index()
 state_cols=["variant","rejection_anomaly_mean","rejection_high_volatility_normal_mean","match_probability_anomaly_mean","match_probability_high_volatility_normal_mean","structural_residual_anomaly_mean","structural_residual_high_volatility_normal_mean","delta_anomaly_mean","delta_high_volatility_normal_mean"]
 focus=eventds[(eventds.dataset.isin(["MSL","SMAP","realAWSCloudwatch","realAdExchange","realTraffic"]))&(eventds.variant.isin(primary))]
 per_dataset=[]
 for metric in M:
  x=multids[multids.variant.isin(GROUPS)].copy(); x["value"]=x.apply(lambda z:f"{z[f'{metric}_mean']:.6f}±{z[f'{metric}_std']:.6f}",axis=1); per_dataset.append(x.pivot(index="dataset",columns="variant",values="value").reset_index())
 training=train.groupby(["variant","seed"],as_index=False).agg(checkpoints=("dataset","count"),training_seconds=("elapsed_seconds","sum"),parameters_all=("parameters_all","max"),parameters_trainable=("parameters_trainable","max"),first_detect=("first_epoch_detect","mean"),final_detect=("final_epoch_detect","mean"),gradient_detect_initial_apra=("gradient_detect_apra","mean"),gradient_detect_after_step_apra=("gradient_detect_after_step_apra","mean"),gradient_detect_after_step_head=("gradient_detect_after_step_acef","mean"))
 grad_cos=cosine.groupby(["group","left","right"],as_index=False).cosine.agg(["mean","median","std"]).reset_index()
 commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=Path(__file__).resolve().parents[2],text=True).strip()
 report=f"""# APRA 损失与内部机制消融：OrionBench 最终报告

## 协议与完整性

- 实验前实现与梯度审计见 `APRA_IMPLEMENTATION_AUDIT.md`；E6--E10 协议提交为 `bb630484b`，最终报告代码为 `{commit}`。
- B3 与 E1--E5 逐点复用；E6 复用已验证、与 E1 严格同配置的 Detection-only 三种子结果；E7--E10 新训练 {validation['checks']['checkpoints']} 个固定最终 Epoch Checkpoint。
- 所有评分固定为 `S_B3 + 0.05 × calibrated_delta_APRA`。训练、合成、校准、权重与模型选择均不使用评价标签。
- 独立验证状态 **{validation['status']}**：{validation['checks']['point_scores']} 个点分数、{validation['checks']['metric_rows']} 行指标；复用结果最大误差 `{max(validation['checks']['reproduction_max_abs_errors'].values()):.3e}`。

## 实现与梯度链路结论

- 三状态前向/后向 DP 返回精确转移边缘概率；没有 argmax、Viterbi、top-k 或训练阶段硬路径。
- B3 proxy、难例权重及冻结视觉投影的 detach/no_grad 是有意冻结；APRA delta、残差、状态概率和对齐 logits 保持可微。
- 零初始化评分头使首次 `L_detect` 反向只更新评分头。最终 E2 Checkpoint 上，同 Checkpoint/Batch/参数集合的梯度如下。

{table(grad)}

### 梯度方向余弦

{table(grad_cos)}

## 配置

- E1：原始 `L_path + L_detect`；E2：`L_select + L_detect`；E3/E4/E5 复用既有消融。
- E6：完整三状态 APRA，仅 `L_detect`。
- E7/E8：分别禁用 Query Reject / Reference Skip。由于等长序列和严格双端点，Reject 与 Skip 数量必须相等，单独禁用任一状态会使另一 gap 状态也不可达。
- E9：完整三状态，评分头的 unmatched 与一阶结构残差输入置零；架构和参数量保持一致。
- E10：仅 Match 可行路径，并移除 unmatched/结构残差评分输入。

## 主种子结果

{table(main)}

## 三随机种子均值与标准差

{table(means)}

![All-11 相对 B3](figures/mechanism_metric_deltas.png)

## 关键成对变化（All-11 三种子均值）

{table(comparisons)}

## 事件召回与误报（三种子）

{table(event_summary)}

### 重点数据集主种子事件统计

{table(focus)}

## 路径、状态与推理成本（主种子）

{table(diag[diag_cols])}

![路径状态](figures/mechanism_path_states.png)

## 真实异常与正常高波动

{table(real[state_cols])}

## 逐数据集三随机种子结果

### Aff-F1

{table(per_dataset[0])}

### VUS-PR

{table(per_dataset[1])}

### VUS-ROC

{table(per_dataset[2])}

## 新训练配置参数、梯度与成本

{table(training)}

E7/E8 的状态耦合、所有新 Checkpoint 的行列概率守恒、禁用状态质量及评分头梯度均由 `validation.json` 检查。

## 逐信号配对差异

Wilcoxon p 值仅作描述，未用于选择模型且未做多重比较校正。

{table(paired)}

## 最终机制筛选

<!-- CONCLUSIONS -->

## 产物

- `checkpoints/`、`training_logs/`、`training_manifest.json`
- `point_scores/`、`feature_diagnostics/`、`inference_manifests/`
- `metrics_per_signal.csv`、`metrics_by_dataset.csv`、`metrics_by_scope.csv`
- `multiseed_summary.csv`、`multiseed_by_dataset.csv`
- `event_counts_*.csv`、`event_differences_per_signal.csv`
- `path_diagnostics_*.csv`、`real_state_separation_*.csv`
- `gradient_path_*.csv`、`gradient_direction_cosines.csv`
- `figures/`、`validation.json`
"""
 (r/"apra_mechanism_final_report.md").write_text(report); print(r/"apra_mechanism_final_report.md")

if __name__=="__main__": main()
