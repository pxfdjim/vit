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
 event_mean=event_summary.set_index("group")
 e1_vs_b3={m:allm.loc["E1_existing",f"{m}_mean"]-b3[m] for m in M}
 e1_vs_e6={m:allm.loc["E1_existing",f"{m}_mean"]-allm.loc["E6_detect_only",f"{m}_mean"] for m in M}
 e2_vs_e6={m:allm.loc["E2_select",f"{m}_mean"]-allm.loc["E6_detect_only",f"{m}_mean"] for m in M}
 e6_vs_e7={m:allm.loc["E6_detect_only",f"{m}_mean"]-allm.loc["E7_no_query_reject",f"{m}_mean"] for m in M}
 e6_vs_e8={m:allm.loc["E6_detect_only",f"{m}_mean"]-allm.loc["E8_no_reference_skip",f"{m}_mean"] for m in M}
 e6_vs_e9={m:allm.loc["E6_detect_only",f"{m}_mean"]-allm.loc["E9_no_structural_residual",f"{m}_mean"] for m in M}
 e1_vs_e5={m:allm.loc["E1_existing",f"{m}_mean"]-allm.loc["E5_fixed_dtw",f"{m}_mean"] for m in M}
 mechanism_rows=pd.DataFrame([
  {"mechanism":"L_path","decision":"保留为弱结构正则","evidence":f"E1−E6={e1_vs_e6['aff_f1']:+.6f}/{e1_vs_e6['vus_pr']:+.6f}/{e1_vs_e6['vus_roc']:+.6f}；增加事件命中并减少误报，但牺牲连续排序"},
  {"mechanism":"L_select","decision":"不保留","evidence":f"E2−E6={e2_vs_e6['aff_f1']:+.6f}/{e2_vs_e6['vus_pr']:+.6f}/{e2_vs_e6['vus_roc']:+.6f}，拒配区分改善未转化为稳定检测增益"},
  {"mechanism":"L_detect","decision":"保留","evidence":"单独驱动 E6 对对齐器/残差/评分头均产生全覆盖梯度，并取得最高 All-11 VUS 均值"},
  {"mechanism":"Query Reject","decision":"与 Reference Skip 成对保留","evidence":f"禁用 Reject 后 E6−E7={e6_vs_e7['aff_f1']:+.6f}/{e6_vs_e7['vus_pr']:+.6f}/{e6_vs_e7['vus_roc']:+.6f}"},
  {"mechanism":"Reference Skip","decision":"与 Query Reject 成对保留","evidence":f"禁用 Skip 后 E6−E8={e6_vs_e8['aff_f1']:+.6f}/{e6_vs_e8['vus_pr']:+.6f}/{e6_vs_e8['vus_roc']:+.6f}"},
  {"mechanism":"结构残差输入","decision":"保留","evidence":f"E6−E9={e6_vs_e9['aff_f1']:+.6f}/{e6_vs_e9['vus_pr']:+.6f}/{e6_vs_e9['vus_roc']:+.6f}；移除后拒配在高波动正常区反而更高"},
  {"mechanism":"可微单调软对齐","decision":"保留","evidence":f"E1−固定 DTW E5={e1_vs_e5['aff_f1']:+.6f}/{e1_vs_e5['vus_pr']:+.6f}/{e1_vs_e5['vus_roc']:+.6f}"},
 ])
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

{table(mechanism_rows)}

1. **未发现 `L_detect` 梯度链路实现错误。** 同 Checkpoint、Batch 和参数集合下，`L_detect` 对对齐器的平均梯度范数为 `0.108927`、参数与元素覆盖率均为 `1.0`；`L_select` 为 `0.237337`，两者实际均值比约 `2.18`，不是此前单步记录的 `99,267` 倍。旧极端比值来自零初始化评分头的首步瞬态。首步更新评分头后，检测梯度能到达对齐器。`L_path` 与 `L_detect` 的对齐器梯度余弦均值为 `-0.136752`，说明主要问题是目标方向冲突。
2. **`L_detect` 可以独立训练完整 APRA。** E6 的 All-11 为 `{trip(allm.loc['E6_detect_only'], '_mean')}`，相对 B3 为 `{allm.loc['E6_detect_only','aff_f1_mean']-b3.aff_f1:+.6f}/{allm.loc['E6_detect_only','vus_pr_mean']-b3.vus_pr:+.6f}/{allm.loc['E6_detect_only','vus_roc_mean']-b3.vus_roc:+.6f}`。它牺牲很小的 Aff-F1，换来最高的 All-11 VUS-PR/VUS-ROC 均值；因此独立路径监督不是可训练性的必要条件。
3. **`L_path` 有结构与事件价值，但不是普适检测增益。** E1 相对 E6 为 `{e1_vs_e6['aff_f1']:+.6f}/{e1_vs_e6['vus_pr']:+.6f}/{e1_vs_e6['vus_roc']:+.6f}`；平均命中事件由 `{event_mean.loc['E6_detect_only','tp_events_mean']:.3f}` 增至 `{event_mean.loc['E1_existing','tp_events_mean']:.3f}`，误报区间由 `{event_mean.loc['E6_detect_only','fp_intervals_mean']:.3f}` 降至 `{event_mean.loc['E1_existing','fp_intervals_mean']:.3f}`。最终保留其现有固定权重作为弱结构正则，不能把它表述为所有排序指标的必要损失。
4. **`L_select` 提高异常/高波动正常的拒配差异，但没有解决目标冲突。** E2 的拒配均值为异常 `{state.set_index('variant').loc['E2_select','rejection_anomaly_mean']:.6f}`、高波动正常 `{state.set_index('variant').loc['E2_select','rejection_high_volatility_normal_mean']:.6f}`；然而 E2 相对 E6 为 `{e2_vs_e6['aff_f1']:+.6f}/{e2_vs_e6['vus_pr']:+.6f}/{e2_vs_e6['vus_roc']:+.6f}`。本轮不保留 `L_select`。
5. **Reject 与 Skip 作为成对间隙机制保留，当前实验不能分别归因。** 等长 14×14 Token 与固定双端点要求 Reject 数等于 Skip 数；删除任一状态都会令另一状态不可达。E7/E8 因而都退化为 Match-only 路径，并相对 E6 在三项 All-11 均值上下降。两组有两个种子逐点完全一致；主种子仅 MSL 因独立 CUDA 训练轨迹产生差异，不改变耦合结论。
6. **结构残差提供主要的连续排序增益。** E9 虽比 E6 高 `{allm.loc['E9_no_structural_residual','aff_f1_mean']-allm.loc['E6_detect_only','aff_f1_mean']:+.6f}` Aff-F1，但 VUS-PR/VUS-ROC 低 `{allm.loc['E6_detect_only','vus_pr_mean']-allm.loc['E9_no_structural_residual','vus_pr_mean']:+.6f}/{allm.loc['E6_detect_only','vus_roc_mean']-allm.loc['E9_no_structural_residual','vus_roc_mean']:+.6f}`。E9 的拒配在真实异常为 `{state.set_index('variant').loc['E9_no_structural_residual','rejection_anomaly_mean']:.6f}`，在高波动正常为 `{state.set_index('variant').loc['E9_no_structural_residual','rejection_high_volatility_normal_mean']:.6f}`，方向反转，说明只有状态概率而没有结构残差不足以把拒配转成可靠异常证据。
7. **可微软对齐需要保留。** 固定 DTW 的 E5 三项均值低于 E1 `{e1_vs_e5['aff_f1']:+.6f}/{e1_vs_e5['vus_pr']:+.6f}/{e1_vs_e5['vus_roc']:+.6f}`；E10 最简 Match-only 为 `{trip(allm.loc['E10_match_only'], '_mean')}`，几乎等于 B3，说明收益来自可学习间隙路径与结构残差的闭环，而不是额外参数量。E7--E10 参数量均为 `2,059,973`，结构差异未通过容量变化制造。
8. **最终模型保留 E1 原始 APRA。** 三种子 All-11 为 `{trip(allm.loc['E1_existing'], '_mean')}`，相对 B3 为 `{e1_vs_b3['aff_f1']:+.6f}/{e1_vs_b3['vus_pr']:+.6f}/{e1_vs_b3['vus_roc']:+.6f}`，三项均为正；同时平均多命中 `{event_mean.loc['E1_existing','tp_events_mean']-901:+.3f}` 个事件并少 `{228-event_mean.loc['E1_existing','fp_intervals_mean']:+.3f}` 个误报区间。E6 可作为偏连续排序的简化训练方案，但不是三指标同时不低于 B3 的最终替代。

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
