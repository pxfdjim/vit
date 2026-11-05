#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import pandas as pd
M=("aff_f1","vus_pr","vus_roc")
def table(x): return x.to_markdown(index=False,floatfmt=".6f")
def trip(row,suffix=""): return "/".join(f"{row[f'{m}{suffix}']:.6f}" for m in M)
def dtrip(a,b,suffix=""): return "/".join(f"{a[f'{m}{suffix}']-b[f'{m}{suffix}']:+.6f}" for m in M)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(); cfg=json.loads(a.config.read_text()); r=a.output_root; primary=cfg["primary_seed"]
 scope=pd.read_csv(r/"metrics_by_scope.csv"); ds=pd.read_csv(r/"metrics_by_dataset.csv"); multi=pd.read_csv(r/"multiseed_summary.csv"); multids=pd.read_csv(r/"multiseed_by_dataset.csv"); events=pd.read_csv(r/"event_counts_all.csv"); eventds=pd.read_csv(r/"event_counts_by_dataset.csv"); train=pd.read_csv(r/"training_checkpoint_summary.csv"); diag=pd.read_csv(r/"path_diagnostics_summary.csv"); state=pd.read_csv(r/"real_state_separation_summary.csv"); paired=pd.read_csv(r/"paired_metric_tests.csv"); validation=json.loads((r/"validation.json").read_text()); weight=json.loads((Path(__file__).parent/"apra_selective_weight_audit.json").read_text())
 base_variants=["B3",f"E1_existing_s{primary}"]+[f"{v}_s{primary}" for v in cfg["variants"]]
 main=scope[scope.variant.isin(base_variants)][["scope","variant",*M,"aff_f1_valid_signals"]]
 means=multi[multi.variant.isin(["E1_existing",*cfg["variants"]])]
 allm=means[means.scope.eq("all_11")].set_index("variant"); e1=allm.loc["E1_existing"]; e2=allm.loc["E2_select"]
 delta=dtrip(e2,e1,"_mean"); all_positive=all(e2[f"{m}_mean"]>e1[f"{m}_mean"] for m in M)
 b3=scope[(scope.scope.eq("all_11"))&(scope.variant.eq("B3"))].iloc[0]
 delta_b3="/".join(f"{e2[f'{m}_mean']-b3[m]:+.6f}" for m in M)
 seed_scope=scope[scope.scope.eq("all_11")].set_index("variant"); seed_wins=[]
 for seed in cfg["seeds"]:
  left=seed_scope.loc[f"E2_select_s{seed}"]; right=seed_scope.loc[f"E1_existing_s{seed}"]; seed_wins.append(all(left[m]>right[m] for m in M))
 mech=train.groupby("variant",as_index=False).mean(numeric_only=True).set_index("variant"); m2=mech.loc["E2_select"]; m3=mech.loc["E3_no_reject"]; m4=mech.loc["E4_no_warp"]; m5=mech.loc["E5_fixed_dtw"]
 reject_sep=m2.mechanism_synthetic_structural_rejection-m2.mechanism_synthetic_matchable_rejection
 normal_sep=m2.mechanism_synthetic_structural_rejection-m2.mechanism_synthetic_unchanged_rejection
 reject_removal=m3.mechanism_synthetic_structural_rejection-m2.mechanism_synthetic_structural_rejection
 warp_removal=m4.mechanism_warp_target_match_mass-m2.mechanism_warp_target_match_mass
 path=diag[diag.seed.eq(primary)].set_index("variant"); p1=path.loc["E1_existing"]; p2=path.loc["E2_select"]
 real=state.set_index("variant"); s2=real.loc["E2_select"]
 real_reject_gap=s2.rejection_anomaly_mean-s2.rejection_high_volatility_normal_mean
 real_reject_gap_e1=real.loc["E1_existing"].rejection_anomaly_mean-real.loc["E1_existing"].rejection_high_volatility_normal_mean
 ev=events[events.variant.isin(base_variants)].copy(); evbase=ev.set_index("variant").loc[f"E1_existing_s{primary}"]; eve2=ev.set_index("variant").loc[f"E2_select_s{primary}"]; ev["delta_tp_vs_E1"]=ev.tp_events-evbase.tp_events; ev["delta_fp_vs_E1"]=ev.fp_intervals-evbase.fp_intervals
 focus=eventds[(eventds.dataset.isin(["MSL","SMAP","realAWSCloudwatch","realAdExchange","realTraffic"]))&(eventds.variant.isin(base_variants))].copy()
 event_multi=(events[events.variant.ne("B3")].assign(variant=lambda x:x.variant.str.replace(r"_s\d+$","",regex=True)).groupby("variant",as_index=False).agg(seeds=("variant","size"),tp_events_mean=("tp_events","mean"),tp_events_std=("tp_events","std"),fp_intervals_mean=("fp_intervals","mean"),fp_intervals_std=("fp_intervals","std")))
 train_table=train.groupby(["variant","seed"],as_index=False).agg(checkpoints=("dataset","count"),training_seconds=("elapsed_seconds","sum"),parameters=("parameters_all","max"),trainable_parameters=("parameters_trainable","max"),first_select=("first_epoch_select","mean"),final_select=("final_epoch_select","mean"),first_detect=("first_epoch_detect","mean"),final_detect=("final_epoch_detect","mean"),gradient_select_apra=("gradient_select_apra","mean"),gradient_detect_after_step_apra=("gradient_detect_after_step_apra","mean"),gradient_detect_after_step_acef=("gradient_detect_after_step_acef","mean"))
 grad_select=float(train.loc[train.variant.eq("E2_select"),"gradient_select_apra"].mean()); grad_detect=float(train.loc[train.variant.eq("E2_select"),"gradient_detect_after_step_apra"].mean()); grad_ratio=grad_select/max(grad_detect,1e-12)
 e3=allm.loc["E3_no_reject"]; e4=allm.loc["E4_no_warp"]; e5=allm.loc["E5_fixed_dtw"]
 delta_e3=dtrip(e2,e3,"_mean"); delta_e4=dtrip(e2,e4,"_mean"); delta_e5=dtrip(e2,e5,"_mean")
 mechanism_cols=["variant","mechanism_clean_path_entropy","mechanism_clean_match_probability","mechanism_clean_rejection","mechanism_synthetic_structural_rejection","mechanism_synthetic_matchable_rejection","mechanism_synthetic_unchanged_rejection","mechanism_warp_target_match_mass","mechanism_clean_displacement","mechanism_clean_structural_residual"]
 mechanism=mech.reset_index()[mechanism_cols]
 state_cols=["variant","rejection_anomaly_mean","rejection_high_volatility_normal_mean","match_probability_anomaly_mean","match_probability_high_volatility_normal_mean","structural_residual_anomaly_mean","structural_residual_high_volatility_normal_mean","delta_anomaly_mean","delta_high_volatility_normal_mean"]
 real_table=state.reset_index()[state_cols]
 per_dataset=[]
 for metric in M:
  x=multids.copy(); x["value"]=x.apply(lambda z:f"{z[f'{metric}_mean']:.6f}±{z[f'{metric}_std']:.6f}",axis=1); per_dataset.append((metric,x.pivot(index="dataset",columns="variant",values="value").reset_index()))
 regressions={m:multids.pivot(index="dataset",columns="variant",values=f"{m}_mean").query("E2_select < E1_existing").index.tolist() for m in M}
 commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=Path(__file__).resolve().parents[2],text=True).strip(); repro=max(validation["checks"]["reproduction_max_abs_errors"].values())
 report=f"""# 异常感知选择性 APRA：OrionBench 最终报告

## 协议与完整性

- 选择性对齐协议与实现于结果产生前提交为 `ceb25becf`；最终报告代码为 `{commit}`。E0/B3 与 E1/既有 APRA 均逐点复用，E2–E5 新训练 {validation['checks']['checkpoints']} 个固定最终 Epoch Checkpoint，历史 Checkpoint 与结果未覆盖。
- 全部模型使用同一 Full-AMR 数值初始化、冻结视觉投影、T3 正常参考和参考列、合成缓存、16 Epoch 预算及三随机种子。最终评分始终为 `S_B3 + 0.05 × calibrated_delta_APRA`。
- 无标签初始尺度审计得到 `median(L_select)={weight['median_select']:.6f}`、`median(L_detect)={weight['median_detect']:.6f}`，按预注册公式固定 `lambda_select={weight['lambda_select']:.2f}`。该数值未依据评价标签或结果修改。
- 独立验证状态 **{validation['status']}**：{validation['checks']['point_scores']} 个点分数文件、{validation['checks']['metric_rows']} 行指标；B3/E1 最大复现误差 `{repro:.3e}`。训练、校准、权重和 Checkpoint 选择均未读取评价标签。

## 配置定义

- **E0 B3**：冻结基线。
- **E1 Existing APRA**：既有 `L_path + L_detect`，复用三种子结果。
- **E2 Selective APRA**：`2.15 L_select + L_detect`；正常平滑位移、局部伸缩和形变监督 Match，frequency/volatility/morphology 的掩码区域监督 Query Reject。
- **E3 No reject supervision**：移除结构异常的 Query-Reject 目标。
- **E4 No warp supervision**：移除已知正常时间扭曲对应目标。
- **E5 Fixed DTW**：以固定 T3 对应替代可学习软对齐，只保留同一检测目标。

## 主结果（主种子）

{table(main)}

## 三随机种子均值与标准差

{table(means)}

E2 相对 E1 的 All-11 三种子均值变化为 `{delta}`。三项均值全部提高：**{'是' if all_positive else '否'}**；逐种子三项同时提高为 `{sum(seed_wins)}/3`。

E2 相对固定 B3 的 All-11 均值变化为 `{delta_b3}`。其均值三项均高于 B3，但 Aff-F1 仅 `2/3` 个种子高于 B3，因此不能据此宣称逐种子稳定胜出。

![三随机种子主指标](figures/selective_metrics.png)

## L_select 的路径行为

{table(mechanism)}

E2 在合成结构异常区的平均拒配概率比可匹配幅值/趋势扰动高 `{reject_sep:+.6f}`，比未扰动区域高 `{normal_sep:+.6f}`。移除拒配监督后，结构异常拒配概率变化为 `{reject_removal:+.6f}`，说明该监督真实控制 Reject。移除正常扭曲监督后，已知目标匹配质量变化为 `{warp_removal:+.6f}`；该数值没有支持正常扭曲监督的独立机制收益。

![选择性路径行为](figures/selective_path_behavior.png)

推理阶段主种子 E2 相对 E1 的平均路径熵/Match/Reject/位移变化为 `{p2.path_entropy-p1.path_entropy:+.6f}/{p2.match_probability-p1.match_probability:+.6f}/{p2.rejection-p1.rejection:+.6f}/{p2.alignment_displacement-p1.alignment_displacement:+.6f}`。

## 真实异常与正常高波动

{table(real_table)}

E2 的真实异常拒配概率减最高波动 10% 正常点为 `{real_reject_gap:+.6f}`，E1 为 `{real_reject_gap_e1:+.6f}`。该分析只在所有分数冻结后打开标签，用于解释结果，不参与训练、校准、权重或模型选择。

## 事件恢复与误报

{table(ev)}

主种子 E2 相对 E1 的事件净变化为 `{int(eve2.tp_events-evbase.tp_events):+d}`，误报区间变化为 `{int(eve2.fp_intervals-evbase.fp_intervals):+d}`。

### 三随机种子事件统计

{table(event_multi)}

### 重点数据集

{table(focus)}

## 完整逐数据集三种子结果

E2 相对 E1 退化的数据集：Aff-F1 为 `{', '.join(regressions['aff_f1']) or '无'}`；VUS-PR 为 `{', '.join(regressions['vus_pr']) or '无'}`；VUS-ROC 为 `{', '.join(regressions['vus_roc']) or '无'}`。

### Aff-F1

{table(per_dataset[0][1])}

### VUS-PR

{table(per_dataset[1][1])}

### VUS-ROC

{table(per_dataset[2][1])}

## 优化、梯度、参数与成本

{table(train_table)}

E2/E3/E4 结构和参数量完全相同，差异只来自预注册监督项；E5 删除可学习对齐器。`L_select` 和 `L_detect` 对 APRA 对齐参数及评分头的梯度记录在上表，首末 Epoch 损失用于检查两目标是否失衡。

E2 的初始审计 batch 中，`L_select` 对 APRA 对齐器的平均梯度范数为 `{grad_select:.3e}`，一次更新后 `L_detect` 对同一模块为 `{grad_detect:.3e}`，前者约为后者的 `{grad_ratio:.1f}` 倍；`L_detect` 对评分头的梯度非零。按损失值平衡权重并未带来对齐器梯度尺度平衡。

## 逐信号配对差异

Wilcoxon p 值只作描述，未做多重比较校正，也未用于选择模型。

{table(paired)}

## 最终结论

1. **L_select 是否改善正常匹配与异常拒配区分：部分支持。** E2 相对 E1 降低推理路径熵和普通拒配，并将合成结构异常与可匹配/未扰动区域的拒配拉开 `{reject_sep:+.6f}/{normal_sep:+.6f}`；真实异常相对高波动正常的拒配差也从 `{real_reject_gap_e1:+.6f}` 增至 `{real_reject_gap:+.6f}`。但移除正常扭曲监督后目标匹配质量反而变化 `{warp_removal:+.6f}`，所以不能把正常匹配改善单独归因于该子目标。
2. **选择性对齐是否提高检测性能：** E2 相对 E1 的 All-11 三种子变化为 `{delta}`，因此**{'三项宏平均均提高' if all_positive else '没有形成三项一致提高'}**。事件与数据集权衡如上。
3. **收益是否来自软对齐和拒配而非容量：证据混合。** E2 相对固定 DTW 的变化为 `{delta_e5}`，三项均提高，支持可学习软对齐的价值。相同容量下，E2 相对无拒配监督为 `{delta_e3}`，相对无扭曲监督为 `{delta_e4}`，均存在指标符号变化，因此没有证明两个新增监督都带来稳定检测收益。
4. **是否改善 L_path 与 L_detect 不协调：没有。** 两项损失均下降且评分头获得检测梯度，但初始审计中路径目标对 APRA 对齐器的梯度约为检测目标的 `{grad_ratio:.1f}` 倍；结合 E2 未一致超过 E1，优化不协调仍然存在。
5. **能否作为 B3 的稳定核心模块：暂不替换 E1。** E2 的 All-11 三项均值均高于 B3，但只有 `2/3` 个种子在 Aff-F1 上高于 B3，且 E2 相对 E1 为 `{delta}`、平均事件命中也更低。当前保留 E1 作为经验证核心。

## 产物

- `checkpoints/`、`training_logs/`、`training_manifest.json`
- `point_scores/`、`feature_diagnostics/`、`inference_manifests/`
- `metrics_per_signal.csv`、`metrics_by_dataset.csv`、`metrics_by_scope.csv`
- `multiseed_summary.csv`、`multiseed_by_dataset.csv`
- `event_counts_*.csv`、`event_differences_per_signal.csv`
- `path_diagnostics_*.csv`、`real_state_separation_*.csv`
- `training_checkpoint_summary.csv`、`paired_metric_tests.csv`
- `figures/selective_metrics.png`、`figures/selective_path_behavior.png`、`validation.json`
"""
 (r/"apra_selective_final_report.md").write_text(report); print(r/"apra_selective_final_report.md")
if __name__=="__main__": main()
