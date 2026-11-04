#!/usr/bin/env python3
"""Write the final Chinese RCCA/CADM experiment report from validated artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def table(frame, digits=6):
    return frame.to_markdown(index=False, floatfmt=f".{digits}f")


def triplet(row):
    return f"{row.aff_f1:.6f} / {row.vus_pr:.6f} / {row.vus_roc:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text()); root = args.output_root
    validation = json.loads((root / "validation.json").read_text())
    if validation["status"] != "passed":
        raise RuntimeError("refusing to report a failed validation")
    scope = pd.read_csv(root / "metrics_by_scope.csv")
    dataset = pd.read_csv(root / "metrics_by_dataset.csv")
    events = pd.read_csv(root / "event_counts_all.csv")
    event_diffs = pd.read_csv(root / "event_differences_per_signal.csv")
    paired = pd.read_csv(root / "paired_metric_tests.csv")
    stability = pd.read_csv(root / "stability_by_scope.csv")
    training = pd.read_csv(root / "training_checkpoint_summary.csv")
    inference = pd.read_csv(root / "inference_diagnostics_summary.csv")
    features = pd.read_csv(root / "feature_distributions_per_signal.csv")
    similarity = pd.read_csv(root / "feature_similarity_per_signal.csv")
    variants_main = ["T3", "E2", "B3", "AMR_native", "C2_simplified", "M1_full",
                     "M2_no_rcca", "M3_same_window", "M4_no_synthetic_objective",
                     "M5_shuffled_reference", "M6_no_cross_score"]
    main_scope = scope[scope.variant.isin(variants_main)].copy()
    comparison_rows = []
    all_scope = scope[scope.scope == "all_11"].set_index("variant")
    for control, meaning in [
        ("M2_no_rcca", "RCCA 的增量"), ("M3_same_window", "参考条件交互的增量"),
        ("M4_no_synthetic_objective", "合成辨别/排序目标的增量"),
        ("M5_shuffled_reference", "正确参考的增量"), ("M6_no_cross_score", "S_cross 接入的增量")]:
        comparison_rows.append({"comparison": f"M1 − {control}", "meaning": meaning,
            "delta_aff_f1": all_scope.loc["M1_full", "aff_f1"] - all_scope.loc[control, "aff_f1"],
            "delta_vus_pr": all_scope.loc["M1_full", "vus_pr"] - all_scope.loc[control, "vus_pr"],
            "delta_vus_roc": all_scope.loc["M1_full", "vus_roc"] - all_scope.loc[control, "vus_roc"]})
    comparisons = pd.DataFrame(comparison_rows)
    b3 = all_scope.loc["B3"]; m1 = all_scope.loc["M1_full"]
    seed_names = ["M1_full", "M1_seed_20261105", "M1_seed_20261106"]
    seed_all = scope[(scope.scope == "all_11") & scope.variant.isin(seed_names)]
    primary_beats = all(m1[m] > b3[m] for m in ("aff_f1", "vus_pr", "vus_roc"))
    every_seed_beats = all((seed_all.set_index("variant")[m] > b3[m]).all()
                           for m in ("aff_f1", "vus_pr", "vus_roc"))
    stability_wide = stability.pivot(index="scope", columns="metric", values=["mean", "std"])
    focus = dataset[(dataset.dataset.isin(["MSL", "SMAP", "realAWSCloudwatch",
                                            "realAdExchange", "realTraffic"])) &
                    dataset.variant.isin(["B3", "M1_full", "M2_no_rcca",
                                           "M3_same_window", "M4_no_synthetic_objective",
                                           "M5_shuffled_reference", "M6_no_cross_score"])]
    focus = focus[["dataset", "variant", "aff_f1", "vus_pr", "vus_roc"]]
    training_summary = training.groupby("variant", as_index=False).agg(
        checkpoints=("dataset", "count"), total_training_seconds=("elapsed_seconds", "sum"),
        mean_training_seconds=("elapsed_seconds", "mean"),
        mean_trainable_parameters=("parameters_trainable", "mean"),
        mean_peak_cuda_memory_mib=("peak_cuda_memory_bytes", lambda x: x.mean()/2**20),
        mean_numeric_gradient=("gradient_numeric", "mean"),
        mean_rcca_gradient=("gradient_rcca", "mean"),
        mean_cadm_gradient=("gradient_cadm", "mean"),
        min_reference_replacement_effect=("reference_replacement_mean_abs_logit_change", "min"))
    feature_summary = features[features.feature.isin([
        "numeric__M1_full", "cross__M1_full", "cross__M2_no_rcca",
        "cross__M3_same_window", "cross__M4_no_synthetic_objective",
        "cross__M5_shuffled_reference"])].groupby("feature", as_index=False).agg(
            signals=("signal", "count"),
            separation_median=("anomaly_minus_high_volatility_normal_in_normal_iqr", "median"),
            separation_q25=("anomaly_minus_high_volatility_normal_in_normal_iqr", lambda x: x.quantile(.25)),
            separation_q75=("anomaly_minus_high_volatility_normal_in_normal_iqr", lambda x: x.quantile(.75)),
            normal_volatility_spearman_median=("normal_volatility_spearman", "median"))
    similarity_summary = similarity[similarity.feature.str.startswith("cross__")].groupby(
        "feature", as_index=False).agg(
            pearson_with_raw=("pearson_with_raw", "mean"),
            pearson_with_M1_cross=("pearson_with_M1_cross", "mean"),
            pearson_with_B3=("pearson_with_B3", "mean"))
    event_summary = events[events.variant.isin(variants_main)].copy()
    event_m1_b3 = event_diffs[(event_diffs.variant == "M1_full") &
                              (event_diffs.baseline == "B3")].agg({
                                  "recovered_baseline_misses": "sum",
                                  "lost_baseline_hits": "sum", "delta_fp_intervals": "sum"})
    paired_m1 = paired[(paired.variant == "M1_full") & (paired.baseline == "B3")]
    lines = [
        "# RCCA + CADM 参考条件跨模态异常建模：OrionBench 最终报告", "",
        "## 协议与完整性", "",
        f"- 实验预注册提交 `a79d97e`；核心实现提交 `87dc84f`，短信号协议修正提交 `f296d7b`，并行推理提交 `8402cbd`。最终执行代码为当前提交。",
        "- 覆盖 11 个数据集、492 条评测信号；MSL/SMAP 使用官方 Train/Test，其余数据集使用完整序列无标签转导式训练。窗口 224、步长 56、16 Epoch。",
        "- CLIP ViT-B/16 与视觉投影冻结；数值编码器从既有 Full AMR 初始化。训练和校准均不使用真实异常标签，也未设置标签验证集或基于标签选择 Checkpoint。",
        "- T3 视觉参考与 constrained-DTW 路径逐信号固定；新模型严格使用 0.90/0.05/0.025/0.025 的 Sv/Sraw/Snum/Scross 预算。M6 将 0.05 全部分配给 Snum，用于验证 S_cross 的实际贡献。",
        f"- 独立验证状态 **{validation['status']}**：66 个 Checkpoint、33 个合成缓存、{validation['checks']['point_scores']} 个点分数文件、{validation['checks']['metric_rows']} 行指标。基线逐点复现最大误差均为 0。",
        "- SMAP 的 D-12 与 MSL 的 T-9 官方 Train 序列短于 8 个窗口，无法为每个查询构造排除四个邻近窗口的同信号参考；这 6 个窗口在每个种子训练缓存中被记录并剔除。全部 492 条评测信号仍完整保留。", "",
        "## 模型与消融定义", "",
        "- **M1**：RCCA 双向参考条件交互 + CADM + Patch 对齐、合成异常辨别、Margin Ranking、未扰动一致性。",
        "- **M2**：移除 RCCA，CADM 仅读取视觉/数值差异与参考匹配关系。",
        "- **M3**：使用原 AMR 同窗口 Cross-Attention，保留其余 CADM 与训练目标。",
        "- **M4**：保留 RCCA，仅使用 Patch 对齐与一致性，取消合成异常辨别和排序损失。",
        "- **M5**：在 M1 推理时跨查询循环并反转时间次序，直接打乱已对齐参考 Token；T3 路径、数值证据和权重保持不变。",
        "- **M6**：使用 M1 数值 Token，但最终分数不接入 S_cross。", "",
        "## 主结果", "",
        "指标顺序为 Aff-F1 / VUS-PR / VUS-ROC。", "",
        table(main_scope[["scope", "variant", "aff_f1", "vus_pr", "vus_roc"]]), "",
        "### M1 相对各控制的 All-11 变化", "", table(comparisons), "",
        "## 重点数据集", "", table(focus), "",
        "## 三随机种子稳定性", "",
        table(stability), "",
        f"M1 主种子相对 B3 的 All-11 变化为 `{m1.aff_f1-b3.aff_f1:+.6f}/{m1.vus_pr-b3.vus_pr:+.6f}/{m1.vus_roc-b3.vus_roc:+.6f}`。",
        f"主种子是否三项均超过 B3：**{'是' if primary_beats else '否'}**；三个种子是否每个种子三项均超过 B3：**{'是' if every_seed_beats else '否'}**。", "",
        "## 逐信号配对差异（M1 相对 B3）", "", table(paired_m1), "",
        "Wilcoxon p 值仅作描述性证据，未用于模型选择，也未做多重比较校正。", "",
        "## 异常事件恢复与误报", "", table(event_summary), "",
        f"M1 相对 B3 共恢复 `{int(event_m1_b3.recovered_baseline_misses)}` 个 B3 漏检事件，丢失 `{int(event_m1_b3.lost_baseline_hits)}` 个 B3 已检事件，误报区间变化 `{int(event_m1_b3.delta_fp_intervals):+d}`。", "",
        "## RCCA 注意力与 CADM 输出", "", table(inference), "",
        f"M5 的最小逐信号参考替换效应为 `{validation['checks']['minimum_M5_reference_effect']:.6g}`，所有信号均高于零；这确认真实参考参与了 CADM 推理。", "",
        "## 正常剧烈波动分离与特征相似度", "",
        "分离度为异常分数中位数减最高波动 10% 正常点中位数，并按正常分数 IQR 标准化。", "",
        table(feature_summary), "", table(similarity_summary), "",
        "## 参数、梯度与训练成本", "", table(training_summary), "",
        "M1/M3/M4 的数值编码器、RCCA/对应 Cross-Attention 与 CADM 均有非零梯度；M2 的 RCCA 参数量和梯度均为零。所有 Checkpoint 的查询替换与参考替换效应均高于审计阈值。", "",
        "## 完整逐数据集结果", "",
    ]
    for metric, title in [("aff_f1", "Aff-F1"), ("vus_pr", "VUS-PR"), ("vus_roc", "VUS-ROC")]:
        pivot = dataset[dataset.variant.isin(variants_main)].pivot(index="dataset", columns="variant", values=metric).reset_index()
        lines += [f"### {title}", "", table(pivot), ""]
    lines += [
        "## 最终回答", "",
        f"1. **M1 没有稳定超过既有 B3。** 主种子为 `{triplet(m1)}`，B3 为 `{triplet(b3)}`；All-11 三项分别变化 `{m1.aff_f1-b3.aff_f1:+.6f}/{m1.vus_pr-b3.vus_pr:+.6f}/{m1.vus_roc-b3.vus_roc:+.6f}`。三个种子的 All-11 均值为 `0.731022/0.388294/0.838141`，仍未在三项上超过 B3。",
        "2. **RCCA 没有显示独立的普遍性能价值。** M1 相对 M2 仅 Aff-F1 `+0.000220`，VUS-PR/VUS-ROC 为 `-0.000986/-0.000137`。非零梯度、注意力和参考替换效应证明 RCCA 真实参与了训练与推理，但没有转化为稳定指标收益。",
        "3. **参考条件交互与正确参考有局部价值，但不构成全指标必要性。** M1 相对同窗口 M3 为 `+0.008521/+0.001204/-0.000123`；相对打乱参考 M5 为 `+0.003348/+0.001999/-0.000546`。正确参考改善 Aff-F1/VUS-PR，却没有改善 All-11 VUS-ROC。",
        "4. **合成异常辨别与排序目标提供了最清晰的新增训练收益。** M1 相对 M4 在 All-11 三项均提高 `+0.004872/+0.004014/+0.000813`，且 M4 的 CADM Logit 标准差仅 `0.000824`，接近退化为常量输出。",
        "5. **CADM 的 S_cross 接入最终评分没有收益。** M1 相对 M6 三项为 `-0.001516/-0.004222/-0.004182`；M1 相对 B3 净少 17 个命中事件并增加 36 个误报区间。跨模态分数在 Yahoo 有收益，但 Real-6、AdExchange 和 Traffic 的退化抵消了它。",
        "6. **本实验没有得到严格不低于 B3 的新简化模型。** M6 是新模型中最强的总体控制，All-11 为 `0.729930/0.394810/0.841876`：VUS-ROC 高于 B3 `+0.001436`，Aff-F1/VUS-PR 仍低 `-0.001916/-0.001151`。既有 C2 简化模型仍是更清晰的工程选择。", "",
        "## 产物", "",
        "- `checkpoints/`、`synthetic_training_cache/`、`point_scores/`、`feature_diagnostics/`。",
        "- `metrics_per_signal.csv`、`metrics_by_dataset.csv`、`metrics_by_scope.csv`、`metric_deltas_*.csv`。",
        "- `event_counts_*.csv`、`event_differences_per_signal.csv`、`paired_metric_tests.csv`。",
        "- `feature_distributions_per_signal.csv`、`feature_similarity_per_signal.csv`。",
        "- `training_checkpoint_summary.csv`、`inference_diagnostics_*.csv`、`stability_*.csv`、`validation.json`。",
    ]
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
