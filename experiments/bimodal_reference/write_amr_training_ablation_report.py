#!/usr/bin/env python3
"""Write the final report for the preregistered AMR training ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIANT_ORDER = [
    "T3", "E2", "B3", "T1_random", "T2_numeric_ssl", "Full",
    "C1_full", "C2_no_cross", "C3_late_joint", "C4_numeric_only",
    "L1_no_patch", "L2_no_temporal_relation", "L3_no_transition_geometry",
    "L4_no_reference", "L5_no_alignment",
]

LABELS = {
    "T1_random": "T1 Random", "T2_numeric_ssl": "T2 Numeric SSL",
    "C1_full": "C1 Full", "C2_no_cross": "C2 No Cross-Attention",
    "C3_late_joint": "C3 Late-joint", "C4_numeric_only": "C4 Numeric-only",
    "L1_no_patch": "L1 −Patch", "L2_no_temporal_relation": "L2 −Temporal Relation",
    "L3_no_transition_geometry": "L3 −Transition Geometry",
    "L4_no_reference": "L4 −Reference", "L5_no_alignment": "L5 −All Alignment",
}


def label(value: str) -> str:
    return LABELS.get(value, value)


def markdown(df: pd.DataFrame, digits: int = 6) -> str:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return out.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol-commit", default="35e6e7b")
    parser.add_argument("--code-commit", default="86ad636")
    args = parser.parse_args()
    run = args.run_dir

    scope = pd.read_csv(run / "metrics_by_scope.csv")
    dataset = pd.read_csv(run / "metrics_by_dataset.csv")
    paired = pd.read_csv(run / "paired_metric_tests.csv")
    training = pd.read_csv(run / "training_summary_by_variant.csv")
    params = pd.read_csv(run / "parameter_counts.csv")
    runtime = pd.read_csv(run / "inference_runtime_by_variant.csv")
    similarity = pd.read_csv(run / "feature_similarity_by_variant.csv")
    distributions = pd.read_csv(run / "feature_distributions_per_signal.csv")
    gradients = pd.concat([
        pd.read_csv(run / "gradient_path_audit.csv"),
        pd.read_csv(run / "gradient_path_audit_full.csv"),
    ], ignore_index=True)
    validation = json.loads((run / "validation.json").read_text())

    scope["_order"] = scope.variant.map({v: i for i, v in enumerate(VARIANT_ORDER)})
    scope = scope.sort_values(["scope", "_order"])
    main_scope = scope[["scope", "variant", "aff_f1", "vus_pr", "vus_roc"]].copy()
    main_scope["variant"] = main_scope.variant.map(label)

    b3 = scope[scope.variant == "B3"].set_index("scope")
    delta_rows = []
    for _, row in scope[scope.variant.isin(VARIANT_ORDER[3:])].iterrows():
        base = b3.loc[row.scope]
        delta_rows.append({
            "scope": row.scope, "variant": label(row.variant),
            "ΔAff-F1 vs B3": row.aff_f1 - base.aff_f1,
            "ΔVUS-PR vs B3": row.vus_pr - base.vus_pr,
            "ΔVUS-ROC vs B3": row.vus_roc - base.vus_roc,
        })
    deltas = pd.DataFrame(delta_rows)

    def experiment_table(names: list[str]) -> pd.DataFrame:
        d = scope[scope.variant.isin(names)][["scope", "variant", "aff_f1", "vus_pr", "vus_roc"]].copy()
        d["variant"] = d.variant.map(label)
        order = {v: i for i, v in enumerate(names)}
        d["_v"] = scope.loc[d.index, "variant"].map(order)
        return d.sort_values(["scope", "_v"]).drop(columns="_v")

    exp1 = experiment_table(["T1_random", "T2_numeric_ssl", "Full"])
    exp2 = experiment_table(["Full", "L1_no_patch", "L2_no_temporal_relation", "L3_no_transition_geometry", "L4_no_reference", "L5_no_alignment"])
    exp3 = experiment_table(["C1_full", "C2_no_cross", "C3_late_joint", "C4_numeric_only"])

    param_summary = params.groupby("variant", as_index=False).agg(
        trainable_parameters=("all", "first"),
        B3_inference_parameters=("inference_numeric", "first"),
        cross_attention_parameters=("cross_attention", "first"),
        fusion_parameters=("fusion", "first"),
    )
    cost = training.merge(param_summary, on="variant", how="left").merge(runtime, on="variant", how="left")
    cost = cost[cost.variant.isin(VARIANT_ORDER[3:])].copy()
    cost["variant"] = cost.variant.map(label)
    cost = cost[["variant", "total_training_seconds", "optimizer_updates", "trainable_parameters",
                 "B3_inference_parameters", "mean_numeric_encoding_seconds"]]

    robust = distributions.groupby("variant", as_index=False).agg(
        separation_median=("anomaly_minus_high_volatility_normal_in_normal_iqr", "median"),
        separation_q25=("anomaly_minus_high_volatility_normal_in_normal_iqr", lambda x: x.quantile(.25)),
        separation_q75=("anomaly_minus_high_volatility_normal_in_normal_iqr", lambda x: x.quantile(.75)),
        normal_volatility_spearman_median=("normal_volatility_spearman", "median"),
    )
    diag = similarity.merge(robust, on="variant", how="left")
    diag = diag[diag.variant.isin(VARIANT_ORDER[3:])].copy()
    diag["variant"] = diag.variant.map(label)
    diag = diag[["variant", "numeric_cka_with_full", "numeric_visual_cosine", "numeric_fused_cosine",
                 "separation_median", "separation_q25", "separation_q75", "normal_volatility_spearman_median"]]

    grad = gradients.groupby(["variant", "loss_group"], as_index=False)[
        ["numeric_transformer", "cross_attention", "fusion", "reconstruction_head"]
    ].mean()
    grad["variant"] = grad.variant.map(label)
    grad = grad.rename(columns={
        "loss_group": "loss", "numeric_transformer": "‖g_numeric‖",
        "cross_attention": "‖g_cross‖", "fusion": "‖g_fusion‖",
        "reconstruction_head": "‖g_reconstruction‖",
    })

    # Descriptive paired signal tests for the most relevant comparisons.
    paired_focus = paired[(paired.baseline == "B3") & paired.variant.isin(
        ["T1_random", "T2_numeric_ssl", "C2_no_cross", "C3_late_joint", "L1_no_patch",
         "L2_no_temporal_relation", "L3_no_transition_geometry", "L4_no_reference", "L5_no_alignment"]
    )][["variant", "metric", "signals", "mean_difference", "positive", "negative", "zero", "wilcoxon_pvalue_descriptive"]].copy()
    paired_focus["variant"] = paired_focus.variant.map(label)

    pivots = []
    for metric, title in [("aff_f1", "Aff-F1"), ("vus_pr", "VUS-PR"), ("vus_roc", "VUS-ROC")]:
        p = dataset.pivot(index="dataset", columns="variant", values=metric)
        cols = [v for v in VARIANT_ORDER if v in p.columns]
        p = p[cols].reset_index().rename(columns={v: label(v) for v in cols})
        pivots.append((title, p))

    all11 = scope[scope.scope == "all_11"].set_index("variant")
    full = all11.loc["Full"]
    t1 = all11.loc["T1_random"]
    t2 = all11.loc["T2_numeric_ssl"]
    c2 = all11.loc["C2_no_cross"]
    c3 = all11.loc["C3_late_joint"]

    lines = [
        "# AMR 训练目标与结构消融：OrionBench 最终报告",
        "",
        "## 协议与完整性",
        "",
        f"- 实验预注册提交 `{args.protocol_commit}`；执行代码提交 `{args.code_commit}`。所有新结果写入独立目录，已有 Checkpoint、历史结果和进程均未覆盖。",
        "- 覆盖 11 个数据集、492 条信号；MSL/SMAP 使用官方 Train/Test，Yahoo/NAB 使用完整序列无标签转导式协议。窗口 224、步长 56、16 Epoch。",
        "- 全部评分固定为 B3：T3 视觉分数 0.90、原始数值残差 0.05、学习数值 Token 距离 0.05。视觉参考和 constrained-DTW 路径逐信号完全复用 T3。",
        "- 训练与校准不使用异常标签。Aff-F1 沿用 `alpha=0.01` 和完整漏检记零修正；VUS-PR/VUS-ROC 使用连续分数。4 条全正常信号在各方法均未定义，宏平均使用共同的 488 条有效信号。",
        f"- 独立验证状态 **{validation['status']}**：{validation['checks']['checkpoints']} 个 Checkpoint、{validation['checks']['point_scores']} 个点分数文件、{validation['checks']['metric_rows']} 行指标；Full 对既有 B3 的最大误差 `{validation['checks']['full_b3_max_abs_error']:.1e}`。",
        "- 本报告只包含用户指定的三个实验，不包含多随机种子实验。",
        "",
        "## 模型与训练目标定义",
        "",
        "- **T1**：随机初始化数值 Transformer，不训练。**T2/C4**：只训练数值 Transformer，目标为 25% masked numeric patch reconstruction；线性重建头只在训练时使用。**T3/Full/C1**：复用现有完整 AMR Checkpoint。",
        "- **C2**：删除双向 Cross-Attention 和融合头；Patch loss 退化为数值 Token 与固定视觉 Patch 的直接一致性，其余数值几何损失保留。**C3**：以逐 Token Late-joint MLP 代替 Cross-Attention。固定 B3 最终只读取数值 Transformer Token。",
        "- **L1–L4** 分别将对应损失权重设为零，不添加替代监督。**L5** 将全部对齐损失设为零。代码与梯度审计确认 Full 没有隐藏的数值自监督目标，因此 L5 的优化步数为 0，数值权重与 T1 完全相同。",
        "- Patch Consistency 组包含直接数值/视觉局部一致性，并在存在融合模块时包含 fused-token alignment；这是唯一对 Cross-Attention 和融合头产生梯度的损失组。Temporal Relation、Transition Geometry、Reference Consistency 只更新数值 Transformer。",
        "",
        "## 主结果",
        "",
        "指标顺序为 Aff-F1 / VUS-PR / VUS-ROC。",
        "",
        markdown(main_scope),
        "",
        "### 相对既有 B3 的变化",
        "",
        markdown(deltas),
        "",
        "## 实验一：数值 Transformer 有效性",
        "",
        markdown(exp1),
        "",
        f"Full 相对随机 T1 的 All-11 变化为 `{full.aff_f1-t1.aff_f1:+.6f}/{full.vus_pr-t1.vus_pr:+.6f}/{full.vus_roc-t1.vus_roc:+.6f}`。这说明完整 AMR 训练带来可测的总体收益，主要体现在 Aff-F1 和 Yahoo；但它不是所有子域、所有指标都单调占优，T1 在 Real-6 的 VUS 排名反而较高。",
        f"T2 相对 T1 的 All-11 变化为 `{t2.aff_f1-t1.aff_f1:+.6f}/{t2.vus_pr-t1.vus_pr:+.6f}/{t2.vus_roc-t1.vus_roc:+.6f}`。masked reconstruction 没有复现 Full 的收益，说明“训练过”本身不足以解释 B3，目标提供的表示约束才是关键。",
        "",
        "## 实验二：跨模态对齐损失消融",
        "",
        markdown(exp2),
        "",
        "没有任何单个损失满足严格的‘移除后九个 scope×metric 全部下降’，因此不能宣称某项损失不可替代。Patch 组的机制作用最清晰：L1 的数值/视觉余弦从 Full 的 0.855 降到 −0.023，且 All-11 Aff-F1、VUS-PR下降；它同时提高 VUS-ROC，说明其价值是平衡阈值检测与跨模态局部对齐，并非所有排名指标的必要条件。",
        "Temporal Relation、Transition Geometry、Reference Consistency 的单项移除仍保持与 Full 很接近的表示。L3 在 All-11 Aff-F1/VUS-ROC 上略高，L4 在 All-11 Aff-F1/VUS-PR 上略高；这些结果不支持它们在固定 B3 下不可替代。L5 与 T1 的权重和分数逐点完全一致，确认移除全部对齐损失后不存在有效训练路径。",
        "",
        "## 实验三：Cross-Attention 必要性",
        "",
        markdown(exp3),
        "",
        f"C2 相对 Full 的 All-11 变化为 `{c2.aff_f1-full.aff_f1:+.6f}/{c2.vus_pr-full.vus_pr:+.6f}/{c2.vus_roc-full.vus_roc:+.6f}`；C3 为 `{c3.aff_f1-full.aff_f1:+.6f}/{c3.vus_pr-full.vus_pr:+.6f}/{c3.vus_roc-full.vus_roc:+.6f}`。两种简化模型都保留了几乎相同的数值表示与最终性能，未观察到 Cross-Attention 在训练阶段的独立普遍收益。",
        "C2 的训练参数从 1,725,985 降至 1,059,936（减少 38.6%），C3 降至 1,282,080（减少 25.7%）。由于固定 B3 推理只编码数值 Token，所有训练结构的评分阶段均使用同一 1,059,936 参数数值编码器，Cross-Attention 不进入最终评分。",
        "",
        "## 梯度路径审计",
        "",
        "下表为各数据集审计 batch 的平均梯度范数。零值表示该损失对该模块没有实际反向路径。",
        "",
        markdown(grad),
        "",
        "审计结论：Patch 组确实更新数值 Transformer、Cross-Attention 和融合头；其余三个对齐损失只更新数值 Transformer；T2/C4 的重建目标同时更新数值 Transformer 与临时重建头；T1/L5 全部梯度严格为零。",
        "",
        "## 参数量、训练时间与推理成本",
        "",
        markdown(cost, digits=4),
        "",
        "Full/C1 的训练时间来自已存在 Checkpoint 的原始训练记录，本轮未重复训练。表中推理时间只统计每条信号的数值 Token 编码；固定 T3 检索、DTW、原始残差和指标计算不计入该列。GPU 并行进程时间不等于串行墙钟时间。",
        "",
        "## 表示相似度与剧烈正常波动分离",
        "",
        "CKA 以 Full 数值 Token 为基准。分离度为异常中位数减去最高波动 10% 正常点中位数，并以正常分数 IQR 标准化；报告 492 条信号的中位数和四分位区间，避免少量近零 IQR 信号支配均值。",
        "",
        markdown(diag),
        "",
        "C2/C3 与 Full 的 CKA 均约 0.988，且数值/视觉余弦约 0.855，说明删除或替换 Cross-Attention 后数值表征基本保持。T1/L5 与 Full 的 CKA 仅 0.229，T2/C4 为 0.300；其波动分离中位数也从 Full 的 1.827 降至 1.053 和 0.736。L1 的 CKA 尚有 0.819，但跨模态余弦接近零，直接验证 Patch 组承担跨模态局部对齐。",
        "",
        "## 逐信号配对差异",
        "",
        "Wilcoxon p 值仅作描述性证据，未用于模型选择，也未做多重比较校正。",
        "",
        markdown(paired_focus),
        "",
        "## 完整逐数据集结果",
        "",
    ]
    for title, pivot in pivots:
        lines += [f"### {title}", "", markdown(pivot), ""]

    lines += [
        "## 最终回答",
        "",
        "1. **数值 Transformer 的收益部分来自训练，而且来自训练目标。** Full 在 All-11 三项指标均优于随机 T1；但随机表示已能提供一部分低权重随机投影距离，且在 Real-6 的 VUS 上有偶然优势。T2 的纯数值重建显著弱于 Full 和 T1，排除了‘任何无标签训练都会产生收益’。",
        "2. **没有一个对齐损失在固定 B3 下被证明严格不可替代。** Patch Consistency 是唯一具有明确独立机制证据的损失：它承担 Cross-Attention/融合梯度和局部跨模态对齐，并改善综合 Aff-F1/VUS-PR。其余三项可单独移除而保持或改善部分指标。",
        "3. **Cross-Attention 没有显示训练阶段的独立普遍价值。** C2 和 C3 在大幅减少训练参数后仍与 Full 基本持平；固定 B3 本身不使用 Cross-Attention 输出。",
        "4. **可以得到更简单且实用上不低于 B3 的模型，但没有简化模型在全部九个 scope×metric 上严格 Pareto 不劣。** C2 是最清晰的简化方案：减少 38.6% 训练参数，All-11 Aff-F1 略高，两个 VUS 指标仅下降 0.0012/0.0005；C3 的 Aff-F1 更高但 VUS 略低。若‘不低于’要求每项数值均不下降，则本实验没有满足该严格条件的简化模型。",
        "",
        "## 产物",
        "",
        "- `checkpoints/`：99 个新训练或受控随机 Checkpoint；Full 的 11 个 immutable Checkpoint 通过路径引用复用。",
        "- `point_scores/`、`metrics_per_signal.csv`、`metrics_by_dataset.csv`、`metrics_by_scope.csv`。",
        "- `gradient_path_audit.csv`、`gradient_path_audit_full.csv`、`parameter_counts.csv`、`training_summary*.csv`、`inference_runtime*.csv`。",
        "- `feature_similarity*.csv`、`feature_distributions_per_signal.csv`、`paired_metric_tests.csv`、`validation.json`。",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
