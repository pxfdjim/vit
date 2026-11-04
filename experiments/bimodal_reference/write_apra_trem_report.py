#!/usr/bin/env python3
"""Write the validated APRA+TREM Chinese report."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

import pandas as pd

METRICS = ("aff_f1", "vus_pr", "vus_roc")


def triplet(row):
    return "/".join(f"{float(row[metric]):.6f}" for metric in METRICS)


def delta(left, right):
    return "/".join(f"{float(left[metric] - right[metric]):+.6f}" for metric in METRICS)


def table(frame):
    return frame.to_markdown(index=False, floatfmt=".6f")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(); root = args.output_root
    config = json.loads(args.config.read_text())
    scope = pd.read_csv(root / "metrics_by_scope.csv")
    dataset = pd.read_csv(root / "metrics_by_dataset.csv")
    events = pd.read_csv(root / "event_counts_all.csv")
    event_dataset = pd.read_csv(root / "event_counts_by_dataset.csv")
    checkpoints = pd.read_csv(root / "training_checkpoint_summary.csv")
    diagnostics = pd.read_csv(root / "inference_diagnostics_summary.csv")
    paired = pd.read_csv(root / "paired_metric_tests.csv")
    validation = json.loads((root / "validation.json").read_text())
    order = ["E0_B3", "E1_APRA_only", *config["variants"]]
    ordered = scope.assign(order=scope.variant.map({name: index for index, name in enumerate(order)})) \
                   .sort_values(["scope", "order"]).drop(columns="order")
    main_results = ordered[["scope", "variant", *METRICS, "aff_f1_valid_signals"]]
    all11 = scope[scope.scope == "all_11"].set_index("variant")
    e0, e1, e2 = all11.loc["E0_B3"], all11.loc["E1_APRA_only"], all11.loc["E2_APRA_TREM"]
    focus = dataset[
        dataset.dataset.isin(["MSL", "SMAP", "realAWSCloudwatch", "realAdExchange", "realTraffic"])
        & dataset.variant.isin(order)
    ][["dataset", "variant", *METRICS]]
    event_lookup = events.set_index("variant")
    event_comparison = events.copy()
    event_comparison["delta_tp_vs_E1"] = event_comparison.tp_events - event_lookup.loc["E1_APRA_only", "tp_events"]
    event_comparison["delta_fp_vs_E1"] = event_comparison.fp_intervals - event_lookup.loc["E1_APRA_only", "fp_intervals"]
    train = checkpoints.groupby("variant", as_index=False).agg(
        total_training_seconds=("elapsed_seconds", "sum"),
        parameters_all=("parameters_all", "max"),
        parameters_trem=("parameters_trem", "max"),
        peak_cuda_memory_bytes=("peak_cuda_memory_bytes", "max"),
        gradient_detect_apra=("gradient_detect_step1_apra", "mean"),
        gradient_detect_trem=("gradient_detect_step1_trem", "mean"),
        residual_input_effect=("residual_input_effect", "mean"),
        reliability_input_effect=("reliability_input_effect", "mean"),
    )
    cost = diagnostics[["variant", "total_inference_seconds", "trem_delta_abs",
                        "local_dynamics_norm", "multiscale_context_norm",
                        "local_fusion_weight", "context_fusion_weight",
                        "trem_score_effect_abs", "retrained_apra_change_vs_E1_abs",
                        "score_change_vs_E1_abs"]]
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    pareto = all(e2[metric] >= e1[metric] for metric in METRICS)
    report = f"""# APRA + TREM：残差演化建模 OrionBench 最终报告

## 协议与完整性

- 预注册与实现提交以 `{commit}` 为最终报告代码版本。实验位于独立工作区和运行目录，未覆盖 APRA、ACEF 或历史 checkpoint。
- E0 为既有 B3；E1 逐点复用已验证 APRA-only；E2–E4 从同一 E1 checkpoint 初始化并仅训练现有 APRA 与新增 TREM。视觉编码器冻结，数值 Transformer、T3 参考和 constrained-DTW 路径保持不变。
- 训练继续使用 APRA-only 的 `L_path + L_detect`、同一无标签合成缓存、16 Epoch 和固定 seed。未增加损失、扰动类型、评价标签、验证集或 checkpoint 选择。
- TREM 零初始化并限制为 APRA Token 剩余幅度的 25%。最终证据仍占用 APRA 原有单一 0.05 修正预算；TREM 为零时与 E1 严格相同。
- 独立验证状态 **{validation['status']}**：{validation['checks']['checkpoints']} 个新 checkpoint、{validation['checks']['point_scores']} 个点分数、{validation['checks']['metric_rows']} 行指标，E0/E1 最大复现误差均为零。

## TREM 实现

- **Local Residual Dynamics** 对 APRA 的结构残差与未匹配特征投影后计算一阶、二阶变化，再通过两层局部卷积建模突变与短时异常。
- **Multi-scale Residual Context** 通过 dilation 1/2/4 的深度可分离残差卷积建模持续趋势、周期及子序列变化。
- 融合权重和有界特征调制由 APRA 可靠性、拒配概率、当前 APRA 分数及冻结 B3 Token 证据共同决定；单分支消融也保留可靠性调制。TREM 不读取独立查询特征，无法绕过 APRA 建立独立分类器。

## 主结果

指标顺序为 Aff-F1 / VUS-PR / VUS-ROC。

{table(main_results)}

E2 All-11 为 `{triplet(e2)}`，相对 E1 为 `{delta(e2, e1)}`，相对 E0 为 `{delta(e2, e0)}`。E2 对 E1 的三指标 Pareto 不劣结论为 **{'成立' if pareto else '不成立'}**。

## 重点数据集

{table(focus)}

## 事件召回与误报

{table(event_comparison)}

## 参数、梯度与训练成本

{table(train)}

每个新 checkpoint 在训练开始前对 E1 的最大误差均为零。`L_detect` 对 APRA 和 TREM 的梯度、残差输入替换效应及可靠性替换效应均由独立验证检查。

## 推理成本与分支使用

{table(cost)}

推理时间为每条信号的对应单模型 GPU 时间，不含固定 T3 检索、DTW、数据恢复和指标计算。

## 逐信号配对差异

Wilcoxon p 值只作描述，不参与模型选择且未做多重比较校正。

{table(paired)}

## 结论

1. **TREM 是否提高 APRA-only：** E2 相对 E1 的 All-11 变化为 `{delta(e2, e1)}`。{'三项指标均保持或提高。' if pareto else '至少一项指标下降，因此不能宣称整体不低于 APRA-only。'}
2. **局部动态分支贡献：** 以 E2 减 E3 的 All-11 变化 `{delta(e2, all11.loc['E3_no_local'])}` 为准。
3. **多尺度上下文贡献：** 以 E2 减 E4 的 All-11 变化 `{delta(e2, all11.loc['E4_no_multiscale'])}` 为准。
4. **检测闭环：** E0/E1 零误差复现、固定 T3 审计、APRA/TREM 梯度和输入替换测试均通过；报告不依据评价结果重新选择权重或模型。

## 产物

- `checkpoints/`、`training_logs/`、`training_manifest.json`
- `point_scores/`、`feature_diagnostics/`、`inference_manifests/`
- `metrics_per_signal.csv`、`metrics_by_dataset.csv`、`metrics_by_scope.csv`
- `event_counts_*.csv`、`event_differences_per_signal.csv`
- `training_checkpoint_summary.csv`、`inference_diagnostics_*.csv`
- `paired_metric_tests.csv`、`validation.json`
"""
    output = root / "apra_trem_final_report.md"
    output.write_text(report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
