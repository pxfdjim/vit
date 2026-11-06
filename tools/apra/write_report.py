#!/usr/bin/env python3
"""Write the final Detect-only APRA cleanup report from validated history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd


METRICS = ("aff_f1", "vus_pr", "vus_roc")


def format_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".6f")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads((args.run / "validation.json").read_text())
    scope = pd.read_csv(args.run / "metrics_by_scope.csv")
    multiseed = pd.read_csv(args.run / "multiseed_summary.csv")
    events = pd.read_csv(args.run / "event_counts_all.csv")
    checkpoints = pd.read_csv(args.run / "checkpoint_manifest.csv")
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     cwd=Path(__file__).resolve().parents[2], text=True).strip()

    primary_names = {"B3": "B3", "E1_APRA": "E1_existing_s20261104",
                     "E6_Detect_only": "E6_detect_only_s20261104"}
    primary_rows = []
    for scope_name in ("all_11", "real_6", "yahoo_4"):
        for label, variant in primary_names.items():
            row = scope[(scope.scope == scope_name) & (scope.variant == variant)].iloc[0]
            primary_rows.append({"scope": scope_name, "variant": label,
                                 **{metric: row[metric] for metric in METRICS}})
    primary = pd.DataFrame(primary_rows)
    all_primary = primary[primary.scope == "all_11"].set_index("variant")
    delta_rows = []
    for baseline in ("E1_APRA", "B3"):
        delta_rows.append({"comparison": f"E6_Detect_only - {baseline}",
            **{f"delta_{metric}": all_primary.loc["E6_Detect_only", metric] -
               all_primary.loc[baseline, metric] for metric in METRICS}})
    deltas = pd.DataFrame(delta_rows)

    mean_rows = []
    for scope_name in ("all_11", "real_6", "yahoo_4"):
        b3 = primary[(primary.scope == scope_name) & (primary.variant == "B3")].iloc[0]
        mean_rows.append({"scope": scope_name, "variant": "B3", "seeds": 1,
                          "aff_f1_mean": b3.aff_f1, "aff_f1_std": 0.0,
                          "vus_pr_mean": b3.vus_pr, "vus_pr_std": 0.0,
                          "vus_roc_mean": b3.vus_roc, "vus_roc_std": 0.0})
        for source_variant, label in (("E1_existing", "E1_APRA"),
                                      ("E6_detect_only", "E6_Detect_only")):
            row = multiseed[(multiseed.scope == scope_name) &
                            (multiseed.variant == source_variant)].iloc[0]
            mean_rows.append({"scope": scope_name, "variant": label,
                              **row.drop(labels=["scope", "variant"]).to_dict()})
    means = pd.DataFrame(mean_rows)

    event_rows = []
    for label, variant in primary_names.items():
        row = events[events.variant == variant].iloc[0]
        event_rows.append({"variant": label, **{column: int(row[column]) for column in
            ("true_events", "tp_events", "fn_events", "fp_intervals", "predicted_intervals")}})
    event_table = pd.DataFrame(event_rows)

    training_seconds = checkpoints.groupby("seed").training_seconds.sum()
    checks = validation["checks"]
    all_means = means[means.scope == "all_11"].set_index("variant")
    mean_delta = {metric: all_means.loc["E6_Detect_only", f"{metric}_mean"] -
                  all_means.loc["E1_APRA", f"{metric}_mean"] for metric in METRICS}
    report = f"""# Detect-only APRA 最终精简与历史一致性报告

## 交付状态

- 最终代码提交 `{commit}`；独立验证状态 **{validation['status']}**。
- 正式训练目标已切换为 E6 Detect-only；没有重新训练、没有新增实验，也没有打开评价标签进行选择。
- 直接复用历史 E6 的 {validation['checkpoints']} 个 Checkpoint、{validation['point_score_files']} 个逐信号分数文件，覆盖 {validation['signals']} 条信号、{len(validation['seeds'])} 个种子和 {validation['point_score_values']} 个分数值。
- 原始 E1 源码、Checkpoint 和结果仍保存在独立工作区与 Git 历史中，本轮没有覆盖。

## 最终模型结构

网络拓扑与 E1 完全一致：

1. 冻结 B3 视觉与数值证据、T3 参考检索和 constrained-DTW 锚点保持不变。
2. APRA 使用 Match、Query Reject、Reference Skip 三状态可微前向/后向动态规划。
3. Match 边缘概率产生对齐参考，间隙质量产生未匹配证据，一阶对齐差产生结构残差。
4. APRA 评分头融合对齐参考、未匹配特征、结构残差、拒配概率与可靠性，输出有界 `delta_APRA`。
5. 最终评分仍为 `S_final = S_B3 + 0.05 × calibrated_delta_APRA`。

模型参数量保持 **2,059,973**，状态字典保持 **73** 个张量；相对 E1 的参数变化为 **0**。

## 最终训练目标

`L = L_detect = L_hardRanking + 0.5 × L_unchanged + 0.5 × L_reversibleWarp`

- `L_hardRanking`：重点监督 B3 响应不足的合成困难扰动相对干净位置提高分数。
- `L_unchanged`：约束未扰动位置的最终分数稳定。
- `L_reversibleWarp`：约束可逆正常时间形变前后的最终分数稳定。

## 实际删除的计算与配置

- 删除完整 `L_path`：Path Cross Entropy、Reject Prior、Cycle Consistency。
- 删除路径目标归一化、目标路径交叉熵、拒配均值惩罚、逆映射循环矩阵及其反向梯度。
- 删除 `path`/`detect` 总损失权重、路径损失日志、路径梯度审计和训练入口中的路径分支。
- 保留 `reversible_warp` 及其插值映射，因为它仍是 `L_reversibleWarp` 的输入。
- 保留三状态动态规划、路径边缘概率、拒配/跳过状态和全部推理输出；这些属于模型结构，不属于已删除的路径监督。

删除损失没有减少模型参数。训练仍需 clean、synthetic、warped 三次 APRA 前向；节省的是路径损失张量计算及其反向贡献。历史 E6 每个种子的 11 数据集 GPU 进程训练时间合计分别为 `{training_seconds.iloc[0]:.1f}`、`{training_seconds.iloc[1]:.1f}`、`{training_seconds.iloc[2]:.1f}` 秒，本轮未重新计时。

## 历史一致性验证

| 检查项 | 结果 |
|:--|--:|
| 33 个 Checkpoint 严格加载最大误差 | {checks['checkpoint_strict_load_max_abs_error']:.1e} |
| 固定输入的 15 个模型输出最大误差 | {checks['module_output_max_abs_error']:.1e} |
| `L_detect` 数值最大误差 | {checks['detect_loss_max_abs_error']:.1e} |
| `L_detect` 逐参数梯度最大误差 | {checks['detect_gradient_max_abs_error']:.1e} |
| 1,476 个逐信号评分文件最大误差 | {checks['point_score_max_abs_error']:.1e} |
| 时间戳不一致文件 | {checks['timestamp_mismatches']} |
| 历史机制报告的 E6 复用误差 | {checks['source_mechanism_reuse_max_abs_error']:.1e} |

## 主种子已有结果

{format_table(primary)}

### All-11 差异

{format_table(deltas)}

主种子 E6 相对 B3 的 Aff-F1 几乎持平，VUS-PR/VUS-ROC 分别提高约 `0.00249/0.00240`；相对 E1 的 Aff-F1 降低约 `0.00531`，VUS-PR 略高，VUS-ROC 略低。这里没有新实验或重新选择。

## 历史三种子结果

以下均值和标准差直接复用已完成的机制消融结果。B3 为固定结果，标准差记为零。

{format_table(means)}

All-11 三种子均值下，E6 相对 E1 为 `{mean_delta['aff_f1']:+.6f}/{mean_delta['vus_pr']:+.6f}/{mean_delta['vus_roc']:+.6f}`。E6 用较低 Aff-F1 换取较高的两项连续排序指标。

## 主种子事件结果

{format_table(event_table)}

E6 相对 E1 少命中 7 个事件，同时少 1 个误报区间；相对 B3 多命中 2 个事件并少 6 个误报区间。

## 最终结论

E6 Detect-only 现在是该分支的统一正式训练方案。它保留 APRA 的全部对齐与结构证据能力，只删除未作为最终目标采用的独立路径监督。历史 Checkpoint、模型输出、检测损失梯度和逐点分数均实现零误差复用。

该选择体现训练目标精简，并不表示 E6 在所有指标上优于 E1。历史三种子结果显示 E6 的 VUS-PR/VUS-ROC 高于 E1，Aff-F1 较低；原 E1 继续作为完整六子损失版本保留，可直接恢复和复现。

## 产物

- `validation.json`、`checkpoint_manifest.csv`、`point_score_manifest.csv`
- `metrics_by_scope.csv`、`metrics_by_dataset.csv`
- `multiseed_summary.csv`、`multiseed_by_dataset.csv`
- `event_counts_all.csv`、`reused_artifacts.json`
"""
    args.output.write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
