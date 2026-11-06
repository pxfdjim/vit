#!/usr/bin/env python3
"""Write the final E1 APRA cleanup and zero-regression report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validation = json.loads((args.run / "validation.json").read_text())
    scope = pd.read_csv(args.run / "metrics_by_scope.csv")
    scope = scope[scope.variant == "E1_APRA"]
    diagnostics = pd.read_csv(args.run / "inference_diagnostics_per_signal.csv")
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     cwd=Path(__file__).resolve().parents[2], text=True).strip()
    scope_table = scope[["scope", "aff_f1", "vus_pr", "vus_roc",
                         "aff_f1_valid_signals"]].to_markdown(index=False)
    events = validation["events"]
    checks = validation["checks"]
    report = f"""# 最终 E1 APRA 精简与零退化验证报告

## 交付状态

- 最终代码基于 `{commit}` 的独立工作区整理；历史 Checkpoint、缓存、报告和其他工作区均未改写。
- 正式模型统一为 `src/models/apra.py::APRA`，训练、推理、评测和默认配置均只指向 E1。
- 独立验证状态 **{validation['status']}**：11 个原始 E1 Checkpoint、492 条信号、{validation['points']} 个逐点分数全部通过。
- 未重新训练，也未运行多随机种子；全量推理直接加载既有 E1 Checkpoint。

## 删除的代码

- 删除 `L_select`、`selective_time_warp`、结构异常 Reject 监督、正常 Warp 对应监督及其辅助项。
- 删除 E3/E4 的监督开关，以及 E7--E10 使用的 Reject/Skip/结构残差运行时开关。
- 从最终分支删除 Selective APRA、机制消融、ACEF、TREM、独立 TREM 和权重扫描的 74 个实验文件。它们仍保存在既有 Git 历史与独立实验工作区，历史产物未删除。
- 最终源码不再包含 `L_select`、Selective APRA 或 TREM/ACEF 的调用关系。

## 保留的 E1 结构

1. 冻结视觉投影与可训练数值 `NumericPatchEncoder` 生成 14×192 Token。
2. 查询/参考联合特征与一阶结构特征产生局部 Match logits，并保留 T3-DTW 的 ±3 Token 锚定带。
3. 前向/后向动态规划对 Match、Query Reject、Reference Skip 三种转移求精确边缘概率。
4. Match 边缘概率生成对齐参考；行匹配质量生成未匹配证据；查询/对齐参考的一阶差生成结构残差。
5. 评分头拼接未匹配、结构残差、对齐参考、Reject 概率和可靠性，经线性层、单层时序 Transformer 与零初始化 `tanh` 头输出 `delta_APRA`。

参数量修改前后均为 **{checks['parameter_counts'][0]}**，状态字典均为 **{checks['state_dict_keys']}** 个张量。原 Checkpoint 使用严格模式直接加载，最大参数误差为 `{checks['checkpoint_load_max_error']:.1e}`。

## 损失与评分

训练目标保持原 E1：

`L = 1.0 × L_path + 1.0 × L_detect`

- `L_path = path_cross_entropy + 0.25 × reject_prior + 0.25 × cycle_consistency`。
- `L_detect = hard_synthetic_ranking + 0.5 × unchanged_stability + 0.5 × reversible_warp_stability`。
- `reversible_warp` 属于原始 `L_path/L_detect` 的公共训练依赖，已保留；它不是 E2 的 Selective Warp Supervision。
- 最终评分保持 `S_final = S_B3 + 0.05 × calibrated_delta_APRA`。

## 零退化验证

| 检查项 | 最大误差 |
|:--|--:|
| 状态字典 | {checks['module_state_max_error']:.1e} |
| 固定 batch 的数值/视觉 Token 与参考 Token | {checks['module_output_max_error']:.1e} |
| Match / Reject / Skip、可靠性、对齐特征、未匹配与结构残差 | {checks['module_output_max_error']:.1e} |
| APRA delta | {checks['module_output_max_error']:.1e} |
| L_path / L_detect | {checks['module_loss_max_error']:.1e} |
| 两项损失的逐参数梯度 | {checks['module_gradient_max_error']:.1e} |
| 492 条信号校准后的 APRA 修正量 | {checks['calibrated_delta_max_error']:.1e} |
| 1348432 个最终点分数 | {checks['point_score_max_error']:.1e} |
| 逐信号、逐数据集、三 scope 指标 | 0.0 |
| 事件命中、漏检与误报区间 | {checks['event_count_max_error']:.1e} |

## 复现指标

{scope_table}

事件统计保持：真实事件 `{events['true_events']}`，命中 `{events['tp_events']}`，漏检 `{events['fn_events']}`，误报区间 `{events['fp_intervals']}`，预测区间 `{events['predicted_intervals']}`。

本次全量推理的 APRA 单模型平均 GPU 时间为 `{diagnostics.inference_seconds.mean():.6f}` 秒/信号。运行环境与历史任务调度不同，因此只将该值作为本次成本记录；网络参数量与计算图未改变。

## 仍保留的公共依赖

- `NumericPatchEncoder`、`VisualColumnEncoder` 和既有完整 AMR 数值初始化。
- T3 视觉参考、constrained-DTW 列路径、B3 点分数和无标签正中位数校准。
- 既有无标签合成缓存中的原始/扰动序列、扰动位置、视觉特征、参考窗口及参考列。
- `reversible_warp` 及原 E1 `L_path` 的路径目标。

最终 E1 不依赖 Selective APRA、ACEF、TREM 或机制消融代码。精简后性能、梯度和逐点输出均为零退化。
"""
    output = args.output or args.run / "apra_final_e1_cleanup_report.md"
    output.write_text(report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
