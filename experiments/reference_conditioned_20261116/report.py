"""Combine preregistered variants without using labels for model selection."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(sys.argv[1]).resolve()
BASELINE = ROOT / "data/runs/model_refactor_v2_20261116_eval"
NAMES = {"baseline": "A_formal", "posterior": "B_posterior",
         "no_reference": "C_no_reference", "t3_direct": "D_t3_direct"}
METRICS = ("aff_f1", "vus_pr", "vus_roc")


def plot_mechanism(frame: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    labels = {"posterior": "Posterior", "no_reference": "No reference",
              "t3_direct": "T3 direct"}
    colors = {"posterior": "#18324F", "no_reference": "#187F83",
              "t3_direct": "#C7674B"}
    panel_data = (
        ("confidence_change_injected", "Match confidence: injected − clean"),
        ("residual_norm_change_injected", "Residual norm: injected − clean"),
        ("final_score_injection_gain", "Score gain over frozen B3 at injection"),
    )
    rng = np.random.default_rng(20261104)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), constrained_layout=True)
    for ax, (metric, title) in zip(axes, panel_data):
        for index, mode in enumerate(labels):
            values = frame.loc[frame.variant == mode, metric].dropna().to_numpy(float)
            x = np.full(len(values), index) + rng.uniform(-.16, .16, len(values))
            ax.scatter(x, values, s=7, color=colors[mode], alpha=.22, rasterized=True)
            median, low, high = np.quantile(values, [.5, .25, .75])
            ax.vlines(index, low, high, color=colors[mode], lw=4, zorder=3)
            ax.scatter([index], [median], s=45, marker="D", facecolor="white",
                       edgecolor=colors[mode], linewidth=1.6, zorder=4)
        ax.axhline(0, color="#97A4AE", lw=.8, ls="--", zorder=0)
        ax.set(title=title, xlim=(-.55, 2.55), xticks=range(3),
               xticklabels=list(labels.values()))
        ax.tick_params(axis="x", labelrotation=18)
    axes[0].set_ylabel("Per-signal contrast")
    output = RUN / "figures" / "unlabeled_mechanism_distributions"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def read_table(file: str, variant: str):
    frame = pd.read_csv(BASELINE / file if variant == "baseline" else
                        RUN / "inference" / variant / file)
    frame = frame[frame.variant == "Model"].copy()
    frame.loc[:, "variant"] = NAMES[variant]
    return frame


def main():
    for mode in NAMES:
        if mode == "baseline":
            continue
        checkpoint_root = RUN / "checkpoints" / mode
        checkpoints = list(checkpoint_root.glob("*.pt"))
        if len(checkpoints) != 11:
            raise RuntimeError(f"{mode}: expected 11 checkpoints, found {len(checkpoints)}")
    tables = {}
    for name in ("metrics_by_scope.csv", "metrics_by_dataset.csv",
                 "metrics_per_signal.csv", "event_counts_all.csv",
                 "event_counts_by_dataset.csv", "event_counts_per_signal.csv"):
        tables[name] = pd.concat([read_table(name, mode) for mode in NAMES],
                                 ignore_index=True)
        tables[name].to_csv(RUN / name, index=False)
    scopes = tables["metrics_by_scope.csv"]
    datasets = tables["metrics_by_dataset.csv"]
    events = tables["event_counts_all.csv"]
    resources = []
    baseline_diagnostics = pd.read_csv(BASELINE / "inference_diagnostics_per_signal.csv")
    baseline_checkpoints = [torch.load(path, map_location="cpu", weights_only=True)
                            for path in sorted((ROOT / "data/checkpoints/model_reference_posterior_v2").glob("*.pt"))]
    resources.append({"variant": "A_formal", "parameters": 1059945,
                      "training_seconds_sum": sum(row["elapsed_seconds"] for row in baseline_checkpoints),
                      "inference_seconds_mean_per_signal": baseline_diagnostics.inference_seconds.mean()})
    for mode in NAMES:
        if mode == "baseline":
            continue
        checkpoints = [torch.load(path, map_location="cpu", weights_only=True)
                       for path in sorted((RUN / "checkpoints" / mode).glob("*.pt"))]
        parameters = [row["parameter_counts"]["all"] for row in checkpoints]
        if len(set(parameters)) != 1:
            raise RuntimeError(f"{mode}: parameter counts differ")
        diagnostics = pd.read_csv(RUN / "inference" / mode /
                                  "inference_diagnostics_per_signal.csv")
        resources.append({"variant": NAMES[mode], "parameters": parameters[0],
                          "training_seconds_sum": sum(row["elapsed_seconds"] for row in checkpoints),
                          "inference_seconds_mean_per_signal": diagnostics.inference_seconds.mean()})
    resources = pd.DataFrame(resources)
    resources.to_csv(RUN / "resources.csv", index=False)
    mechanism = pd.read_csv(RUN / "mechanism_diagnostics.csv")
    plot_mechanism(mechanism)
    mechanism_summary = mechanism.groupby("variant", as_index=False).agg(
        signals=("signal", "count"),
        normal_match_confidence=("match_confidence_normal", "mean"),
        clean_match_confidence_at_injection=("match_confidence_clean_at_injection", "mean"),
        injected_match_confidence=("match_confidence_injected", "mean"),
        confidence_change_injected=("confidence_change_injected", "mean"),
        normal_residual_norm=("joint_norm_normal", "mean"),
        clean_residual_norm_at_injection=("joint_norm_clean_at_injection", "mean"),
        injected_residual_norm=("joint_norm_injected", "mean"),
        residual_norm_change_injected=("residual_norm_change_injected", "mean"),
        normal_delta_abs=("delta_normal_abs", "mean"),
        injected_delta_abs=("delta_injected_abs", "mean"),
        raw_delta_change_injected=("raw_delta_change_injected", "mean"),
        numeric_view_removal_effect=("numeric_view_removal_effect", "mean"),
        visual_view_removal_effect=("visual_view_removal_effect", "mean"),
        injected_score_gain=("final_score_injection_gain", "mean"),
        unchanged_score_shift_abs=("final_score_unchanged_shift_abs", "mean"),
        normal_warp_alignment_error=("normal_warp_alignment_error", "mean"),
        normal_warp_delta_error=("normal_warp_delta_error", "mean"))
    mechanism_summary.to_csv(RUN / "mechanism_summary.csv", index=False)
    kind_summary = pd.read_csv(RUN / "mechanism_by_perturbation.csv").groupby(
        ["variant", "kind"], as_index=False).agg(
        windows=("signal", "count"),
        confidence_change=("confidence_change", "mean"),
        residual_norm_change=("residual_norm_change", "mean"),
        raw_delta_change=("delta_change", "mean"))
    kind_summary.to_csv(RUN / "mechanism_by_perturbation_summary.csv", index=False)

    def shown(frame, columns, decimals=6):
        copy = frame[list(columns)].copy()
        for column in copy.select_dtypes(include="number"):
            if pd.api.types.is_integer_dtype(copy[column]):
                copy[column] = copy[column].map(lambda x: f"{int(x):,}" if np.isfinite(x) else "—")
            else:
                copy[column] = copy[column].map(lambda x: f"{x:.{decimals}f}" if np.isfinite(x) else "—")
        return copy.to_markdown(index=False, disable_numparse=True)

    baseline = scopes[scopes.variant == "A_formal"].set_index("scope")
    deltas = scopes.copy()
    for metric in METRICS:
        deltas[f"delta_{metric}_vs_A"] = [row[metric] - baseline.loc[row.scope, metric]
                                             for _, row in deltas.iterrows()]
    deltas = deltas[deltas.variant != "A_formal"]
    all11 = scopes[scopes.scope == "all_11"].set_index("variant")
    gained = [metric for metric in METRICS if all11.loc["B_posterior", metric] >
              all11.loc["A_formal", metric]]
    conditional = []
    for comparator in ("C_no_reference", "D_t3_direct"):
        signs = ["+" if all11.loc["B_posterior", metric] > all11.loc[comparator, metric]
                 else "−" if all11.loc["B_posterior", metric] < all11.loc[comparator, metric]
                 else "=" for metric in METRICS]
        conditional.append(f"B 相对 {comparator} 的 All-11 三指标方向为 " + "/".join(signs))
    b, a = all11.loc["B_posterior"], all11.loc["A_formal"]
    c, d = all11.loc["C_no_reference"], all11.loc["D_t3_direct"]
    event_index = events.set_index("variant")
    b_events, a_events = event_index.loc["B_posterior"], event_index.loc["A_formal"]
    b_mechanism = mechanism_summary.set_index("variant").loc["posterior"]
    def triple(left, right):
        return "/".join(f"{left[m] - right[m]:+.6f}" for m in METRICS)
    sections = ["# 后验条件双视图异常表征实验", "",
                "## 实验协议", "",
                "- A 逐点复用当前正式模型；B/C/D 使用相同冻结视觉特征、T3 参考、三状态对齐、数值初始化、16 Epoch、seed 20261104 和 Detect-only 损失。",
                "- C 屏蔽新增表征网络中的参考 Token 与对齐置信度，参数量保持相同；原有 B3 分支仍含参考机制。D 用 T3 直接对应 Token 替代后验参考聚合，保留后验置信度以单独检验聚合方式。",
                "- B/C/D 的训练与无标签校准均不读取真实异常标签；评价只在固定最终 Epoch Checkpoint 推理完成后进行。评分公式均为 S_B3 + 0.05 × calibrated(delta)。",
                "- 本轮每组只有一个种子；差异是单次实验观察，不能宣称统计稳定性。", "",
                "## 训练与推理一致性", "",
                "在 realTraffic/occupancy_6005 的独立检查中，归一化窗口、视觉 Patch、参考 ID、T3 位置先验、匹配后验、学习残差和原始 delta 的训练/推理最大误差均为 `0.0`；完整 492 信号与评分公式的审计见 `validation.json`。", "",
                "新网络只需要匹配后验与拒配质量，因此实验对齐器省去原六统计量的计算；后验和对齐参数梯度与正式对齐器逐元素一致，真实信号的 2,380 个最终点分数在省略前后逐点零误差。", "",
                "## 后验条件网络", "",
                "`p_i=Σ_j P_ij`，`π_ij=softmax_j[-softplus(c)(j-a_i)^2]`，`R_match=Σ_jP_ijR_j/p_i`，`R_prior=Σ_jπ_ijR_j`。",
                "每视图分别做 192→64 投影，共享两层 64 维关系编码器 `F(Q,R)=MLP[Q,R,Q−R,|Q−R|]`，得到 `Hᵐ=pF(Qᵐ,R_matchᵐ)+uF(Qᵐ,R_priorᵐ)`；联合层输入 `[Hⁿ,Hᵛ,|Hⁿ−Hᵛ|]`，两层 64 维 MLP 和零初始化线性头产生 `tanh` 修正。",
                "训练目标保持 `L_hardRanking + 0.5 L_unchanged + 0.5 L_reversibleWarp`。", "",
                "## 三个 scope 的 Aff-F1 / VUS-PR / VUS-ROC", "",
                shown(scopes, ("scope", "variant", *METRICS)), "",
                "## 相对正式模型 A 的变化", "",
                shown(deltas, ("scope", "variant", *(f"delta_{m}_vs_A" for m in METRICS))), "",
                "## 11 数据集逐项结果", "",
                shown(datasets, ("dataset", "variant", *METRICS)), "",
                "## 事件与成本", "",
                shown(events, ("variant", "true_events", "tp_events", "fn_events",
                               "fp_intervals", "predicted_intervals"), 3), "",
                shown(resources, ("variant", "parameters", "training_seconds_sum",
                                  "inference_seconds_mean_per_signal"), 3), "",
                "训练时间是 11 个数据集 GPU 进程耗时之和；推理时间为每条信号的模型评分段均值，不含固定的视觉缓存读取、T3 参考生成和指标计算。", "",
                "## 无标签机制诊断", "",
                "诊断使用 MSL、SMAP、Traffic、YAHOOA1 的合成缓存，不使用真实异常标签。局部注入位置由合成扰动掩码给出；正常形变使用预先固定的可逆时间映射。",
                "C 在新增表征网络内部将匹配系数固定为 1、拒配系数固定为 0，因此其匹配置信度列按定义为常数，不能解读为 C 学会了完美匹配。",
                "视图移除效应是固定模型上的输入干预敏感性，用于检查数值与视觉分支是否影响输出；它不是重新训练后的性能贡献估计。", "",
                "### 匹配与学习残差", "",
                shown(mechanism_summary, ("variant", "signals", "normal_match_confidence",
                                          "clean_match_confidence_at_injection",
                                          "injected_match_confidence", "confidence_change_injected",
                                          "normal_residual_norm", "clean_residual_norm_at_injection",
                                          "injected_residual_norm", "residual_norm_change_injected")), "",
                "### 评分与双视图敏感性", "",
                shown(mechanism_summary, ("variant", "normal_delta_abs", "injected_delta_abs",
                                          "raw_delta_change_injected", "injected_score_gain",
                                          "unchanged_score_shift_abs", "numeric_view_removal_effect",
                                          "visual_view_removal_effect", "normal_warp_alignment_error",
                                          "normal_warp_delta_error")), "",
                "### 合成扰动类型", "",
                shown(kind_summary, ("variant", "kind", "windows", "confidence_change",
                                     "residual_norm_change", "raw_delta_change")), "",
                "![后验匹配、条件残差与评分响应](figures/posterior_reference_mechanism.svg)", "",
                "上图为 MSL 训练缓存中首个信号的首个合成扰动窗口，按固定顺序取样，未依据真实标签或结果强弱挑选。", "",
                "![无标签机制效应的逐信号分布](figures/unlabeled_mechanism_distributions.svg)", "",
                "## 结果解释", "",
                f"1. **正式模型对照：** B 相对 A 的 All-11 三项变化为 `{triple(b,a)}`，正向指标数为 {len(gained)}/3。Real-6 的 VUS-ROC 与 Yahoo-4 的 Aff-F1、VUS-PR 仍低于 A，不能称为各 scope 全面不劣。",
                f"2. **参考条件：** B 相对等参数 C 的 All-11 三项变化为 `{triple(b,c)}`。同一训练设置下移除参考条件后总体退化，支持新增网络需要参考信息；这是单种子结果。",
                f"3. **后验聚合：** B 相对等参数 D 的 All-11 三项变化为 `{triple(b,d)}`。后验聚合只在 VUS-ROC 上占优，未证明它优于直接 T3 对应的全部检测指标。",
                f"4. **事件权衡：** B 相对 A 命中事件 {int(b_events.tp_events-a_events.tp_events):+d} 个、误报区间 {int(b_events.fp_intervals-a_events.fp_intervals):+d} 个。连续排序的微小改善未转化为事件检测净收益。",
                f"5. **无标签机制：** B 在注入位置的同位配对匹配可信度变化为 `{b_mechanism.confidence_change_injected:+.6f}`、64 维残差范数变化为 `{b_mechanism.residual_norm_change_injected:+.6f}`，说明表示对扰动有响应；但注入位置相对 B3 的平均评分增益 `{b_mechanism.injected_score_gain:+.6f}` 小于未扰动位置的平均绝对偏移 `{b_mechanism.unchanged_score_shift_abs:.6f}`，不能据此宣称选择性异常保留已经解决。正常形变路径误差与这些合成诊断不能替代真实标签上的判断。", "",
                "**模型状态：** B 是有继续验证价值的实验模型；本次单种子结果不足以替换正式模型 A，也不足以把后验聚合单独定为已证实的核心收益。", "",
                "## 解释边界", "",
                "- B 相对 C 检验新增网络对参考 Token 的使用；B 相对 D 检验后验参考聚合。二者仍可能受到训练随机轨迹影响，单种子不能证明普遍提升。",
                "- 正常形变与局部注入的无标签机制诊断见 `mechanism_diagnostics.csv` 和对应图。",
                "- 逐信号指标与事件变化见 `metrics_per_signal.csv`、`event_counts_per_signal.csv`；逐数据集事件统计见 `event_counts_by_dataset.csv`。",
                "- 所有数值文件保存在 Git 忽略的实验目录，不纳入正式模型或仓库提交。", ""]
    (RUN / "reference_conditioned_report.md").write_text("\n".join(sections))
    print(scopes[["scope", "variant", *METRICS]].to_string(index=False))


if __name__ == "__main__":
    main()
