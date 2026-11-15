# Final model layout

正式代码暂用中性名称 `Model`。算法名称确定后，只需调整公开名称，不需要改动模型结构、Checkpoint 张量键或计算流程。

## 代码结构

- `src/models/model.py`：模型、三状态可微单调对齐和证据头。
- `src/core/visual_features.py`：定位、读取并校验预生成的视觉特征文件。
- `src/core/reference.py`：T3 参考检索、constrained-DTW、归一化和逐点分数恢复。
- `src/training/build_frozen_b3.py`：构建训练所需的冻结 B3 反事实分数和 T3 路径。
- `src/training/train.py`：正式训练入口；损失直接作用于校准后的最终逐点分数。
- `src/inference/predict.py`：单进程推理与分数生成。
- `src/inference/predict_workers.py`：多 GPU 推理启动器。
- `src/evaluation/evaluate.py`：OrionBench 指标与事件评测。
- `src/evaluation/evaluate_workers.py`：并行评测启动器。
- `configs/model.json`：当前评分一致性版本的默认配置。
- `tests/test_model.py`、`tests/test_reference.py`：模型、梯度、T3 参考和分数恢复测试。

`experiments/` 只存放临时消融、诊断和本地探索脚本，并由 Git 忽略。正式训练、推理和评测代码均在 `src/`。

## 视觉特征文件接口

这里的“缓存接口”不是服务，也不在运行时生成缓存。它只负责读取已生成的 final-layer 与 Block-6 视觉特征文件，并检查：

1. 输入序列摘要是否一致；
2. 窗口数与张量形状是否正确；
3. 特征版本、Block 编号和归一化标记是否匹配；
4. 张量是否全部为有限值。

因此正式名称改为 `visual_features.py`。该模块不训练模型、不修改特征，也不写入实验结果。

## 命令

当前推理默认读取 `data/checkpoints/model_score_consistent/` 的 11 个权重。原 Detect-only E6 权重保留在 `data/checkpoints/model/`。权重和训练缓存位于 Git 忽略的 `data/` 中。

训练一个数据集（已有冻结 B3 缓存时）：

```bash
python -m src.training.train \
  --config configs/model.json \
  --training-data /path/to/dataset-training-data.pt \
  --output /path/to/checkpoint.pt \
  --device cuda:0
```

`--training-data` 应传入与配置中随机种子一致的无标签合成缓存；`--frozen-b3-root` 和 `--initial-checkpoint` 可以覆盖配置默认路径。训练从原 E6 权重开始，固定 16 Epoch，只更新模型参数。若需重新构建冻结 B3 缓存：

```bash
/home/pengxf/anaconda3/envs/vit/bin/python src/training/build_frozen_b3.py \
  --config configs/model.json \
  --dataset realTraffic \
  --device cuda:0
```

该构建脚本需使用已安装 `open_clip` 的 `vit` 环境并以脚本方式运行，因为它直接导入历史冻结 B3 评分代码。数据路径在 `configs/model.json` 中声明，历史实现保存在 `.worktrees/rcca-cadm-e6aad7d/`。

当前训练公式为 `S_final = S_B3 + 0.05 × calibrated_delta`，`L = L_hardRanking + 0.5 L_unchanged + 0.5 L_reversibleWarp`。B3 和 T3 在训练中冻结，校准尺度不参与梯度；逐点修正保留梯度。

多 GPU 推理：

```bash
python -m src.inference.predict_workers \
  --config configs/model.json \
  --output-root /path/to/run \
  --gpus 0 1 2 3
```

评测：

```bash
python -m src.evaluation.evaluate_workers \
  --config configs/model.json \
  --output-root /path/to/run \
  --workers 8
```

测试：

```bash
python -m pytest tests -q
```

## 结构与复现检查

- 参数量：`2,059,973`
- Checkpoint 状态张量：`73`
- 11 个新权重与实验归档逐字节相同，并全部严格加载。
- 相同 CPU 输入与实验归档的正式推理分数逐点误差为 `0.0`；CPU 与历史 GPU 结果在所检信号上的最大差异为 `1.37e-6`。
- 正式训练入口与实验实现对同一信号的损失、校准尺度和各模块梯度范数完全相同。

复制的实验 Checkpoint 保留原有兼容标记；加载器仍严格检查三状态拓扑和全部状态张量。
