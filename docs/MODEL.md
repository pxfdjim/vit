# Final model layout

正式代码暂用中性名称 `Model`。算法名称确定后，只需调整公开名称，不需要改动模型结构、Checkpoint 张量键或计算流程。

## 代码结构

- `src/models/model.py`：模型、三状态可微单调对齐、证据头和检测损失。
- `src/core/visual_features.py`：定位、读取并校验预生成的视觉特征文件。
- `src/core/reference.py`：T3 参考检索、constrained-DTW、归一化和逐点分数恢复。
- `src/training/train.py`：正式训练入口。
- `src/inference/predict.py`：单进程推理与分数生成。
- `src/inference/predict_workers.py`：多 GPU 推理启动器。
- `src/evaluation/evaluate.py`：OrionBench 指标与事件评测。
- `src/evaluation/evaluate_workers.py`：并行评测启动器。
- `configs/model.json`：已验证的默认配置。
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

训练一个数据集：

```bash
python -m src.training.train \
  --config configs/model.json \
  --training-data /path/to/dataset-training-data.pt \
  --output /path/to/checkpoint.pt \
  --device cuda:0
```

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

## 已验证不变量

- 参数量：`2,059,973`
- Checkpoint 状态张量：`73`
- 历史 Detect-only Checkpoint 严格加载误差：`0.0`
- 模型输出、检测损失和梯度误差：`0.0`
- 4,045,296 个历史逐点分数误差：`0.0`

旧 Checkpoint 中的历史模型标记仍由兼容加载器识别；新 Checkpoint 写入中性标记 `model` 与 `evidence_head`。
