# Model layout

正式代码暂用中性名称 `Model`。当前架构的数学定义与新旧职责对照见 [ARCHITECTURE_REFACTOR.md](ARCHITECTURE_REFACTOR.md)。

## 代码结构

- `src/models/model.py`：数值与视觉列编码、T3 先验下的三状态动态规划和六项证据读出。
- `src/core/visual_features.py`：定位、读取并校验预生成的视觉特征文件。
- `src/core/reference.py`：冻结 T3 参考检索与 constrained-DTW 的公共实现。
- `src/core/scoring.py`：训练、推理共用的可微逐点恢复和无标签分数校准。
- `src/training/build_frozen_b3.py`：构建训练所需的冻结 B3 反事实分数和 T3 路径。
- `src/training/train.py`：正式训练入口；损失直接作用于校准后的最终逐点分数。
- `src/inference/predict.py`：读取冻结 T3 路径并生成逐点分数；不重复运行 T3 检索和 DTW。
- `src/inference/predict_workers.py`：多 GPU 推理启动器。
- `src/evaluation/evaluate.py`：OrionBench 指标与事件评测。
- `src/evaluation/evaluate_workers.py`：并行评测启动器。
- `configs/model.json`：重构后的架构版本与默认配置。
- `tests/test_model.py`、`tests/test_reference.py`：模型、梯度、T3 参考和分数恢复测试。

`experiments/` 只存放临时消融、诊断和本地探索脚本，并由 Git 忽略。正式训练、推理和评测代码均在 `src/`。

## 视觉特征文件接口

这里的“缓存接口”只负责读取已生成的视觉特征文件。正式推理只读取 Block-6 特征；冻结 B3 分数与 T3 路径沿用先前由完整视觉特征生成的产物。接口检查：

1. 输入序列摘要是否一致；
2. 窗口数与张量形状是否正确；
3. 特征版本、Block 编号和归一化标记是否匹配；
4. 张量是否全部为有限值。

因此正式名称改为 `visual_features.py`。该模块不训练模型、不修改特征，也不写入实验结果。

## 命令

重构后的严格加载器读取 `data/checkpoints/model_reference_posterior_v2/`。新旧对齐器及评分头张量结构不同，旧权重只用于初始化数值编码器；需要先为各数据集训练新版 Checkpoint。所有权重和训练缓存位于 Git 忽略的 `data/` 中。

训练一个数据集（已有冻结 B3 缓存时）：

```bash
python -m src.training.train \
  --config configs/model.json \
  --training-data /path/to/dataset-training-data.pt \
  --output /path/to/checkpoint.pt \
  --device cuda:0
```

`--training-data` 应传入与配置中随机种子一致的无标签合成缓存；`--frozen-b3-root` 和 `--initial-checkpoint` 可以覆盖配置默认路径。默认只迁移旧版正式模型的数值编码器参数，然后初始化新的对齐器和读出头；训练预算为 16 Epoch。现有冻结 B3 缓存内的 T3 窗口 ID 与列中心可直接复用。若需重新构建缓存：

```bash
/home/pengxf/anaconda3/envs/vit/bin/python src/training/build_frozen_b3.py \
  --config configs/model.json \
  --dataset realTraffic \
  --device cuda:0
```

该构建脚本需使用已安装 `open_clip` 的 `vit` 环境并以脚本方式运行，因为它直接导入历史冻结 B3 评分代码。数据路径在 `configs/model.json` 中声明，历史实现保存在 `.worktrees/rcca-cadm-e6aad7d/`。

当前训练公式为 `S_final = S_B3 + 0.05 × calibrated_delta`，`L = L_hardRanking + 0.5 L_unchanged + 0.5 L_reversibleWarp`。Warp 项按已知时间映射比较对应位置。B3 和 T3 在训练中冻结，校准尺度不参与梯度；逐点修正保留梯度。

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

## 结构与无标签检查

- 可训练参数量：`1,059,945`，其中对齐器为 2 个标量、读出头为 7 个参数。
- 三状态 DP 的 Match/Reject 行质量与 Match/Skip 列质量在单元测试中守恒。
- 全部 492 份冻结 T3 路径通过窗口 ID、单调列中心与固定端点格式审计。
- `vit` 环境在宿主执行环境可识别 8 张 CUDA GPU；受限沙箱不映射 GPU 设备。四卡队列已完成 11 个数据集各 16 Epoch 的新版 Checkpoint；严格加载、有限损失、参数更新和无标签元数据均通过独立核查。
- 492 条信号、1,348,432 个点已完成推理；冻结 B3 对历史结果最大误差为零，最终逐点分数对 `B3 + 0.05 × calibrated_delta` 的最大误差也为零。realAdExchange 的 5 条信号训练/推理输入、T3 窗口及列中心、局部修正均零误差一致；SMAP 官方 Test 信号推理通过。
- 训练日志、Checkpoint 诊断和评价结果位于 Git 忽略的 `data/runs/model_refactor_v2_20261116/` 与 `data/runs/model_refactor_v2_20261116_eval/`。评价结果不参与结构选择。

严格加载器检查 `model_config.architecture == 2` 和全部状态张量。旧 Checkpoint 的正式代码保留在 Git 历史中，历史文件没有改写。
