# ViT4TS 第一阶段方法核查

- 论文：[arXiv 2506.06836](https://arxiv.org/pdf/2506.06836)，附录 Tables 4、7、8、13、14。
- 官方代码：[ZLHe0/VLM4TS](https://github.com/ZLHe0/VLM4TS)，校验提交 8ab8c16414eb2c1a861dfc3e76f458180035a879。当前项目的 vit4ts.py、model_utils.py、preprocess.py 与该提交逐字一致；evaluate.py 的 TP 计数有一处实质差异，见下表。
- 本实验固定 OpenAI CLIP ViT-B/16，SHA256 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f；论文主实验说明为 LAION-400M，故论文数值仅作参照，不宣称完全同配置复现。
- 不运行 VLM4TS / Qwen；原始代码与先前结果保留。独立实验代码、缓存、区间与指标均在本目录。

## 论文与公开代码

| 环节 | 论文描述 | 公开代码／实验脚本 |
|---|---|---|
| 滑窗 | 224 点、步长 56、图像 224×224 | run_experiment.py 指定 240 点、步长 60、图像 224×224 |
| 时间映射 | 每张图映射回对应原始时间点并按真实步长叠加 | stitch_anomaly_maps() 用图像宽 224 推出步长 56；align_anomaly_vector() 再把整个结果插值到 240/60 的覆盖长度 |
| 预处理 | 先 Min-Max、后线性去趋势 | preprocess_time_series() 先去趋势、后 Min-Max；draw_image() 用固定 [0,1] y 轴 |
| 图像列聚合 | 叠加二维图后按行取 q=0.25 分位数 | 每张图逐列取最高 25% 像素均值，再叠加一维向量 |
| 多尺度融合 | 文中写 harmonic averaging | vit4ts.py 中三尺度分数是算术平均；本实验不改此算法 |
| 阈值 | α=0.1/0.01/0.001，EWMA、Gaussian μ+zσ | 同三个 α；EWMA span=max(1,int(T*0.01))，再做 Gaussian 阈值 |
| 区间评价 | 论文公式按“有交集的预测区间”计 TP；未覆盖的真区间计 FN | 当前项目 evaluate.py 与论文公式一致；作者仓库对每条预测区间累计其覆盖的真区间个数，可能一条预测计多个 TP。本次已保存的 1350 个评价行中两种 TP 计数完全一致，因此实际 F1 差为 0 |
| F1-max | 论文表 13/14 是数据集内固定 α 平均 F1 后取三者最大 | aggregate_results.py 先在单条序列上取 α 最优，再平均；两个口径均保存，主表以论文口径为准 |

## 控制变量

- official_240：官方实验脚本 240/60、官方预处理、官方映射。
- paper_window_224：相对基线只改窗口至 224/56；每张图仍为 224×224。
- paper_preprocess_240：相对基线只改 Min-Max 与去趋势顺序，绘图仍为代码的固定 [0,1]。
- 每个配置的 corrected_mapping：同一批已缓存的窗口异常向量，先把 224 列插值到对应原始窗口的 224/240 点，再按真实 56/60 步长叠加；末尾沿用官方 extrapolation。
- data_dedup：只对重复时间戳的四条序列分别保留首次值、末次值；其余序列沿用基线结果，不用标签选择策略。
- quantile_diagnostic：额外后处理诊断；在 224/56 条件下使用同一网络生成的 224×224 异常图，先按真实窗口叠加二维图，再按行取 q=0.25 分位数。由于公开代码已把 14×14 patch 图上采样到 224×224，此诊断不能等同于论文未公开的完整实现。

缓存 cache/ 是逐窗口的一维异常向量；映射对照不重复计算 CLIP。所有 scores/ 保存逐时间点分数，results/ 保存每个 α 的预测区间与 TP/FP/FN。comparison.csv 是数据集对照表，per_signal_metrics.csv 是逐序列明细，manifest.json 保存权重与源文件校验值。
