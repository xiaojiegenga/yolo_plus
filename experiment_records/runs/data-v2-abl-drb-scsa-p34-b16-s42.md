# Run 记录：data-v2-abl-drb-scsa-p34-b16-s42

## 基本信息

| 项目 | 内容 |
|---|---|
| 状态 | 正式训练完成，保留为结构负结果 |
| 实验目的 | 在冻结配方下，评估全 C3k2-DRB 与 P3/P4 独立残差 SCSA 这一完整结构候选 |
| 配置文件 | `experiments/data-v2-abl-drb-scsa-p34-b16-s42.yaml` |
| 代码分支 / commit | `feature/data-v2-abl-drb-scsa` / `3a19da2` |
| 对照 | `data-v2-abl-000-y26m-b16-s42` |
| 云端原始目录 | `/root/yolo_plus/runs/data-v2-abl-drb-scsa-p34-b16-s42/` |
| 回传归档 | `exports/data-v2-abl-drb-scsa-p34-b16-s42.zip` |
| 本地保存目录 | `runs/data-v2-abl-drb-scsa-p34-b16-s42/` |
| 数据 / split | `rice-pest-data-v2` / Val，117 张图、557 个实例 |
| GPU / 环境 | RTX 5090；Python 3.12.3；PyTorch 2.12.1+cu130；Ultralytics 8.4.80 |
| 记录日期 | 2026-09-11 |
| 开始 / 结束时间 | 日志未提供完整墙钟起止时间 |

## 实验参数

| 参数 | 实际值 |
|---|---|
| model | `ultralytics-main/ultralytics/cfg/models/26/yolo26m-drb-scsa-seg.yaml` |
| data | `experiments/yolo_data_v2_cloud.yaml`，data-v2 |
| pretrained | 官方 `yolo26m-seg.pt` |
| epochs / patience | 300 / 100 |
| batch / nbs | 16 / 64 |
| imgsz / workers / device | 640 / 8 / 0 |
| optimizer / lr0 / lrf | AdamW / 0.001667 / 0.01 |
| momentum / weight_decay | 0.9 / 0.0005 |
| warmup_epochs / warmup_bias_lr | 3 / 0.1 |
| seed / deterministic / amp | 42 / true / true |
| mask_ratio / mixup / copy_paste | 2 / 0.0 / 0.3 |
| mosaic / close_mosaic | 1.0 / 15 |
| degrees / translate / scale | 15 / 0.1 / 0.3 |
| flipud / fliplr | 0.5 / 0.5 |
| cache / cos_lr | false / false |
| DRB 位置 | Baseline 第 2、4、6、8、13、16、19、22 层；共 15 个内部 DRB |
| DRB 分支 | 主分支 K=9；辅助 (k,d)=(5,1)、(5,2)、(3,3)、(3,4) |
| SCSA 位置 | Baseline 第 4、6 层之后，新模型第 5、8 层 |
| SCSA 包装 | `Y = X + β × SCSA(X)`；两处 β 独立、初值为 0 |
| 保留结构 | 外层 CSP/C3k 投影、C2PSA、末级 PSA、原始 Segment26/Proto |
| best.pt 选择 | 官方 fitness = Val Box mAP50-95 + Val Mask mAP50-95 |
| 主指标 | Val Mask mAP50-95，同时报告 mAP50 |

实际 `args.yaml` 与正式 000 只有 model、pretrained 的表示方式及 name、save_dir 不同，其余字段一致。

## 训练结果

总体精确指标使用 `results.csv` 的 epoch 231，与 `best.pt` 的 `train_metrics` 一致。
差值为本轮减 Baseline；百分比指标的差值单位为百分点。

| 指标 | 正式 000 | 本轮 | 差值 |
|---|---:|---:|---:|
| 实际 epoch | 300 | 300 | 0 |
| Best epoch | 216 | 231 | +15 |
| Official fitness | 0.81977 | 0.72224 | -0.09753 |
| Mask P | 0.64822 | 0.66955 | +2.133 pp |
| Mask R | 0.67252 | 0.60800 | -6.452 pp |
| Mask mAP50 | 0.71132 | 0.67664 | -3.468 pp |
| Mask mAP50-95 | 0.36348 | 0.31719 | -4.629 pp |
| Box mAP50 | 0.72980 | 0.69323 | -3.657 pp |
| Box mAP50-95 | 0.45629 | 0.40505 | -5.124 pp |
| 总训练时间 | 4416.89 s（1.227 h） | 4725.85 s（1.313 h） | +6.995% |
| 日志 GPU_mem 峰值 | 15.6 GB | 15.7 GB | +0.1 GB |
| 融合后参数量 | 23,509,010 | 20,372,628 | -13.34% |
| 日志融合后 GFLOPs | 121.2 | 109.7 | -11.5 |

本轮到 best epoch 累计 3649.68 s，权重大小 48.610 MB。日志最终验证 inference 为
2.1 ms/image，Baseline 为 1.3 ms/image；二者不是统一条件的独立测速。

### 分类型 Val Mask

以下来自训练结束重新加载、融合 `best.pt` 后的验证日志，保留日志精度。

| 类别 | Instances | 本轮 P | 本轮 R | Baseline mAP50 | 本轮 mAP50 | Baseline mAP50-95 | 本轮 mAP50-95 | ΔmAP50-95/pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rice leaffolder | 462 | 0.689 | 0.571 | 0.677 | 0.660 | 0.266 | 0.245 | -2.1 |
| Rice stemborers | 95 | 0.658 | 0.632 | 0.745 | 0.693 | 0.461 | 0.389 | -7.2 |

最终复核 all 的 Mask P/R/mAP50/mAP50-95 为 0.674/0.601/0.677/0.317。
总体汇总沿用选取最佳权重时的精确值，分类别汇总采用此次最终复核值。

### 收敛与训练状态

完成全部 300 epoch，未触发早停。训练及验证 CSV 数值全程有限，无训练崩溃或显存不足记录。
train/Val 分别为 938/117 张，corrupt 均为 0；两条重复标签由加载器移除，与 Baseline 口径相同。

| epoch 区间 | 平均 train seg_loss | 平均 Val seg_loss | 平均 Mask mAP50-95 |
|---|---:|---:|---:|
| 1–50 | 1.52778 | 1.60831 | 0.15716 |
| 51–100 | 1.29095 | 1.33774 | 0.24532 |
| 101–150 | 1.21268 | 1.30010 | 0.27880 |
| 151–200 | 1.15409 | 1.28721 | 0.29196 |
| 201–250 | 1.10304 | 1.28409 | 0.30404 |
| 251–300 | 1.05326 | 1.29821 | 0.29897 |

Mask mAP50 与 mAP50-95 的全程峰值均在 epoch 187，为 0.68145 / 0.31777。
即使改用 Mask mAP50-95 单独挑选轮次，仍比 Baseline 0.36348 低 4.571 个百分点。
因此这次负结果不能由“官方 fitness 选错最佳轮次”解释。

## 本地分析：失败原因

### 已确认的现象

这次失败指未达到精度提升目标，训练运行本身正常。Box 与 Mask 两套 mAP 都下降，
两个类别都退步，说明问题不限于掩膜边界。Mask P 上升而 R 下降，表明当前评估工作点
偏向更少的检出；AP 曲线整体结果也下降，不只是一个 P/R 工作点的取舍。
钻心虫的严格 Mask AP 降幅更大，但这不等价于已证明“小目标漏检是唯一原因”。

### 首要结构疑点：整个 Bottleneck 被纯深度卷积替换

`C3k2DRB._replace_bottlenecks()` 将每个完整 `Bottleneck` 替换为 `DilatedReparamBlock`。
原 Bottleneck 包含两层 Conv-BN-激活，以及满足条件时的 `x + F(x)` 局部残差。
本轮 DRB 只有多分支 depthwise Conv-BN 求和，每个通道独立处理，不包含通道混合、
激活或自身的 identity 残差。外层 CSP/C3k 投影与旁路仍在，但不等于原内部结构仍在。

这不是单纯扩大卷积核：同时减少了内部通道交互、非线性变换和局部信息直通路径。
而且变化遍及 Backbone 与 Neck 的全部 8 个 C3k2，影响从浅层纹理到多尺度融合。
因此“替换单元过于简化、范围过大”是当前最值得优先验证的解释。

原作者的 UniRepLKNetBlock 把 DRB 用作 depthwise 空间算子，外围另有 SE、含通道投影的
FFN、激活及残差。本项目的整个 Bottleneck 替换方式不等同于该完整块。
参见 [UniRepLKNet 官方实现](https://github.com/AILab-CVC/UniRepLKNet/blob/main/unireplknet.py)。

### 第二个疑点：预训练特征复用减少

日志首次迁移为 724/1210 个状态项。被替换的 Bottleneck 权重与新增注意力参数不能按原结构复用。
后续 1196/1210 是训练器重建同结构模型时的迁移，不代表从官方权重加载了 1196 项。
状态项包含 BN 缓冲区等，不能把该比例当成预训练参数量百分比。

938 张训练图上，同时重建大量内部特征变换，可能增加重新学习有效表征的难度。
这与前述结构变化共同发生，当前无法拆分各自贡献。

### SCSA 确实参与了计算，但贡献正负未确定

直接读取最佳权重得到：新模型第 5 层 β=0.273681640625，第 8 层 β=0.321533203125。
两处都已离开零初始化，排除了“注意力始终关闭”的解释。

当前实现的 SCSA 输出为 `X × 空间门控 × 通道门控`，各门控位于 0～1。
因此在这份最佳权重的正 β 下，整体输出相当于 `X × (1 + β × 门控)`，
并不是直接把主干 X 乘以一个小于 1 的系数。可以改变不同位置和通道的相对权重，
但不能仅凭“注意力抑制特征”就断言它导致了本次漏检。
SCSA 可能补偿 DRB，也可能引入不利重加权；这一轮整体实验不能区分。

### 后期已进入泛化平台，缺少单纯延长训练的依据

末 50 轮训练 seg_loss 继续下降，但 Val seg_loss 上升，平均 Mask mAP50-95 回落。
这与后期过拟合或泛化受限相符，而非“训练尚未开始收敛”。
继续增加 epoch 不应作为首选处理办法。

## 下一步

保留本轮源码、权重和负结果，不将当前整体结构列为已验证有效的改进，也不据此直接与 D1 组合。

下一版优先讨论 DRB 的封装：保留通道混合、激活和局部残差，再引入 DRB 空间算子。
另一个可选方向是收窄替换范围，先保留浅层与 Neck 的原始 C3k2。原论文也提出根据下游任务
选择核尺寸，并通常把大核放在中高层；这为收窄范围提供设计依据，而不是本数据集的效果保证。
参见 [UniRepLKNet，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ding_UniRepLKNet_A_Universal_Perception_Large-Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.pdf)。

本次仅归档和诊断，下一版结构尚未确定。

## 限制

- 只有 seed=42 的整体结构实验，未独立测量 DRB、SCSA 或二者交互的贡献；上述原因是有源码依据的假设，不是已证实的因果归因。
- 未完成逐实例配对分析或按尺寸分组评估，不能把下降具体归于小目标、密集目标或某类边界错误。
- 分类别日志只有三位小数；训练阶段与最终融合验证是不同次验证，P/R 不强行混用。
- GFLOPs 的统计不完整覆盖注意力中的函数式矩阵运算；训练用时和日志速度不是统一效率基准。参数减少不能据此认定计算效率提升。
- 本轮结论只覆盖冻结配方与当前 Val，不代表其他封装、位置或数据集上的 DRB/SCSA 均无效；未使用 Test。

## 数据与源码来源

- [本轮 results.csv](../../runs/data-v2-abl-drb-scsa-p34-b16-s42/results.csv)：最佳轮次、总体指标、训练曲线与用时。
- [本轮 args.yaml](../../runs/data-v2-abl-drb-scsa-p34-b16-s42/args.yaml)：实际参数。
- [本轮训练日志](../../runs/data-v2-abl-drb-scsa-p34-b16-s42/train_data-v2-abl-drb-scsa-p34-b16-s42.log)：环境、训练完成、显存、分类型验证与预训练迁移。
- [本轮 best.pt](../../runs/data-v2-abl-drb-scsa-p34-b16-s42/weights/best.pt)：train_metrics 与 SCSA β。
- [正式 000 记录](data-v2-abl-000-y26m-b16-s42.md)及其原始 Run：统一对照。
- [DRB/SCSA 源码](../../ultralytics-main/ultralytics/nn/modules/drb_scsa.py)、[原始 Bottleneck/C3k2](../../ultralytics-main/ultralytics/nn/modules/block.py)：结构诊断。
