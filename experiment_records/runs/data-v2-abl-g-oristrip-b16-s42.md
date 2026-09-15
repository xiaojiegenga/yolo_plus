# Run 记录：data-v2-abl-g-oristrip-b16-s42

G（Neck P3 定向条带上下文）本次未获得正式分割收益。`best.pt` 对应 epoch 206，Val Mask
mAP50-95=0.36020，比 Baseline 低 0.00328；official fitness 0.81416，比 Baseline 低 0.00561。
差额小于两次训练各自平台区单轮波动中位数，因此本轮判为**与 Baseline 不可区分**，按门控规则
（须严格高于 0.36348）为未通过。分类型看，卷叶螟 Mask mAP50-95 与 Baseline 逐位相同（0.266），
两类 Mask AP50 同幅下降 0.015，第 2 节的"收益集中在卷叶螟"预期再次不成立。P3 条带/朝向这一路
（F、F2、G）至此已有三条独立负结果，后续结构改进不再在该方向加码。

## 基本信息与实际参数

| 项目 | 内容 |
|---|---|
| 状态 | 已完成并核验，300 epoch，未触发早停 |
| 实验目的 | 验证 P3 输出上横、纵、主对角、副对角四向条带上下文对细长小目标实例分割的作用 |
| 源码分支 / 实际训练提交 | `feature/data-v2-abl-g-oristrip` / `73ec6c0faade099dc89586b172e4e4b387baa2c0`；日志首行记录 |
| 配置 | G 分支 `experiments/data-v2-abl-g-oristrip-b16-s42.yaml` |
| 本地原始 Run | `runs/data-v2-abl-g-oristrip-b16-s42/` |
| GPU / 环境 | RTX 5090，Python 3.12.3，PyTorch 2.12.1+cu130，Ultralytics 8.4.80 |
| model / pretrained | `yolo26m-p3-oristrip-seg.yaml` / 官方 `yolo26m-seg.pt` |
| data / split | `yolo_data_v2_cloud.yaml` / Val |
| epochs / patience | 300 / 100 |
| batch / nbs / imgsz | 16 / 64 / 640 |
| optimizer / lr0 / lrf | AdamW / 0.001667 / 0.01 |
| momentum / weight_decay | 0.9 / 0.0005 |
| warmup_epochs / warmup_bias_lr / cos_lr | 3 / 0.1 / false |
| seed / deterministic / AMP | 42 / true / true |
| workers / cache | 8 / false |
| mask_ratio / mixup / copy_paste | 2 / 0 / 0.3 |
| mosaic / close_mosaic | 1.0 / 15 |
| degrees / flipud / fliplr / scale | 15 / 0.5 / 0.5 / 0.3 |

G 与 000 的 `args.yaml` 差异仅为 model、pretrained 的表达方式、name、save_dir。训练配方一致，
从官方预训练重新训练，`resume=false`。相对 Baseline 的唯一结构改动是 Neck 第 16 层
`C3k2 [256, True]` → `C3k2OrientedStrip [256, True]`，层号、层数、`Segment26` 的输入层
`[16, 19, 22]` 与检测 stride=8/16/32 均不变。

## 结构与权重迁移核验

构造摘要与运行单预测一致：`YOLO26m-p3-oristrip-seg summary: 342 layers, 27,088,550 parameters,
27,088,550 gradients, 133.2 GFLOPs`（Baseline 329 层）。该模型 942 个状态张量中，新增的
`oriented_context` 恰好占 38 个。

日志出现两次权重迁移，均已核对：

| 阶段 | 日志行 | 说明 |
|---|---|---|
| 入口 `model.load(yolo26m-seg.pt)` | `Transferred 904/942` | 目标模型 942 键，来源官方权重 904 键，全部匹配；未迁移的 38 个即新模块自身张量 |
| 训练器按 nc=2 重建模型后 | `Transferred 928/942` | 未迁移的 14 个为类别相关张量 `model.23.cv3.*`、`one2one_cv3.*`、`proto.semseg.*`，属 80→2 类头重建，与 Baseline 相同 |

即官方预训练的每一个张量都落到了 G 模型上，未迁移项只有新模块自身与类别头，不存在从随机
初始化重训的情况。

`best.pt` 复核：942 个张量全部有限；`model.16` 为 `C3k2OrientedStrip`，38 个 `oriented_context`
张量齐全；投影权重 81,920 个元素**全部非零**（absmax 0.342773），六条分支权重 absmax 分别为
reduce 0.387207、local 0.498535、horizontal 0.523438、vertical 0.573242、diagonal 0.295166、
antidiagonal 0.250488。两条对角分支都被训练激活，不是死分支。

训练结束时日志对 `best.pt` 做 `fuse=True` 重载并完成整轮 Val，融合摘要为
`156 layers, 23,625,426 parameters, 122.4 GFLOPs`；相对 Baseline 融合模型 149 层 / 23,508,626
参数 / 121.2 GFLOPs，增量恰为 +7 层、+116,800 参数（+0.497%），与设计值一致。对角掩码分支
在融合后正常运行，未出现异常。

## 正式结果

总体使用 CSV 中 official fitness 最大轮次，并与 `best.pt` 的 `train_metrics` 逐项核对一致
（fitness 0.81416、Box mAP50-95 0.45396、Mask mAP50 0.69628、Mask mAP50-95 0.36020）。

| 指标 | Baseline（216） | G（206） | G−Baseline |
|---|---:|---:|---:|
| Official fitness，CSV 两项之和 | 0.81977 | 0.81416 | **-0.00561** |
| Box P | 0.67918 | 0.70083 | +0.02165 |
| Box R | 0.64991 | 0.64537 | -0.00454 |
| Box mAP50 | 0.72980 | 0.72794 | -0.00186 |
| Box mAP50-95 | 0.45629 | 0.45396 | -0.00233 |
| Mask P | 0.64822 | 0.59990 | -0.04832 |
| Mask R | 0.67252 | 0.73019 | +0.05767 |
| Mask mAP50 | 0.71132 | 0.69628 | **-0.01504** |
| Mask mAP50-95 | 0.36348 | 0.36020 | **-0.00328** |

G 的全程最高 Mask mAP50-95 也在 epoch 206，改按掩膜单项选权重不会使结果超过 Baseline。
两项分割主指标均低于 Baseline，检测框 AP50-95 同样下降 0.00233，本轮不构成正向结构。

工作点发生明显位移：Mask P 下降 0.04832 而 Mask R 上升 0.05767，召回增幅是 AP 变化的
十几倍。新分支确实改变了掩膜输出的置信分布，但没有转化为排序质量。

## 分类型 Val Mask

下表来自训练结束后重新加载并融合 `best.pt` 的 Val 日志，保留其三位精度。

| 类别 | Images | Instances | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 117 | 557 | 0.702 | 0.644 | 0.727 | 0.454 | 0.605 | 0.729 | 0.696 | 0.360 |
| Rice leaffolder | 87 | 462 | 0.648 | 0.645 | 0.689 | 0.401 | 0.542 | 0.721 | 0.662 | 0.266 |
| Rice stemborers | 38 | 95 | 0.756 | 0.642 | 0.766 | 0.506 | 0.669 | 0.737 | 0.730 | 0.453 |

与 Baseline 的 `best.pt` 复核值逐项对比：

| 类别 | 指标 | Baseline | G | G−Baseline |
|---|---|---:|---:|---:|
| Rice leaffolder | Mask P | 0.630 | 0.542 | -0.088 |
| Rice leaffolder | Mask R | 0.656 | 0.721 | +0.065 |
| Rice leaffolder | Mask mAP50 | 0.677 | 0.662 | -0.015 |
| Rice leaffolder | Mask mAP50-95 | 0.266 | 0.266 | 0.000 |
| Rice leaffolder | Box mAP50-95 | 0.404 | 0.401 | -0.003 |
| Rice stemborers | Mask P | 0.671 | 0.669 | -0.002 |
| Rice stemborers | Mask R | 0.687 | 0.737 | +0.050 |
| Rice stemborers | Mask mAP50 | 0.745 | 0.730 | -0.015 |
| Rice stemborers | Mask mAP50-95 | 0.461 | 0.453 | -0.008 |
| Rice stemborers | Box mAP50-95 | 0.508 | 0.506 | -0.002 |

## 收敛与噪声尺度

| 统计量 | Baseline | G |
|---|---:|---:|
| 前 5 个最高 official fitness 均值 | 0.80882 | 0.80896 |
| 前 10 个最高 official fitness 均值 | 0.80463 | **0.80700** |
| 前 10 个最高 Mask mAP50-95 均值 | 0.35328 | **0.35377** |
| 全程最高 Mask mAP50-95 | 0.36348（216） | 0.36020（206） |
| epoch 150–300 单轮 Mask mAP50-95 变化绝对值中位数 | 0.00509 | 0.00473 |
| epoch 150–300 单轮 official fitness 变化绝对值中位数 | 0.00946 | 0.00996 |
| 最后一轮 Mask mAP50-95 | 0.34131 | 0.34143 |

G 的 top-5、top-10 fitness 均值和 top-10 Mask 均值都略高于 Baseline，差距最大的单项是两次
训练各自峰值的 0.00328，而该轮次所在的平台区单轮波动中位数就有 0.00473～0.00509。峰值
来自 100 多个 epoch 的高位波动，不由任何稳定的优势支撑。按运行单第 8 节的门控规则，本轮
既未达到"严格高于 0.36348"，差额又落在单轮波动之内，判定为与 Baseline 不可区分。

两边都在 epoch 300 结束后回落（G 0.34143、Baseline 0.34131），比较必须使用 `best.pt`。
G 的峰值也在 206，而 Baseline 在 216，两者相差 10 个 epoch，进一步说明这点差异不具有区分度。

## 机制判定

运行单第 2 节的预期是：收益应集中在短边中位数 7.17 px 的卷叶螟，短边 22.27 px 的钻心虫
基本不动；若两类同幅变化，则条带几何这套解释不成立。

实测结果为两类 Mask AP50 各自下降 0.015，完全同幅；卷叶螟 Mask mAP50-95 与 Baseline 逐位
相同（0.266），钻心虫下降 0.008。**预期被否**。四向条带与对角分支没有带来任何类别选择性
的收益，与 F、E、DRB/SCSA 的表现一致。

把同类结构的卷叶螟 Mask mAP50-95 放在一起看，八次训练落在 0.245～0.266 的窄带内，Baseline
的 0.266 本身就在带顶：

| 结构 | 卷叶螟 Mask mAP50-95 |
|---|---:|
| Baseline（000） | 0.266 |
| A1 SR-CBAM | 0.262 |
| A2 ZR-CBAM | 0.262 |
| B Dice | 0.260 |
| DRB/SCSA | 0.245 |
| E DySample | 0.248 |
| F 条带上下文 | 0.262 |
| G 定向条带上下文 | 0.266 |

八种不同机制的结构改动，卷叶螟严格掩膜 AP 的变动不超过 0.021，且无一超过 Baseline。这一
稳定性说明该指标当前的瓶颈不在网络的结构容量、感受野形状或上下文聚合方式上，继续在同一
方向更换模块不会改变结论。

## 训练行为与运行成本

| 运行项 | Baseline | G |
|---|---:|---:|
| 实际 / 计划 epoch | 300 / 300 | 300 / 300 |
| Best epoch | 216 | 206 |
| 总训练时间 | 4416.89 s（1.227 h） | 4438.29 s（1.233 h） |
| epoch 10 后中位单轮时间 | 14.70 s | 14.55 s |
| 峰值 GPU_mem | 15.6 GB | 15.6 GB |
| 融合后结构 | 149 层 / 23.5086 M 参数 / 121.2 GFLOPs | 156 层 / 23.6254 M 参数 / 122.4 GFLOPs |
| `best.pt` 大小 | 54,523,953 B（54.524 MB） | 54,773,801 B（54.774 MB） |
| Val loss NaN 单元 | 10（epoch 7、8、36） | 4（仅 epoch 2） |
| 全部权重张量有限 | 是 | 是 |

参数量增加 116,800（+0.497%），计算量增加约 0.99%，训练总时长增加 0.48%，显存与 Baseline
同档，单轮耗时未见可测差别。本轮结果不能由"新增参数或算力堆出来的"解释，也不是训练成本
导致的退化。

训练结束日志对 `best.pt` 的复核速度为 1.5 ms 推理 + 0.5 ms 后处理每张图（Baseline 同口径
记录为 1.3 ms + 0.7 ms）。该数值来自训练收尾的单张 Val 流程，不是统一 batch=1 基准，仅作
参考，表 10 仍保留"待统一测速"。

## 结论

- G 未通过门控。Val Mask mAP50-95 低于 Baseline 0.00328，差额小于单轮波动中位数，判为与
  Baseline 不可区分；两项分割主指标与检测框 AP50-95 均未超过 Baseline。
- 第 2 节的机制预期被否：两类 Mask AP50 同幅下降，卷叶螟 Mask mAP50-95 与 Baseline 相同。
- P3 条带/朝向这一路已有 F、F2、G 三条独立负结果，本轮之后不再在该方向继续加码。
- G 不进入 D1 组合，不替换 F，表 1 的对照基准保持 `data-v2-abl-000-y26m-b16-s42`。
- 本轮属结构消融，结果登记进 `comparison.csv` 与实验总表表 1、表 8、表 9、表 10；不进表 2。
- 只有 seed=42 单次证据；与 Baseline 的差异未做多 seed 复验，本轮结论限定为"未见收益"，
  不外推为"该结构在所有设置下无效"。

## 异常与限制

- epoch 2 有 4 个 Val loss 字段为 NaN，数量少于 Baseline 的 10 个；训练损失、全部 P/R/mAP
  指标、best epoch 以及两份权重的全部张量均为有限值，不属于训练崩溃。
- 未对卷叶螟做原图配对专项评估。E、F、C、D1 在该协议下出现过的局部收益与正式主指标方向
  不一致，本轮不具备启动该专项的收益依据。
- 未使用 Test。Test 保留到全部方案冻结后统一评估。

## 证据

| 内容 | 位置 |
|---|---|
| 完整 Run | `runs/data-v2-abl-g-oristrip-b16-s42/` |
| 实际参数 | `runs/data-v2-abl-g-oristrip-b16-s42/args.yaml` |
| 逐 epoch 指标 | `runs/data-v2-abl-g-oristrip-b16-s42/results.csv` |
| 训练日志 | `runs/data-v2-abl-g-oristrip-b16-s42/train_data-v2-abl-g-oristrip-b16-s42.log` |
| 训练曲线 | `runs/data-v2-abl-g-oristrip-b16-s42/results.png` |
| Mask PR 曲线 | `runs/data-v2-abl-g-oristrip-b16-s42/MaskPR_curve.png` |
| 传输归档 | `exports/data-v2-abl-g-oristrip-b16-s42.zip` |
| Baseline 对照 | `experiment_records/runs/data-v2-abl-000-y26m-b16-s42.md` |
