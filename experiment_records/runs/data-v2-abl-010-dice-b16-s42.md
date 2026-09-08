# Run 记录：data-v2-abl-010-dice-b16-s42

## 结论

**B：原实例 BCE + 0.5 × Soft Dice 未通过单模块门控，不进入组合实验。**
完整 Run 已核验，best epoch 为 243。相对正式 `000`，Mask mAP50 / mAP50-95 分别
下降 0.03116 / 0.01723，official fitness 下降 0.01618；两个类别 Mask AP 均下降。
该结论适用于本次 data-v2、冻结配方和 seed=42 下的 B 实现，不推广为所有 Dice 方案无效。

## 基本信息

| 项目 | 内容 |
|---|---|
| 状态 | 已完成 300 epoch，完整 Run 已核验并正式登记；未通过门控 |
| 更新日期 | 2026-09-07 |
| 实验目的 | 只增加实例掩膜 Soft Dice 项，检验掩膜重叠质量与严格 IoU 指标 |
| 配置文件 | B 分支 `experiments/data-v2-abl-010-dice-b16-s42.yaml` |
| 实现分支 / commit | `feature/data-v2-abl-b-dice`；源码 `1d1a71ed2daef0e59611f790efe76b9bd213d708`，文档 `cef6b0b` |
| 实现基点 | 正式 Baseline `c0f4f35`，独立于 A1/A2 |
| 云端原始目录 | `/root/yolo_plus/runs/data-v2-abl-010-dice-b16-s42/` |
| 本地完整目录 | `runs/data-v2-abl-010-dice-b16-s42/`，共 29 个文件 |
| 本地训练日志 | `runs/data-v2-abl-010-dice-b16-s42/data-v2-abl-010-dice-b16-s42.log` |
| 数据 / split | `rice-pest-data-v2` / Val，117 张图、557 个实例 |
| GPU / 环境 | RTX 5090（32111 MiB）；Python 3.12.3；PyTorch 2.12.1+cu130；Ultralytics 8.4.80 |
| 权重保存日期 | 2026-09-04，时间字段不含时区 |

## 参数与实现核验

`args.yaml` 相对正式 `000` 只有四项差异：新增 `instance_dice_gain=0.5`、
`instance_dice_smooth=1.0`，以及 Run 的 `name`、`save_dir`。B 分支配置与实际参数一致，
其中 `project: runs` 被解析为云端绝对路径。两份权重的 `train_args` 与 args.yaml 一致，
均保存了两个 Dice 参数。best.pt 与 Baseline 的状态键和张量形状完全相同，没有注意力参数。

| 参数 | 实际值 |
|---|---:|
| model / pretrained | 官方 `yolo26m-seg.pt` |
| epochs / patience | 300 / 100 |
| batch / nbs / accumulate | 16 / 64 / 4 |
| imgsz / workers / device | 640 / 8 / 0 |
| optimizer | AdamW（显式） |
| lr0 / lrf | 0.001667 / 0.01 |
| momentum / weight_decay | 0.9 / 0.0005 |
| warmup_epochs / warmup_bias_lr | 3 / 0.1 |
| seed / deterministic | 42 / true |
| amp / cache / cos_lr | true / false / false |
| mask_ratio | 2 |
| instance_dice_gain / instance_dice_smooth | 0.5 / 1.0 |
| mosaic / close_mosaic | 1.0 / 15 |
| mixup / copy_paste | 0.0 / 0.3 |
| degrees / translate / scale | 15 / 0.1 / 0.3 |
| flipud / fliplr | 0.5 / 0.5 |
| 预训练迁移 | 890 / 904 项；AMP 检查通过 |
| 最佳权重选择 | Val official fitness = Box mAP50-95 + Mask mAP50-95 |

源码 `1d1a71e` 中，BCE 保留原裁框与面积归一化。Dice 使用 float32 sigmoid 概率，
在每个匹配实例的目标框内计算：

```text
p = sigmoid(mask_logits)
DiceLoss = 1 - (2 × Σ(p × target) + 1) / (Σp + Σtarget + 1)
InstanceMaskLoss = original_BCE + 0.5 × DiceLoss
```

按匹配实例求和后，沿用原前景归一化与分割增益；模型结构、检测损失和语义辅助损失不变。
训练日志未记录实际 Git commit，因此上述提交是实现来源，不能作为已核实的训练提交。

## 最佳轮次与资源

CSV 的官方 fitness 最大值位于 epoch 243，与 best.pt 的全部 train_metrics 一致。
last.pt 的 train_metrics 与 CSV epoch 300 一致，两份权重的全部状态张量均为有限值。

| 指标 | 结果 |
|---|---:|
| 计划 / 实际 epoch | 300 / 300，未触发 EarlyStopping |
| 最佳 epoch | 243 |
| Official fitness | 0.80359 |
| Val Box P / R | 0.76448 / 0.60003 |
| Val Box mAP50 / mAP50-95 | 0.71822 / 0.45734 |
| Val Mask P / R | 0.70942 / 0.59795 |
| Val Mask mAP50 / mAP50-95 | 0.68016 / 0.34625 |
| Val Mask F1 | 0.64893 |
| 到最佳轮次累计时间 | 3889.95 s（1.081 h） |
| 总训练时间 | 4792.11 s（1.331 h） |
| 日志 GPU_mem 峰值 | 16.3 GB |
| 融合后模型 | 149 layers / 23,509,010 Params / 121.2 GFLOPs（日志精度） |
| best.pt / last.pt 大小 | 各 54.524209 MB |
| 训练结束 Val 速度 | 1.4 ms inference + 0.9 ms postprocess / image，非统一 batch=1 测速 |

B 与 Baseline 的推理结构一致，参数量不增加；但训练增加 375.22 s（8.50%），日志峰值
显存由 15.6 GB 增至 16.3 GB。两组均完成 300 epoch，平均每轮分别约 15.974 / 14.723 s。

## 与正式 Baseline 比较

总体指标统一使用官方 fitness 选出的最佳 epoch 精确值：Baseline epoch 216、B epoch 243。
差值单位为指标原始小数；Mask mAP50 / mAP50-95 的下降对应 3.116 / 1.723 个百分点。

| 指标 | 000 Baseline | B Dice | B−000 |
|---|---:|---:|---:|
| Box mAP50 | 0.72980 | 0.71822 | -0.01158 |
| Box mAP50-95 | 0.45629 | 0.45734 | +0.00105 |
| Mask P | 0.64822 | 0.70942 | +0.06120 |
| Mask R | 0.67252 | 0.59795 | -0.07457 |
| Mask mAP50 | 0.71132 | 0.68016 | -0.03116 |
| Mask mAP50-95 | 0.36348 | 0.34625 | -0.01723 |
| Official fitness | 0.81977 | 0.80359 | -0.01618 |
| Mask F1 | 0.66015 | 0.64893 | -0.01121 |
| Top-5 fitness 均值 | 0.80882 | 0.79582 | -0.01300 |
| Top-10 fitness 均值 | 0.80463 | 0.79233 | -0.01230 |

精确率提高伴随召回率明显下降，Mask F1、两项 Mask AP 与总体 fitness 都降低。
top-5 / top-10 fitness 均低约 0.013 / 0.012，负结果不只体现在单个最佳 epoch。

即使仅用于诊断地查看整条曲线，B 的最高 Mask mAP50 也只有 0.70114（epoch 283），
最高 Mask mAP50-95 为 0.35045（epoch 206），仍分别低于 Baseline 正式值
0.71132 / 0.36348。因此本次退化不能仅归因于官方 fitness 选择了非 Mask AP 最高轮次。
正式汇总仍使用 epoch 243，不更换选权重规则。

## 分类型 Val Mask

下表来自训练结束后重新加载并融合 best.pt 的 Val 日志，保留其三位精度。

| 类别 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | ΔmAP50 | ΔmAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Rice leaffolder | 0.678 | 0.602 | 0.651 | 0.260 | -0.026 | -0.006 |
| Rice stemborers | 0.739 | 0.596 | 0.711 | 0.433 | -0.034 | -0.028 |

两类别的召回率分别下降 0.054 / 0.091，两项 AP 均下降，钻心虫在严格 Mask IoU 下的
退化更明显。当前结果没有支持 B 改善区域重叠质量；也未进行边界专项评估，不能直接
解释为某种具体边界错误。

训练结束融合验证的总体 Mask P/R/mAP50/mAP50-95 为 0.708 / 0.599 / 0.681 / 0.346，
与 epoch 243 元数据存在轻微差异。与 `000`、A1、A2 一样，总体登记使用 CSV/权重元数据，
分类别使用日志复核，保持同一记录口径。

## 收敛、数值与限制

- CSV 连续覆盖 epoch 1–300，无缺轮或重复；日志完成训练、best.pt 验证与保存，无异常回溯或 OOM。
- epoch 4 的 Val box/seg/cls/dfl loss 为 NaN；epoch 6 的 Val seg/cls loss 为 NaN。其余 CSV 字段、所有训练损失、P/R/mAP、学习率和权重张量均有限，不能表述为 Val loss 全程正常。
- 第 286 轮按冻结配方关闭 Mosaic。epoch 251–285 的平均 fitness / Mask mAP50-95 为 0.77310 / 0.33284，epoch 286–300 为 0.75071 / 0.32158；后期没有追回性能，但时序变化不能单独证明由关闭 Mosaic 导致。
- 训练 seg_loss 前 10 / 后 10 轮均值为 2.21882 / 1.31033，下降约 40.9%；训练目标得到优化，但没有转化为 Val AP 收益。B 的 seg_loss 包含额外 Dice 项，不能直接拿其绝对值与 Baseline 的 BCE-only seg_loss 判断拟合优劣。
- 日志未分别输出 BCE、Dice 项或梯度贡献，不能由当前结果确定是否因 λ=0.5 过大、两项梯度冲突或其他优化机制导致下降。
- train / Val 均为 0 corrupt，与 Baseline 一样有两张训练图各移除 1 个重复标签。数据划分与冻结参数未发生额外变化。
- 两份权重均已 strip optimizer，epoch=-1；最佳轮次通过 CSV 和 train_metrics 对齐确认。best.pt 保存字段为 2026-09-04T18:44:52.018117，last.pt 为 18:59:54.189197，未含时区。
- 当前只有 seed=42，没有统计显著性或多 seed 稳定性结论；最终 Test 与边界、小目标专项评估均未开展。

## 决策与后续

本次 `BCE + 0.5 × Soft Dice` 同时降低总体 Mask AP 和两类别 AP，增加训练成本，按预设
规则淘汰本次 B 实现，不进入 A+B、B+C 或 A+B+C。保留其正式负结果，不据此修改既有
冻结配方或在本 Run 上继续调整 λ。

下一步按计划独立准备 C：P2Head 的定义和配置，再进行其单模块验证。若以后重新尝试
Dice，需要明确不同的研究假设和新 Run ID，不能覆盖本次结果；正式训练仍由用户手动启动。

## 证据来源

- `runs/data-v2-abl-010-dice-b16-s42/args.yaml`：实际参数；与 `runs/data-v2-abl-000-y26m-b16-s42/args.yaml` 比较。
- `runs/data-v2-abl-010-dice-b16-s42/results.csv`：300 轮指标、最佳轮次、top-k、非有限值与训练时间。
- `runs/data-v2-abl-010-dice-b16-s42/weights/best.pt`、`weights/last.pt`：train_metrics、train_args、状态键与有限值核验。
- `runs/data-v2-abl-010-dice-b16-s42/data-v2-abl-010-dice-b16-s42.log`：完成状态、模型规模、显存和训练结束 best.pt 分类型复核。
- `experiment_records/runs/data-v2-abl-000-y26m-b16-s42.md`：Baseline 分类别与资源对照。
- B 分支 `experiments/data-v2-abl-010-dice-b16-s42.yaml`、提交 `1d1a71e` 的 `ultralytics-main/ultralytics/utils/loss.py`：冻结配置和实现。
