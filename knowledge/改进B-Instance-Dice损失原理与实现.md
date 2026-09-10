# 改进 B：Instance Dice 损失原理与实现

## 1. 本次改进是什么

改进 B 不修改 YOLO26m-seg 的 Backbone、Neck 或 Head，只修改训练阶段的实例掩膜损失：

```text
Baseline：L_instance = L_BCE
B：       L_instance = L_BCE + 0.5 × L_Dice
```

正式 Run ID 为 `data-v2-abl-010-dice-b16-s42`。B 分支从正式 Baseline 提交 `c0f4f35`
创建，损失源码和配置提交为 `1d1a71e`。该分支不包含 A1/A2 注意力模块，因此能作为只改变损失函数的单项消融实验。

## 2. 先理解 YOLO26 的实例掩膜

YOLO26 的分割 Head 不会为每个目标直接输出一张完整掩膜。它先输出一组共享的 Prototype
掩膜，再为每个预测目标输出一组 mask coefficients。源码通过矩阵组合得到实例掩膜 logits：

```text
Z_i = coefficients_i × prototypes
```

其中 `Z_i` 仍是没有经过 sigmoid 的实数。正式 Baseline 对每个已经匹配到真实目标的正样本：

1. 用 `BCEWithLogits` 比较 `Z_i` 与真实二值掩膜；
2. 只保留目标框内的损失；
3. 按目标框面积归一化；
4. 对正样本求和，最后除以正样本数量。

B 完整保留这条 BCE 路径，只在同一个目标框内增加 Soft Dice。

## 3. BCE 解决什么问题

二值交叉熵把每个像素分别看作前景或背景分类：

```text
L_BCE = -[g × log(sigmoid(z)) + (1-g) × log(1-sigmoid(z))]
```

`g` 是 0 或 1 的真实标签，`z` 是 mask logit。代码使用 `binary_cross_entropy_with_logits`，
把 sigmoid 与 BCE 合并计算，比先手动 sigmoid 再取对数更稳定。

BCE 的优点是每个像素都有直接监督；不足是它主要关心逐像素分类，没有直接优化预测区域与
真实区域的整体重叠比例。小目标的前景像素较少时，整体重叠信息尤其有补充价值。

## 4. Soft Dice 怎样计算

对某个匹配实例，在其目标框区域 `C` 内定义：

```text
p = sigmoid(z)
intersection = Σ_C(p × g)
L_Dice = 1 - (2 × intersection + smooth) / (Σ_C p + Σ_C g + smooth)
```

本项目固定：

| 项目 | 固定值 |
|---|---:|
| `instance_dice_gain` | 0.5 |
| `instance_dice_smooth` | 1.0 |
| sigmoid 位置 | 只在 Dice 内把 logits 转成概率 |
| 计算范围 | 每个匹配实例的目标框内 |
| 聚合方式 | 每个正样本先算 BCE+Dice，再沿用原正样本数量归一化 |

预测与真实掩膜重叠越好，Dice 系数越接近 1，`L_Dice` 越接近 0。`smooth=1.0` 相当于在
分子和分母中加入一个像素尺度的平滑项，避免极小区域出现除零或过强数值波动。

## 5. 为什么组合 BCE 和 Dice

BCE 与 Dice 提供的监督角度不同：

- BCE 逐像素判断前景和背景，提供细粒度、稳定的局部梯度；
- Dice 直接关心预测区域和真实区域的整体交集，降低前景/背景像素数量不平衡的影响；
- 二者组合后，保留 Baseline 已验证的 BCE 优化路径，同时补充区域重叠目标。

Dice 系数目标被用于分割损失的典型工作包括 V-Net。这里采用的是适配 YOLO 实例匹配和
目标框裁剪路径的 Soft Dice，而不是直接复制医学图像的整图训练方式。

## 6. 源码改动位置

| 文件 | 改动 |
|---|---|
| `ultralytics-main/ultralytics/utils/loss.py` | 增加框内实例 Soft Dice，并与原 BCE 相加 |
| `ultralytics-main/ultralytics/cfg/default.yaml` | 增加默认 `instance_dice_gain=0` 和 `instance_dice_smooth=1` |
| `ultralytics-main/ultralytics/cfg/__init__.py` | 把两个参数注册为数值型配置 |
| `ultralytics-main/tests/test_instance_dice_loss.py` | 验证公式、BCE 等价、梯度和配置传递 |
| `experiments/data-v2-abl-010-dice-b16-s42.yaml` | B 的唯一正式训练配置 |

核心计算对应：

```python
pred_probability = pred_mask.float().sigmoid()
intersection = (pred_probability * target).sum(dim=(1, 2))
denominator = pred_probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
dice_loss = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
instance_loss = bce_loss + dice_gain * dice_loss
```

实际源码在计算交集和分母前，还会用目标框掩码同时裁剪预测概率与真实掩膜。

## 7. 为什么默认 gain 是 0

`default.yaml` 中使用 `instance_dice_gain=0.0`，是为了保持非 B 配置的官方实例 BCE 行为。
B 的正式 YAML 才显式设置为 0.5。源码在 gain 为 0 时直接返回原 BCE，不计算 Dice，因此：

- 正式 Baseline 配置的损失语义不变；
- 其他检测、分割任务不会因为存在这段源码就自动启用 B；
- `args.yaml` 会明确记录正式 B 使用的 gain 和 smooth。

## 8. 哪些部分没有改变

- 模型参数量和 GFLOPs 不变；
- 推理阶段不计算 Dice，推理速度和输出结构不变；
- Box、分类、DFL 和语义辅助损失不变；
- 数据、增强、batch、学习率、epoch 和随机种子不变；
- Val 指标与 official fitness 的定义不变；
- Test 仍保留到最终方案冻结后使用。

训练日志中的 `seg_loss` 会变成 BCE 与 Dice 组合后的值，因此不能拿它的绝对大小与 Baseline
的纯 BCE `seg_loss` 直接比较收敛优劣。正式判断仍使用同一 Val 指标，主指标是 Mask
mAP50-95，同时报告 Mask mAP50 和分类别结果。

## 9. 实验后怎样判断

B 训练结束后与 `data-v2-abl-000-y26m-b16-s42` 比较：

- Official fitness；
- 总体 Mask mAP50-95 和 Mask mAP50；
- 卷叶螟、钻心虫各自的 Mask Recall、mAP50 和 mAP50-95；
- 最佳结果是否只是一轮尖峰；
- 训练时间、显存和数值稳定性。

本轮固定 seed=42。单次正提升可用于工程门控，但不能表述为已经证明统计显著。

## 参考资料

- Milletari、Navab、Ahmadi，V-Net：[Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
