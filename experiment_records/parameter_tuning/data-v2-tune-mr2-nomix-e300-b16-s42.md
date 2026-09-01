# RTX 5090 参数优化第 2 轮掩膜精调分析：data-v2-tune-mr2-nomix-e300-b16-s42

## 技术摘要

- 本次 Run 完成连续 epoch 1～300，未发生 OOM 或训练中断；总时长 4410.58 s（1.225 h）。
- Ultralytics 官方分割 fitness 在 **epoch 216 达到 0.81977**，对应现有 `best.pt`；该轮也同时取得全程最高 Mask mAP50 `0.71132`。
- 相对 P1，official fitness 提升 **+0.00984**，Box mAP50-95 提升 **+0.00441**，Mask mAP50 提升 **+0.02167**，Mask mAP50-95 提升 **+0.00543**。
- Mask P/R 从 P1 的 `0.72069 / 0.57720` 变为 `0.64822 / 0.67252`：Precision 下降、Recall 明显提高，但 Mask F1 从 `0.64101` 提高到 `0.66015`，且两项 Mask mAP 均提高。
- 分类型 Mask AP50 为卷叶螟 `0.677`、钻心虫 `0.745`；相对 P1 分别提高 `+0.006 / +0.037`，本轮主要收益并非专门来自卷叶螟。
- P2 相对 P1 实际同时改变了 `mask_ratio`、`mixup` 和有效 `warmup_bias_lr`，因此不能把收益单独归因于某一个参数。
- 用户于 2026-09-01 决定停止继续调参且不补跑 seed=2、3，接受当前只有 seed=42 单次证据的限制，将本配方冻结为 data-v2 后续正式实验统一训练参数。

本 Run 属于参数优化实验，用于确定 `云服务器实验设计与记录表.md` 表 1，不作为期刊模型对比 Run，不写入 `experiment_records/comparison.csv`。

## 实验范围与证据

| 项目 | 值 |
|---|---|
| Run ID | `data-v2-tune-mr2-nomix-e300-b16-s42` |
| 任务 | 无人机航拍水稻害虫实例分割 |
| 数据 | `rice-pest-data-v2` |
| 选择 split | Val（117 images / 557 instances） |
| 模型选择口径 | Val official fitness = Box mAP50-95 + Mask mAP50-95 |
| Test | 未使用 |
| 原始 Run | [`runs/data-v2-tune-mr2-nomix-e300-b16-s42/`](../../runs/data-v2-tune-mr2-nomix-e300-b16-s42/) |
| 实际参数 | [`args.yaml`](../../runs/data-v2-tune-mr2-nomix-e300-b16-s42/args.yaml) |
| 逐 epoch 指标 | [`results.csv`](../../runs/data-v2-tune-mr2-nomix-e300-b16-s42/results.csv) |
| 训练曲线 | [`results.png`](../../runs/data-v2-tune-mr2-nomix-e300-b16-s42/results.png) |
| Mask PR 曲线 | [`MaskPR_curve.png`](../../runs/data-v2-tune-mr2-nomix-e300-b16-s42/MaskPR_curve.png) |
| 现有最佳权重 | `runs/data-v2-tune-mr2-nomix-e300-b16-s42/weights/best.pt`（官方 fitness，epoch 216） |

## 本次实际训练参数

| 参数 | 实际值 | 冻结说明 |
|---|---:|---|
| model | `yolo26m-seg.pt` | 官方预训练权重 |
| epochs / patience | 300 / 100 | 训练满 300 epoch，未早停 |
| batch / nbs / accumulate | 16 / 64 / 4 | 物理 batch=16，有效名义 batch 约 64 |
| imgsz | 640 | 固定，不做分辨率调参 |
| workers / cache | 8 / false | 保持 |
| optimizer | `AdamW`（显式） | 固定 |
| lr0 / lrf | 0.001667 / 0.01 | 线性学习率，`cos_lr=false` |
| momentum / weight_decay | 0.9 / 0.0005 | 固定 |
| warmup_epochs / warmup_bias_lr | 3 / 0.1 | 固定 P2 实际值 |
| seed / deterministic | 42 / true | 参数筛选与后续正式实验统一使用 seed=42 |
| amp | true | 固定 |
| mask_ratio | 2 | 相对 P1 从 4 降至 2 |
| mosaic / mixup / copy_paste | 1.0 / 0.0 / 0.3 | 固定 |
| degrees / flipud / fliplr | 15 / 0.5 / 0.5 | 固定航拍方向增强 |
| scale | 0.3 | 固定 |
| close_mosaic | 15 | epoch 285 后关闭组合增强 |
| val split / iou / max_det | val / 0.7 / 300 | 保持官方验证口径 |

## 与 P1 的核心结果对比

两轮均在 data-v2、YOLO26m-seg、batch=16、imgsz=640、seed=42 和 Val split 下比较，并分别按官方分割 fitness 选取最佳 epoch。

| 指标 | P1 best（epoch 251） | P2 best（epoch 216） | P2 − P1 |
|---|---:|---:|---:|
| **Official fitness** | 0.80993 | **0.81977** | **+0.00984** |
| Box P | 0.67611 | 0.67918 | +0.00307 |
| Box R | 0.65062 | 0.64991 | −0.00071 |
| Box mAP50 | 0.70931 | 0.72980 | +0.02049 |
| Box mAP50-95 | 0.45188 | 0.45629 | +0.00441 |
| Mask P | **0.72069** | 0.64822 | −0.07247 |
| Mask R | 0.57720 | **0.67252** | +0.09532 |
| Mask F1 | 0.64101 | **0.66015** | +0.01914 |
| Mask mAP50 | 0.68965 | **0.71132** | +0.02167 |
| Mask mAP50-95 | 0.35805 | **0.36348** | +0.00543 |

| 运行项 | P2 结果 |
|---|---:|
| 计划 / 实际 epoch | 300 / 300（未早停） |
| Official fitness 最佳 epoch | 216 |
| Mask mAP50 曲线峰值 epoch | 216（与官方 best 重合） |
| last.pt / epoch 300 fitness | 0.78002 |
| Top-5 official fitness 均值 | 0.80882（较 P1 +0.00896） |
| Top-10 official fitness 均值 | 0.80463（较 P1 +0.00848） |
| 总训练时间 | 4410.58 s（1.225 h） |
| epoch 10 后中位单轮时间 | 14.63 s |
| best.pt / last.pt 大小 | 各约 54.52 MB |
| 参数量 / GFLOPs | 23.509 M / 121.2 |

## 分类型 Mask AP50

分类型 AP50 来自 `best.pt` 对应的 Mask PR 曲线图例。当前归档没有独立保存完整的分类型 P/R/mAP50-95 文本，因此本记录不推断或补写这些值。

| 类别 | P0 | P1 | P2 | P2 − P1 | P2 − P0 |
|---|---:|---:|---:|---:|---:|
| Rice leaffolder（卷叶螟） | 0.649 | 0.671 | **0.677** | +0.006 | +0.028 |
| Rice stemborers（钻心虫） | 0.718 | 0.708 | **0.745** | +0.037 | +0.027 |

P2 同时提高两类 Mask AP50，但相对 P1 的主要增益来自钻心虫。卷叶螟只有 `+0.006`，不能把本轮解释为已经解决卷叶螟小目标或漏检问题。

## 收敛与稳定性诊断

| epoch 区间 | Mask mAP50 均值 | 区间最高 | Mask mAP50-95 均值 | Official fitness 均值 | 区间最高 fitness |
|---|---:|---:|---:|---:|---:|
| 1–50 | 0.51215 | 0.63110 | 0.23646 | 0.52374 | 0.68860 |
| 51–100 | 0.61783 | 0.66505 | 0.29996 | 0.67959 | 0.74340 |
| 101–150 | 0.64760 | 0.67428 | 0.31457 | 0.72419 | 0.76176 |
| 151–200 | 0.66441 | 0.69658 | 0.32894 | 0.75614 | 0.80437 |
| 201–250 | 0.68175 | **0.71132** | 0.33983 | 0.77761 | **0.81977** |
| 251–285 | **0.68762** | 0.70629 | **0.34502** | **0.78997** | 0.80748 |
| 286–300 | 0.67452 | 0.68394 | 0.33622 | 0.76913 | 0.78002 |

best epoch 位于 216；251～285 区间的平均水平仍较高，但 close_mosaic 生效后的 286～300 区间出现回落。现有结果不支持继续增加训练轮数，应统一使用 `best.pt`，而不是 `last.pt`。

![训练与验证曲线](../../runs/data-v2-tune-mr2-nomix-e300-b16-s42/results.png)

### Precision 与 Recall 的方向性权衡

P2 的 Mask Precision 比 P1 低 `0.07247`，Recall 高 `0.09532`。这不是单纯的性能下降：Mask F1、Mask mAP50 和 Mask mAP50-95 同时提高，说明模型覆盖了更多真实实例，代价是增加了一部分误检。Mask F1 曲线显示 all classes 的峰值约为 `0.66 @ confidence 0.320`。

后续若课题应用强调低误检，应在冻结模型后单独使用 Val 校准置信度阈值，并在 Test 前冻结阈值；置信度阈值属于推理工作点，不改变本次训练参数冻结结论，也不得用 Test 调参。

### 数值异常不构成训练崩溃

`results.csv` 的训练损失与全部 mAP 指标均保持有限。Val loss 在 epoch 7、8 的 box/seg/cls/dfl 字段，以及 epoch 36 的 seg/cls 字段出现 NaN；异常没有传播到训练损失、mAP、权重或后续 epoch。故本 Run 可用于参数筛选，但不能表述为“所有 Val loss 全程有限”。

归一化混淆矩阵显示两类之间几乎没有直接互相混淆，主要错误仍是目标与 background 之间的漏检/误检。

![归一化混淆矩阵](../../runs/data-v2-tune-mr2-nomix-e300-b16-s42/confusion_matrix_normalized.png)

## 参数差异与归因边界

P2 名义改动为 `mask_ratio 4→2`、`mixup 0.1→0`，并把 P1 自动选择的 AdamW 显式写入配置。实际还存在一个容易遗漏的差异：

- P1 使用 `optimizer=auto`，Trainer 会把运行时 `warmup_bias_lr` 改为 `0.0`；
- P2 使用显式 AdamW，保留默认 `warmup_bias_lr=0.1`。

因此，P2 的收益只能表述为“第 2 轮组合配方优于 P1”，不能单独声称由 `mask_ratio=2`、关闭 mixup 或 warmup bias 中任何一项造成。降低 `mask_ratio` 增加的是掩膜监督网格密度，不会提高 Proto 的原生特征分辨率。

## 冻结决定

2026-09-01，用户决定停止本阶段参数搜索，不再补跑 seed=2、3。依据 seed=42 的 Val 结果，正式冻结 P2 为 data-v2 后续 Baseline、源码消融、模型尺度和跨代对比实验的统一训练配方：

```text
epochs=300, patience=100, batch=16, imgsz=640, workers=8
optimizer=AdamW, lr0=0.001667, lrf=0.01, momentum=0.9
weight_decay=0.0005, warmup_epochs=3, warmup_bias_lr=0.1
mask_ratio=2, mosaic=1.0, mixup=0.0, copy_paste=0.3
degrees=15, flipud=0.5, fliplr=0.5, scale=0.3
close_mosaic=15, seed=42, deterministic=true, amp=true
```

冻结表示后续各正式对比实验必须统一使用上述参数；不表示 P2 已经过多 seed 稳定性证明。论文方法或实验设置中应如实写明使用固定 seed=42，并将“缺少多 seed 均值与标准差”列为实验限制。

## 后续执行

1. 不再运行参数优化 P3，也不补跑 P2 seed=2、3。
2. 在总表表 1 登记上述冻结值，后续正式 Run 使用唯一且不与参数优化 Run 冲突的 Run ID。
3. 开始期刊正式 Baseline、源码消融、模型尺度和跨代对比；这些 Run 才进入 `experiment_records/comparison.csv`。
4. Val 继续用于方案比较；Test 保留到模型与阈值全部冻结后统一执行。
