# RTX 5090 参数诊断轮次 mask_ratio=1 分析：data-v2-tune-mr1-e300-b16-s42

## 技术摘要

- 本次 Run 训练到 **epoch 287 触发早停**（patience=100，best 出现在 epoch 187），未发生 OOM 或训练中断；总时长 6977.54 s（1.938 h）。
- Ultralytics 官方分割 fitness 在 **epoch 187 达到 0.85378**，对应现有 `best.pt`；该轮也同时取得全程最高 Mask mAP50 `0.72887`。
- 相对 mr=2 的冻结配方，官方网格上 fitness **+0.03401**、Mask mAP50-95 **+0.03121**、Mask mAP50 **+0.01755**，而 Box mAP50-95 只 **+0.00280**。增益几乎全部落在掩膜项。
- 增益并非卷叶螟专属：分类型 Mask AP50 卷叶螟 `0.691`、钻心虫 `0.766`，相对 mr=2 分别 `+0.014 / +0.021`。第 2 节"收益集中在卷叶螟掩膜"的可证伪预期**不成立**。
- **但两次验证的掩膜真值不在同一网格上。** `mask_ratio` 同时决定 Val 数据集的 GT 多边形栅格化网格（mr=2 → 320²，mr=1 → 640²），因此基线与本轮的 Mask 列不是在同一份真值上测得的，本轮的掩膜增益不能直接采信。
- 据此判定为参数诊断结论：配方维持 `mask_ratio=2`，表 1 的对照基准不改，本 Run 不进入表 1 与 `comparison.csv`。
- 与 P2 不同，本轮 `results.csv` 未出现任何 NaN 损失单元。
- 只有 seed=42 单次证据。

本 Run 属于参数优化实验，用于确定 `云服务器实验设计与记录表.md` 表 2，不作为期刊模型对比 Run，不写入 `experiment_records/comparison.csv`。

## 实验范围与证据

| 项目 | 值 |
|---|---|
| Run ID | `data-v2-tune-mr1-e300-b16-s42` |
| 任务 | 无人机航拍水稻害虫实例分割 |
| 数据 | `rice-pest-data-v2` |
| 选择 split | Val（117 images / 557 instances） |
| 模型选择口径 | Val official fitness = Box mAP50-95 + Mask mAP50-95 |
| Test | 未使用 |
| 原始 Run | [`runs/data-v2-tune-mr1-e300-b16-s42/`](../../runs/data-v2-tune-mr1-e300-b16-s42/) |
| 实际参数 | [`args.yaml`](../../runs/data-v2-tune-mr1-e300-b16-s42/args.yaml) |
| 逐 epoch 指标 | [`results.csv`](../../runs/data-v2-tune-mr1-e300-b16-s42/results.csv) |
| 训练日志 | [`train_data-v2-tune-mr1-e300-b16-s42.log`](../../runs/data-v2-tune-mr1-e300-b16-s42/train_data-v2-tune-mr1-e300-b16-s42.log) |
| 训练曲线 | [`results.png`](../../runs/data-v2-tune-mr1-e300-b16-s42/results.png) |
| Mask PR 曲线 | [`MaskPR_curve.png`](../../runs/data-v2-tune-mr1-e300-b16-s42/MaskPR_curve.png) |
| 现有最佳权重 | `runs/data-v2-tune-mr1-e300-b16-s42/weights/best.pt`（官方 fitness，epoch 187） |

## 本次实际训练参数

与冻结配方逐项相同，唯一差异是 `mask_ratio`。

| 参数 | 实际值 | 冻结说明 |
|---|---:|---|
| model | `yolo26m-seg.pt` | 官方预训练权重 |
| epochs / patience | 300 / 100 | epoch 287 早停，best 在 187 |
| batch / nbs / accumulate | 16 / 64 / 4 | 物理 batch=16，有效名义 batch 约 64 |
| imgsz | 640 | 固定，不做分辨率调参 |
| workers / cache | 8 / false | 保持 |
| optimizer | `AdamW` | 固定 |
| lr0 / lrf | 0.001667 / 0.01 | 线性学习率，`cos_lr=false` |
| momentum / weight_decay | 0.9 / 0.0005 | 固定 |
| warmup_epochs / warmup_bias_lr | 3 / 0.1 | 固定 |
| seed / deterministic | 42 / true | 与全部正式实验统一 |
| amp | true | 固定 |
| **mask_ratio** | **1** | 本轮唯一改动，相对冻结配方从 2 降至 1 |
| mosaic / mixup / copy_paste | 1.0 / 0.0 / 0.3 | 固定 |
| degrees / flipud / fliplr | 15 / 0.5 / 0.5 | 固定航拍方向增强 |
| scale | 0.3 | 固定 |
| close_mosaic | 15 | epoch 285 后关闭组合增强 |
| val split / iou / max_det | val / 0.7 / 300 | 保持官方验证口径 |

## 与冻结配方的核心结果对比

对照为 `data-v2-abl-000-y26m-b16-s42`（与 `data-v2-tune-mr2-nomix-e300-b16-s42` 同配方同 seed，指标一致），仅 `mask_ratio` 与本轮不同。两轮均按官方分割 fitness 选取最佳 epoch。

| 指标 | mr=2 best（epoch 216） | mr=1 best（epoch 187） | mr1 − mr2 |
|---|---:|---:|---:|
| **Official fitness** | 0.81977 | **0.85378** | **+0.03401** |
| Box P | 0.67918 | **0.71588** | +0.03670 |
| Box R | 0.64991 | **0.65955** | +0.00964 |
| Box mAP50 | 0.72980 | **0.74766** | +0.01786 |
| Box mAP50-95 | 0.45629 | **0.45909** | +0.00280 |
| Mask P | 0.64822 | **0.69972** | +0.05150 |
| Mask R | **0.67252** | 0.64543 | −0.02709 |
| Mask mAP50 | 0.71132 | **0.72887** | +0.01755 |
| Mask mAP50-95 | 0.36348 | **0.39469** | +0.03121 |

Box mAP50-95 只提高 `0.00280`，Mask mAP50-95 提高 `0.03121`，两者相差一个量级。`mask_ratio` 只改变掩膜监督目标的栅格化精度、不改变框目标，增益集中在掩膜项与这一机制方向一致；但方向一致不等于增益可用，"评价网格"一节给出不能直接采信的理由。

| 运行项 | mr1 结果 |
|---|---:|
| 计划 / 实际 epoch | 300 / 287（早停） |
| Official fitness 最佳 epoch | 187 |
| Mask mAP50 曲线峰值 epoch | 187（与官方 best 重合） |
| last.pt / epoch 287 fitness | 0.79347 |
| Top-5 official fitness 均值 | 0.84588（epoch 187 / 237 / 186 / 262 / 283） |
| Top-10 official fitness 均值 | 0.84050 |
| 总训练时间 | 6977.54 s（1.938 h） |
| epoch 10 后中位单轮时间 | 24.25 s |
| best.pt / last.pt 大小 | 各约 54.5 MB |
| 参数量 / GFLOPs | 26.971 M / 131.9（融合后 23.509 M / 121.2） |
| Val loss NaN 单元 | 0 |

单轮耗时从 mr=2 的 14.63 s 升到 24.25 s，掩膜监督分辨率翻倍使每个 epoch 的验证与损失计算量增加约 66%。

## 收敛与稳定性诊断

| epoch 区间 | Mask mAP50 均值 | 区间最高 | Mask mAP50-95 均值 | Official fitness 均值 | 区间最高 fitness |
|---|---:|---:|---:|---:|---:|
| 1–50 | 0.53576 | 0.64726 | 0.26385 | 0.56110 | 0.70266 |
| 51–100 | 0.62168 | 0.69457 | 0.32260 | 0.69742 | 0.78739 |
| 101–150 | 0.65949 | 0.69683 | 0.34986 | 0.75732 | 0.82636 |
| 151–200 | 0.68173 | **0.72887** | 0.36465 | 0.79079 | **0.85378** |
| 201–250 | 0.69649 | 0.71992 | 0.37393 | 0.81328 | 0.84655 |
| 251–285 | **0.69707** | 0.71161 | **0.37376** | **0.82124** | 0.84257 |
| 286–287 | 0.67186 | 0.67209 | 0.35907 | 0.79237 | 0.79347 |

区间均值在 201–285 保持高位，但**峰值出现在 151–200**，且 251–285 的均值（0.82124）高于 151–200（0.79079）却没有新的峰值，说明后半程处于高位波动而非持续上升。close_mosaic 生效后的 286–287 出现回落。现有结果支持统一使用 `best.pt` 而非 `last.pt`。

![训练与验证曲线](../../runs/data-v2-tune-mr1-e300-b16-s42/results.png)

## 分类型结果

本轮的完整分类型指标来自日志末尾对 `best.pt` 的验证输出，属归档内的原始证据。

| 类别 | 实例数 | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 557 | 0.748 | 0.459 | 0.698 | 0.647 | 0.729 | 0.394 |
| Rice leaffolder（卷叶螟） | 462 | 0.704 | 0.397 | 0.675 | 0.630 | 0.691 | 0.316 |
| Rice stemborers（钻心虫） | 95 | 0.791 | 0.521 | 0.722 | 0.663 | 0.766 | 0.472 |

Val 分类型实例数为卷叶螟 **462**、钻心虫 **95**，合计 557，与 `PROGRESS.md` 早期记录的卷叶螟 174 不一致；462 与总量 557 自洽，是应采用的值。

`best.pt` 没有为 mr=2 对照臂留下独立的分类型 mAP50-95 文本，对照只能取两张 Mask PR 曲线图例的 AP50：

| 类别 | mr=2 | mr=1 | mr1 − mr2 |
|---|---:|---:|---:|
| Rice leaffolder（卷叶螟） | 0.677 | **0.691** | +0.014 |
| Rice stemborers（钻心虫） | 0.745 | **0.766** | +0.021 |
| all classes | 0.711 | **0.729** | +0.018 |

两类同时上升，且**钻心虫的涨幅大于卷叶螟**。若细长几何与监督分辨率的方向成立，涨幅应集中在短边中位数 7.17 px 的卷叶螟上，而不是短边 22.27 px 的钻心虫。该项预期被否。

![归一化混淆矩阵](../../runs/data-v2-tune-mr1-e300-b16-s42/confusion_matrix_normalized.png)

## 评价网格与归因边界

这是本轮结论的关键限制，也是不据此改写表 1 基准的原因。

`mask_ratio` 不只影响训练。`YOLODataset.build_transforms` 在 `if self.augment` 分支之外无条件构造 `Format(..., mask_ratio=hyp.mask_ratio, ...)`（`ultralytics/data/dataset.py`），因此 **Val 数据集同样按 `1/mask_ratio` 栅格化 GT 多边形**（`polygons2masks(..., downsample_ratio=mask_ratio)`）。项目的验证器把预测掩膜统一压到 stride-4 网格（`ultralytics/models/yolo/segment/val.py` 的 `postprocess`），随后掩膜 IoU 在该网格上比较预测与 GT。

由此：

- mr=2 的 Val GT 在 320² 上栅格化后送入比较，mr=1 的在 640² 上栅格化后送入比较；
- 两轮的 Mask P/R/mAP 因此**不是在同一份真值上测得的**，掩膜列的差值不是同口径差值；
- Box 列不受影响，仍是同口径比较；
- 论文只报官方网格数值，官方网格按项目口径固定为 stride-4 掩膜预测网格；`mask_ratio` 改变的是监督与真值的栅格化，不改变该预测网格。

要闭合这一问题，必须把两个 `best.pt` 用**同一个 `mask_ratio`** 各评一次。该复评没有留下任何归档产物（无 Val Run、无 JSON、无日志），本次归档无法引用其数值，因此本记录只陈述口径不一致这一事实，不引用未归档的比较结果。

## 结论与冻结决定

- 配方维持 `mask_ratio=2`；表 1 的对照基准保持 `data-v2-abl-000-y26m-b16-s42`，不因本轮改写。
- 本轮登记进《云服务器实验设计与记录表》表 2；不运行 `fill_results_table.py`，不进 `comparison.csv`，不进表 1。
- 官方网格上的掩膜增益幅度较大且方向与机制一致，但评价网格不一致使其不可采信；若后续要将其写入论文或据此改配方，须先完成共同网格复评并归档。
- 本轮不构成"已解决卷叶螟小目标掩膜"的证据。

## 后续执行

1. 不据此修改冻结配方。
2. 表 2 登记本轮的参数与结果数据。
3. 若需要闭合网格口径问题，另起一个共同网格复评 Run（新 Run ID），把两个 `best.pt` 在同一 `mask_ratio` 下的 Val 结果作为独立产物打包回传。
4. Val 继续用于方案比较；Test 保留到模型与阈值全部冻结后统一执行。
