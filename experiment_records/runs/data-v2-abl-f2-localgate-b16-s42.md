# Run 记录：data-v2-abl-f2-localgate-b16-s42

F2（F 的局部特征引导条带门控）本次未获得正式分割收益，且是 P3 条带这一路三条负结果里最弱的一条。
`best.pt` 对应 epoch 104，Val Mask mAP50-95=0.34733，比 Baseline 低 0.01615（1.615 个百分点），
比原 F 低 0.01149；official fitness 0.79673，比 Baseline 低 0.02304。训练在第 204 轮按
patience=100 早停，全程最高 Mask mAP50-95 与最高 fitness 都落在 best epoch，不是训练中断或选权重
口径造成的差异。新增门控与投影全部学到非零权重，模块确实参与了前向变化。本轮不进入组合，
P3 条带方向在 F、F2、G 三条独立负结果后关闭。

## 基本信息与实际参数

| 项目 | 内容 |
|---|---|
| 状态 | 已完成并核验，204 epoch 早停（patience=100），未跑满 300 |
| 实验目的 | 验证用局部外观调节条带分支的横、纵门控，能否在保留召回的同时改善误检与掩膜匹配 |
| 源码分支 / 实际训练提交 | `feature/data-v2-abl-f2-localgate` / `c13cef0c65f1cbad9dd4b60347e499a558519db3`；日志首行记录 |
| 配置 | F2 分支 `experiments/data-v2-abl-f2-localgate-b16-s42.yaml` |
| 本地原始 Run | `runs/data-v2-abl-f2-localgate-b16-s42/` |
| GPU / 环境 | RTX 5090，Python 3.12.3，PyTorch 2.12.1+cu130，Ultralytics 8.4.80 |
| model / pretrained | `yolo26m-p3-localgate-seg.yaml` / 官方 `yolo26m-seg.pt` |
| data / split | `yolo_data_v2_cloud.yaml` / Val |
| epochs / patience | 300 / 100（实际 204 轮早停） |
| batch / nbs / imgsz | 16 / 64 / 640 |
| optimizer / lr0 / lrf | AdamW / 0.001667 / 0.01 |
| momentum / weight_decay | 0.9 / 0.0005 |
| warmup_epochs / warmup_bias_lr / cos_lr | 3 / 0.1 / false |
| seed / deterministic / AMP | 42 / true / true |
| workers / cache | 8 / false |
| mask_ratio / mixup / copy_paste | 2 / 0 / 0.3 |
| mosaic / close_mosaic | 1.0 / 15 |
| degrees / flipud / fliplr / scale | 15 / 0.5 / 0.5 / 0.3 |

F2 与 000 的 `args.yaml` 差异仅为 model、pretrained 的路径表达、name、save_dir。训练配方逐项一致，
从官方预训练重新训练，`resume=false`。相对 Baseline 的唯一结构改动是 Neck 第 16 层
`C3k2 [256, True]` → `C3k2LocalGate [256, True]`，层号、层数、`Segment26` 的输入层 `[16, 19, 22]`
与检测 stride=8/16/32 均不变。相对原 F，本次只增加局部引导门控：门控卷积 weight/bias 初始为 0，
两个空间门初值恒为 1，因此训练起点与 F 等价。

## 结构与权重迁移核验

构造摘要与运行单一致：`YOLO26m-p3-localgate-seg summary: 339 layers, 27,039,656 parameters,
27,039,656 gradients, 132.7 GFLOPs`；融合后 `155 layers, 23,576,660 parameters, 122.0 GFLOPs`，
相对原 F 的 23,576,530 恰好多 130 个参数，与设计值一致。

日志出现两次权重迁移，均已核对：

| 阶段 | 日志行 | 说明 |
|---|---|---|
| 入口 `model.load(yolo26m-seg.pt)` | `Transferred 904/932 items` | 目标模型 932 键，未迁移的 28 项即模块自身张量（原 F 的 26 项 + 门控 2 项） |
| 训练器按 nc=2 重建模型后 | `Transferred 918/932 items` | 未迁移的 14 项为类别相关张量，属 80→2 类头重建，与 Baseline 相同 |

`best.pt` 复核：932 个参数与缓冲张量全部有限。第 16 层为 F2 模块，新增的两个入口都被训练激活：

| 张量 | 形状 | absmax | 非零元素 |
|---|---|---:|---|
| `strip_context.gate.weight` | (2, 64, 1, 1) | 0.311768 | 128 / 128 |
| `strip_context.gate.bias` | (2,) | 1.750000 | 2 / 2 |
| `strip_context.project.weight` | (256, 192, 1, 1) | 0.406494 | 49,152 / 49,152 |
| `strip_context.project.bias` | (256,) | 0.880859 | 256 / 256 |

门控卷积的初值为全零，训练后 weight 与 bias 全部非零，说明两个空间门已经偏离单位门、确实在调制
横纵条带分支。三条降维/条带分支的权重同样非零（reduce 0.395508、local 0.531250、horizontal
0.570801、vertical 0.623047 的 absmax）。因此本轮负结果不能归因于新分支保持零输出。

## 正式结果

总体使用 CSV 中 official fitness 最大轮次，与 `best.pt` 的 `train_metrics` 逐项核对一致
（fitness 0.79673、Box mAP50-95 0.44940、Mask mAP50 0.70155、Mask mAP50-95 0.34733）。

| 指标 | Baseline（216） | F2（104） | F2−Baseline |
|---|---:|---:|---:|
| Official fitness，CSV 两项之和 | 0.81977 | 0.79673 | **-0.02304** |
| Box P | 0.67918 | 0.68186 | +0.00268 |
| Box R | 0.64991 | 0.63885 | -0.01106 |
| Box mAP50 | 0.72980 | 0.71792 | -0.01188 |
| Box mAP50-95 | 0.45629 | 0.44940 | -0.00689 |
| Mask P | 0.64822 | 0.63879 | -0.00943 |
| Mask R | 0.67252 | 0.63051 | -0.04201 |
| Mask mAP50 | 0.71132 | 0.70155 | **-0.00977** |
| Mask mAP50-95 | 0.36348 | 0.34733 | **-0.01615** |

与 Baseline 相比，检测框与掩膜两类指标的 P/R/mAP 全部下降，工作点没有出现 G 那种"召回上升换精确率"
的位移，而是整体退到更差的水平。F2 的全程最高 Mask mAP50-95 与最高 fitness 都在 epoch 104，
改用单项选权重也不能改变本轮结论。

## 分类型 Val Mask

下表来自训练结束后重新加载并融合 `best.pt` 的 Val 日志，保留其三位精度；与上表的训练期精确指标
分别记录。

| 类别 | Images | Instances | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 117 | 557 | 0.676 | 0.638 | 0.718 | 0.449 | 0.641 | 0.625 | 0.702 | 0.347 |
| Rice leaffolder | 87 | 462 | 0.654 | 0.613 | 0.666 | 0.385 | 0.628 | 0.619 | 0.654 | 0.262 |
| Rice stemborers | 38 | 95 | 0.699 | 0.663 | 0.770 | 0.512 | 0.653 | 0.632 | 0.750 | 0.432 |

与 Baseline 及原 F 的 `best.pt` 复核值对比：

| 类别 | 指标 | Baseline | F | F2 | F2−Baseline |
|---|---|---:|---:|---:|---:|
| Rice leaffolder | Mask P | 0.630 | 0.638 | 0.628 | -0.002 |
| Rice leaffolder | Mask R | 0.656 | 0.610 | 0.619 | -0.037 |
| Rice leaffolder | Mask mAP50 | 0.677 | 0.658 | 0.654 | -0.023 |
| Rice leaffolder | Mask mAP50-95 | 0.266 | 0.262 | 0.262 | -0.004 |
| Rice stemborers | Mask P | 0.671 | 0.644 | 0.653 | -0.018 |
| Rice stemborers | Mask R | 0.687 | 0.705 | 0.632 | -0.055 |
| Rice stemborers | Mask mAP50 | 0.745 | 0.750 | 0.750 | +0.005 |
| Rice stemborers | Mask mAP50-95 | 0.461 | 0.455 | 0.432 | -0.029 |

门控没有带来任何类别选择性收益：卷叶螟 AP50 下降 0.023，钻心虫 AP50 与 Baseline 相同但 AP50-95
下降 0.029。相对原 F，卷叶螟 Mask P/R 小升、AP50 微降，钻心虫则从 F 的召回偏高转为召回偏低，
两类严格掩膜 AP 都低于 F。

## 收敛行为

| 统计量 | Baseline | F | F2 |
|---|---:|---:|---:|
| 前 5 个最高 official fitness 均值 | 0.808818 | 0.812830 | 0.784220 |
| 前 10 个最高 official fitness 均值 | 0.804628 | 0.810723 | 0.777970 |
| 前 10 个最高 Mask mAP50-95 均值 | 0.353278 | 0.353603 | 0.340845 |
| 全程最高 Mask mAP50-95 | 0.36348（216） | 0.35882（237） | 0.34733（104） |
| 最后一轮 Mask mAP50-95 | 0.34131（300） | 0.33796（300） | 0.32817（204） |
| epoch 105–204 单轮 Mask mAP50-95 变化绝对值中位数 | 0.00941 | — | 0.00975 |
| epoch 105–204 单轮 official fitness 变化绝对值中位数 | 0.01717 | — | 0.02182 |

F2 的 top-5、top-10 fitness 与 top-10 Mask 均值分别比 Baseline 低 0.024598、0.026658、0.012433，
负向不体现在单个轮次上；差额 0.01615 也远大于同窗口的单轮波动中位数 0.00975。因此本轮不属于
G 那种"落在噪声地板内、不可区分"的情形，而是全程更低。

早停发生在第 204 轮，best epoch 104 之后连续 100 轮没有刷新 fitness。这个峰值不是孤立尖峰：
其后第 2 至第 4 高的 Mask mAP50-95 为 0.34470（epoch 167）、0.34262（epoch 197）、0.34168
（epoch 160），前 10 高全部落在 epoch 104–197 之间，说明 epoch 104 之后仍有若干轮处在接近峰值的
水平，只是没有超过它。峰值出现在第 104 轮、且后段没有二次上升，是早停在本轮成立的原因；Baseline
与 F 的峰值分别出现在 216 与 237，需要跑满 300 轮。

## 运行成本

| 运行项 | Baseline（300 轮） | F2（204 轮） |
|---|---:|---:|
| 实际 / 计划 epoch | 300 / 300 | 204 / 300（早停） |
| Best epoch | 216 | 104 |
| 总训练时间 | 4416.89 s（1.227 h） | 3025.20 s（0.841 h） |
| epoch 50–204 中位单轮时间 | 14.69 s | 14.76 s |
| 峰值 GPU_mem | 15.6 GB | 15.6 GB |
| 融合后结构 | 149 层 / 23.5086 M 参数 / 121.2 GFLOPs | 155 层 / 23.5767 M 参数 / 122.0 GFLOPs |
| Val loss NaN 单元 | 10（epoch 7、8、36） | 6（epoch 2、3） |
| 全部权重张量有限 | 是 | 是 |

相对 Baseline 融合模型，F2 增加 130 个参数（+0.0006%）与约 0.002 GFLOPs，中位单轮时间多 0.07 s
（0.5%），显存同档。总训练时间短是早停的结果，不是结构更快。本轮退化不能由参数量或算力增加解释。

## 机制判定

设计意图是让局部外观决定横纵条带分支的增益（门范围 (0,2)，可同时减弱或分别增强），从而在保留
召回的同时压低相似稻叶纹理造成的误检。权重核验确认门控已经偏离单位门、被训练激活，但实测结果是
Mask R 下降 0.042、Mask P 下降 0.009，两类严格掩膜 AP 都低于 Baseline 与原 F：门控既没有换来
精确率，又损失了召回。**预期被否**，且是"有学习、无收益"的形态，不是模块未生效。

把 P3 条带这一路的三次训练放在一起，卷叶螟 Mask mAP50-95 依次为 Baseline 0.266、F 0.262、
F2 0.262、G 0.266。八次正式训练（含 A1、A2、B、DRB/SCSA、E）中该指标始终落在 0.245～0.266
的窄带内且无一超过 Baseline，与 G 记录中的结论一致：当前瓶颈不在 P3 的感受野形状或上下文聚合方式上。
F2 是本方向里唯一触发早停的一次，说明在这条方向上增加门控这类额外自由度会让收敛更早停滞，而不是
提供新的可分信息。

## 结论

- F2 未通过门控。Val Mask mAP50-95=0.34733，比 Baseline 低 0.01615，比原 F 低 0.01149；
  检测框与掩膜两类 P/R/mAP 全部低于 Baseline。
- 收敛全程更低：top-10 fitness 均值低 0.026658，差额大于同窗口单轮波动中位数 0.00975，
  不属于与 Baseline 不可区分的情形。204 轮早停，best epoch 104。
- 第 7 节机制预期被否。门控卷积与投影权重全部非零、门已偏离单位门，属"有学习、无收益"。
- F2 不进入 D1+F2，也不进入 DSS 相关组合；表 1 的对照基准保持 `data-v2-abl-000-y26m-b16-s42`。
- P3 条带/朝向方向（F、F2、G）至此三条独立负结果，F2 是该方向的最差一次，后续不再加码。
- 本轮属结构消融，结果登记进 `comparison.csv` 与实验总表表 4、表 21、表 17；不进表 2。
- 只有 seed=42 单次证据，未做多 seed 复验；结论限定为"本轮实现未见收益"。

## 异常与限制

- epoch 2、3 各有 3 个 Val loss 字段为 NaN（共 6 个单元），少于 Baseline 的 10 个；训练损失、
  全部 P/R/mAP 指标、best epoch 以及两份权重的全部张量均为有限值，不属于训练崩溃。
- 未做原图尺寸与小目标配对专项评估。F2 在原 F 的 CPU 配对协议下没有启动该专项的收益依据，
  本轮正式主指标比 F 低 0.01149，补充协议不作为成功标准。
- 未使用 Test。Test 已在最终模型冻结后统一完成（见表 15），本轮结构不参与。
- F2 是"F + 局部引导门控"这一具体实现，不能把本轮结论外推为所有门控式条带结构都无效。

## 证据

| 内容 | 位置 |
|---|---|
| 完整 Run | `runs/data-v2-abl-f2-localgate-b16-s42/` |
| 实际参数 | `runs/data-v2-abl-f2-localgate-b16-s42/args.yaml` |
| 逐 epoch 指标 | `runs/data-v2-abl-f2-localgate-b16-s42/results.csv` |
| 训练日志 | `runs/data-v2-abl-f2-localgate-b16-s42/train_data-v2-abl-f2-localgate-b16-s42.log` |
| 训练曲线 | `runs/data-v2-abl-f2-localgate-b16-s42/results.png` |
| Mask PR 曲线 | `runs/data-v2-abl-f2-localgate-b16-s42/MaskPR_curve.png` |
| 原理与实现 | `knowledge/改进F2-局部引导条带门控原理与实现.md` |
| Baseline 对照 | `experiment_records/runs/data-v2-abl-000-y26m-b16-s42.md` |
| 原 F 对照 | `experiment_records/runs/data-v2-abl-f-p3strip-b16-s42.md` |
