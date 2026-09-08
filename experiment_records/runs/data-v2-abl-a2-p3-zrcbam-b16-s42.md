# Run 记录：data-v2-abl-a2-p3-zrcbam-b16-s42

## 基本信息

| 项目 | 内容 |
|---|---|
| 状态 | 已完成，epoch 276 正常早停；相对正式 `000` 近似持平，未通过组合保留门控 |
| 实验目的 | 在冻结训练配方下，只在 Backbone P3 加入零初始化加法残差 CBAM，检验 A1 失败后新的注意力候选 |
| 配置文件 | `experiments/data-v2-abl-a2-p3-zrcbam-b16-s42.yaml` |
| 配置实现分支 / commit | `feature/data-v2-abl-a-attention` / `a38aabfa2caf36979646d20ef2bb657aad1347ae`；训练日志未记录 Git commit |
| 云端原始目录 | `/root/yolo_plus/runs/data-v2-abl-a2-p3-zrcbam-b16-s42/` |
| 本地保存目录 | `runs/data-v2-abl-a2-p3-zrcbam-b16-s42/` |
| 数据 / split | `rice-pest-data-v2` / Val（117 张图、557 个实例） |
| GPU / 环境 | RTX 5090；Python 3.12.3；PyTorch 2.12.1+cu130；Ultralytics 8.4.80 |
| 权重保存日期 | 2026-09-04（`best.pt` 元数据）；时间字段不含时区 |
| 开始 / 结束时间 | 日志未写墙钟时间，不登记推算的起止时刻 |
| 训练日志 | `runs/data-v2-abl-a2-p3-zrcbam-b16-s42/train_data-v2-abl-a2-p3-zrcbam-b16-s42.log` |

## 实验参数

正式配置中 33 个 `train` 字段与 `args.yaml` 一致，只有 `project` 被解析为云端绝对路径。
相对正式 `000`，全部实际参数只有 `model`、`pretrained`、`name`、`save_dir` 四项差异：
模型由官方权重直接建模改为 A2 YAML 建模后迁移同一份官方权重，Run 名称和保存目录随之改变。
数据、损失和冻结训练参数一致。

| 参数 | 实际值 |
|---|---:|
| model | `ultralytics-main/ultralytics/cfg/models/26/yolo26m-p3-zrcbam-seg.yaml` |
| pretrained | 官方 `yolo26m-seg.pt` |
| data | `experiments/yolo_data_v2_cloud.yaml` |
| epochs / patience | 300 / 100 |
| batch / nbs / accumulate | 16 / 64 / 4 |
| imgsz / device / workers | 640 / 0 / 8 |
| optimizer | AdamW（显式） |
| lr0 / lrf | 0.001667 / 0.01 |
| momentum / weight_decay | 0.9 / 0.0005 |
| warmup_epochs / warmup_bias_lr | 3 / 0.1 |
| seed / deterministic | 42 / true |
| amp / cache / cos_lr | true / false / false |
| mask_ratio | 2 |
| mosaic / close_mosaic | 1.0 / 15 |
| mixup / copy_paste | 0.0 / 0.3 |
| degrees / translate / scale | 15 / 0.1 / 0.3 |
| flipud / fliplr | 0.5 / 0.5 |
| 源码改进 | Backbone 第 4 层 `C3k2` 输出后加入 ZR-CBAM，仅 P3/8；P4 保持普通 `C3k2` |
| ZR-CBAM | `Y=X+β×CBAM(X)`；reduction=16；kernel=7；可学习标量 β 初值 0 |
| 预训练迁移 | 建模阶段 904 / 908 项；Trainer 阶段 894 / 908 项 |
| AMP 检查 | 通过 |
| `best.pt` 选择 | Val official fitness = Box mAP50-95 + Mask mAP50-95 |
| 论文主指标 | Val Mask mAP50-95；同时报告 Mask mAP50 |

## 最佳权重结果

`results.csv` 的官方 fitness 最大值位于 epoch 176，与日志报告的最佳轮次及 `best.pt`
的全部 `train_metrics` 一致。连续 100 个 epoch 未改善后，训练在 epoch 276 按
`patience=100` 正常结束；`last.pt` 的 `train_metrics` 对应 epoch 276。

| 指标 | 结果 |
|---|---:|
| 计划 / 实际 epoch | 300 / 276 |
| 最佳 epoch | 176 |
| Official fitness | 0.82146 |
| Val Box P / R | 0.67696 / 0.68711 |
| Val Box F1 | 0.68200 |
| Val Box mAP50 | 0.73641 |
| Val Box mAP50-95 | 0.46037 |
| Val Mask P / R | 0.66346 / 0.67319 |
| Val Mask F1 | 0.66829 |
| Val Mask mAP50 | 0.71506 |
| Val Mask mAP50-95 | 0.36109 |
| 到最佳轮次累计时间 | 2602.77 s（0.723 h） |
| 总训练时间 | 4082.59 s（1.134 h） |
| 日志 GPU_mem 峰值 | 15.6 GB（日志显示精度 0.1 GB） |
| 融合后模型 | 154 layers / 23.541877 M Params / 121.2 GFLOPs（日志精度） |
| `best.pt` 大小 | 54.587549 MB（52.059 MiB） |
| best.pt 复核速度 | 1.7 ms inference + 0.7 ms postprocess / image（训练结束 Val；非统一 batch=1 基准测速） |

### 训练结束 best.pt 复核

训练结束后重新加载并融合 `best.pt` 的 Val 日志只保留 3 位小数，逐类别结果如下。

| 类别 | Images | Instances | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 117 | 557 | 0.677 | 0.690 | 0.737 | 0.460 | 0.663 | 0.675 | 0.714 | 0.362 |
| Rice leaffolder | 87 | 462 | 0.616 | 0.669 | 0.682 | 0.386 | 0.598 | 0.649 | 0.655 | 0.262 |
| Rice stemborers | 38 | 95 | 0.738 | 0.711 | 0.792 | 0.534 | 0.727 | 0.701 | 0.773 | 0.463 |

总体正式汇总使用用于选择 `best.pt` 的 epoch 176 精确值，逐类别指标使用日志中的
`best.pt` 复核值。融合后复核的总体数值存在轻微差异，保持与 `000`、A1 相同的登记口径。

## 与正式 000、A1 的比较

三个 Run 采用同一数据、RTX 5090 环境、冻结配方和 seed=42。A2 相对 `000` 的研究变量为
P3 ZR-CBAM；相对 A1 同时改变了插入位置和残差形式，因此不能把 A2 与 A1 的差值归因于
单独移除 P4 或单独改变初始化。

| 指标 | 000 Baseline | A1 SR-CBAM | A2 P3 ZR-CBAM | A2−000 | A2−A1 |
|---|---:|---:|---:|---:|---:|
| Official fitness | 0.81977 | 0.80644 | 0.82146 | +0.00169 | +0.01502 |
| Box mAP50 | 0.72980 | 0.72641 | 0.73641 | +0.00661 | +0.01000 |
| Box mAP50-95 | 0.45629 | 0.45547 | 0.46037 | +0.00408 | +0.00490 |
| Mask P | 0.64822 | 0.69962 | 0.66346 | +0.01524 | -0.03616 |
| Mask R | 0.67252 | 0.64107 | 0.67319 | +0.00067 | +0.03212 |
| Mask F1 | 0.66015 | 0.66907 | 0.66829 | +0.00814 | -0.00078 |
| Mask mAP50 | 0.71132 | 0.70884 | 0.71506 | +0.00374 | +0.00622 |
| Mask mAP50-95 | 0.36348 | 0.35097 | 0.36109 | -0.00239 | +0.01012 |
| Top-5 official fitness 均值 | 0.80882 | 0.80225 | 0.80978 | +0.00096 | +0.00753 |
| Top-10 official fitness 均值 | 0.80463 | 0.79979 | 0.80685 | +0.00222 | +0.00706 |

A2 相对 A1 恢复了总体 AP，但相对 `000` 的两项 Mask AP 和 official fitness 差值绝对值均小于 0.01。official fitness 的微小
提高来自 Box mAP50-95；论文主指标 Mask mAP50-95 仍低于 `000`。top-5 / top-10 fitness
只有约 0.001 / 0.002 的微小增加，支持近似持平的判断，不能据此宣称稳定有效。

### 分类型变化

逐类别差值均来自训练结束 `best.pt` 复核日志，精度为 3 位小数。

| 类别 | 指标 | 000 Baseline | A1 SR-CBAM | A2 P3 ZR-CBAM | A2−000 |
|---|---|---:|---:|---:|---:|
| Rice leaffolder | Mask P | 0.630 | 0.676 | 0.598 | -0.032 |
| Rice leaffolder | Mask R | 0.656 | 0.582 | 0.649 | -0.007 |
| Rice leaffolder | Mask mAP50 | 0.677 | 0.662 | 0.655 | -0.022 |
| Rice leaffolder | Mask mAP50-95 | 0.266 | 0.262 | 0.262 | -0.004 |
| Rice stemborers | Mask P | 0.671 | 0.724 | 0.727 | +0.056 |
| Rice stemborers | Mask R | 0.687 | 0.690 | 0.701 | +0.014 |
| Rice stemborers | Mask mAP50 | 0.745 | 0.753 | 0.773 | +0.028 |
| Rice stemborers | Mask mAP50-95 | 0.461 | 0.438 | 0.463 | +0.002 |

总体 Mask mAP50 的小幅增加主要来自钻心虫，卷叶螟的四项 Mask 指标均低于 `000`，
其中 mAP50 下降 0.022。当前没有证据说明 A2 改善了卷叶螟分割或小目标表现。

## 资源、收敛与限制

- `best.pt` 中 P3 的可学习 β 为 0.481689453125，`last.pt` 为 0.5107421875（`model.4.zrcbam.residual_scale`），均偏离初值 0，说明注意力分支参与了学习；这不构成其提高分割精度的证据。
- A2 fused Params 为 23,541,877，较 `000` 增加 32,867（约 0.140%）；既有结构测算 GFLOPs@640 为 121.231795，较 `000` 增加 0.060646。训练结束融合模型日志与此规模一致。
- 总用时较 `000` 少 334.30 s（7.57%），但 A2 只运行 276 epoch；按实际轮数平均为 14.792 s/epoch，`000` 为 14.723 s/epoch，不能把早停后的较短总时长解释为单轮加速。
- 日志显存峰值与 `000` 同为 15.6 GB；当前 Val 复核速度不替代统一 batch=1 FP16 效率评估。
- `results.csv` 包含连续 epoch 1–276，无重复或缺轮；epoch 276 的 fitness 为 0.79412、Mask mAP50 / mAP50-95 为 0.68557 / 0.34141，正式比较使用 epoch 176。
- 首 10 到末 10 epoch 的训练损失均下降：box -36.55%、seg -38.91%、cls -56.67%、dfl -43.54%、sem -58.29%。由于在 epoch 276 早停，未进入计划中的 epoch 286 关闭 Mosaic 阶段；这是冻结 EarlyStopping 规则产生的实际结果。
- epoch 3 的 Val box/seg/cls/dfl loss 为 NaN；所有训练损失、P/R/mAP、学习率、最佳轮次指标及两份权重的全部状态张量均为有限值。日志无 OOM、Traceback、RuntimeError 或 Exception，AMP 检查和训练结束 `best.pt` 验证均完成。
- train / Val 均为 0 corrupt；与 `000`、A1 相同，`Rice_leaffolder_3x3_p0020_r0c2.jpg`、`Rice_leaffolder_3x3_p0032_r1c1.jpg` 各移除 1 个重复标签。
- 本地 Run 共 27 个文件，包含参数、完整 epoch CSV、训练日志、曲线、混淆矩阵、预测图以及 `best.pt` / `last.pt`；日志正常完成验证和保存。
- 两份权重已完成 optimizer strip，因此 checkpoint 的 `epoch=-1`、`best_fitness=None`；实际最佳轮次以 CSV、日志及 `train_metrics` 的一致性确认。保存时间字段分别为 `2026-09-04T16:53:32.985349` 和 `2026-09-04T17:18:12.834788`，未包含时区。
- 逐类别指标只有日志中的 3 位小数；当前仅有 seed=42，未进行多 seed 稳定性复验或小目标专项评估。

## 结论与下一步

- A2 相对 A1 明显恢复，但相对正式 `000` 属于近似持平，主指标 Mask mAP50-95 和卷叶螟 Mask AP 没有提高。
- 按预设门控，A2 未获得进入组合所需的明确总体收益或目标专项收益，保留本次结果但不进入 A+B、A+C 或 A+B+C。A1 的负结果保持独立记录。
- 当前注意力方向暂停组合使用；独立 B：Dice 的完整 Run 已回传并核验，本次实现未通过门控。下一步独立准备 C：P2Head；Test 保留到最终方案冻结后统一评估。阶段结论见 [A、B 改进完成总结](../data-v2-ab-summary.md)。

## 证据文件

- `runs/data-v2-abl-a2-p3-zrcbam-b16-s42/args.yaml`
- `runs/data-v2-abl-a2-p3-zrcbam-b16-s42/results.csv`
- `runs/data-v2-abl-a2-p3-zrcbam-b16-s42/train_data-v2-abl-a2-p3-zrcbam-b16-s42.log`
- `runs/data-v2-abl-a2-p3-zrcbam-b16-s42/weights/best.pt`
- `runs/data-v2-abl-a2-p3-zrcbam-b16-s42/weights/last.pt`
- `experiment_records/runs/data-v2-abl-000-y26m-b16-s42.md`
- `experiment_records/runs/data-v2-abl-100-srcbam-b16-s42.md`
- `experiment_records/data-v2-source-ablation-plan.md`
