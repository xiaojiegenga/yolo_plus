# Run 记录：data-v2-abl-100-srcbam-b16-s42

## 基本信息

| 项目 | 内容 |
|---|---|
| 状态 | 已完成并核验；改进 A 未通过单模块门控，不进入后续组合 |
| 实验目的 | 在冻结训练配方下，只在 YOLO26m-seg Backbone P3/P4 加入 SR-CBAM，检验注意力机制能否改善复杂水稻背景中的实例分割 |
| 配置文件 | `experiments/data-v2-abl-100-srcbam-b16-s42.yaml` |
| 代码分支 / commit | `feature/data-v2-abl-a-attention` / `9d0c479030ea8124fd6260592d3e31ce855cebd3` |
| 云端原始目录 | `/root/yolo_plus/runs/data-v2-abl-100-srcbam-b16-s42/` |
| 回传归档 | `exports/data-v2-abl-100-srcbam-b16-s42.zip` |
| 本地保存目录 | `runs/data-v2-abl-100-srcbam-b16-s42/` |
| 数据 / split | `rice-pest-data-v2` / Val（117 张图、557 个实例） |
| GPU / 环境 | RTX 5090；Python 3.12.3；PyTorch 2.12.1+cu130；Ultralytics 8.4.80 |
| 训练日期 | 2026-09-03（`best.pt` 元数据） |
| 训练日志 | `runs/data-v2-abl-100-srcbam-b16-s42/train_data-v2-abl-100-srcbam-b16-s42.log` |

## 实验参数

实际解析参数来自 `runs/data-v2-abl-100-srcbam-b16-s42/args.yaml`。除模型 YAML 和预训练
迁移外，数据、训练参数、硬件与正式 `000 Baseline` 保持一致。

| 参数 | 实际值 |
|---|---:|
| model | `ultralytics-main/ultralytics/cfg/models/26/yolo26m-srcbam-seg.yaml` |
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
| 源码改进 | Backbone 第 4、6 层 `C3k2` 输出后加入 SR-CBAM（P3/8、P4/16） |
| SR-CBAM | reduction=16；kernel=7；每层独立可学习残差混合系数，α 初值 0.1 |
| 预训练迁移 | 建模阶段 904 / 912 项；Trainer 阶段 898 / 912 项 |
| AMP 检查 | 通过 |
| `best.pt` 选择 | Val official fitness = Box mAP50-95 + Mask mAP50-95 |
| 论文主指标 | Val Mask mAP50-95；同时报告 Mask mAP50 |

## 最佳权重结果

`best.pt` 元数据与 `results.csv` 一致，最佳轮次为 epoch 235。

| 指标 | 结果 |
|---|---:|
| 计划 / 实际 epoch | 300 / 300 |
| 最佳 epoch | 235 |
| Official fitness | 0.80644 |
| Val Box P / R | 0.72298 / 0.64998 |
| Val Box F1 | 0.68454 |
| Val Box mAP50 | 0.72641 |
| Val Box mAP50-95 | 0.45547 |
| Val Mask P / R | 0.69962 / 0.64107 |
| Val Mask F1 | 0.66907 |
| Val Mask mAP50 | 0.70884 |
| Val Mask mAP50-95 | 0.35097 |
| 到最佳轮次累计时间 | 3672.53 s（1.020 h） |
| 总训练时间 | 4645.58 s（1.290 h） |
| 日志 GPU_mem 峰值 | 15.1 GB（日志显示精度 0.1 GB） |
| 融合后模型 | 159 layers / 23.574744 M Params / 121.286586 GFLOPs |
| `best.pt` 大小 | 54.661 MB（52.129 MiB） |
| best.pt 复核速度 | 1.7 ms inference + 0.7 ms postprocess / image（训练结束 Val；非统一 batch=1 基准测速） |

### 训练结束 best.pt 复核

训练结束后重新加载并融合 `best.pt` 的 Val 日志只保留 3 位小数，逐类别结果如下。

| 类别 | Images | Instances | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 117 | 557 | 0.722 | 0.649 | 0.726 | 0.455 | 0.700 | 0.636 | 0.707 | 0.350 |
| Rice leaffolder | 87 | 462 | 0.699 | 0.589 | 0.691 | 0.403 | 0.676 | 0.582 | 0.662 | 0.262 |
| Rice stemborers | 38 | 95 | 0.745 | 0.709 | 0.761 | 0.507 | 0.724 | 0.690 | 0.753 | 0.438 |

总体正式汇总使用用于选择 `best.pt` 的 epoch 235 精确值，逐类别指标使用日志中的
`best.pt` 复核值。

## 与正式 000 Baseline 的严格配对比较

两次训练的数据、冻结训练参数、RTX 5090 环境和 seed=42 相同，唯一研究变量为 P3/P4
SR-CBAM。正值表示 SR-CBAM 高于 Baseline。

| 指标 | 000 Baseline | 100 SR-CBAM | 差值 |
|---|---:|---:|---:|
| Official fitness | 0.81977 | 0.80644 | -0.01333 |
| Box mAP50 | 0.72980 | 0.72641 | -0.00339 |
| Box mAP50-95 | 0.45629 | 0.45547 | -0.00082 |
| Mask P | 0.64822 | 0.69962 | +0.05140 |
| Mask R | 0.67252 | 0.64107 | -0.03145 |
| Mask F1 | 0.66015 | 0.66907 | +0.00892 |
| Mask mAP50 | 0.71132 | 0.70884 | -0.00248 |
| Mask mAP50-95 | 0.36348 | 0.35097 | -0.01251 |
| Top-5 official fitness 均值 | 0.80882 | 0.80225 | -0.00656 |
| Top-10 official fitness 均值 | 0.80463 | 0.79979 | -0.00484 |

SR-CBAM 提高了当前工作点的 Mask 精确率和 F1，但召回率下降，Mask mAP50 没有提高，
Mask mAP50-95 下降 0.01251。top-5 与 top-10 fitness 均低于 Baseline，负差异不是单个
最佳 epoch 的孤立波动。

### 分类型变化

逐类别差值基于两次训练结束的 `best.pt` 复核日志，精度为 3 位小数。

| 类别 | 指标 | 000 Baseline | 100 SR-CBAM | 差值 |
|---|---|---:|---:|---:|
| Rice leaffolder | Mask P | 0.630 | 0.676 | +0.046 |
| Rice leaffolder | Mask R | 0.656 | 0.582 | -0.074 |
| Rice leaffolder | Mask mAP50 | 0.677 | 0.662 | -0.015 |
| Rice leaffolder | Mask mAP50-95 | 0.266 | 0.262 | -0.004 |
| Rice stemborers | Mask P | 0.671 | 0.724 | +0.053 |
| Rice stemborers | Mask R | 0.687 | 0.690 | +0.003 |
| Rice stemborers | Mask mAP50 | 0.745 | 0.753 | +0.008 |
| Rice stemborers | Mask mAP50-95 | 0.461 | 0.438 | -0.023 |

卷叶螟的精确率提高，但召回率下降 0.074，mAP50 下降 0.015；钻心虫 mAP50 小幅提高
0.008，但严格 mAP50-95 下降 0.023。注意力机制没有改善本实验最关注的卷叶螟分割表现。

## 注意力门与资源代价

`best.pt` 中两处可学习残差混合系数为：

| 插入位置 | α 初值 | best.pt α |
|---|---:|---:|
| P3/8（Backbone 第 4 层） | 0.100 | 0.16722 |
| P4/16（Backbone 第 6 层） | 0.100 | 0.09860 |

P3 门明显偏离初值，说明注意力分支参与了学习；P4 门基本保持初值。该现象只支持“P3
比 P4 使用得更多”的结构诊断，不能据此把 P3 单独认定为有效模块。

| 资源指标 | 000 Baseline | 100 SR-CBAM | 差值 |
|---|---:|---:|---:|
| Params | 23.509010 M | 23.574744 M | +0.065734 M（+0.280%） |
| GFLOPs@640 | 121.171149 | 121.286586 | +0.115437（+0.095%） |
| best.pt | 54.524 MB | 54.661 MB | +0.137 MB |
| 峰值显存 | 15.6 GB | 15.1 GB | -0.5 GB（日志精度有限，不作为显存降低结论） |
| 训练时间 | 1.227 h | 1.290 h | +0.064 h（+5.18%） |

模型计算量只小幅增加，但没有获得与代价相匹配的精度收益。

## 收敛、异常与限制

- 首 10 到末 10 epoch 的训练损失均明显下降：box -33.28%、seg -41.51%、cls -52.72%、dfl -29.53%、sem -70.06%，训练正常收敛。
- epoch 286 起关闭 Mosaic，与 `close_mosaic=15` 一致；训练损失的末段阶跃属于增强策略切换。
- epoch 300 的 fitness 为 0.79625，低于 epoch 235，但差距小于 Baseline 的末轮回落；正式比较仍必须使用 `best.pt`。
- epoch 3 的 Val box/seg/cls/dfl loss 为 NaN；训练损失、P/R/mAP、后续 Val、权重和最佳轮次均为有限值，日志没有 OOM、Traceback、RuntimeError 或 Exception。
- train / Val 均为 0 corrupt；与 Baseline 相同的两个训练图像各移除 1 个重复标签，数据处理口径一致。
- 逐类别结果只以日志中的 3 位小数保存；若最终论文需要更高精度逐类别指标，应对冻结模型统一独立 Val。
- 当前比较只有 seed=42，门控结论属于本项目预先规定的工程决策，不是统计显著性结论。

## 结论与下一步

- 当前 P3/P4 SR-CBAM 未通过单模块门控：总体 official fitness、Mask mAP50 和主指标 Mask mAP50-95 均未提高，目标类别卷叶螟的 Recall 与两项 Mask AP 也下降。
- 改进 A 按负结果保留记录，但不进入 A+B、A+C 或 A+B+C 组合，不为凑完整矩阵继续长训。
- 下一步冻结改进 B 的 `BCE + λ × Dice` 唯一公式、λ、smooth/epsilon 和聚合方式；B 必须从正式 Baseline 提交 `c0f4f35` 建立兄弟分支，不能从当前 A 分支继续开发。
- 当前不使用 Test，不启动下一次 300 epoch 长训。

## 证据文件

- `runs/data-v2-abl-100-srcbam-b16-s42/train_data-v2-abl-100-srcbam-b16-s42.log`
- `runs/data-v2-abl-100-srcbam-b16-s42/args.yaml`
- `runs/data-v2-abl-100-srcbam-b16-s42/results.csv`
- `runs/data-v2-abl-100-srcbam-b16-s42/results.png`
- `runs/data-v2-abl-100-srcbam-b16-s42/MaskPR_curve.png`
- `runs/data-v2-abl-100-srcbam-b16-s42/confusion_matrix_normalized.png`
- `runs/data-v2-abl-100-srcbam-b16-s42/weights/best.pt`
