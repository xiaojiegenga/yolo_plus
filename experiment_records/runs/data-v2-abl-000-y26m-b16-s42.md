# Run 记录：data-v2-abl-000-y26m-b16-s42

## 基本信息

| 项目 | 内容 |
|---|---|
| 状态 | 已完成并核验，可作为正式消融 `000 Baseline` |
| 实验目的 | 在冻结训练配方下重新训练不含 Attention、Dice、P2Head 的官方 YOLO26m-seg，建立后续源码消融的同环境基准 |
| 配置文件 | `experiments/data-v2-abl-000-y26m-b16-s42.yaml` |
| 代码分支 / commit | `cloud/data-v2-5090` / `1c63ee418f20cd4a5fdcd62b33547a40d24b69d4` |
| 云端原始目录 | `/root/yolo_plus/runs/data-v2-abl-000-y26m-b16-s42/` |
| 回传归档 | `exports/data-v2-abl-000-y26m-b16-s42.zip` |
| 本地保存目录 | `runs/data-v2-abl-000-y26m-b16-s42/` |
| 数据 / split | `rice-pest-data-v2` / Val（117 张图、557 个实例） |
| GPU / 环境 | RTX 5090；Python 3.12.3；PyTorch 2.12.1+cu130；Ultralytics 8.4.80 |
| 训练日期 | 2026-09-02（`best.pt` 元数据）；日志未写墙钟时间，不登记推算的起止时刻 |
| 训练日志 | `runs/data-v2-abl-000-y26m-b16-s42/train_data-v2-abl-000-y26m-b16-s42.log` |

## 实验参数

实际解析参数来自 `runs/data-v2-abl-000-y26m-b16-s42/args.yaml`。正式配置中的 33 个
`train` 字段均与实际值一致；`model`、`data`、`project` 仅被云端入口解析成绝对路径。

| 参数 | 实际值 |
|---|---:|
| model | `yolo26m-seg.pt`（官方预训练权重） |
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
| 源码改进 | 无；A/B/C 全部关闭 |
| 预训练迁移 | 890 / 904 项 |
| AMP 检查 | 通过 |
| `best.pt` 选择 | Val official fitness = Box mAP50-95 + Mask mAP50-95 |
| 论文主指标 | Val Mask mAP50-95；同时报告 Mask mAP50 |

## 最佳权重结果

`best.pt` 与 `results.csv` 一致，最佳轮次为 epoch 216。

| 指标 | 结果 |
|---|---:|
| 计划 / 实际 epoch | 300 / 300 |
| 最佳 epoch | 216 |
| Official fitness | 0.81977 |
| Val Box P / R | 0.67918 / 0.64991 |
| Val Box F1 | 0.66422 |
| Val Box mAP50 | 0.72980 |
| Val Box mAP50-95 | 0.45629 |
| Val Mask P / R | 0.64822 / 0.67252 |
| Val Mask F1 | 0.66015 |
| Val Mask mAP50 | 0.71132 |
| Val Mask mAP50-95 | 0.36348 |
| 到最佳轮次累计时间 | 3183.75 s（0.884 h） |
| 总训练时间 | 4416.89 s（1.227 h） |
| 日志 GPU_mem 峰值 | 15.6 GB（日志显示精度 0.1 GB） |
| 融合后模型 | 149 layers / 23.509 M Params / 121.2 GFLOPs |
| `best.pt` 大小 | 54.524 MB（52.0 MiB） |
| best.pt 复核速度 | 1.3 ms inference + 0.7 ms postprocess / image（训练结束 Val；非统一 batch=1 基准测速） |

### 训练结束 best.pt 复核

日志在训练结束后重新加载并融合 `best.pt`，得到以下 Val 输出。日志只保留 3 位小数。

| 类别 | Images | Instances | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 117 | 557 | 0.679 | 0.650 | 0.730 | 0.456 | 0.651 | 0.672 | 0.711 | 0.364 |
| Rice leaffolder | 87 | 462 | 0.673 | 0.647 | 0.707 | 0.404 | 0.630 | 0.656 | 0.677 | 0.266 |
| Rice stemborers | 38 | 95 | 0.684 | 0.653 | 0.752 | 0.508 | 0.671 | 0.687 | 0.745 | 0.461 |

训练阶段 `results.csv` 的 epoch 216 行及 checkpoint `train_metrics` 保存了更高精度的总体
结果，训练结束复核日志则提供逐类别结果。两次验证的总体 Mask P 分别为 0.64822 和 0.651，
存在 0.00278 的轻微差异；总体正式汇总沿用用于选择 `best.pt` 的 epoch 216 精确值，逐类别
指标使用日志中的 `best.pt` 复核值。

## 收敛与稳定性分析

训练损失整体持续下降，后期进入平台区。首 10 个 epoch 与末 10 个 epoch 的均值如下：

| 训练损失 | 首 10 epoch 均值 | 末 10 epoch 均值 | 变化 |
|---|---:|---:|---:|
| box_loss | 1.66984 | 1.11455 | -33.25% |
| seg_loss | 1.62770 | 0.95962 | -41.04% |
| cls_loss | 1.90525 | 0.89108 | -53.23% |
| dfl_loss | 0.00704 | 0.00499 | -29.10% |
| sem_loss | 1.78895 | 0.53801 | -69.93% |

最佳轮次附近不是单点异常：official fitness 的 top-5 / top-10 均值分别为 0.80882 和
0.80463。epoch 300 的 fitness 已回落到 0.78002，Mask mAP50 / mAP50-95 回落到
0.68074 / 0.34131，因此后续比较必须使用 `best.pt`，不能使用 `last.pt`。

## 与冻结配方 P2 的复现对照

正式 `000` 与参数优化 P2 `data-v2-tune-mr2-nomix-e300-b16-s42` 的实际参数除 Run 名称和
保存目录外一致。两份 `results.csv` 在全部 300 个 epoch 上，除累计 `time` 外的其余 22 列
逐值完全一致；正式 Run 仅比 P2 多用 6.31 s（0.143%）。两次训练的 checkpoint 日期和 Git
commit 不同，说明本次是冻结配方的独立确定性重跑，不是旧 Run 改名。

该结果证明 seed=42、`deterministic=true` 和当前环境下训练轨迹可精确复现，可将本 Run
冻结为后续消融的 `000 Baseline`。它仍是同一 seed 的复现，不替代多 seed 均值与标准差。

## 异常与限制

- `results.csv` 中 epoch 7、8 的 Val box/seg/cls/dfl loss，以及 epoch 36 的 Val seg/cls loss 为 NaN；训练日志本身未打印 NaN，训练损失、全部 P/R/mAP、学习率和最佳权重均为有限值，与 P2 的同轮现象一致，不影响本次最佳结果登记。
- 数据加载器报告 train / Val 均为 0 corrupt；训练集有两个文件各移除 1 个重复标签：`Rice_leaffolder_3x3_p0020_r0c2.jpg`、`Rice_leaffolder_3x3_p0032_r1c1.jpg`。这是实际数据加载行为，后续消融必须保持同一数据缓存与处理口径。
- AMP 检查通过，训练完成全部 300 epoch，未触发 EarlyStopping；日志中没有 OOM、Error、Exception 或 Traceback。
- 原 ZIP 的 28 项标准 Run 产物齐全；单独补下载日志后，本地 Run 目录共 29 个文件，包含 `args.yaml`、`results.csv`、曲线、混淆矩阵、可视化、两份权重和完整训练日志。
- 逐类别结果只以日志中的 3 位小数保存；若论文最终需要更高精度逐类别指标，应在所有冻结模型上按同一命令统一独立 Val。
- 当前正式实验固定 seed=42，不能据此估计随机波动范围。

## 结论与下一步

- 本 Run 核验通过，正式承担消融 `000`、YOLO26m 同环境 Baseline 和跨模型对比中的 YOLO26m 基准行。
- 下一步只冻结 B：Dice 的唯一公式、权重和聚合方式，再实现源码与正式配置；在新 commit 上先做 10 epoch 预检。
- 当前不启动下一次 300 epoch 长训，不使用 Test 调整 Dice 或其他模块。

## 证据文件

- `runs/data-v2-abl-000-y26m-b16-s42/train_data-v2-abl-000-y26m-b16-s42.log`
- `runs/data-v2-abl-000-y26m-b16-s42/args.yaml`
- `runs/data-v2-abl-000-y26m-b16-s42/results.csv`
- `runs/data-v2-abl-000-y26m-b16-s42/results.png`
- `runs/data-v2-abl-000-y26m-b16-s42/MaskPR_curve.png`
- `runs/data-v2-abl-000-y26m-b16-s42/confusion_matrix_normalized.png`
- `runs/data-v2-abl-000-y26m-b16-s42/weights/best.pt`
