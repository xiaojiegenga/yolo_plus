# Run 记录：data-v2-abl-e-dysample-b16-s42

本次 E（Neck 两处 DySample LP）未获得分割收益。官方 best.pt 对应 epoch 264，Val Mask mAP50-95=0.34649，较 Baseline 下降 0.01699（1.699 个百分点），较 D1 下降 0.02678。保留本次负结果，当前不进入 D1+E 组合。

## 基本信息与实际参数

| 项目 | 内容 |
|---|---|
| 状态 | 已完成并核验，300 epoch |
| 实验目的 | 独立验证 Neck 动态上采样对实例分割的作用 |
| 源码分支 | `feature/data-v2-abl-e-dysample` |
| 实现提交 | `5d91a2b`；回传日志未记录实际训练 commit |
| 配置 | E 分支 `experiments/data-v2-abl-e-dysample-b16-s42.yaml` |
| 本地原始 Run | `runs/data-v2-abl-e-dysample-b16-s42/` |
| GPU / 环境 | RTX 5090，Python 3.12.3，PyTorch 2.12.1+cu130，Ultralytics 8.4.80 |
| model / pretrained | `yolo26m-dysample-seg.yaml` / 官方 `yolo26m-seg.pt` |
| data / split | `yolo_data_v2_cloud.yaml` / Val |
| epochs / patience | 300 / 100 |
| batch / nbs / imgsz | 16 / 64 / 640 |
| optimizer / lr0 / lrf | AdamW / 0.001667 / 0.01 |
| momentum / weight_decay | 0.9 / 0.0005 |
| warmup_epochs / warmup_bias_lr | 3 / 0.1 |
| seed / deterministic / AMP | 42 / true / true |
| workers / cache | 8 / false |
| mask_ratio / mixup / copy_paste | 2 / 0 / 0.3 |
| mosaic / close_mosaic | 1.0 / 15 |
| degrees / flipud / fliplr / scale | 15 / 0.5 / 0.5 / 0.3 |

实际 args 与 Baseline 的差异仅为模型路径、预训练权重表达方式和 Run 输出名称；训练配方一致。E 保留标准 Segment26、检测 stride=8/16/32，仅将第 11、14 层改为 DySample（scale=2，groups=4，LP，固定偏移系数 0.25）。

## 正式结果

总体数值取 official fitness（Box mAP50-95 + Mask mAP50-95）最大轮次，并以 best.pt 的 train_metrics 核对。

| 指标 | Baseline（epoch 216） | D1（epoch 278） | E（epoch 264） | E−Baseline |
|---|---:|---:|---:|---:|
| Official fitness | 0.81977 | 0.82836 | 0.80522 | -0.01455 |
| Box mAP50-95 | 0.45629 | 0.45509 | 0.45873 | +0.00244 |
| Mask P | 0.64822 | 0.64137 | 0.70605 | +0.05783 |
| Mask R | 0.67252 | 0.68318 | 0.61999 | -0.05253 |
| Mask mAP50 | 0.71132 | 0.71245 | 0.69353 | -0.01779 |
| Mask mAP50-95 | 0.36348 | 0.37327 | 0.34649 | -0.01699 |

E 的检测框 AP50-95 略升，但掩膜 AP 和召回率下降。日志工作点下 Precision 提高，不能据此认定整体分割改善；不同 Run 的 P/R 工作点不等同于固定置信度的误检对比。

### 分类型 Val Mask

以下是训练结束后的 best.pt 验证日志值，保留日志的三位小数精度。

| 类别 | Baseline AP50 | E AP50 | Baseline AP50-95 | E AP50-95 | E P | E R |
|---|---:|---:|---:|---:|---:|---:|
| Rice leaffolder | 0.677 | 0.646 | 0.266 | 0.248 | 0.644 | 0.600 |
| Rice stemborers | 0.745 | 0.742 | 0.461 | 0.446 | 0.770 | 0.642 |

两类 AP50-95 均下降。卷叶螟 AP50 下降 0.031，未体现本次结构预期的分割收益。

## 收敛与模型选择

| 统计量 | Baseline | E |
|---|---:|---:|
| 前 5 个最高 fitness 的均值 | 0.808818 | 0.799984 |
| 前 10 个最高 fitness 的均值 | 0.804628 | 0.796374 |
| 上述前 10 轮的 Mask mAP50-95 均值 | 0.351075 | 0.344045 |
| 全程最高 Mask mAP50-95 | 0.36348（216） | 0.34816（234） |
| epoch 251–285 Mask mAP50-95 均值 | 0.345024 | 0.335119 |
| epoch 286–300 Mask mAP50-95 均值 | 0.336217 | 0.333324 |
| 最后一轮 Mask mAP50-95 | 0.34131 | 0.34006 |

负结果不只体现于单个最佳轮次。即使另看 E 全程最高 Mask AP，也低于 Baseline；正式结果仍使用 epoch 264，不能换成 epoch 234。后期训练没有表现出需要延长训练的明确依据。

## 资源与完成情况

- CSV 连续包含 epoch 1–300，日志正常完成，best.pt 和 last.pt 可加载；两份权重的全部状态张量有限。
- best.pt 的指标与 epoch 264 一致，last.pt 的指标与 epoch 300 一致，checkpoint 中的训练参数与 args.yaml 一致。
- 训练时间 4473.95 s（74.57 分钟，1.243 h），较 Baseline 的 4416.89 s 增加约 1.29%；日志峰值 GPU_mem=15.4 GB。
- 日志 fused Params=23,541,842，较 Baseline 增加 32,832（约 0.14%）；best.pt 大小 54.591699 MB。

## 限制与已知问题

- 第 2 轮 val/box_loss、val/seg_loss、val/cls_loss、val/dfl_loss 为 NaN。全部训练损失、P/R/mAP 及最终权重有限；未定位该轮验证损失异常的根因，不能表述为全程无 NaN。
- 当前仅 seed=42；CUDA grid_sample 反向出现非确定性警告，不能保证逐位复现。本次结果支持淘汰当前 E 配置，不推广为所有动态上采样方案都无效。
- 日志未保存实际训练 commit；`5d91a2b` 是实现来源，不能代替实际训练版本证据。
- THOP 121.228493 GFLOPs 未计 grid_sample 等函数算子，不代表完整计算量。训练时长是单次实测；日志推理时间不是统一 batch=1 延迟基准。
- 分类别日志为训练结束后复核值，总体精确值为 CSV/checkpoint 记录，二者不混用。未运行新的小目标、固定阈值误检或 Test 评估。

## 下一步

保留 D1 作为第一个正向结构，当前 E 不进入组合。第二个结构需要重新设计并独立验证，继续固定数据集和训练配方；本次结果不足以认定具体失败机制，也没有支持直接叠加 D1+E 的证据。

## 证据入口

- [原始逐轮结果](../../runs/data-v2-abl-e-dysample-b16-s42/results.csv)
- [实际参数](../../runs/data-v2-abl-e-dysample-b16-s42/args.yaml)
- [训练与分类别验证日志](../../runs/data-v2-abl-e-dysample-b16-s42/train_data-v2-abl-e-dysample-b16-s42.log)
- [最佳权重](../../runs/data-v2-abl-e-dysample-b16-s42/weights/best.pt)
- [Baseline 正式记录](data-v2-abl-000-y26m-b16-s42.md)
- [D1 正式记录](data-v2-abl-d1-p2proto-r2-b16-s42.md)
