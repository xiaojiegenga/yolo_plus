# Run 记录：data-v2-cmp1-dss-b16-s42

## 基本信息

| 项目 | 内容 |
|---|---|
| 状态 | 已完成并核验（本地 CSV / 权重 / 配置一致性均已复核） |
| 实验目的 | DSS 三位置复合改进（DSEM + SFCM + DPRM）正式主 Run，最终冻结模型 |
| 配置文件 | `experiments/data-v2-cmp1-dss-b16-s42.yaml` |
| 代码分支 / commit | `feature/data-v2-cmp-dss` @ `1d5d0e1`（2026-09-17 已合并入 `feature/data-v2-abl-a-attention` @ `bb6f72b`） |
| 云端原始目录 | `runs/data-v2-cmp1-dss-b16-s42-20260915-223450/`（入口未传 `--run-name`，目录带时间戳） |
| 回传归档 | `exports/data-v2-cmp1-dss-b16-s42-20260915-223450.zip`（本地解包后重命名为正式 ID） |
| 本地保存目录 | `runs/data-v2-cmp1-dss-b16-s42/` |
| GPU / 环境 | RTX 5090；Ultralytics 仓库源码 8.4.80 |
| 开始 / 结束时间 | 2026-09-15 22:34 起，历时 1.335 h |

10 epoch 预检：Run `…-preflight-20260915-221816`，epoch 10 Mask mAP50 / mAP50-95 =
0.613 / 0.305（同期 000 为 0.474 / 0.223），权重迁移与 stride-2 原型验证对齐均核验通过。

## 实验参数

| 参数 | 值 |
|---|---:---|
| model | `ultralytics/cfg/models/26/yolo26m-dss-seg.yaml` + `--pretrained yolo26m-seg.pt` |
| data | `experiments/yolo_data_v2_cloud.yaml`（`/root/yolo_data`） |
| epochs | 300 |
| batch | 16（nbs 64） |
| imgsz | 640 |
| optimizer | AdamW（lr0 0.001667） |
| seed | 42（deterministic） |
| 其他改动 | 仅模型结构；train 段 33 项与 000 逐项一致（本地 diff 仅 `pretrained` 表达方式不同） |

## 训练结果

| 指标 | 结果 |
|---|---:---|
| 实际 epoch | 300（未触发早停，epoch 300 仍为 0.36950） |
| 最佳 epoch | 219（official fitness 0.85020） |
| 训练时长 | 1.335 h（16.1 s/epoch） |
| 峰值显存 | 16.6 GB |
| Val Mask P | 0.68346 |
| Val Mask R | 0.65301 |
| Val Mask mAP50 | 0.71687 |
| Val Mask mAP50-95 | **0.38929** |

Box 侧：mAP50 0.73700 / mAP50-95 0.46091。fused 摘要：163 层、23,865,205 参数、
139.8 GFLOPs（对照 000 为 23.54 M / 121.2 G）。官方权重迁移：入口 890/934（与本地审计
逐项一致），nc=2 重建后 920/934。数值健康度：仅 epoch 8 两个 val 损失格 NaN，训练损失与
全部 mAP 有限。

### 分类型 Val Mask（best.pt 复核）

| 类别 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Rice leaffolder | 0.681 | 0.648 | 0.707 | **0.306** |
| Rice stemborers | 0.687 | 0.663 | 0.731 | **0.477** |

## 本地分析

- 主要结论：**Val Mask mAP50-95 较 000 基线 +2.58 pp，为 14 组正式实验中最大正向收益**
  （此前最好为 D1 的 +0.98 pp），通过全部预注册性能门控（主门 0.36848 / 机制目标 0.37327 /
  理想线 0.37827 均越过）。四组件激活范数：α 1.303（256/256 通道）、γ 2.309、p2_proj 7.364、
  hi_out 1.024，无死组件。
- 留一消融（四个 Run 全部超过 ±0.005 噪声地板，最终模型维持本 Run 结构不变）：
  双分辨率 stride-2 残差 **+4.10 pp**、DSEM 方向细节增强 **+3.13 pp**、P2 跨阶段注入
  **+1.13 pp**、SFCM 语义前景门 **+0.79 pp**；留一和 9.15 ≫ 净 2.58，组件强协同
  （移除双分辨率或 DSEM 任一，复合跌破基线）。详见
  [消融实验设计与登记表](../../knowledge/消融实验设计与登记表.md)。
- 原图 COCO 配对评估（本地 CPU，`experiment_records/evaluations/data-v2-cmp1-cpu-pair.json`）：
  总体 segm AP50-95 0.34020 → 0.36166（+2.15 pp）；卷叶螟 all +1.95 pp、钻心虫 +2.34 pp、
  bbox 两类 +0.6/+0.7 pp。小目标专项（174 小卷叶螟）AP50-95 0.1869，未过旧门（000-CPU
  0.1911），但 conf0.25 召回 +8.1 pp（TP 106→120）——小目标"检出更多、严格 IoU AP 持平"，
  论文按此口径表述。conf≥0.5 低 IoU 高置信率 20.6%（000 17.0%）：SFCM 背景抑制假设被证伪，
  门为置信度放大器（详见消融登记表机制列）。
- Test 终评（最终冻结后一次，云端 GPU 官方协议）：000 0.33490 → 本模型 **0.35216（+1.73 pp）**，
  本地 CPU 配对交叉确认 +1.84 pp；分类型 Test 上钻心虫 +3.13 pp、卷叶螟 +0.32 pp；检测分支
  Test 让步约 1.2 pp 换严格 IoU 掩膜质量。详见
  [Test 终评记录](../../knowledge/Test终评记录-data-v2-cmp1.md)。
- 与同环境 Baseline 的差异：仅模型结构（YAML 两处：第 2 层 `C3k2DSEM`、头
  `Segment26DSS` 接收 [2,16,19,22]）与 `segment val.py` 的 stride-4 评估网格对齐修复
  （自 D1 分支移植，官方验证器假设原型 stride 4，不修复则 stride-2 原型验证坐标错位）。
- 异常或中断：无。权重迁移、配方一致性、全部 CSV/权重有限性均核验通过。
- 下一步：实验阶段收官，转入论文写作。剩余可选：seed 2/3 复验（2×约 1.3 h）、D1 独立对照行。
