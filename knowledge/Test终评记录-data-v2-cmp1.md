# Test 终评记录（data-v2-cmp1-dss · 最终冻结模型）

评估日期：2026-09-17。脚本：`yolo_plus/scripts/eval_test_split.py`（提交 `d4fd4f1`）。
协议：官方 Ultralytics 验证（imgsz 640、conf 0.001、iou 0.7、max_det 300、FP32），
每个模型先跑 val（复现校验）再跑 test（118 图 / 511 实例，方案冻结后首次使用）。

## 1. 本地 CPU 配对结果（000 与 cmp1 同机同协议，已完成）

| 指标（Test split） | 000 基线 | cmp1（DSS） | Δ |
|---|---:|---:|---:|
| **Mask mAP50-95（主指标）** | 0.34288 | **0.36129** | **+1.84 pp** |
| Mask mAP50 | 0.69702 | 0.68970 | −0.73 pp |
| Mask P / R | 0.587 / 0.713 | 0.648 / 0.674 | P +6.1 pp / R −3.9 pp |
| 卷叶螟 Mask mAP50-95 | 0.27998 | 0.28347 | +0.35 pp |
| 钻心虫 Mask mAP50-95 | 0.40578 | 0.43910 | +3.33 pp |
| Box mAP50 / mAP50-95 | 0.68672 / 0.42192 | 0.66122 / 0.41050 | −2.55 / −1.14 pp |

配对 val（同协议参照）：000 0.36137 vs cmp1 0.38514 → **+2.38 pp**（与训练网格口径 +2.58 pp 一致）。

**结论：主指标收益泛化到 Test（+1.84 pp），方向与量级成立。** 三点如实记录：

1. **Test 收益小于 Val**（+1.84 vs +2.58/+2.38 pp），属正常的留出集收缩；
2. **分类型画像翻转**：Val 上收益集中在卷叶螟（+4.0 pp），Test 上集中在钻心虫（+3.33 pp）、
   卷叶螟仅 +0.35 pp——论文中分类型结论应表述为"两类均正向、幅度随划分波动"，不承诺类别专属收益；
3. **检测分支在 Test 上让步**：Box mAP50-95 −1.14 pp、Mask mAP50 −0.73 pp，换来严格 IoU 下掩膜
   质量的提升——与消融阶段发现的"置信度-误检/精度权衡"一致，写进讨论节。

## 2. val 复现校验（协议有效性）

- cmp1：训练期记录 0.71687 / 0.38929 vs 独立本机 0.71322 / 0.38514（−0.37 / −0.42 pp，
  独立 val 的 rect/批尺寸/硬件正常差异，均在噪声地板量级）；
- 000：训练期记录 0.71132 / 0.36348 vs 独立本机 0.69371 / 0.36137（mAP50-95 −0.21 pp 吻合）。
- 注意：脚本输出的"复现校验"块以 cmp1 记录值为参照，对 000 权重无意义（其 MD 中该块请忽略）。

## 3. 云端 GPU 官方口径（已完成，2026-09-17，论文正式 Test 数值）

环境：RTX 5090 / Ultralytics 8.4.80 / torch 2.12.1+cu130；协议与本地一致（batch 16）。
原始文件：`yolo_plus/experiment_records/evaluations/test-eval-{cmp1,000}-5090-gpu.{json,md}`，
日志与图表：`exports/5090/`（含 `runs/test-eval/eval-{cmp1,000}-{val,test}/` 混淆矩阵与 PR 曲线）。

| 指标（Test，118 图 / 509 实例¹） | 000 基线 | cmp1（DSS） | Δ |
|---|---:|---:|---:|
| **Mask mAP50-95（主指标）** | 0.33490 | **0.35216** | **+1.73 pp** |
| Mask mAP50 | 0.69452 | 0.68964 | −0.49 pp |
| Mask P / R | 0.584 / 0.703 | 0.644 / 0.673 | P +6.0 pp / R −3.1 pp |
| 卷叶螟 Mask mAP50-95 | 0.27241 | 0.27559 | +0.32 pp |
| 钻心虫 Mask mAP50-95 | 0.39739 | 0.42873 | +3.13 pp |
| Box mAP50 / mAP50-95 | 0.68740 / 0.42214 | 0.66216 / 0.41001 | −2.52 / −1.21 pp |

配对 val（同协议同硬件）：000 0.35081 vs cmp1 0.37239 → **+2.16 pp**。

¹ 数据集文档登记 Test 为 511 实例，加载器移除 2 个重复标签后实际评估 509（日志可见
`Rice_leaffolder_3x3_p0017_r0c0/r1c1.jpg` 各 1 条重复）；与 Train/Val 的既有处理口径一致。

**独立评估 vs 训练期验证的系统性差异（论文脚注用，如实记录）**：独立 val 复现低于训练期数值，
对两模型同时存在——000：0.35081 vs 0.36348（−1.27 pp）；cmp1：0.37239 vs 0.38929（−1.69 pp）；
本地 CPU 独立评估同样偏低（−0.21 / −0.42 pp）。原因属协议路径差异（独立评估做 Conv+BN 融合、
剥离仅训练组件、rect 分组不同），非权重或数据问题。**配对差值在三个协议下一致**：
val +2.58（训练口径）/ +2.38（CPU）/ +2.16（GPU）；Test +1.84（CPU）/ +1.73（GPU）。
论文建议：Val 列沿用训练日志数值（与全部消融表同口径），Test 列用本节数值，并加脚注说明
独立评估协议的系统性偏移对两模型同向、配对比较不受影响。

## 4. 三协议交叉汇总（论文可直接引用）

| 比较 | Val Δ (Mask mAP50-95) | Test Δ |
|---|---:|---:|
| 训练网格口径（best.pt，正式记录） | +2.58 pp | — |
| 独立评估 · 本地 CPU 配对 | +2.38 pp | +1.84 pp |
| 独立评估 · 云端 GPU 官方协议 | +2.16 pp | **+1.73 pp** |

Test 分类型画像在两硬件上一致：卷叶螟微升（+0.32/+0.35 pp）、钻心虫领涨（+3.13/+3.33 pp）；
检测分支在 Test 上让步约 1.1–1.2 pp，换来严格 IoU 掩膜质量——与消融阶段的置信度-精度权衡一致。

## 5. 文件索引

| 内容 | 位置 |
|---|---|
| cmp1 云端 GPU 终评（论文正式） | `yolo_plus/experiment_records/evaluations/test-eval-cmp1-5090-gpu.{json,md}` |
| 000 云端 GPU 终评（论文正式） | `yolo_plus/experiment_records/evaluations/test-eval-000-5090-gpu.{json,md}` |
| cmp1 本地 CPU 终评（交叉验证） | `yolo_plus/experiment_records/evaluations/test-eval-cmp1-local-cpu.{json,md}` |
| 000 本地 CPU 终评（交叉验证） | `yolo_plus/experiment_records/evaluations/test-eval-000-local-cpu.{json,md}` |
| 消融阶段 CPU 配对（5 份） | `yolo_plus/experiment_records/evaluations/data-v2-*.json` |
| Test 图表（混淆矩阵、PR 曲线） | `yolo_plus/exports/5090/test-eval/eval-{cmp1,000}-{val,test}/` |
