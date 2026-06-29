# YOLO26 改进代码 — 工作流与修改记录

> **最后更新**: 2026-06-29 17:30 | **当前阶段**: v1-CBAM 代码已完成，可以开始训练

---

## 1. 代码生效机制（我改的代码如何影响训练？）

### 1.1 当前环境架构

```
┌─────────────────────────────────────────────────────────┐
│  训练脚本 (运行时)                                        │
│  E:\Study\DeepCNN\yolo26\code\train_yolov26_seg.py      │
│  → from ultralytics import YOLO                         │
│  → model.train(...)                                     │
└────────────────────┬────────────────────────────────────┘
                     │ Python import 查找路径
                     ▼
┌─────────────────────────────────────────────────────────┐
│  pip 安装的 ultralytics (当前实际加载的)                   │
│  D:\tool\Anaconda3\Lib\site-packages\ultralytics\       │
│  版本: 8.4.80                                            │
│  ❌ 这个位置我们不方便改，改了也容易丢                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  源码文件夹 (我们要改的地方)                                │
│  e:\Study\DeepCNN\yolo26\yolo_plus\ultralytics-main\    │
│  → ultralytics/nn/modules/conv.py   (CBAM 在这里)        │
│  → ultralytics/nn/modules/block.py  (C3k2, SPPF 在这里)  │
│  → ultralytics/nn/tasks.py          (注册模块在这里)       │
│  → ultralytics/cfg/models/26/       (YAML 配置在这里)      │
│  版本: 8.4.41  ⚠️ 比 pip 版本旧！                         │
│  ❌ 目前训练时不会加载这里的代码                             │
└─────────────────────────────────────────────────────────┘
```

**一句话**：训练脚本 `from ultralytics import YOLO` 加载的是 pip 安装的版本（8.4.80），不是我们源码文件夹里的版本（8.4.41）。直接改源码文件夹的代码**不会**影响训练。

### 1.2 解决方案：pip 可编辑安装（Development Mode）

用 `pip install -e`（editable install）把 Python 的 import 路径从 pip 包指向我们的源码文件夹。之后改源码 → 立即生效。

```
pip install -e .
      ↑
      创建符号链接，让 Python 从源码文件夹加载 ultralytics
      而不是从 site-packages
```

**安装后的架构**：

```
训练脚本 → from ultralytics import YOLO
                │
                ▼  (pip install -e 改变了指向)
         e:\Study\DeepCNN\yolo26\yolo_plus\ultralytics-main\ultralytics\
         ✅ 我们改这里的代码 → 训练时立即生效
```

### 1.3 环境搭建步骤（✅ 已完成）

```powershell
# ===== 步骤 1：从 pip site-packages 复制 8.4.80 到源码文件夹 ✅ =====
Copy-Item -Recurse -Force `
  "D:\tool\Anaconda3\Lib\site-packages\ultralytics\" `
  "e:\Study\DeepCNN\yolo26\yolo_plus\ultralytics-main\ultralytics\"

# ===== 步骤 2：pip 可编辑安装（在 yolo26 环境中）✅ =====
& "D:\tool\Anaconda3\envs\yolo26\python.exe" -m pip install -e "e:\Study\DeepCNN\yolo26\yolo_plus\ultralytics-main"

# ===== 步骤 3：验证安装 ✅ =====
# 期望：version 8.4.80，path 指向源码文件夹
```

### 1.4 如何恢复

如果改坏了，想回到官方版本：

```powershell
pip install ultralytics==8.4.80 --force-reinstall
# 恢复后 python -c "import ultralytics; print(ultralytics.__file__)" 
# 应该回到 D:\tool\Anaconda3\Lib\site-packages\ultralytics\
```

### 1.5 训练目录说明

有两个目录，各有不同用途：

| 目录 | 用途 |
|---|---|
| `E:\Study\DeepCNN\yolo26\code\` | **训练工作区**：训练脚本、数据集、预训练权重、训练结果（runs） |
| `e:\Study\DeepCNN\yolo26\yolo_plus\` | **代码开发区**：ultralytics 源码、YAML 配置、改进文档 |

当你运行 `E:\Study\DeepCNN\yolo26\code\train_yolov26_seg.py` 时，它 import 的是哪个 ultralytics 取决于 pip 安装状态。pip install -e 之后就会加载我们修改过的版本。

---

## 2. 版本号与记录规范

版本号：`baseline → v1 → v2 → v3 → v4`，每个版本对应一次独立改动。

每次改进后：
1. 在 section 3 追加一条改进记录（日期、代码改动清单、训练结果）
2. 将关键指标填入 section 4 的消融实验记录表
3. 更新 section 4 的每类详细对比表

---

## 3. 改进记录

> 以下为实际改进记录，每次改进追加在下面。

### 当前基线（Baseline）

**训练结果文件夹**: `yolo26m_seg_20260628_172809/`
- **模型**: yolo26m-seg.pt (pretrained, fine-tuned)
- **数据集**: 768/95/95, 2 classes (leaffolder + stemborers)
- **Best epoch**: 246 / 400

| 指标 | 值（val, epoch 246） |
|---|---|
| Precision(M) | **0.717** (71.7%) |
| Recall(M) | **0.620** (62.0%) |
| mAP50(M) | **0.683** |
| mAP50-95(M) | **0.329** |
| 卷叶螟 mAP50(M) | 0.604 (漏检率 52%) |
| 钻心虫 mAP50(M) | 0.763 (漏检率 22%) |
| 卷叶螟最佳 F1 | 0.57 @ conf=0.35 |
| 钻心虫最佳 F1 | 0.72 @ conf=0.38 |

> 注：Precision/Recall 取 mAP50(M) 最高的 epoch（246），数据来自 `results.csv` 第 247 行。

---

### v1 — CBAM 注意力机制（✅ 代码完成，⬜ 待训练）

**日期**: 2026-06-29 | **Git 分支**: `v1-cbam` | **GitHub commit**: `ae38878`

**改动文件**:
| # | 文件 | 改动 |
|---|---|---|
| 1 | `ultralytics/nn/modules/conv.py:593` | CBAM 构造函数添加 `c2=None` 参数（框架兼容） |
| 2 | `ultralytics/nn/tasks.py:40` | import 添加 CBAM |
| 3 | `ultralytics/nn/tasks.py:1836` | `base_modules` 注册 CBAM |
| 4 | `ultralytics/cfg/models/26/yolo26-cbam.yaml` | **新建** — CBAM 版模型配置 |

**模型参数对比**:

| 指标 | 原始 yolo26m | yolo26m-cbam | 变化 |
|---|---|---|---|
| 层数 | 280 | 300 | +20 |
| 参数量 | 21.78 M | 22.63 M | +0.85 M (+3.9%) |
| 计算量 | 74.7 GFLOPs | 75.4 GFLOPs | +0.7 GFLOPs (+0.9%) |

**详细讲解**: 见 [node.md](node.md) 第一章。

**训练状态**: ⬜ 待训练 → 训练后填入下方指标

| 指标 | Baseline | v1 CBAM |
|---|---|---|
| P(M) | 0.717 | ? |
| R(M) | 0.620 | ? |
| mAP50(M) | 0.683 | ? |
| mAP50-95(M) | 0.329 | ? |
| 卷叶螟 AP50 | 0.604 | ? |
| 钻心虫 AP50 | 0.763 | ? |
| 卷叶螟漏检率 | 52% | ? |

---

## 4. 改进计划

> ⏱️ **中期答辩策略**：每个改进独立与 Baseline 对比（不是论文那种累积叠加的消融）。
> 采用独立对比方式：v1 vs Baseline、v2 vs Baseline、v3 vs Baseline、v4 vs Baseline。
> 累积消融实验（Baseline → +A → +A+B → +A+B+C）留到后续论文阶段补充。

| 版本 | 改进内容 | 针对问题 | 对比对象 |
|---|---|---|---|
| **Baseline** | yolo26m-seg 原始 | — | — |
| **v1 — CBAM** | Backbone 嵌入 CBAM 注意力 | 卷叶螟特征弱 | vs Baseline |
| **v2 — P2** | Neck 增加 P2 高分辨率层 | 卷叶螟小目标漏检 | vs Baseline |
| **v3 — Dice** | 分割 Loss 加入 Dice Loss | Mask 边缘粗糙 | vs Baseline |
| **v4 — Combined** | CBAM + P2 + Dice 三项集成 | 综合优化 | vs Baseline |

### 中期答辩对比表（每做完一个版本填入）

| Model | P(M) | R(M) | mAP@0.5(M) | mAP@0.5:0.95(M) | 卷叶螟 AP50 | 钻心虫 AP50 | 卷叶螟漏检率 | Params(M) | FLOPs(G) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.717 | 0.620 | **0.683** | 0.329 | 0.604 | 0.763 | 52% | ? | ? |
| v1 CBAM | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| v2 P2 | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| v3 Dice | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| v4 Combined | ? | ? | ? | ? | ? | ? | ? | ? | ? |

> 注：每个版本独立训练，使用与 Baseline 完全相同的训练配置，公平对比。
> F1(M) = 2×P×R/(P+R)，可自行计算；Size = best.pt 文件大小。
> 每类 mAP50 和漏检率来自 confusion_matrix 和 PR 曲线。
> **消融实验（累积对比）留到论文阶段再做**。

### 当前改动清单

| # | 模块 | 涉及文件 | 状态 |
|---|---|---|---|
| 1 | CBAM 注册 | `nn/tasks.py` | ✅ (2026-06-29) |
| 2 | v1 YAML | `cfg/models/26/yolo26-cbam.yaml` | ✅ (2026-06-29) |
| 3 | P2 特征层 | `cfg/models/26/yolo26-p2.yaml` | ⬜ |
| 4 | Dice Loss | loss 计算或 head 修改 | ⬜ |
| 5 | Combined YAML | `cfg/models/26/yolo26-plus.yaml` | ⬜ |

---

## 5. 数据获取指南

### results.csv 自动包含的指标

| results.csv 列名 | 消融表格对应 |
|---|---|
| `metrics/precision(M)` | ✅ Precision(M) |
| `metrics/recall(M)` | ✅ Recall(M) |
| `metrics/mAP50(M)` | ✅ mAP@0.5(M) |
| `metrics/mAP50-95(M)` | ✅ mAP@0.5:0.95(M) |

### ⚠️ 需手动获取的指标

| 指标 | 获取方式 |
|---|---|
| **每类 mAP50 / mAP50-95** | `yolo segment val model=best.pt data=yolo_data.yaml split=test` |
| **每类漏检率** | 查看 `confusion_matrix_normalized.png` |
| **每类 F1 曲线 / 最佳阈值** | 查看 `MaskF1_curve.png` |
| **Params / FLOPs** | Python 中 `model.info()` |
| **模型文件大小** | `best.pt` 文件属性 |
| **整体 F1(M)** | 用 P 和 R 计算：`2*P*R/(P+R)` |

### 每次训练后检查清单

- [ ] `results.png` — loss 收敛正常，无过拟合
- [ ] `confusion_matrix_normalized.png` — 记录每类漏检率
- [ ] `MaskF1_curve.png` — 记录每类最佳 F1 和 conf 阈值
- [ ] `MaskPR_curve.png` — 记录每类 mAP50
- [ ] `results.csv` — 提取 best epoch 的 P/R/mAP
- [ ] 运行 `model.info()` — 获取 Params / FLOPs
- [ ] 将数据填入消融实验表格

---

## 6. 参考论文（YOLO-Pineapple）核心数据

**论文标题**: YOLO-pineapple: enhanced pineapple detection in UAV images using an optimized YOLOv8 model
**发表**: Expert Systems with Applications, 2026 | **基线模型**: YOLOv8

### 论文四大改进模块

| 模块 | 全称 | 作用 |
|---|---|---|
| **DITAH** | Dynamic Interactive Task Alignment Head | 解耦检测头，减少分类/定位任务冲突 |
| **GMSC** | Grouped Multi-Scale Convolution | 分组多尺度卷积（含 CBAM 注意力），减少冗余计算 |
| **SCSA** | Spatial and Channel Synergistic Attention | 空间+通道协同注意力，增强语义交互 |
| **Focaler_SIoU** | Focaler + SIoU Loss | 自适应样本加权 + 角度感知的 IoU Loss |

### 论文消融实验（逐步叠加）

| Model | P | R | mAP50 | mAP50-95 | F1 |
|---|---|---|---|---|---|
| Baseline(YOLOv8) | 92.2% | 87.5% | 92.8% | — | — |
| + DITAH | 92.3% | 89.2% | 93.7% | — | — |
| + GMSC(CBAM) | — | 89.1% | 93.8% | 61.8% | — |
| + SCSA | 93.1% | — | 94.0% | — | — |
| + Focaler_SIoU | **93.8%** | — | **94.3%** | — | **91.2%** |

### 论文注意力机制对比（GMSC 模块中测试）

| 注意力 | P | R | mAP50 | mAP50-95 | F1 | Params(M) | FLOPs(G) |
|---|---|---|---|---|---|---|---|
| CA | 91.6 | 89.1 | 93.4 | 62.8 | 90.3 | 2.13 | 8.5 |
| BAM | 92.7 | 88.7 | 93.3 | 62.3 | 90.7 | 2.13 | 8.5 |
| EMA | 92.0 | 89.0 | 93.1 | 62.7 | 90.5 | 2.12 | 8.6 |
| SE | 91.7 | 88.8 | 93.3 | 63.1 | 90.2 | 2.11 | 8.5 |
| **CBAM** ⭐ | **92.7** | **89.1** | **93.8** | 62.9 | **90.9** | 2.18 | 8.5 |

> 论文结论：CBAM 在 mAP50 和 F1 上均最优，这是我们选 CBAM 作为第一步改进的依据。
> ⚠️ 注意论文用的是 YOLOv8 轻量模型（~2M params），我们用的是 YOLO26m-seg（~22M params），数值不可直接对比，但改进思路可迁移。
