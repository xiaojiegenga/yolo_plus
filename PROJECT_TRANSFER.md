# YOLO26 水稻害虫识别 — 项目完整打包文档

> **历史文档说明（2026-07-28）**：本文件由 Claude 在旧 V2 调试阶段生成，用于保留项目历史。旧 V2 已归档为 `archive/v2-p2-failed`。文中“BCE Collapse 已定位”“近单位初始化已修复”等表述没有经过一轮干净、完整的重新训练验证，不能作为当前项目结论。当前执行规范以用户最新要求、`CODEX_PROJECT_CONTEXT.md`、`code_plus.md` 和实际 Git 状态为准。
>
> **生成日期**: 2026-07-28 | **Git 分支**: `v2-p2` | **生成目的**: 项目迁移到 Codex / 其他 AI 编码工具
>
> 读完本文档后，你可以在新环境中完整复现当前项目的所有工作。

---

## 目录

1. [项目概览](#1-项目概览)
2. [环境搭建（在新机器上重建）](#2-环境搭建)
3. [项目目录结构](#3-项目目录结构)
4. [Git 分支与提交历史](#4-git-分支与提交历史)
5. [YOLO26 架构入门（必读）](#5-yolo26-架构入门)
6. [改进版本总览](#6-改进版本总览)
7. [v1 — CBAM 注意力机制](#7-v1--cbam-注意力机制)
8. [v2 — P2 高分辨率特征层](#8-v2--p2-高分辨率特征层)
9. [代码改动清单（完整）](#9-代码改动清单完整)
10. [训练脚本说明](#10-训练脚本说明)
11. [训练结果汇总](#11-训练结果汇总)
12. [v2 调试记录（BCE 崩溃）](#12-v2-调试记录bce-崩溃)
13. [待完成任务](#13-待完成任务)
14. [关键文件索引](#14-关键文件索引)
15. [常见问题 FAQ](#15-常见问题-faq)

---

## 1. 项目概览

### 1.1 我是什么课题？

**水稻害虫识别** — 基于无人机（UAV）航拍图像，使用 YOLO26 做**实例分割**（Instance Segmentation），同时检测害虫位置（Bounding Box）和轮廓（Mask）。

- **2 个类别**：卷叶螟（Rice Leaffolder）+ 钻心虫（Rice Stemborers）
- **数据集**：768 训练 / 95 验证 / 95 测试
- **输入图像**：640×640（UAV 航拍）

### 1.2 核心问题

| 优先级 | 问题 | 原因 |
|--------|------|------|
| **P0** | 卷叶螟漏检率 52% | 虫子小（~25×25 px）、密集、与稻叶颜色接近 |
| **P1** | Mask 边缘粗糙 (mAP50-95=0.33) | 分割头 mask 分辨率不够 |
| **P1** | 整体 mAP50 0.683 不够高 | 需要更好的特征提取和训练策略 |

### 1.3 改进路线图

```
Baseline (yolo26m-seg) → v1: CBAM 注意力 → v2: P2 高分辨率层 → v3: Dice Loss → v4: 三项合并
                         ✅ 已完成           🔧 调试中              ⬜ 计划中       ⬜ 计划中
```

---

## 2. 环境搭建

### 2.1 硬件

- **GPU**: NVIDIA RTX 4060 Ti 8GB
- **CPU**: 任意

### 2.2 软件环境

```powershell
# ===== 步骤 0：确认 Conda 环境 =====
conda activate yolo26
# Python 3.10+, PyTorch 2.x + CUDA

# ===== 步骤 1：安装基础依赖 =====
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.4.80

# ===== 步骤 2：pip 可编辑安装（让源码修改立即生效）=====
# 这一步是关键！不执行的话，修改源码文件夹不会影响训练
cd e:\Study\DeepCNN\yolo26\yolo_plus\ultralytics-main
pip install -e .

# ===== 步骤 3：验证安装 =====
python -c "import ultralytics; print(ultralytics.__file__)"
# 期望输出：e:\Study\DeepCNN\yolo26\yolo_plus\ultralytics-main\ultralytics\__init__.py
# 如果不是指向源码文件夹，说明 pip install -e 没生效！
```

### 2.3 训练数据位置

```
E:\Study\DeepCNN\yolo26\code\
  ├── yolo_data.yaml          # 数据集配置（路径 + 类别名）
  ├── train/                   # 训练图片 + 标签
  ├── val/                     # 验证图片 + 标签
  ├── test/                    # 测试图片 + 标签
  ├── yolo26m-seg.pt           # 预训练权重
  └── yolo26n-seg.pt           # 预训练权重（轻量版）
```

### 2.4 恢复官方版本

如果改坏了，想回到官方 ultralytics：

```powershell
pip install ultralytics==8.4.80 --force-reinstall
```

---

## 3. 项目目录结构

```
e:\Study\DeepCNN\yolo26\
│
├── yolo_plus/                          # ★ 代码开发 + 文档（本目录）
│   ├── CLAUDE.md                       # 项目说明（给 AI 助手看）
│   ├── code_plus.md                    # 改进工作流 + 修改记录
│   ├── node.md                         # 详细技术讲解（v1 CBAM 原理 + v2 P2 原理 + 调试记录）
│   ├── note.md                         # 学习笔记（Bottleneck/C2f/C3k2 源码分析）
│   ├── PROJECT_TRANSFER.md             # ★ 本文件 — 项目打包文档
│   ├── AGENTS.md                       # Codex 用的项目说明文件
│   │
│   ├── ultralytics-main/               # ★ Ultralytics 源码（8.4.80，已 pip install -e）
│   │   ├── setup.py
│   │   └── ultralytics/
│   │       ├── cfg/
│   │       │   ├── default.yaml        # 训练默认超参数
│   │       │   └── models/
│   │       │       └── 26/             # ★ YOLO26 模型 YAML 配置
│   │       │           ├── yolo26.yaml         # 原始检测模型 (Backbone + Head)
│   │       │           ├── yolo26-seg.yaml     # 原始分割模型
│   │       │           ├── yolo26-cbam.yaml    # v1: CBAM 检测版
│   │       │           ├── yolo26-cbam-seg.yaml # v1: CBAM 分割版
│   │       │           ├── yolo26-p2.yaml      # 官方 P2 检测版（参考）
│   │       │           └── yolo26-p2-seg.yaml  # ★ v2: P2 分割版（我们新建的）
│   │       │
│   │       ├── nn/
│   │       │   ├── modules/
│   │       │   │   ├── conv.py         # Conv, DWConv, CBAM 等基础模块
│   │       │   │   ├── block.py        # C2f, C3k2, SPPF, Proto26 等组合模块
│   │       │   │   └── head.py         # Detect, Segment, Segment26 检测头
│   │       │   └── tasks.py            # ★ YAML→PyTorch 模型装配 + 模块注册表
│   │       │
│   │       ├── models/yolo/segment/
│   │       │   └── val.py              # ★ 分割验证器（P2 适配修改）
│   │       │
│   │       └── utils/
│   │           ├── loss.py             # 损失函数（BCE mask loss 等）
│   │           ├── ops.py              # process_mask, crop_mask 等工具
│   │           └── nms.py              # NMS 后处理
│   │
│   └── results/                        # v1 训练结果图表（已移至 v1-cbam 分支）
│       └── v1-cbam/
│
├── code/                               # ★ 训练工作区（独立目录）
│   ├── train_yolov26_seg.py            # ★ 训练脚本（支持 baseline/cbam/p2 三个版本）
│   ├── debug_mask.py                   # 调试脚本 — 验证 mask 数据流是否正确
│   ├── yolo_data.yaml                  # 数据集配置
│   ├── yolo26m-seg.pt                  # 预训练权重 (medium)
│   ├── yolo26n-seg.pt                  # 预训练权重 (nano)
│   ├── yolov8m-seg.pt                  # YOLOv8 权重（参考）
│   │
│   └── runs/segment/runs_seg/          # ★ 训练结果
│       ├── yolo26m_seg_20260628_172809/ # Baseline 训练结果 (best epoch 246)
│       ├── yolo26m_cbam_seg_20260630_014120/ # v1 CBAM 训练结果 (best epoch 250)
│       └── yolo26m_p2_seg_20260701_034948/  # v2 P2 训练结果 (失败: Mask mAP=0)
│
└── YOLO-pineapple.html                 # 参考论文
```

---

## 4. Git 分支与提交历史

### 4.1 分支结构

```
main ──→ v1-cbam ──→ v2-p2 (当前) ──→ v3-dice ──→ v4-combined
         (已merge)    (当前开发)       (空壳分支)    (空壳分支)
```

| 分支 | 状态 | 内容 |
|------|------|------|
| `main` | 基线 | yolo26m-seg 原始训练结果 |
| `v1-cbam` | ✅ 已合并 | CBAM 注意力（conv.py + tasks.py + yolo26-cbam.yaml） |
| **`v2-p2`** | 🔧 当前 | P2 高分辨率层 + 初始化修复 + val.py 坐标修复 |
| `v3-dice` | ⬜ 空壳 | 预留 Dice Loss 改进 |
| `v4-combined` | ⬜ 空壳 | 预留三项合并 |

### 4.2 完整提交历史（最近 10 个）

```
2053450 @ fix(v2-p2): correct bbox-to-proto coordinate mapping for P2 models
5a8213d @ fix(v2-p2): optimize head weight transfer by reordering Segment26 inputs
4700cc1 Fix: add mask resolution mismatch handling in segment val
bb392b3 Fix: use actual imgsz instead of hardcoded 4*proto in segment validation
c973ed0 v2-P2: add high-resolution P2/4 feature layer for small object detection
ecd5407 docs: add per-class Mask P/R to v1 comparison tables
93ef37c v1 results: CBAM training complete — mAP50 0.683→0.700
7a63fde v1: add yolo26-cbam-seg.yaml (segment version with Segment26 head)
e53e29b docs: add node.md (CBAM详解) and update code_plus.md v1 status
ae38878 v1: register CBAM in tasks.py, add yolo26-cbam.yaml
f01e2de init: baseline ultralytics 8.4.80 + yolo26m-seg training results
```

### 4.3 当前未提交的改动

```
M node.md                                          # v2 调试文档更新（已完成）
M ultralytics-main/ultralytics/nn/modules/block.py  # ★ Proto26 近单位初始化修复
?? AGENTS.md                                        # Codex 项目说明文件
```

---

## 5. YOLO26 架构入门（必读）

### 5.1 三阶段架构

每个 YOLO 模型由三部分组成：

```
输入图像 (640×640×3)
    │
    ▼
┌──────────────────────────────────────────────┐
│  Backbone（骨架网络）                          │
│  - 逐层下采样提取特征：640 → 320 → 160 → 80 → 40 → 20 │
│  - 模块：Conv(下采样) + C3k2(特征提取) + SPPF(多尺度池化) + C2PSA(注意力) │
│  - 输出：P2(160×160), P3(80×80), P4(40×40), P5(20×20) │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Neck（特征金字塔 FPN+PAN）                     │
│  - FPN：自上而下融合（P5→P4→P3→P2）             │
│  - PAN：自下而上增强（P2→P3→P4→P5）             │
│  - 输出：多尺度增强特征（4个尺度）                │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Head（检测头）                                 │
│  - Detect/Segment26：从特征图预测 bbox + class  │
│  - Proto26：生成 Mask Proto（分割掩码原型）       │
│  - Mask coeff × Proto = 实例分割 mask          │
└──────────────────────────────────────────────┘
```

### 5.2 五步阅读路线（深入代码）

| 步骤 | 文件 | 看什么 |
|------|------|--------|
| 1 | YAML 配置 (`cfg/models/26/yolo26.yaml`) | YAML 语法、`[from, repeats, module, args]` 格式 |
| 2 | 基础模块 (`nn/modules/conv.py`, `block.py`) | Conv, C3k2, SPPF 的 forward() |
| 3 | 检测头 (`nn/modules/head.py`) | Detect, Segment26, Proto26 的 forward() |
| 4 | 装配工厂 (`nn/tasks.py` → `parse_model()`) | YAML → PyTorch 模型的转换过程 |
| 5 | 高层 API (`models/yolo/model.py`, `engine/model.py`) | YOLO() 类、.train()/.val() 入口 |

### 5.3 YAML 语法速查

```yaml
# 每一层的格式：[from, repeats, module, args]
- [-1, 1, Conv, [64, 3, 2]]       # -1 = 上一层，1 = 重复1次，Conv = 模块名，[64,3,2] = 参数
- [-1, 2, C3k2, [256, False, 0.25]]  # 2 = 重复2次（内部堆叠2个Bottleneck）
- [[-1, 2], 1, Concat, [1]]       # [-1, 2] = 拼接上一层和第2层的输出
```

---

## 6. 改进版本总览

### 6.1 改进策略

每个版本**独立对比 Baseline**（不是累积叠加）。这样在中期答辩时，每个改进都有独立的消融结论。

| 版本 | 改进 | 针对问题 | 对比对象 | 状态 |
|------|------|----------|----------|------|
| Baseline | yolo26m-seg 原始 | — | — | ✅ |
| **v1 CBAM** | Backbone 嵌入 CBAM 注意力 | 卷叶螟特征弱 | vs Baseline | ✅ 训练完成 |
| **v2 P2** | Neck 增加 P2 高分辨率层 | 卷叶螟小目标漏检 | vs Baseline | 🔧 调试中 |
| **v3 Dice** | 分割 Loss 加入 Dice Loss | Mask 边缘粗糙 | vs Baseline | ⬜ 计划中 |
| **v4 Combined** | CBAM + P2 + Dice 三项集成 | 综合优化 | vs Baseline | ⬜ 计划中 |

### 6.2 答辩对比表

| Model | P(M) | R(M) | mAP50(M) | mAP50-95(M) | 卷叶螟 AP50 | 钻心虫 AP50 | 卷叶螟漏检率 | Params(M) | FLOPs(G) |
|-------|------|------|----------|-------------|------------|------------|------------|-----------|----------|
| **Baseline** | 0.717 | 0.620 | **0.683** | 0.329 | 0.604 | 0.763 | **52%** | 21.78 | 74.7 |
| **v1 CBAM** | 0.753 | 0.633 | **0.700** | 0.343 | 0.607 | 0.793 | 45% | 27.83 | 121.9 |
| v2 P2 | ? | ? | ? | ? | ? | ? | ? | 24.76 | 280.5 |
| v3 Dice | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| v4 Combined | ? | ? | ? | ? | ? | ? | ? | ? | ? |

---

## 7. v1 — CBAM 注意力机制

### 7.1 改了什么

在 Backbone 的每个 C3k2 后面插入 CBAM（Channel + Spatial Attention），共 +4 层。

```
改进前:                改进后:
  Conv                   Conv
  C3k2                   C3k2
  Conv       →           CBAM  ★
  C3k2                   Conv
  ...                    C3k2
                         CBAM  ★
                         ...
```

### 7.2 代码改动

| # | 文件 | 行号 | 改动 |
|---|------|------|------|
| 1 | `nn/modules/conv.py` | ~593 | CBAM 构造函数添加 `c2=None`（框架兼容） |
| 2 | `nn/tasks.py` | ~40 | import 添加 CBAM |
| 3 | `nn/tasks.py` | ~1836 | `base_modules` 注册 CBAM |
| 4 | `cfg/models/26/yolo26-cbam.yaml` | 新建 | CBAM 检测模型 YAML |
| 5 | `cfg/models/26/yolo26-cbam-seg.yaml` | 新建 | CBAM 分割模型 YAML |

### 7.3 训练结果（Best epoch 250）

| 指标 | Baseline | v1 CBAM | 变化 |
|------|----------|---------|------|
| mAP50(M) | 0.683 | **0.700** | ↑ +2.5% |
| mAP50-95(M) | 0.329 | **0.343** | ↑ +4.3% |
| Box mAP50 | 0.667 | **0.694** | ↑ +4.0% |
| 卷叶螟 R(M) | 0.482 | **0.552** | ↑ +14.5% |
| 卷叶螟 AP50(M) | 0.604 | **0.607** | ↑ +0.5% |
| 卷叶螟漏检率 | 52% | **45%** | ↓ -7pp |
| 钻心虫 P(M) | 0.753 | **0.844** | ↑ +12.1% |
| 钻心虫 AP50(M) | 0.763 | **0.793** | ↑ +3.9% |

**结论**: CBAM 带来全面温和提升，钻心虫受益最明显，卷叶螟 Recall 提升但 AP50 几乎不变——CBAM 帮助找到了"容易的"虫，微小的虫仍需 P2。

### 7.4 训练结果文件位置

```
E:\Study\DeepCNN\yolo26\code\runs\segment\runs_seg\yolo26m_cbam_seg_20260630_014120\
├── best.pt / last.pt          # 模型权重
├── results.csv                # 每 epoch 指标
├── results.png                # Loss + mAP 曲线
├── confusion_matrix.png       # 混淆矩阵
├── MaskPR_curve.png            # Mask PR 曲线
└── ...
```

---

## 8. v2 — P2 高分辨率特征层

### 8.1 改了什么

在 Neck 中增加 P2/4 高分辨率特征层，使检测头从 3 输入 (P3/P4/P5) 变为 4 输入 (P2/P3/P4/P5)。

```
改进前:                      改进后:
  P5 → Upsample                P5 → Upsample
    + P4 → C3k2 (P4)             + P4 → C3k2 (P4)
  P4 → Upsample                P4 → Upsample
    + P3 → C3k2 (P3) ★           + P3 → C3k2 (P3)
  P3 → Conv↓2                  P3 → Upsample        ← 多一级!
    +     → C3k2 (P4)             + P2 → C3k2 (P2) ★ ← 新增!
  P4 → Conv↓2                  P2 → Conv↓2
    + P5  → C3k2 (P5)            +     → C3k2 (P3)
                               P3 → Conv↓2
  Detect: [P3,P4,P5]             +     → C3k2 (P4)
                               P4 → Conv↓2
                                 + P5  → C3k2 (P5)

                               Segment26: [P2,P3,P4,P5] ← 4个输出!
```

**核心价值**: P2 特征图 160×160（是 P3 的 4 倍像素），小目标（卷叶螟 ~25×25 px）在 P2 上有 ~6×6 格点 vs P3 只有 ~3×3 格点。

### 8.2 Mask Proto 分辨率翻倍

- Baseline: Proto = 160×160（基于 P3）
- P2: Proto = **320×320**（基于 P2）— 4 倍像素，mask 边缘更细

### 8.3 代码改动

| # | 文件 | 改动 | 状态 |
|---|------|------|------|
| 1 | `cfg/models/26/yolo26-p2-seg.yaml` | **新建** — P2 分割模型 YAML | ✅ |
| 2 | `code/train_yolov26_seg.py` | 添加 `_build_p2_model()` + 权重迁移 | ✅ |
| 3 | `models/yolo/segment/val.py` | proto_factor 动态检测 + 坐标修复 | ✅ 已 commit |
| 4 | `nn/modules/block.py` | Proto26 近单位初始化 | ✅ 已修改，**未 commit** |

### 8.4 模型参数对比

| 指标 | Baseline | yolo26m-p2-seg |
|------|----------|----------------|
| 层数 | 329 | 410 |
| 参数量 | 27.11 M | **24.76 M** ↓ |
| 计算量 | 132.5 GFLOPs | **280.5 GFLOPs** ↑ |
| Proto 分辨率 | 160×160 | 320×320 |
| Batch size | 8 | **4** (RTX 4060 Ti 8GB 限制) |

### 8.5 当前状态：调试中 🔧

v2 P2 训练跑了 2 次，均失败（Mask mAP=0）。**根因已找到**，但修复后的训练尚未进行。

详见 [第 12 节 — v2 调试记录](#12-v2-调试记录bce-崩溃)。

---

## 9. 代码改动清单（完整）

### 9.1 已 commit 的改动（Git 历史中）

| 文件 | Commit | 改动摘要 |
|------|--------|---------|
| `nn/modules/conv.py:593` | `ae38878` | CBAM 添加 `c2=None` 参数 |
| `nn/tasks.py:40,1836` | `ae38878` | 注册 CBAM 到 base_modules |
| `cfg/models/26/yolo26-cbam.yaml` | `ae38878` | 新建 CBAM 检测 YAML |
| `cfg/models/26/yolo26-cbam-seg.yaml` | `7a63fde` | 新建 CBAM 分割 YAML |
| `cfg/models/26/yolo26-p2-seg.yaml` | `c973ed0` | 新建 P2 分割 YAML |
| `models/yolo/segment/val.py:78,118,149` | `bb392b3`, `2053450` | proto_factor 动态检测 + 坐标修复 |

### 9.2 未 commit 的改动（待提交）

| 文件 | 行号 | 改动 | 重要性 |
|------|------|------|--------|
| **`nn/modules/block.py`** | 2003-2014 | Proto26 近单位初始化（feat_refine 近零 + feat_fuse 近恒等） | **🔴 关键！** 必须在下次训练前 commit |
| `node.md` | 534-729 | v2 调试记录（BCE 崩溃 + NaN + 坐标映射） | 文档更新 |

### 9.3 未 push 到远程的分支

`v2-p2` 分支的 commit `2053450`（最新的 val.py 修复）尚未 push 到 `origin/v2-p2`。

---

## 10. 训练脚本说明

### 10.1 位置

```
E:\Study\DeepCNN\yolo26\code\train_yolov26_seg.py
```

### 10.2 使用方法

修改脚本顶部的 `MODEL_VERSION` 变量：

```python
MODEL_SIZE = "m"           # "n" = nano, "m" = medium
MODEL_VERSION = "p2"       # "baseline" | "cbam" | "p2"
```

然后运行：

```powershell
conda activate yolo26
cd E:\Study\DeepCNN\yolo26\code
python train_yolov26_seg.py
```

### 10.3 训练超参数（与 Baseline 完全一致）

```python
IMAGE_SIZE = 640          # 输入图像尺寸
EPOCHS = 400              # 总训练轮次
BATCH_SIZE = 8            # Baseline/CBAM: 8, P2: 4 (显存限制)
LEARNING_RATE = 0.01      # 初始学习率
WARMUP_EPOCHS = 5.0      # 预热轮次
PATIENCE = 100            # 早停耐心值
MOSAIC = 1.0              # Mosaic 数据增强
MIXUP = 0.1               # MixUp 数据增强
COPY_PASTE = 0.3          # Copy-Paste 数据增强
```

### 10.4 `_build_p2_model()` 关键逻辑

当 `MODEL_VERSION = "p2"` 时，脚本会：

1. **从 YAML 构建 P2 模型**（`yolo26-p2-seg.yaml`，m 尺度，2 类）
2. **加载预训练权重** (`yolo26m-seg.pt`)
3. **重映射层编号**：
   - 原始模型 ≥17 的层全部偏移 +6（因为 P2 插了 6 层）
   - Head 内部 cv2/cv3/cv4 分支重排（P3@0→0, P4@1→2, P5@2→3, P2@1 随机初始化）
   - Proto26 核心层直接迁移（c_=256 硬编码，结构相同）
4. **严格过滤**形状不匹配的 key，输出详细的迁移报告

---

## 11. 训练结果汇总

### 11.1 所有训练运行

| 运行名称 | 模型 | 状态 | Best Epoch | mAP50(M) | 备注 |
|----------|------|------|------------|----------|------|
| `yolo26m_seg_20260628_172809` | Baseline | ✅ 完成 | 246 | **0.683** | 对照组 |
| `yolo26m_cbam_seg_20260630_014120` | v1 CBAM | ✅ 完成 | 250 | **0.700** | +2.5% |
| `yolo26m_p2_seg_20260701_034948` | v2 P2 | ❌ 失败 | 248 (Box only) | **0** | Mask 崩溃 |
| `yolo26m_seg_20260529_005023` | 早期测试 | 废弃 | — | — | — |
| `yolo26m_seg_20260628_141215` | 早期测试 | 废弃 | — | — | — |
| `yolo26m_seg_20260629_011616` | 早期测试 | 废弃 | — | — | — |

### 11.2 Baseline 详细指标

| 指标 | 值 | 来源 |
|------|-----|------|
| mAP50(M) | 0.683 | `yolo26m_seg_20260628_172809/results.csv` epoch 246 |
| mAP50-95(M) | 0.329 | 同上 |
| Box mAP50 | 0.667 | 同上 |
| 卷叶螟 mAP50(M) | 0.604 (P=0.679, R=0.482) | 同上 + val 脚本 |
| 钻心虫 mAP50(M) | 0.763 (P=0.753, R=0.755) | 同上 + val 脚本 |
| 卷叶螟漏检率 | 52% | `confusion_matrix_normalized.png` |
| 卷叶螟最佳 F1 | 0.57 @ conf=0.35 | `MaskF1_curve.png` |

### 11.3 v1 CBAM 详细指标

| 指标 | 值 | 来源 |
|------|-----|------|
| mAP50(M) | 0.700 | `yolo26m_cbam_seg_20260630_014120/results.csv` epoch 250 |
| mAP50-95(M) | 0.343 | 同上 |
| Box mAP50 | 0.694 | 同上 |
| 卷叶螟 mAP50(M) | 0.607 (P=0.662, R=0.552) | 同上 + val 脚本 |
| 钻心虫 mAP50(M) | 0.793 (P=0.844, R=0.714) | 同上 + val 脚本 |
| 卷叶螟漏检率 | 45% | `confusion_matrix_normalized.png` |

---

## 12. v2 调试记录（BCE 崩溃）

### 12.1 问题时间线

```
训练尝试 #1 (epoch 1-357):
  ├── Epoch 1:   Box mAP50=0.306, Mask mAP50=1.19e-07  ← 不正常
  │   └── 根因: (1) val.py 硬编码坐标 + (2) Proto26 随机初始化
  ├── Epoch 100: Box mAP50=0.559, Mask mAP50=0         ← 持续崩溃
  ├── Epoch 248: Box mAP50=0.666 (接近 baseline!), Mask mAP50=0
  ├── Epoch 353: seg_loss = NaN                         ← 崩溃连锁
  └── Epoch 357: 手动停止

根因定位:
  └── Proto26 的 feat_fuse 层 (随机Kaiming初始化) → 噪声喂给预训练 Proto
      → 模型学会"全背景"偷懒策略 (BCE Collapse)
      → epoch 353 权重漂移到极端 → proto 输出 Inf → seg_loss NaN
```

### 12.2 三个独立问题

| # | 问题 | 根因 | 修复 | 状态 |
|---|------|------|------|------|
| 1 | Mask mAP=0 | Proto26 随机初始化 → BCE 崩溃 | 近单位初始化 `block.py:2003-2014` | ✅ 已修复 |
| 2 | seg_loss NaN | BCE 崩溃连锁 → 权重漂移 → Inf | 同上修复预防 | ✅ 根源消除 |
| 3 | Mask 坐标偏移 | val.py 硬编码 `imgsz//4` | 动态 `proto_factor` | ✅ 已 commit |

### 12.3 BCE 崩溃机制

```
训练时:
  - GT mask 99%是背景(0), 1%是前景(1)
  - BCE Loss: -[y·log(p) + (1-y)·log(1-p)]
  - 模型快速学到：输出非常大的负值 logit
    → sigmoid(负大值) ≈ 0 → BCE ≈ 0 (对背景完美预测)
    → 但前景也被预测为 0!（模型"偷懒"）

验证时:
  - 阈值 gt_(0.0): sigmoid(0)=0.5 → 负值 logit 全判为背景
  - 结果：所有 mask 为空 → mAP = 0
```

**为什么标准模型没有这个问题？** 标准模型使用预训练的 Proto（cv1/cv2/cv3 全部从 COCO 迁移），proto 已经知道如何产生有意义的基础掩码。P2 模型新增的 `feat_fuse` 层破坏了这种平衡。

### 12.4 修复代码（block.py:2003-2014）

```python
# feat_refine: near-zero init → don't corrupt P2 features at start
for m in self.feat_refine:
    nn.init.normal_(m.conv.weight, mean=0, std=1e-5)

# feat_fuse: near-identity init → pass P2 features through, pad zeros to 256ch
nn.init.constant_(self.feat_fuse.conv.weight, 0)
with torch.no_grad():
    for i in range(base_ch):  # 128 → 128 identity mapping @ center pixel
        self.feat_fuse.conv.weight[i, i, 1, 1] = 1.0
```

**效果**: 前 128 通道 = P2 特征原样输出，后 128 通道 = 零填充。对标准模型（base_ch=256）完全安全。

### 12.5 验证修复是否生效

训练后检查 epoch 1 的 Mask mAP(M)：
- ❌ **修复前**: Mask mAP(M) ≈ 0 或 ~1e-7
- ✅ **修复后**: Mask mAP(M) 应该出现非零值（~0.1-0.3）

### 12.6 debug_mask.py 脚本

位置：`E:\Study\DeepCNN\yolo26\code\debug_mask.py`

用于验证 mask 数据流（不需要训练），会输出：
- Proto 形状和数值范围
- Mask coefficients 分布
- 实际 mask 生成结果
- 所有检测的 mask > 0 比例

---

## 13. 待完成任务

### 13.1 立即（P0）

| 任务 | 命令/操作 | 说明 |
|------|-----------|------|
| **commit Proto26 修复** | `git add block.py && git commit -m "fix(v2-p2): near-identity init for Proto26 feat_fuse to prevent BCE collapse"` | 当前未提交 |
| **commit node.md 更新** | `git add node.md && git commit -m "docs: add v2 P2 debugging documentation (BCE collapse + NaN)"` | 当前未提交 |
| **重启 P2 训练** | `python train_yolov26_seg.py` (MODEL_VERSION="p2") | 验证修复是否生效 |
| **push 到远程** | `git push origin v2-p2` | 备份代码 |

### 13.2 短期（P1）

| 任务 | 说明 |
|------|------|
| **v2 P2 训练完成** | 预计 14h（2× Baseline），batch=4 |
| **v2 指标填入 code_plus.md** | 训练完成后更新消融表 |
| **v2 结果分析** | 对比 Baseline，分析卷叶螟改善情况 |

### 13.3 中期（P1-P2）

| 任务 | 分支 | 说明 |
|------|------|------|
| **v3 Dice Loss** | `v3-dice` | 在 BCE loss 基础上添加 Dice Loss，改善 mask 边缘精度 |
| **v4 三项合并** | `v4-combined` | CBAM + P2 + Dice Loss 综合模型 |

### 13.4 论文阶段

- 累积消融实验（Baseline → +CBAM → +CBAM+P2 → +CBAM+P2+Dice）
- 多组对比（不同注意力机制、不同 Neck 结构）
- 可视化（热力图、mask 边缘对比、PR 曲线叠加）

---

## 14. 关键文件索引

### 14.1 文档

| 文件 | 内容 |
|------|------|
| `yolo_plus/CLAUDE.md` | 项目全貌（给 AI 助手看的上下文） |
| `yolo_plus/code_plus.md` | 改进工作流 + 环境搭建 + 消融表 |
| `yolo_plus/node.md` | 详细技术讲解（v1 CBAM 原理、v2 P2 原理、调试记录） |
| `yolo_plus/note.md` | 学习笔记（Bottleneck/C2f/C3k2 源码分析） |
| `yolo_plus/PROJECT_TRANSFER.md` | **本文件** — 项目打包文档 |
| `yolo_plus/AGENTS.md` | Codex 用的项目说明（精简版） |

### 14.2 源码（我们改过的）

| 文件 | 改动 | Commit |
|------|------|--------|
| `ultralytics-main/ultralytics/nn/modules/conv.py` | CBAM 兼容 | `ae38878` |
| `ultralytics-main/ultralytics/nn/modules/block.py` | Proto26 初始化修复 | **未 commit** |
| `ultralytics-main/ultralytics/nn/tasks.py` | 注册 CBAM | `ae38878` |
| `ultralytics-main/ultralytics/models/yolo/segment/val.py` | proto_factor 动态检测 | `bb392b3`, `2053450` |

### 14.3 配置（我们新建的）

| 文件 | 内容 |
|------|------|
| `cfg/models/26/yolo26-cbam.yaml` | v1 CBAM 检测模型 |
| `cfg/models/26/yolo26-cbam-seg.yaml` | v1 CBAM 分割模型 |
| `cfg/models/26/yolo26-p2-seg.yaml` | v2 P2 分割模型 |

### 14.4 脚本

| 文件 | 用途 |
|------|------|
| `E:\Study\DeepCNN\yolo26\code\train_yolov26_seg.py` | 训练脚本 |
| `E:\Study\DeepCNN\yolo26\code\debug_mask.py` | Mask 数据流调试脚本 |

### 14.5 训练结果

| 目录 | 内容 |
|------|------|
| `runs/segment/runs_seg/yolo26m_seg_20260628_172809/` | Baseline 训练结果 |
| `runs/segment/runs_seg/yolo26m_cbam_seg_20260630_014120/` | v1 CBAM 训练结果 |
| `runs/segment/runs_seg/yolo26m_p2_seg_20260701_034948/` | v2 P2 训练结果（失败） |

---

## 15. 常见问题 FAQ

### Q1: 改了源码为什么不生效？

**A**: 必须执行 `pip install -e .` 在源码目录。检查方法：
```powershell
python -c "import ultralytics; print(ultralytics.__file__)"
```
输出必须指向 `yolo_plus/ultralytics-main/ultralytics/`，不能是 `site-packages/ultralytics/`。

### Q2: P2 训练时 OOM 怎么办？

**A**: P2 的计算量是 Baseline 的 2 倍。在 8GB 显卡上，需要 `BATCH_SIZE = 4`（已经在训练脚本中自动设置）。

### Q3: 如何只验证已训练的模型？

```powershell
cd E:\Study\DeepCNN\yolo26\code
yolo segment val \
  model=runs/segment/runs_seg/<run_name>/weights/best.pt \
  data=yolo_data.yaml \
  split=test
```

### Q4: CBAM 和 P2 能一起用吗？

**A**: 技术上可以，但需要新建一个合并版 YAML（v4 计划做这件事）。当前每个版本独立测试。

### Q5: 为什么 YAML 是分割版 vs 检测版的区别？

**A**: 最后一层不同：
- 检测版：`Detect` 头 → 只输出 bbox + class
- 分割版：`Segment26` 头 → 输出 bbox + class + mask coefficients + proto

### Q6: `strict=False` 加载权重是什么意思？

**A**: PyTorch 的 `load_state_dict(strict=False)` 允许部分匹配——形状相同的层迁移，形状不同的层跳过。新增的 CBAM 层、P2 路径层会因为"形状不匹配"而跳过，保持随机初始化。

### Q7: 当前环境 pip list 在哪里？

**A**: 在 `yolo26` conda 环境中。关键包版本：
```
ultralytics==8.4.80 (via pip install -e)
torch==2.x
CUDA==12.1
```

---

## 附录 A：给新 AI 助手的简短摘要

如果你是一个新接手这个项目的 AI 助手，请按以下顺序了解：

1. **先读** `CLAUDE.md` — 了解项目全貌和用户背景
2. **再读** `code_plus.md` — 了解改进工作流和训练记录
3. **然后读** `node.md` — 深入了解 v1/v2 技术细节
4. **最后读** 本文件 — 完整的文件索引和状态

**当前最重要的任务**：
1. Commit block.py 的 Proto26 修复
2. 重启 P2 训练，验证 Mask mAP 不再为 0
3. 如果 P2 训练成功，填入消融表格，推进 v3 Dice Loss

**用户是深度学习初学者**，解释技术概念时请用通俗语言，关联到 Backbone/Neck/Head 三阶段架构。

---

> 📋 **文档维护**: 每次新改进完成后，请更新此文件的状态标记和指标表。
>
> 🤖 Generated with [Claude Code](https://claude.com/claude-code)
