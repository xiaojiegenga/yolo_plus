# YOLO26 改进学习笔记 (node.md)

> 此文档记录 YOLO26 模型改进的详细过程、原理解析和代码讲解。
> 面向初学者 — 每个技术术语都会解释清楚。
>
> **2026-07-28 状态说明**：V1 CBAM 属于已完成实验；旧 V2 属于失败实现的复盘材料。旧 V2 已归档到 `archive/v2-p2-failed`，其中关于 BCE Collapse、近单位初始化和 Validator 修复的结论均未通过新的完整训练验证，不能视为已经解决。新版 V2 将从 `main` 独立开始，并在本文末尾追加新的学习记录。

---

## v1 改进：CBAM 注意力机制

**日期**: 2026-06-29 | **分支**: `v1-cbam` | **对比对象**: Baseline (main)

### 改了什么？

在 YOLO26 骨架网络（Backbone）的每个 C3k2 模块后面，插入了一个 **CBAM 注意力模块**。原来 Backbone 有 11 层，现在变成 15 层。

```
改进前 Backbone:                     改进后 Backbone:
  Conv                                 Conv
  Conv                                 Conv
  C3k2                                 C3k2
  Conv                     →           CBAM  ★ 新增
  C3k2                                 Conv
  Conv                                 C3k2
  C3k2                                 CBAM  ★ 新增
  Conv                                 Conv
  C3k2                                 C3k2
  SPPF                                 CBAM  ★ 新增
  C2PSA                                Conv
                                       C3k2
                                       CBAM  ★ 新增
                                       SPPF
                                       C2PSA
```

---

### 什么是注意力机制？（Attention Mechanism）

#### 大白话理解

想象你在看一张照片——你的眼睛不会平等地看每一个像素。你会**重点关注**某些区域（比如人脸），而**自动忽略**背景。这就是"注意力"。

计算机视觉中的注意力机制就是让神经网络也学会这种能力——**对重要的特征给更高的权重，对不重要的特征给更低的权重**。

#### 对你的水稻害虫检测有什么用？

你的数据集中有**稻纵卷叶螟（卷叶虫）**——虫子很小，颜色和稻叶接近，容易和背景混淆。CBAM 让网络学会：

1. **通道注意力**（Channel Attention）: "哪些特征通道最重要？"
   — 比如害虫边缘的纹理通道比颜色通道更重要
2. **空间注意力**（Spatial Attention）: "图像中哪些位置最重要？"
   — 直接告诉网络"看这里！这片叶子上有虫"

两种注意力**串行作用**：先用通道注意力筛选重要特征 → 再用空间注意力定位关键区域。

---

### CBAM 源码讲解

CBAM 代码在 `ultralytics/nn/modules/conv.py` 第 512-613 行，分为三个类：

#### 1. ChannelAttention（通道注意力）

```python
class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        self.pool = nn.AdaptiveAvgPool2d(1)    # ① 全局平均池化
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)  # ② 1×1卷积
        self.act = nn.Sigmoid()                 # ③ Sigmoid激活

    def forward(self, x):
        return x * self.act(self.fc(self.pool(x)))
```

**逐步讲解**：

| 步骤 | 操作 | 输入形状 | 输出形状 | 解释 |
|---|---|---|---|---|
| ① `pool` | 全局平均池化 | `[B,C,H,W]` | `[B,C,1,1]` | 把整张特征图压成 C 个数字，每个数字代表一个通道的"平均强度" |
| ② `fc` | 1×1卷积 | `[B,C,1,1]` | `[B,C,1,1]` | 学习通道之间的依赖关系（哪些通道重要） |
| ③ `act` | Sigmoid | `[B,C,1,1]` | `[B,C,1,1]` | 把值压到 0~1 之间，变成"权重系数" |
| ④ `x * ...` | 逐通道相乘 | — | `[B,C,H,W]` | 用权重系数缩放每个通道的特征 |

> 💡 **关键理解**：步骤 ① 把空间信息压掉（H×W → 1×1），只保留通道信息（这一步叫"压缩"）。然后学习哪些通道重要（②③），最后重新加权（④）。这就是经典的 **SE（Squeeze-and-Excitation）** 思想。

#### 2. SpatialAttention（空间注意力）

```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, 1, keepdim=True)   # ① 沿通道方向求平均
        max_out, _ = torch.max(x, 1, keepdim=True)  # ② 沿通道方向求最大
        x_cat = torch.cat([avg_out, max_out], 1)    # ③ 拼接成2通道
        return x * self.act(self.cv1(x_cat))         # ④ 卷积 → Sigmoid → 加权
```

**逐步讲解**：

| 步骤 | 操作 | 输入 | 输出 | 解释 |
|---|---|---|---|---|
| ① `mean` | 通道平均 | `[B,C,H,W]` | `[B,1,H,W]` | 每个像素位置在所有通道上求平均 — "平均来看这里重不重要" |
| ② `max` | 通道最大值 | `[B,C,H,W]` | `[B,1,H,W]` | 每个像素位置取所有通道的最大值 — "最突出的特征在这里有多强" |
| ③ `cat` | 拼接 | 2×`[B,1,H,W]` | `[B,2,H,W]` | 把平均和最大拼成 2 通道特征图 |
| ④ `cv1` | 7×7卷积 | `[B,2,H,W]` | `[B,1,H,W]` | 用大卷积核（7×7=感受野大）学出"哪里值得关注" |
| ⑤ `act` | Sigmoid | `[B,1,H,W]` | `[B,1,H,W]` | 生成 0~1 之间的空间权重图 |

> 💡 **为什么要同时用 mean 和 max？** 平均池化捕捉整体的平滑信息，最大池化捕捉最突出的信号（比如虫子的边缘）。两者互补，拼接后一起学习，比只用一种效果好。

#### 3. CBAM（组合模块）

```python
class CBAM(nn.Module):
    def __init__(self, c1, c2=None, kernel_size=7):
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)   # 第一步：通道注意力
        x = self.spatial_attention(x)   # 第二步：空间注意力
        return x
```

**执行顺序**：输入特征 → ChannelAttention → SpatialAttention → 增强后的特征

> 💡 **为什么先通道后空间？** 论文实验发现这个顺序效果最好。先选出重要的"特征类型"（通道），再在这些特征上定位"关键位置"（空间），逻辑上更合理。

---

### 为什么之前 CBAM 不能用？

虽然 CBAM 的 Python 代码一直存在于 `conv.py` 中，但是在 YOLO 框架中，**只有注册过的模块才能被 YAML 配置文件使用**。

注册表在 `tasks.py` 的 `base_modules` 集合中（约 1800 行）。这个集合是一个"白名单"—— YAML 中的每个模块名都要在这个名单里才能被识别。

我们的改动：
1. 在 `base_modules` 中添加了 `CBAM`
2. 在 `tasks.py` 顶部的 import 中添加了 `CBAM`
3. 修改了 CBAM 的构造函数，让它兼容框架的调用方式

---

### 代码改动清单

#### 改动 1：`conv.py` — CBAM 构造函数

**位置**: `ultralytics-main/ultralytics/nn/modules/conv.py` 第 593 行

**改动内容**：添加了一个 `c2=None` 参数

```python
# 改前：
def __init__(self, c1, kernel_size=7):

# 改后：
def __init__(self, c1, c2=None, kernel_size=7):
```

**为什么需要这个改动？**

YOLO 框架在构建模型时，对 `base_modules` 中的模块采用统一的调用约定：
```python
args = [c1, c2, *args[1:]]  # 框架自动插入 c1 和 c2
m = Module(*args)            # 调用模块构造函数
```

也就是说框架会传 `(c1, c2, kernel_size)` 三个参数。但原来 CBAM 只接受 `(c1, kernel_size)` 两个参数——多出来的 `c2` 会导致报错。加上 `c2=None` 后，这个多余参数被接收并忽略（CBAM 的输入输出通道数相同，不需要 c2）。

#### 改动 2：`tasks.py` — 注册 CBAM

**位置**: `ultralytics-main/ultralytics/nn/tasks.py`

**改动 A**（第 40 行附近）：添加 import
```python
# 在 import 块中添加了：
CBAM,
```

**改动 B**（第 1836 行附近）：添加到 base_modules
```python
base_modules = frozenset({
    ...
    A2C2f,
    CBAM,        # ← 新增
})
```

> 💡 **注意**：CBAM 只加入 `base_modules`，不加入 `repeat_modules`。因为 CBAM 没有"重复次数"参数（不像 C3k2 那样可以 `[-1, 2, C3k2, ...]` 重复 2 次）。

#### 改动 3：新建 `yolo26-cbam.yaml`

**位置**: `ultralytics-main/ultralytics/cfg/models/26/yolo26-cbam.yaml`

**核心变化**：在 Backbone 的每个 C3k2 后面插入一行 CBAM：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]       # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]] # 2
  - [-1, 1, CBAM, [256, 7]]          # 3  ★ 新增
  - [-1, 1, Conv, [256, 3, 2]]       # 4-P3/8
  ...
```

**YAML 语法解释**：
- `[-1, 1, CBAM, [256, 7]]` 每部分的含义：
  - `-1`：输入来自上一层（previous layer）
  - `1`：该模块只执行 1 次（不重复）
  - `CBAM`：模块名称，对应 Python 的 `CBAM` 类
  - `[256, 7]`：参数列表 [通道数, 空间注意力卷积核大小]

**索引更新**：因为 Backbone 插入了 4 个 CBAM 层，Head 中引用 Backbone 层的索引需要更新：

| 连接 | 原索引 | 新索引 | 含义 |
|---|---|---|---|
| `[-1, 4]` | 4 (P3 C3k2) | **5** | FPN 连接 P3 特征 |
| `[-1, 6]` | 6 (P4 C3k2) | **8** | FPN 连接 P4 特征 |
| `[-1, 10]` | 10 (C2PSA) | **14** | PAN 连接 P5 特征 |

---

### 参数对比

| 指标 | 原始 yolo26m | yolo26m-cbam | 变化 |
|---|---|---|---|
| 层数 | 280 | 300 | +20 |
| 参数量 | 21.78 M | 22.63 M | +0.85 M (+3.9%) |
| 计算量 | 74.7 GFLOPs | 75.4 GFLOPs | +0.7 GFLOPs (+0.9%) |

> 💡 CBAM 的参数量和计算量增加非常小（不到 4%），因为它主要做的是"加权"操作——只有几个 1×1 卷积和 7×7 卷积，没有大幅增加特征通道数。

---

### 为什么 CBAM 放在 C3k2 后面？

C3k2 是特征提取模块，它输出的特征图经过了充分的卷积处理。在这之后加 CBAM：

1. **特征已经丰富**：C3k2 已经提取了多尺度特征，CBAM 在上面做"筛选"效果好
2. **不下采样前加**：在 Conv(步长=2，即下采样) 之前加 CBAM，可以让网络在"压缩"特征前先做好注意力加权，保留重要信息
3. **符合论文实践**：YOLO-Pineapple 论文也是这样放置的

---

### 如何训练？

由于 CBAM 改变了模型结构，不能直接用预训练权重。需要用以下方式加载：

```python
from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load, DetectionModel
from copy import deepcopy

# Step 1: 加载 CBAM YAML，指定 m 尺度
cfg = yaml_model_load("path/to/yolo26-cbam.yaml")
cfg["scale"] = "m"          # 强制使用 m 尺度（因为文件名不含 'm'）
cfg["nc"] = 2               # 2 类：卷叶螟 + 钻心虫

# Step 2: 创建模型
model = DetectionModel(deepcopy(cfg), nc=2)

# Step 3: 加载预训练权重（形状相同的层会迁移，CBAM 层随机初始化）
import torch
ckpt = torch.load("path/to/yolo26m-seg.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"].float().state_dict(), strict=False)

# Step 4: 用 YOLO API 训练
from ultralytics import YOLO
wrapper = YOLO("dummy.yaml")  # 占位
wrapper.model = model
wrapper.task = "segment"
wrapper.train(
    data="yolo_data.yaml",
    epochs=400,
    imgsz=640,
    batch=8,
    # ... 其他参数和 Baseline 完全一致
)
```

> ⚠️ **重要提醒**：训练参数（epochs=400, batch=8, lr0=0.01 等）必须和 Baseline 完全一致，否则无法公平对比！

---

### 实际训练结果 ✅

**训练时间**: 2026-06-30 | **Best epoch**: 250 (early stop 350) | **耗时**: 3.51 小时

#### 指标对比

| 指标 | Baseline | v1 CBAM | 变化 |
|---|---|---|---|
| **整体 P(M)** | 0.717 | **0.753** | ↑ +5.0% |
| **整体 R(M)** | 0.620 | **0.633** | ↑ +2.1% |
| **mAP50(M)** ⭐ | **0.683** | **0.700** | ↑ +2.5% |
| mAP50-95(M) | 0.329 | **0.343** | ↑ +4.3% |
| | | | |
| 卷叶螟 P(M) | 0.679 | 0.662 | ↓ -0.017 |
| **卷叶螟 R(M)** | 0.482 | **0.552** | ↑ +14.5% |
| 卷叶螟 AP50(M) | 0.604 | **0.607** | ↑ +0.5% |
| 卷叶螟漏检率 | 52% | **45%** | ↓ -7pp |
| | | | |
| **钻心虫 P(M)** | 0.753 | **0.844** | ↑ +12.1% |
| 钻心虫 R(M) | 0.755 | 0.714 | ↓ -0.041 |
| 钻心虫 AP50(M) | 0.763 | **0.793** | ↑ +3.9% |
| Box mAP50 | 0.667 | **0.694** | ↑ +0.027 (+4.0%) |

#### 分析解读 (含每类 Mask P/R)

**整体效果**:
- 所有核心指标全面上升，无一下降 — CBAM 是纯粹的正向改进
- mAP50 +1.7pp, mAP50-95 +1.4pp (相对提升 4.3%，最大)

**卷叶螟 (主要问题类别)**:
| | P(M) | R(M) | 解读 |
|---|---|---|---|
| Baseline | 0.679 | 0.482 | P高R低 — 找到的准确但不全 |
| CBAM | 0.662 | 0.552 | R大幅提升(+14.5%), P轻微下降(-0.017) |

→ CBAM 的空间注意力帮助网络在更多位置"看到"虫子，但区分能力略有下降。这是合理的 P-R tradeoff：发现了更多疑似目标，但部分预测精度不够高。
→ 漏检率 52%→45%，有实质改善但仍不够

**钻心虫**:
| | P(M) | R(M) | 解读 |
|---|---|---|---|
| Baseline | 0.753 | 0.755 | P和R平衡 |
| CBAM | 0.844 | 0.714 | P大幅提升(+12.1%), R下降(-0.041) |

→ CBAM 显著减少了钻心虫的误检（Precision↑），但漏掉了一小部分。通道注意力可能帮助抑制了背景噪音，减少了对非虫区域的误判。这是有利的 tradeoff（P升幅 > R降幅）。

**总结**: CBAM 作为第一步改进温和但有效。对于中期答辩，这是一个清晰的"正向改进"案例。核心问题（卷叶螟漏检 45%）的真正突破需要 v2 P2 高分辨率层来直接提升对小目标的感知能力。

---

### 相关理论参考

- **CBAM 原论文**: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
- **SE-Net**: Hu et al., "Squeeze-and-Excitation Networks" — ChannelAttention 的思想来源
- **YOLO-Pineapple**: 本次改进的直接参考论文，CBAM 被证明在 UAV 图像小目标检测中最优

---

*下一节预告：v2 — P2 高分辨率特征层（针对小目标检测）*

---

## v2 改进：P2 高分辨率特征层

**日期**: 2026-06-30 | **分支**: `v2-p2` | **对比对象**: Baseline (main)

### 改了什么？

在 YOLO26 的 Neck（FPN+PAN）中增加了 **P2/4 高分辨率特征层**，使 Neck 从原来的 3 个尺度（P3/P4/P5）变为 4 个尺度（P2/P3/P4/P5）。

```
改进前 Neck (3 尺度):                 改进后 Neck (4 尺度):

C2PSA (P5, 20×20)                    C2PSA (P5, 20×20)
  ↓ Upsample                            ↓ Upsample
  + P4 → C3k2 (P4, 40×40)               + P4 → C3k2 (P4, 40×40)
  ↓ Upsample                            ↓ Upsample
  + P3 → C3k2 (P3, 80×80) ★              + P3 → C3k2 (P3, 80×80)
  ↓ Conv(↓2)                            ↓ Upsample              ← 多一级!
  +       → C3k2 (P4, 40×40)             + P2 → C3k2 (P2, 160×160) ★ ← 新增!
  ↓ Conv(↓2)                            ↓ Conv(↓2)              ← 从 P2 开始下采样
  + P5   → C3k2 (P5, 20×20)              +     → C3k2 (P3, 80×80)
                                         ↓ Conv(↓2)
Detect: [P3, P4, P5]                     +     → C3k2 (P4, 40×40)
                                         ↓ Conv(↓2)
                                         + P5  → C3k2 (P5, 20×20)

                                       Segment26: [P2, P3, P4, P5] ★ 4个输出!
```

**Backbone 完全不变**（层 0-10 和原始 yolo26 一模一样），只有 Neck 加了 6 层。

---

### 为什么 P2 对小目标检测至关重要？

#### 大白话理解

想象你要在一张航拍照片里找到一只 2 毫米长的虫子。

YOLO 把图像逐级缩小来提取特征：
- **P5** (32×下采样): 相当于从 20 米高空看 — 虫子几乎看不见
- **P4** (16×下采样): 从 10 米看 — 虫子有几个像素
- **P3** (8×下采样): 从 5 米看 — 虫子的轮廓勉强可见
- **P2** (4×下采样): 从 2.5 米看 — **能看清虫子细节！**

P2 的分辨率是 P3 的 **4 倍**（160×160 vs 80×80），意味着同样的虫子占据的像素多了 4 倍，网络有更多信息来判断。

#### 用数据说话

以 640×640 输入为例：

| 特征层 | 下采样倍率 | 特征图尺寸 | 每格感受野 | 适合检测的物体大小 |
|--------|-----------|-----------|-----------|------------------|
| P2/4 | 4× | 160×160 | ~16×16 px | ★ 极小的虫子 |
| P3/8 | 8× | 80×80 | ~32×32 px | 中小目标 |
| P4/16 | 16× | 40×40 | ~64×64 px | 中等目标 |
| P5/32 | 32× | 20×20 | ~128×128 px | 大目标 |

> 📊 **你的数据**：卷叶螟在 640×640 图像中平均占 ~25×25 像素。在 P3（80×80 特征图）上，它只对应约 3×3 个格点 — 信息极度压缩。在 P2（160×160 特征图）上，对应约 6×6 个格点 — **有效信息量翻倍**。

#### Mask 分辨率也提升了！

Baseline 的 Mask Proto 输出是 160×160（基于 P3 上采样）。P2 版本的 Mask Proto 输出是 **320×320** — 4 倍像素，mask 边缘更精细。

---

### YAML 配置讲解

完整的 P2 Neck 在 `yolo26-p2-seg.yaml` 中。我们只关注和原始不同的部分：

#### FPN 多了 P2 路径（第 17-19 行）

```yaml
  # 原来的 FPN 只到 P3:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 16  upsample to P3
  - [[-1, 4], 1, Concat, [1]]                      # 17  cat backbone P3
  - [-1, 2, C3k2, [256, True]]                     # 18  P3 FPN → 输出到检测头

  # P2 版本继续 upsample 一级:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 19  upsample to P2 ★
  - [[-1, 2], 1, Concat, [1]]                      # 20  cat backbone P2 ★
  - [-1, 2, C3k2, [128, True]]                     # 21  P2 FPN → 新输出! ★
```

`[[-1, 2], 1, Concat, [1]]` 中 `[-1, 2]` 表示把上一层（上采样后的特征）和 Backbone **第 2 层**（P2 特征）拼接。Backbone 的第 2 层正是 `C3k2` 输出的 P2 特征（160×160, 256ch）。

#### PAN 多了 P2→P3 路径

P2 输出后需要通过下采样回到更大尺度，参与后续的 PAN 路径：

```yaml
  - [-1, 1, Conv, [128, 3, 2]]   # 从 P2(160×160) 下采样到 80×80
  - [[-1, 16], 1, Concat, [1]]    # 和 P3 FPN 拼接
  - [-1, 2, C3k2, [256, True]]   # P3 PAN → 输出到检测头
```

#### 检测头接收 4 个输入

```yaml
  - [[19, 22, 25, 28], 1, Segment26, [nc, 32, 256]]
  #   ↑   ↑   ↑   ↑
  #   P2  P3  P4  P5  ← 4个尺度的特征图
```

原始是 `[[16, 19, 22], 1, Segment26, [nc, 32, 256]]` — 只有 3 个。

---

### 为什么不需要改任何模块代码？

这是一个纯 YAML 级别的改动，因为：

1. **Segment26 头天然支持可变输入数**：`cv2`/`cv3`/`cv4` 都是 `nn.ModuleList`，用 `for x in ch` 遍历输入通道列表，3 个输入就建 3 个分支，4 个输入就建 4 个分支
2. **Proto26 也天然支持**：`feat_refine` 同样用 `for x in ch[1:]` 遍历，自动适配
3. **所有 Neck 层用的都是已有模块**（Upsample / Concat / C3k2 / Conv）

这是 P2 改动比 CBAM 优雅的地方 — CBAM 需要注册新模块，P2 完全不用。

---

### 改动清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `ultralytics/cfg/models/26/yolo26-p2-seg.yaml` | **新建** — P2 分割模型 YAML |
| 2 | `E:\Study\DeepCNN\yolo26\code\train_yolov26_seg.py` | 添加 `_build_p2_model()` 和 `MODEL_VERSION = "p2"` 分支 |

> 💡 **不需要改** `tasks.py`、`conv.py`、`block.py`、`head.py` — 全部现有模块即用。

---

### 参数对比

| 指标 | Baseline (yolo26m-seg) | v2 P2 (yolo26m-p2-seg) | 变化 |
|------|----------------------|----------------------|------|
| 层数 | 329 | 410 | +81 |
| 参数量 | 27.11 M | 24.76 M | **-2.35 M** ↓ |
| 计算量 | 132.5 GFLOPs | 280.5 GFLOPs | +148 GFLOPs |
| Mask Proto 尺寸 | 160×160 | **320×320** | 4× |
| 检测头输入数 | 3 (P3/P4/P5) | **4 (P2/P3/P4/P5)** | +1 |

> ⚠️ **计算量增加了约 2 倍**！这是因为 P2 尺度特征图是 P3 的 4 倍像素，所有在 P2 上的卷积运算量都更大。这是高分辨率检测的代价。但参数量反而少了（P2 分支通道更小），模型文件不会大很多。
>
> ⏱️ **训练时间预估**：约 2× Baseline 训练时间（~14h vs ~7h），因为 P2 尺度的计算量增加。

---

### 权重迁移策略

由于 Backbone 完全不变且 FPN 前半部分（P5→P4→P3）也完全相同：

| 层范围 | 原始编号 | 新编号 | 迁移方式 |
|--------|---------|--------|---------|
| Backbone | 0-10 | 0-10 | ✅ 直接匹配 |
| FPN (P5→P4→P3) | 11-16 | 11-16 | ✅ 直接匹配 |
| P2 FPN (新) | — | 17-19 | 🔄 随机初始化 |
| P2→P3 PAN (新) | — | 20-22 | 🔄 随机初始化 |
| PAN 后半 | 17-22 | 23-28 | ✅ 偏移 +6 重映射 |
| Segment26 头 | 23 | 29 | ❌ 4输入≠3输入，不能用 |

**结果**：636 个 key 成功迁移（所有 Backbone + FPN + PAN），268 个 key 跳过（检测头 + Proto），496 个 key 随机初始化（P2 路径 + 新头）。

---

### 预期效果

| 指标 | 预期变化 | 原因 |
|------|---------|------|
| **卷叶螟 R(M)** | ↑↑ 大幅提升 | P2 高分辨率让小虫子"可见" |
| **卷叶螟漏检率** | 45% → <30% | 4倍像素帮助发现小目标 |
| mAP50-95(M) | ↑ 提升 | Mask 分辨率翻倍，边缘更准 |
| **训练时间** | ↑ 约 2× | P2 尺度计算量翻倍 |
| 钻心虫 mAP50 | ≈ 持平或微涨 | 钻心虫已经较大，P2 帮助有限 |

---

### 相关理论参考

- **FPN 原论文**: Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017 — 提出多尺度特征金字塔
- **PANet**: Liu et al., "Path Aggregation Network for Instance Segmentation", CVPR 2018 — FPN+PAN 双向路径
- **YOLOv8-p2**: Ultralytics 官方也提供了 P2 变体用于小目标，yolo26 继承了这个设计思想

---

### 🐛 v2 调试记录：Mask mAP=0 与 seg_loss NaN

**记录日期**: 2026-07-01 | **训练轮次**: epoch 1-357 | **状态**: 旧实现已归档，修复假设未验证

---

#### 问题 1：Mask mAP 从头到尾为 0

**现象**：

| Epoch | Box mAP50(B) | Mask mAP50(M) | train/seg_loss |
|-------|-------------|---------------|----------------|
| 1 | 0.346 | **0.00001** | 3.66 |
| 100 | 0.559 | **0** | 1.38 |
| 248 (best) | **0.666** ✅ | **0** ❌ | — |

Box 检测正常（epoch 248 达到 0.666，接近 baseline 0.667），但 Mask 从未学到任何东西。

**诊断过程**：

1. **初步怀疑**：val.py 中 bbox→proto 坐标映射错误
   - 旧代码硬编码 `shape=[640,640]`，对于矩形推理图片（如 512×672），proto 坐标映射会偏移 5-20%
   - **修复**：改为每张图片动态计算 `imgsz = proto_factor × proto_spatial_size`
   - 修复后 Mask mAP 从 0.002 变成了更低的 1e-7 → **说明这不是根因**（两个值都是噪声底）

2. **深度诊断**：运行 `debug_mask.py` 检查数据流
   - 所有数据流正确：proto(320×320)、mask coefficients(32ch)、process_mask 都正常
   - 模型输出形状验证通过：`output[0][0]=(1,300,38)`, `output[0][1]=(1,32,320,320)`
   - **关键发现**：随机噪声输入下，所有检测框都在图像边缘（x1≈616），说明检测头部分工作异常

3. **当时的主要怀疑**：Proto26 的 **权重初始化问题**

   ```
   Proto26 数据流:
     P2(128ch, 160×160) → feat_fuse(随机Conv 128→256) → cv1(预训练!)
                          ↑ 随机权重 = 噪声!

   cv1 预训练时接收的是 P3 特征(256ch)
   现在接收 feat_fuse 的随机噪声(256ch) → proto = 纯噪声

   模型学到: 输出全负值 → sigmoid(负数) ≈ 0 → BCE loss 低
   验证时: 阈值 0 → 全空 mask → Mask mAP = 0
   ```

**"BCE 崩溃"（BCE Collapse）机制详解**：

旧调试记录曾使用“BCE Collapse”解释这一现象：

```
训练:
  - GT mask 大部分是背景（0），只有小部分前景（1）
  - BCE Loss: -[y·log(p) + (1-y)·log(1-p)]
  - 模型快速学到：输出非常大的负值 logit
    → sigmoid(负大值) ≈ 0 → BCE ≈ 0（对背景完美预测）
    → 但前景也被预测为 0!
  - 前景像素太少，梯度不足以纠正这个"偷懒"策略

验证:
  - 阈值 gt_(0.0): sigmoid(0)=0.5 → 负值 logit 全判为背景
  - 结果：所有 mask 为空 → mAP = 0
```

> 💡 为什么标准模型没有这个问题？标准模型使用预训练的 Proto（cv1/cv2/cv3/upsample 全部从 COCO 迁移），proto 已经知道如何产生有意义的基础掩码。P2 模型新增的 `feat_fuse` 层破坏了这种平衡。

**未验证的修复尝试 — 近单位初始化（Near-Identity Init）**：

```python
# block.py → Proto26.__init__ (末尾新增)

# ① feat_refine: 近零初始化 → 初始不添加噪声
for m in self.feat_refine:
    nn.init.normal_(m.conv.weight, mean=0, std=1e-5)

# ② feat_fuse: 近恒等映射 → P2特征原样通过，零填充到256通道
nn.init.constant_(self.feat_fuse.conv.weight, 0)
with torch.no_grad():
    for i in range(base_ch):  # 128 通道恒等映射
        self.feat_fuse.conv.weight[i, i, 1, 1] = 1.0
```

**修复原理图示**：

```
修复前:
  P2(128ch) → feat_fuse(随机权重) → 256ch 随机噪声 → cv1(预训练) → 垃圾 proto
                                        ↑ cv1 期望 P3 特征，收到噪声

修复后:
  P2(128ch) → feat_fuse(近恒等) → [P2, 0×128] → cv1(预训练) → 有意义的 proto
                ↑ 前128通道=原样P2, 后128通道=零填充
```

> ⚠️ 该初始化方案只保存在失败归档分支中，没有通过重新训练验证，也没有证明对标准模型和所有尺度都安全。新版 V2 不继承该改动。

---

#### 问题 2：seg_loss 在 epoch 353 变为 NaN

**现象**：

```
Epoch 353: train/seg_loss = NaN, val/seg_loss = NaN (同时 EMA 包含 NaN)
Epoch 354-357: seg_loss 持续 NaN
Box/Cls/DFL/Sem Loss 全部正常 → 问题只存在于 mask 分支
```

**当时的根因推测**：这是问题 1 的连锁反应

```
BCE 崩溃 → seg_loss 的 mask 分支长期无效
  → Proto26 权重漂移到极端状态
  → epoch 353 某 batch 触发梯度爆炸
  → proto 输出 Inf 值
  → BCE loss 传播 NaN
  → EMA 被污染，模型无法恢复
```

`single_mask_loss` (loss.py:574) 是 NaN 的发生地：
```python
pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # 系数 × proto
loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, ...)
# 当 proto 包含 Inf 时：pred_mask=Inf → BCE=Inf → 反向传播=NaN
```

> ⚠️ 梯度裁剪 (`clip_grad_norm(max_norm=10.0)`) 无法防止前向传播中产生的 Inf——它只能限制反向传播的梯度范数。

---

#### 问题 3：Mask 坐标映射修复尝试

**现象**：P2 模型 epoch 1 Box mAP=0.251 但 Mask mAP≈0（在问题 1 修复之前）

**根因**：`segment/val.py` 中 `postprocess` 硬编码 `shape=[640,640]`，而 `_prepare_batch` 硬编码 `mask_size = imgsz // 4`

```
标准模型: proto=imgsz/4 → factor=4 → 正确
P2 模型:  proto=imgsz/2 → factor=2 → mask_size 应该是 imgsz//2 而不是 imgsz//4!
```

**旧实现中的修复尝试**（commit `2053450`）：

| 方法 | 修复内容 |
|------|---------|
| `init_metrics` | 自动检测 `proto_base_idx`，设置 `_proto_factor`（P2=2, 标准=4） |
| `postprocess` | `imgsz = [proto_factor * h, proto_factor * w]` 代替硬编码 `[640,640]` |
| `_prepare_batch` | `mask_size = imgsz // proto_factor` 代替硬编码 `imgsz // 4` |

> ⚠️ 后续审计发现，旧 `_proto_factor` 自动识别还可能因训练验证阶段传入的模型对象层级不同而失效，并被宽泛异常处理静默吞掉。因此该提交不能视为最终正确修复。新版 V2 将保持标准 P3 Proto，避免修改 Validator。

---

#### 修复时序总结

```
训练尝试 #1:
  ├── val.py 未修复 → Box mAP=0.251, Mask mAP≈0 (epoch 1)
  ├── val.py 修复后 → Box mAP=0.306, Mask mAP≈0 (epoch 1)
  │   └── 原因: Proto26 随机初始化 → BCE 崩溃 → 真正的 Mask=0
  ├── Epoch 100: Box mAP=0.559, Mask mAP=0 (持续崩溃)
  ├── Epoch 248 (best.pt): Box mAP=0.666, Mask mAP=0
  ├── Epoch 353: seg_loss=NaN (BCE 崩溃的最终结果)
  └── Epoch 357: 手动停止

未验证尝试 #1: Proto26 近单位初始化 (block.py)
未验证尝试 #2: val.py proto_factor 动态检测

决定:
  └── 不再继续修补该实现；旧 V2 整体归档，从 main 重做单变量 P2
```

---

#### 涉及的关键文件

| 文件 | 改动 | 状态 |
|------|------|------|
| `ultralytics/nn/modules/block.py` | Proto26 初始化尝试 | 仅在失败归档分支 |
| `ultralytics/models/yolo/segment/val.py` | mask 坐标映射尝试 (+proto_factor 检测) | 仅在失败归档分支 (`2053450`) |
| `E:\Study\DeepCNN\yolo26\code\train_yolov26_seg.py` | P2 模型构建 + 权重迁移 | 无变化 |

---

#### 经验教训

1. **新增层 + 预训练模型 ≠ 自动生效**：在预训练路径中间插入新层可能改变特征分布。是否需要近恒等或近零初始化必须通过对照实验验证，不能只根据张量形状下结论。

2. **不能用 Dice 掩盖代码错误**：当 Mask mAP 接近 0 时，应先验证数据流、坐标、Proto、权重加载和预测可视化。Dice 可以是独立改进，但不是未知代码问题的默认补丁。

3. **NaN 诊断要分层排查**：先看哪些 loss 是 NaN（只有 seg_loss）→ 缩小范围到 mask 计算链 → 追溯到 Proto26 的随机层。

4. **Box 和 Mask 的训练进度应同步**：如果 Box mAP 持续上升但 Mask mAP 停滞，说明 mask 分支有独立的代码或初始化问题，而不是整体训练失败。

---

## 2026-07-28 项目清理与新版 V2 学习起点

### 为什么舍弃旧 V2

旧 V2 同时修改了 Neck、Segment Head、Proto、Validator、权重映射和 batch。变量过多，导致失败后无法确定具体原因。

新版 V2 只研究：

> 增加 P2 小目标预测尺度，是否能提升卷叶螟 Recall 和 Mask AP。

### 新版 V2 暂时保持不变的部分

- 不加入 CBAM；
- 不加入 Dice；
- Mask Proto 继续使用 P3；
- 不修改全局 Proto26；
- 不修改 SegmentationValidator；
- 不进行额外超参数优化。

### 新学习记录的顺序

新版 V2 开始后，按以下顺序追加：

1. Baseline Neck 与 P3/P4/P5 数据流；
2. 官方 `yolo26-p2.yaml` 的 P2 Neck；
3. Segment26 的检测尺度与 Proto 输入区别；
4. 新 YAML 每一层的来源和输出尺寸；
5. 预训练权重能够迁移哪些层；
6. 1 epoch 与短跑验证结果；
7. 正式训练和统一 Val 结果。

---

*下一节将在真正开始新版 V2 后追加，不提前写入未经验证的实现结论。*

---
