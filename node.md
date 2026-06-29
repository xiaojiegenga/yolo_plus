# YOLO26 改进学习笔记 (node.md)

> 此文档记录 YOLO26 模型改进的详细过程、原理解析和代码讲解。
> 面向初学者 — 每个技术术语都会解释清楚。

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

### 预期效果

参考 YOLO-Pineapple 论文中 CBAM 的效果：
- CBAM 在注意力机制对比中 **mAP50 最高**（93.8%，优于 CA、SE、BAM、EMA）
- CBAM 在 **F1 分数上最优**（90.9%）
- 参数量增加极小（+0.07M，约 3%）

针对我们水稻害虫识别的两个问题：

| 问题 | CBAM 如何帮助 |
|---|---|
| 🔥 卷叶螟漏检率高 (52%) | 空间注意力定位小目标，通道注意力增强害虫纹理特征 |
| ⚠️ 整体 mAP50 | 加强特征表达，让 Backbone 提取的特征更有判别力 |

---

### 相关理论参考

- **CBAM 原论文**: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
- **SE-Net**: Hu et al., "Squeeze-and-Excitation Networks" — ChannelAttention 的思想来源
- **YOLO-Pineapple**: 本次改进的直接参考论文，CBAM 被证明在 UAV 图像小目标检测中最优

---

*下一节预告：v2 — P2 高分辨率特征层（针对小目标检测）*

---

