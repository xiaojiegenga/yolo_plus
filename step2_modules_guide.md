# Step 2 — 模块实现详解：从 Conv 到 C2PSA

> **本文档目的**：把 `ultralytics-main/ultralytics/nn/modules/` 下你需要读懂的 12 个核心类，按 "从原子到复合、从简单到复杂、并对齐 YOLO-pineapple 论文" 的顺序整理成一份**自洽**的讲解，方便你脱离对话独立阅读。
>
> **阅读前提**：已完成 Step 1（理解 `yolo26.yaml` 的 backbone / head 拓扑和 `[from, repeats, module, args]` 语法）。
>
> **建议节奏**：分 4 天读完。每读完一个类，建议做以下三件事：
> 1. 在 IDE 里打开对应行号，**亲眼看一遍源码**
> 2. 在白纸上**画一遍** `forward()` 的张量流图
> 3. 回到 `yolo26.yaml`，找到这个类被用到的那一行，**对照 args 是怎么映射到 `__init__` 参数的**

---

## 目录

- [通用心法：怎么读一个 nn.Module](#通用心法怎么读一个-nnmodule)
- [Tier 1 必读核心（按顺序）](#tier-1--必读核心按顺序)
  - [1. autopad — 自动 padding 助手](#1-autopad--自动-padding-助手)
  - [2. Conv — 网络的原子积木](#2-conv--网络的原子积木重中之重)
  - [3. Concat — 通道拼接](#3-concat--通道拼接)
  - [4. Bottleneck — 残差瓶颈块](#4-bottleneck--残差瓶颈块)
  - [5. C2f — YOLOv8 主力 CSP 块](#5-c2f--yolov8-主力-csp-块最重要的一节)
  - [6. C3k2 — YOLO26 的主力块](#6-c3k2--yolo26-的主力块)
  - [6.5 深入对比：C2f vs C3k2（含练习 Q&A）](#65-深入对比c2f-vs-c3k2含练习-qa)
- [Tier 2 与论文强相关](#tier-2--与论文强相关)
  - [7. SPP → SPPF — 多尺度池化](#7-spp--sppf--多尺度池化)
  - [8. PSA → C2PSA — 自注意力块](#8-psa--c2psa--自注意力块)
- [Tier 3 选读拓展](#tier-3--选读拓展)
  - [9. DWConv — 深度可分离卷积](#9-dwconv--深度可分离卷积)
  - [10. GhostConv — Ghost 模块](#10-ghostconv--ghost-模块)
  - [11. C3 — YOLOv5 时代的 block](#11-c3--yolov5-时代的-block)
- [整体回顾](#整体回顾)
- [下一步](#下一步)

---

## 通用心法：怎么读一个 nn.Module

PyTorch 中所有网络层都是 `nn.Module` 的子类，**只需要看两个方法**：

```
┌─────────────────────────────────────┐
│  __init__(...)  ← 准备好工具（一次） │
│                  → 看构造参数         │
│                  → 对照 YAML 的 args │
│                                     │
│  forward(x)     ← 实际干活（每次跑）  │
│                  → 看张量怎么流       │
│                  → 注意形状变化       │
└─────────────────────────────────────┘
```

类比：
- `__init__` = 厨师把刀、砧板、锅准备好（**只在创建模型时跑一次**）
- `forward` = 客人点菜后，厨师真正动手做菜（**每次喂一张图都跑一次**）

> ⚠️ **重要约定**：在 PyTorch 里你**不会**手动调 `model.forward(x)`，而是写 `model(x)` —— PyTorch 会自动调 forward 并埋好梯度记录。**但读代码时看 `forward()` 就是看模型的实际行为**。

**张量形状约定**（在 YOLO 整个仓库通用）：
```
[B, C, H, W]
 │  │  │  │
 │  │  │  └─ 宽 Width
 │  │  └──── 高 Height
 │  └─────── 通道数 Channels
 └────────── 批次大小 Batch
```

---

## Tier 1 — 必读核心（按顺序）

### 1. `autopad` — 自动 padding 助手

**位置**：`ultralytics/nn/modules/conv.py:30`

```python
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
```

**做什么**：当用户没指定 `padding` 时，自动算一个让"输出空间尺寸 = 输入 / stride"的 padding。

**口诀**：`padding = kernel_size // 2`
- `k=3` → `p=1`（这是 YOLO 里最常见的情况）
- `k=5` → `p=2`
- `k=1` → `p=0`（1×1 卷积不需要 padding）

**为什么重要**：这就是你看 YAML `Conv [64, 3, 2]` 里**没人显式写 padding** 的原因 —— autopad 帮你填了。

**难度**：⭐ —— 6 行代码扫一眼即可。

---

### 2. `Conv` — 网络的原子积木（重中之重）

**位置**：`ultralytics/nn/modules/conv.py:39`

```python
class Conv(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

YOLO26 整张网络的 **90% 操作都是 `Conv2d + BatchNorm + SiLU`** 这三件套，`Conv` 就是它们的打包。

#### 构造参数对照表

| 参数 | 含义 | YAML 中位置 |
|---|---|---|
| `c1` | 输入通道数 | 自动推断，**不在 YAML 写** |
| `c2` | 输出通道数 | `args[0]` |
| `k` | 卷积核大小 | `args[1]` |
| `s` | 步长 | `args[2]` |
| `p` | padding | 默认 None → autopad |
| `g` | 分组卷积组数 | 默认 1 |
| `d` | 空洞卷积扩张率 | 默认 1 |
| `act` | 激活函数 | 默认 True = SiLU |

举例：YAML 第 22 行 `[-1, 1, Conv, [64, 3, 2]]` →
- `c2=64, k=3, s=2`
- `c1=3`（因为输入是 RGB 三通道，框架自动算）

#### `forward()` 拆解 —— **必须读懂这一行**

```python
return self.act(self.bn(self.conv(x)))
```

这是 Python 函数嵌套调用，**从里往外读**：

```
       self.act( self.bn( self.conv(x) ) )
           ↑         ↑         ↑
        第3步     第2步     第1步
       激活函数  批归一化   卷积
```

等价于：

```python
def forward(self, x):
    y1 = self.conv(x)   # 第1步：卷积
    y2 = self.bn(y1)    # 第2步：批归一化
    y3 = self.act(y2)   # 第3步：激活
    return y3
```

#### 形状追踪：具体例子

以 `Conv(3, 64, k=3, s=2)` 作用在 `[1, 3, 640, 640]` 的图像上：

```
输入 x: [1, 3, 640, 640]   ← batch=1, RGB, 640×640
         │
         ▼  ┌─ self.conv (nn.Conv2d, autopad=1) ─┐
         │  │ 64 个 3×3 核扫过, stride=2 砍半     │
         │  └────────────────────────────────────┘
         ▼
       [1, 64, 320, 320]   ← 通道 3→64, 空间 640→320
         │
         ▼  ┌─ self.bn (BatchNorm2d, 64 通道) ──┐
         │  │ 标准化每通道分布, 形状不变          │
         │  └────────────────────────────────────┘
         ▼
       [1, 64, 320, 320]
         │
         ▼  ┌─ self.act (SiLU) ────────────────┐
         │  │ 非线性映射 y = x·sigmoid(x), 形状不变 │
         │  └────────────────────────────────────┘
         ▼
       [1, 64, 320, 320]   ← 最终输出 = YAML 第 0 行的 `P1/2`
```

#### 输出形状公式（必记）

```
H_out = floor( (H_in + 2*p - k) / s ) + 1
```

例：`H_in=640, p=1, k=3, s=2` → `(640 + 2 - 3) / 2 + 1 = 319 + 1 = 320` ✓

#### 卷积核如何"扫过"特征图（直观）

```
输入 5×5:                   3×3 核 sliding:                  输出特征图:
┌─┬─┬─┬─┬─┐                ┌─┬─┬─┐                          ┌─┬─┬─┐
│a│b│c│d│e│         ┌───►  │a│b│c│  ◄── 第1个窗口    ───►   │X│ │ │
├─┼─┼─┼─┼─┤         │      ├─┼─┼─┤                          ├─┼─┼─┤
│f│g│h│i│j│         │      │f│g│h│   对应位置相乘+求和         │ │ │ │
├─┼─┼─┼─┼─┤         │      ├─┼─┼─┤   得到 1 个标量 X         ├─┼─┼─┤
│k│l│m│n│o│         │      │k│l│m│                          │ │ │ │
├─┼─┼─┼─┼─┤         │      └─┴─┴─┘                          └─┴─┴─┘
│p│q│r│s│t│         │
├─┼─┼─┼─┼─┤         │  X = a·w₀+b·w₁+c·w₂+f·w₃+g·w₄+h·w₅+k·w₆+l·w₇+m·w₈
│u│v│w│x│y│         │  （w₀…w₈ 是这个核的 9 个可训练权重）
└─┴─┴─┴─┴─┘         │
                    │  然后窗口往右滑 stride 步（stride=2 表示跳 2 格）
```

**64 个不同的核 = 64 个不同的"探测器"**，每个负责识别不同的局部模式。输出通道数 = 核的数量。

#### SiLU 激活函数曲线

```
      y
      ↑          ╱─────  x 很大 → y ≈ x（接近直线）
      │        ╱
      │      ╱
      │    ╱
      │  ╱
  ────┼────────────► x
      │ ╲
      │  ╲___      x 是负数 → y 接近 0 但不完全为 0
      │            （比 ReLU 平滑, 梯度更友好）

公式：y = x · sigmoid(x) = x / (1 + e^(-x))
```

**为什么需要激活函数？** 没有它，无论叠多少层 Conv 都等价于 1 层（线性变换的组合还是线性）。激活函数引入**非线性**，才能让网络学习复杂模式。

#### 几个常见疑问

> **Q: 为什么 `nn.Conv2d(..., bias=False)`？**  
> A: 紧跟着的 BN 自带 β 偏置项，再加 bias 是冗余的。这是常见的"省参数小技巧"。

> **Q: BN 到底在做什么？**  
> A: 对每个通道，把一批数据的分布标准化：`y = (x - μ) / √(σ² + ε) · γ + β`。`γ`、`β` 是 2 个可训练参数（每通道一对）。**作用**：让训练更稳定、收敛更快。

> **Q: `forward_fuse` 是什么？**  
> A: 推理优化路径 —— 把 BN 的参数提前融进 Conv 的权重里，少做一次运算。**训练时不会走**，可暂时无视。

#### 练习题（验证理解）

1. `Conv [128, 3, 2]` 喂进 `[1, 64, 320, 320]`，输出形状？
2. 把 `forward` 改成 `bn(act(conv(x)))`（顺序调换）会怎样？
3. `Conv(3, 64, 3, 2)` 有多少可训练参数？

> **参考答案**（先自己想，再展开看）：
> 1. `[1, 128, 160, 160]`（c2=128 决定通道；s=2 决定空间砍半）
> 2. 先 SiLU 再 BN：SiLU 让负数趋近 0、正数大致保留 → BN 紧接着会重新调整分布，结果不一定坏，但破坏了"BN 标准化送进激活"的统计假设，训练稳定性会下降。这就是为什么标准模式是 Conv→BN→Act。
> 3. Conv2d 权重：`64 × 3 × 3 × 3 = 1728`（无 bias）。BN：`γ` 64 + `β` 64 = 128。**总共 1856 个可训练参数**。

**难度**：⭐⭐ —— 这一节读懂，YOLO 网络看懂一半。

---

### 3. `Concat` — 通道拼接

**位置**：`ultralytics/nn/modules/conv.py:616`

```python
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
```

整个类不到 10 行。

**做什么**：把一个**张量列表**沿指定维度拼接。在 YOLO 里 `dimension=1` 表示**通道维**拼接 —— 高和宽不变，通道数相加。

**对应 YAML**：所有 `[[-1, 6], 1, Concat, [1]]` 这种跳接拼接都靠它。

#### 形状示意

```
张量 A: [1, 256, 80, 80]   ┐
                            ├──► torch.cat([A, B], dim=1) ──► [1, 768, 80, 80]
张量 B: [1, 512, 80, 80]   ┘                                       └─256+512─┘
                                                          只通道相加, H/W 不变!
```

⚠️ **前提**：参与拼接的张量除了通道维其他维度**必须完全一致**。如果 H/W 不一致就会报错 —— 所以 Neck 里 `Concat` 之前总有 `Upsample` 把空间尺寸对齐。

**难度**：⭐ —— 1 分钟看懂。

---

### 4. `Bottleneck` — 残差瓶颈块

**位置**：`ultralytics/nn/modules/block.py:457`

```python
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)              # 隐藏层通道数 = c2 * expansion ratio
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```

**核心思想**：两个 Conv 串联，必要时加一条**残差连接（skip connection）**。

#### 数据流图

```
            ┌───────────────────────────────────────────────┐
            │  Bottleneck (shortcut=True, c1==c2)            │
            │                                                │
   x ────┬──┤ ┌──────────┐    ┌──────────┐                  │
         │  │ │   cv1    │    │   cv2    │                  │
         │  │ │ Conv 3×3 │───►│ Conv 3×3 │──┐               │
         │  │ └──────────┘    └──────────┘  │               │
         │  │                               ▼               │
         └─────────────────────────────────►(+)───► y       │
                  shortcut（残差直连）                       │
            └───────────────────────────────────────────────┘

   注意：cv1 把通道从 c1 压到 c_=c2*0.5（瓶颈），
        cv2 再从 c_ 扩回 c2 —— "先窄后宽" 的瓶颈结构
```

#### 关键变量

- `c_ = int(c2 * e)`：隐藏层通道数。`e=0.5` 表示中间通道数减半（"瓶颈"由此得名）。
- `self.add`：是否启用残差连接。**两个条件**：用户允许 (`shortcut=True`) **且** 输入输出通道一致 (`c1 == c2`)。
- `forward()` 用三目运算符：通道一致就 `x + cv2(cv1(x))`，否则就 `cv2(cv1(x))`。

#### 为什么要残差？

让深层网络易于训练。`y = x + F(x)` 意味着即使 `F(x)` 学得不好（接近 0），输出至少等于输入（信息不丢失），梯度也能通过 `+x` 这条路直接传到浅层 —— **缓解梯度消失**。这是 ResNet（2015）的核心贡献。

**难度**：⭐⭐ —— 5 分钟。

---

### 5. `C2f` — YOLOv8 主力 CSP 块（最重要的一节）

**位置**：`ultralytics/nn/modules/block.py:288`

```python
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)                                  # 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)                 # 入口：c1 → 2c
        self.cv2 = Conv((2 + n) * self.c, c2, 1)              # 出口：(2+n)c → c2
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))    # ① 先 Conv 升通道, 再切两半
        y.extend(m(y[-1]) for m in self.m)   # ② 把"右半"反复喂给 n 个 Bottleneck, 每个输出都留着
        return self.cv2(torch.cat(y, 1))     # ③ 全部 concat, 再 Conv 压回 c2
```

**这是整份文档最重要的一个类**。读懂它，后面 C3k2、C2PSA、PSA 全是它的变体。

#### 设计思想：CSP（Cross Stage Partial）

把特征通道**一分为二**：
- 一半"直通"，**保留浅层信息**
- 一半"深加工"，经过若干 Bottleneck **提取深层特征**
- 最后**全部 concat 起来**，把浅深特征都送到下一层

**好处**：既加深网络又不丢信息，参数效率比"全部走 Bottleneck"高得多。

#### `forward()` 三步逐行拆解

假设 `C2f(c1=64, c2=128, n=2, shortcut=True)`，输入 `x: [1, 64, 80, 80]`，`self.c = 128 * 0.5 = 64`。

**第 ① 步：`y = list(self.cv1(x).chunk(2, 1))`**

```
x: [1, 64, 80, 80]
   │
   ▼  cv1: Conv 1×1, 64 → 2*self.c = 128
   │
[1, 128, 80, 80]
   │
   ▼  .chunk(2, dim=1) — 沿通道维切成 2 块
   │
y = [
  y[0]: [1, 64, 80, 80],  ← 左半，"直通保留"
  y[1]: [1, 64, 80, 80],  ← 右半，"准备深加工"
]
```

**第 ② 步：`y.extend(m(y[-1]) for m in self.m)`**

每次 Bottleneck 处理上一次的输出（注意 `y[-1]` —— 总是取列表最后一个），处理结果**追加**到列表：

```
n = 2 个 Bottleneck

Bottleneck 0:                Bottleneck 1:
  输入 = y[-1] = y[1]          输入 = y[-1] = y[2]
  输出 → 追加为 y[2]            输出 → 追加为 y[3]

y = [
  y[0]: [1, 64, 80, 80]   ← cv1 切出来的左半
  y[1]: [1, 64, 80, 80]   ← cv1 切出来的右半
  y[2]: [1, 64, 80, 80]   ← Bottleneck 0 加工后
  y[3]: [1, 64, 80, 80]   ← Bottleneck 1 加工后
]
```

**第 ③ 步：`return self.cv2(torch.cat(y, 1))`**

```
torch.cat(y, 1): 4 个 [1, 64, 80, 80] 沿通道维拼接
                 = [1, 256, 80, 80]   ← 64 × 4 = (2+n)*c = 4*64

     │
     ▼  cv2: Conv 1×1, 256 → c2 = 128
     │
   [1, 128, 80, 80]   ← 最终输出
```

#### 完整数据流图（n=2 的情况）

```
              ┌──── y[0] ─────────────────────────────────┐
              │                                            │
              │                                            │
              │              ┌─── y[2] ────────────┐       │
              │              │                     │       │
              │              │       ┌─ y[3] ──┐   │       │
              │              │       │         │   │       │
              ▼              ▼       ▼         ▼   ▼       ▼
x ──► cv1 ─┬─split──► [Bottleneck 0] ──► [Bottleneck 1] ─► concat ──► cv2 ──► out
   1×1 升通道 │                                                       1×1 压通道
            └──► y[1] ────► y[2]            ─► y[3]
                  (右半经反复加工，每步都留下)

通道数变化（c1=64, c2=128, n=2, e=0.5 → self.c=64）：
   x          : 64 通道
   cv1 后      : 128 通道 (= 2c)
   切两半     : 64 + 64
   每个 Bot 输出 : 64
   concat 4 个 : 256 通道 (= (2+n)c)
   cv2 后      : 128 通道
```

#### 与传统 CSP 的对比（看一眼即可）

- **传统 CSP（C3）**：只保留 `y[0]`（左半）和最后一个 Bottleneck 的输出，**两个**张量 concat。
- **C2f**：保留 `y[0]`、`y[1]` 和**每一个** Bottleneck 的输出，**(2+n) 个**张量 concat。

→ C2f 让更多中间特征参与最终融合，**提升表达力**。

#### `forward_split` 是干嘛的？

`block.py:314` 还有个 `forward_split`：

```python
def forward_split(self, x):
    y = self.cv1(x).split((self.c, self.c), 1)
    ...
```

与 `forward` 等价，只是用 `.split()` 替代 `.chunk()`。**作用**：某些导出场景（如 ONNX）`chunk` 不被某些后端支持，要用 `split` 重写。读代码时无视即可。

**难度**：⭐⭐⭐ —— **C2f 是整个 Step 2 的高潮**，至少花 20 分钟把上面的流图自己画一遍。

---

### 6. `C3k2` — YOLO26 的主力块

**位置**：`ultralytics/nn/modules/block.py:1069`

```python
class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5,
                 attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn                                    # 分支 ①：注意力增强（YOLO26 新增）
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k                                     # 分支 ②：更复杂的 C3k 块
            else Bottleneck(self.c, self.c, shortcut, g)
                                                       # 分支 ③：朴素 Bottleneck
            for _ in range(n)
        )
```

**关键观察**：`class C3k2(C2f)` —— **C3k2 直接继承 C2f**，唯一改的是 `self.m`（中间那 `n` 个加工单元）。**`forward()` 没重写**，完全沿用 C2f 的逻辑。

#### `self.m` 的三种模式（三分支表达式）

| 条件 | 中间块用什么 | YAML 例子 |
|---|---|---|
| `attn=True` | `Bottleneck + PSABlock`（带注意力） | YOLO26 部分变体 |
| `c3k=True` | `C3k` 块（更深的嵌套 CSP） | `yolo26.yaml` 第 28 行 `C3k2 [512, True]` |
| 都 False | 朴素 `Bottleneck` | `yolo26.yaml` 第 24 行 `C3k2 [256, False, 0.25]` |

**YAML 参数怎么映射？**

YAML 第 24 行：`- [-1, 2, C3k2, [256, False, 0.25]]`
- `args[0]=256` → `c2=256`
- `args[1]=False` → `c3k=False`
- `args[2]=0.25` → `e=0.25`
- 结论：用**朴素 Bottleneck**，且隐藏通道是 `256 * 0.25 = 64`（更瘦更快，适合浅层）

YAML 第 28 行：`- [-1, 2, C3k2, [512, True]]`
- `c3k=True`, `e` 用默认 0.5
- 结论：用**更复杂的 C3k**，适合深层做精细加工

#### 设计哲学：浅层用简单的，深层用复杂的

回看 `yolo26.yaml` Backbone：

```yaml
- [-1, 2, C3k2, [256, False, 0.25]]   # 层 2，浅层 → 朴素 Bottleneck
- [-1, 2, C3k2, [512, False, 0.25]]   # 层 4，仍较浅 → 朴素 Bottleneck
- [-1, 2, C3k2, [512, True]]          # 层 6，进入深层 → C3k
- [-1, 2, C3k2, [1024, True]]         # 层 8，最深 → C3k
```

**直觉**：图像浅层信息冗余多（边缘、纹理），简单 block 就够了；越往深越要"精雕细琢"。

**难度**：⭐⭐⭐ —— 主要难在搞清楚三分支表达式。继承自 C2f 这点要记牢。

---

### 6.5 深入对比：C2f vs C3k2（含练习 Q&A）

> 上一节说 C3k2 "只比 C2f 多开关"。这一节把这句话展开成**源码级的精确对比**，并补充一个前文没提的隐藏差异。

#### 一句话总结

**C3k2 继承自 C2f（`class C3k2(C2f)`），骨架和 `forward()` 完全一样；唯一的区别是中间那排"加工单元" `self.m` 里装的东西不同 —— C2f 永远装朴素 Bottleneck，C3k2 是一个三选一的"可换芯"插槽。**

#### 相同点：骨架与 forward 完全共用

C3k2 **没有重写 `forward()`**，它跑的就是 C2f 的那三行（chunk 切两半 → 右半逐个加工 → 全部 concat）：

```
                共享骨架（CSP 结构）
   x ─► cv1 ─┬─► y[0] ────────────────────┐
       1×1   │                            ├─► concat ─► cv2 ─► out
             └─► y[1] ─► [芯] ─► [芯] ────┘            1×1
                          │       │
                          └───────┴── 每个芯的输出都进 concat

   C2f 和 C3k2 的差别只在 [芯] 这个插槽里装什么！
```

#### 三种"芯"的结构图解

```
分支③ 朴素模式 (c3k=False)          分支② c3k 模式 (c3k=True)
┌─────────────────────┐            ┌──────────────────────────────────┐
│  Bottleneck(e=0.5)  │            │  C3k —— 完整的迷你 C3 块！        │
│                     │            │                                  │
│  64 ─3×3─► 32       │            │  64 ─┬─cv1(1×1)─►32─►Bot─►Bot─┐  │
│     ─3×3─► 64       │            │      └─cv2(1×1)─►32──────────┤  │
│  (+残差)            │            │            concat──►cv3(1×1)─►64 │
└─────────────────────┘            │  里面又藏了 2 个 Bottleneck！    │
   2 层卷积深                       └──────────────────────────────────┘
                                      6 层卷积深（块中块，CSP 套 CSP）

分支① attn 模式 (attn=True)
┌──────────────────────────────┐
│  Bottleneck ──► PSABlock     │
│  (普通卷积)    (自注意力+FFN) │
└──────────────────────────────┘
   卷积 + 注意力混合
```

名字 "C3k2" 的来历：**C3k 芯 + C2f 框架**。`c3k=True` 时外层是 C2f 式结构、每个芯又是一个 C3 式结构 —— 块中块、嵌套 CSP。

#### 隐藏差异：朴素模式的 C3k2 ≠ C2f！

对比两者创建 Bottleneck 的那一行源码：

| | C2f 的芯（`block.py:306`） | C3k2 朴素芯（`block.py:1104`） |
|---|---|---|
| 调用 | `Bottleneck(c, c, shortcut, g, k=((3,3),(3,3)), e=1.0)` | `Bottleneck(c, c, shortcut, g)` ← 全用默认值 |
| 内部隐藏通道 | `e=1.0` → **不收窄**（64→64→64） | 默认 `e=0.5` → **收窄一半**（64→32→64） |
| 单芯参数量（c=64，忽略 BN） | 64×64×9×2 ≈ **73.7k** | 64×32×9 + 32×64×9 ≈ **36.9k（正好一半）** |

另外构造函数默认值也不同：C2f 默认 `shortcut=False`，C3k2 默认 `shortcut=True`。

**结论**：即使三个开关全关，C3k2 也比 C2f 更"瘦" —— 这是 YOLO11/26 整体轻量化设计的一部分。

#### 三种芯的参数量与深度对比（self.c=64 实算）

| 芯 | 参数量 | 串行卷积深度 | 特点 |
|---|---|---|---|
| C2f 的 Bottleneck（e=1.0） | ≈73.7k | 2 层 | 宽而浅 |
| C3k2 朴素 Bottleneck（e=0.5） | ≈36.9k | 2 层 | 窄而浅，最省 |
| C3k2 的 C3k 芯（内含 2 个 Bottleneck） | ≈45.1k | 6 层 | 参数适中但**深 3 倍**，非线性更强 |

→ `c3k=True` 不是单纯堆参数，而是用接近的参数量换 **3 倍深度**，让深层特征被"精雕细琢"。
（attn 芯的参数量取决于 Attention 内部实现，暂不展开。）

#### 对照 yolo26.yaml：三种芯全都在用！

```yaml
backbone:
  - [-1, 2, C3k2, [256, False, 0.25]]   # 层2  → 分支③ 朴素芯（浅层，特征冗余，省着用）
  - [-1, 2, C3k2, [512, False, 0.25]]   # 层4  → 分支③ 朴素芯
  - [-1, 2, C3k2, [512, True]]          # 层6  → 分支② C3k 芯（深层，精加工）
  - [-1, 2, C3k2, [1024, True]]         # 层8  → 分支② C3k 芯
head:
  - [-1, 2, C3k2, [512, True]]          # 层13/16/19 → 分支② C3k 芯（Neck 融合）
  - [-1, 1, C3k2, [1024, True, 0.5, True]]  # 层22 → 分支①！args=[c2, c3k, e, attn]
                                        #        attn=True → Bottleneck+PSABlock 注意力芯
```

两个关键细节：

1. **分支优先级**：Python 链式三目运算符从前往后判断 —— 层 22 同时有 `c3k=True` 和 `attn=True`，但 `attn` 排在三目链最前面，所以**分支①赢**。它在 Neck 分辨率最低的 P5 分支上加注意力，和 C2PSA 只放 Backbone 末端是同一个逻辑（注意力贵，只在小特征图上用）。
2. **parse_model 的隐式缩放**（Step 4 详讲）：对 m/l/x 尺寸，`tasks.py:1657` 会强制把 `c3k` 改成 `True` —— 同一份 YAML，小模型用朴素芯、大模型自动换 C3k 芯。

#### 演进脉络与口诀

```
C3 (YOLOv5)          C2f (YOLOv8)              C3k2 (YOLO11/26)
─────────────        ──────────────            ─────────────────
固定骨架+固定芯   →   改进骨架：保留所有     →   保留 C2f 骨架，
只保留最后一个        中间结果一起 concat        把芯做成可换插槽
单元的输出           （特征更丰富）            （一个类适配三种需求）
```

> **口诀**：*C2f 改骨架，C3k2 改芯；骨架管特征怎么流，芯管特征怎么加工。*

#### 练习 Q&A

**Q1：层 6 的 `C3k2 [512, True]` 在 yolo26n（depth=0.50, width=0.25）下，`self.m` 里有几个芯？每个芯内部又有几个 Bottleneck？**

**A1**：**1 个芯，每芯 2 个 Bottleneck，合计 2 个。**

- **芯的个数**：YAML repeats=2 → parse_model 深度缩放 `max(round(2 × 0.50), 1) = 1` → `self.m` 里只有 **1 个芯**。
- **芯的类型**：`c3k=True` 且 attn 未设置 → 走分支② → 芯是 **C3k**。
- **每芯内部**：C3k2 调用时写死 `C3k(self.c, self.c, 2, shortcut, g)` —— 这个 `2` 是硬编码，**不受 depth 缩放**（它不经过 parse_model）→ 每个 C3k 含 **2 个 Bottleneck**。
- **通道追踪**（width=0.25）：c2 = 512×0.25 = **128** → self.c = 128×0.5 = **64** → C3k 内部 c_ = 64×0.5 = **32** → 最里面的 Bottleneck 是 32→32（k=3×3, e=1.0）。

对比 yolo26l（depth=1.00, width=1.00）：芯数 = max(round(2×1.0),1) = **2**，合计 4 个 Bottleneck；且 c2=512、self.c=256 —— 又深又宽。

**易错点**：两层"重复次数"只有外层受缩放 —— YAML 的 repeats 被 depth 缩放，但 C3k 内部的 "2" 是写死的。这就是为什么 n/s 小模型不会把 C3k 内部也缩成 1 个 Bottleneck。

---

**Q2：如果想把 CBAM 塞进 C3k2 当"第四种芯"，应该改 `block.py` 的哪几行？**

**A2**：先说一个重要发现 —— **Ultralytics 已经内置了 CBAM！** 位置：

- `conv.py:512` — `ChannelAttention`（通道注意力）
- `conv.py:549` — `SpatialAttention`（空间注意力）
- `conv.py:583` — `CBAM(c1, kernel_size=7)`，forward 为 `spatial_attention(channel_attention(x))`

CBAM 的输出形状与输入**完全相同**（注意力只做加权、不改通道/分辨率）→ 天然适合当芯。**不用自己从零写 CBAM 类**，YOLO-pineapple 复现时可以直接用。

需要改 **3 处，全部在 `block.py`**：

1. **第 12 行 import**：`from .conv import Conv, DWConv, ...` 的列表里加上 `CBAM`。
2. **C3k2 的 `__init__` 签名**（约 1072 行）：加一个参数 `cbam: bool = False`（建议加在签名最后，避免打乱现有 YAML 的位置参数顺序）。
3. **`self.m` 的三目链**（约 1096 行）：仿照 attn 分支，加一个新分支：

```python
self.m = nn.ModuleList(
    nn.Sequential(Bottleneck(self.c, self.c, shortcut, g), CBAM(self.c, 7))
    if cbam                                  # 新分支：放最前 = 优先级最高
    else nn.Sequential(Bottleneck(...), PSABlock(...))
    if attn
    else C3k(self.c, self.c, 2, shortcut, g)
    if c3k
    else Bottleneck(self.c, self.c, shortcut, g)
    for _ in range(n)
)
```

两个关键洞察：

- **不用动 `tasks.py`！** 因为改的是 C3k2 的内部，而 C3k2 早已注册在 `base_modules` 里。对比 Step 4 文档的"添加全新模块 4 步流程"，这种**借壳改芯**少走两步，是论文复现时最省事的路径之一。
- **分支位置 = 优先级**：链式三目从前往后判断，新分支放最前面意味着 `cbam=True` 时其他开关全部失效（参考层 22 的 attn 优先于 c3k）。

**注意事项**：YAML 的 args 按位置传参，`cbam` 排在签名第 9 位（c1, c2, n, c3k, e, attn, g, shortcut, cbam），在 YAML 里启用就得把前面的参数写全：`- [-1, 2, C3k2, [256, False, 0.5, False, 1, True, True]]`。嫌丑的话，更优雅的做法是定义子类 `class C3k2CBAM(C3k2)` 把 cbam 固定为 True —— 但新类名就需要走 Step 4 的完整注册流程了。

> ⚠️ 当前还在读代码阶段，以上是"路线图"而非行动指令 —— 先不要真的改源码。

---

## Tier 2 — 与论文强相关

### 7. SPP → SPPF — 多尺度池化

> 📌 **与你毕设强相关**：YOLO-pineapple 论文里的 "SPP 增强" 改的就是这一块。先读懂原版 YOLO26 用的 SPPF，再去看论文怎么改。

#### `SPP` —— 原始版（`block.py:185`）

```python
class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
```

**做什么**：用 3 个**不同 kernel size**（5、9、13）的池化**并行**扫过同一张特征图，把不同感受野的结果拼起来 → 让网络同时"看清近处和远处"。

**数据流**：

```
        ┌────────────────────────────────────────────┐
        │            原始 SPP（并行）                 │
        │                                            │
   x ──► cv1 ──┬──► (identity, 不池化) ──┐          │
              ├──► MaxPool k=5 ────────┤           │
              ├──► MaxPool k=9 ────────┤── concat ─► cv2 ──► out
              └──► MaxPool k=13 ───────┘           │
        │                                          │
        └──────────────────────────────────────────┘

   4 个张量沿通道拼接, 通道数 = 4 × c_
```

#### `SPPF` —— 优化版（`block.py:208`）

```python
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y
```

**关键改进**：**同一个 kernel=5 的池化反复串行 3 次**，数学上**等价**于 SPP 的 `[5, 9, 13]` 并行，但速度快得多。

#### 为什么"k=5 串行 3 次 ≈ k=5/9/13 并行"？

**池化感受野的叠加规律**：
- 1 次 `MaxPool(k=5, s=1)` → 感受野 5
- 2 次串行 → 感受野 5+(5-1) = 9
- 3 次串行 → 感受野 9+(5-1) = 13

→ **三次串行 kernel=5 的累积感受野正好覆盖 {5, 9, 13}**。

**数据流**：

```
        ┌────────────────────────────────────────────┐
        │            SPPF（串行）                     │
        │                                            │
   x ──► cv1 ──► y₀ ──┬─────────────────────┐       │
                      ▼                      │       │
                  MaxPool k=5                │       │
                      │                      │       │
                      ▼  y₁ ─────────────────┼──┐    │
                  MaxPool k=5                │  │    │
                      │                      │  │    │
                      ▼  y₂ ─────────────────┼──┼──┐ │
                  MaxPool k=5                │  │  │ │
                      │                      │  │  │ │
                      ▼  y₃ ─────────────────┴──┴──┴─┴── concat ──► cv2 ──► out
        │                                                            │
        └────────────────────────────────────────────────────────────┘

   4 个张量沿通道拼接 (y₀, y₁, y₂, y₃), 通道数 = (n+1) × c_ = 4 × c_
```

**YAML 中的样子**（`yolo26.yaml` 第 31 行）：

```yaml
- [-1, 1, SPPF, [1024, 5, 3, True]]   # c2=1024, k=5, n=3, shortcut=True
```

#### 与 YOLO-pineapple 论文的对照点

读完 SPPF 之后**立刻打开 `YOLO-pineapple.html`**，找它讲 SPP 的章节（论文里搜 "SPP"），对照：
1. 原版 SPPF 长什么样
2. 论文改成了什么（可能加了不同 dilation、可能换成 ASPP、可能并行更多 kernel）
3. 为什么这样改对小目标 / UAV 图像更有效

**这是你第一次把代码和论文挂钩的关键时刻**。

**难度**：⭐⭐ —— 不难，但需要理解"感受野的串行累加"。

---

### 8. PSA → C2PSA — 自注意力块

> 📌 **与你毕设强相关**：YOLO-pineapple 论文的核心改进之一是引入 **CBAM 注意力**。先读懂 YOLO26 自带的 PSA 注意力，再看论文怎么换成 CBAM。

#### `PSA` —— Position-Sensitive Attention（`block.py:1381`）

```python
class PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1),
            Conv(self.c * 2, self.c, 1, act=False),
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))
```

#### 数据流图

```
       ┌─────────────────────────────────────────────────┐
       │                  PSA                             │
       │                                                  │
   x ──► cv1 ─┬─split──► a ────────────────────────┐     │
        1×1   │                                     │     │
              └──► b ──► Attention ──► (+) ──► FFN ─► (+) ──► concat ──► cv2 ──► out
                            │           ▲        │     ▲                       1×1
                            └───────────┘        └─────┘
                          (注意力残差)         (FFN 残差)
       └────────────────────────────────────────────────┘

   思想：和 C2f 类似的 "split 一半保留, 一半深加工" 的 CSP 结构,
        但深加工模块换成了 [Attention + FFN] (Transformer Encoder 的标配)
```

#### 注意力机制本质（直觉版，不抠数学）

**Attention 在做什么**：对特征图上的每个位置，计算"该关注图上哪些其他位置"，再用注意力权重加权求和。

```
位置 (3,5) 处的输出 = 加权求和(整张特征图所有位置的特征向量)
                  其中权重 = 学到的"与每个位置的相关程度"
```

**通俗类比**：人看图找鸟时，眼睛会自动忽略天空、聚焦树枝 —— 注意力模块教网络做同样的事。

`Attention` 类本身在 `transformer.py` 里，**先不深抠**，知道它"做位置间的特征融合"即可。

#### `C2PSA` —— 多个 PSABlock 堆叠（`block.py:1436`）

```python
class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))
        )

    def forward(self, x):
        # forward 类似 PSA，只是中间从单个 attention+ffn 变成 n 个 PSABlock 串联
```

**与 PSA 的区别**：PSA 内部只有 1 个 Attention+FFN，C2PSA 可以堆 `n` 个 `PSABlock`（**每个 PSABlock = 1 个 Attention + 1 个 FFN**）。结构更深、表达力更强。

**YAML 中的样子**（`yolo26.yaml` 第 32 行）：

```yaml
- [-1, 2, C2PSA, [1024]]   # c2=1024, n=2 (从 YAML 第 2 个字段)
```

#### 为什么 C2PSA 只在 Backbone 末端用？

```
yolo26.yaml backbone:
  ...
  - [-1, 1, SPPF, [1024, 5, 3, True]]   # 9
  - [-1, 2, C2PSA, [1024]]              # 10  ← 只在这一处！
```

注意力计算量是 **O(H²·W²)** 量级，在大尺寸特征图上爆炸。**只在分辨率最低的 P5/32 上用**（640×640 输入 → 20×20 特征图），既能享受注意力的好处，又不至于跑不动。

#### 与 YOLO-pineapple 论文 CBAM 的对照

读完 C2PSA 之后，去论文里找 CBAM 章节。对照思考：
| 维度 | C2PSA（YOLO26） | CBAM（论文用） |
|---|---|---|
| 注意力类型 | Self-Attention（自注意力） | Channel + Spatial Attention |
| 计算开销 | 高（自注意力是 O(N²)） | 低（CBAM 是轻量的） |
| 适合放在哪 | 只放深层（小特征图） | 几乎任何位置都能放 |
| 论文为什么换 | CBAM 更轻量，适合 UAV 实时推理 | （读论文确认你的猜测）|

**这一步比对**就是你"第一次把架构改进的论文动机和代码对应起来"的训练。

**难度**：⭐⭐⭐⭐ —— PSA 涉及 Transformer 的 Attention 机制。先建立"形状 + 数据流"的直觉，数学推导可以慢慢补。

---

## Tier 3 — 选读拓展

> 这三个类不必精读，**扫一眼建立全貌**即可。它们要么是 Tier 1 的轻量变体，要么是历史版本。

### 9. `DWConv` — 深度可分离卷积

**位置**：`conv.py:185`

```python
class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
```

**就一行**：继承 `Conv`，把 `groups` 设为 `gcd(c1, c2)`。当 `c1==c2` 时 `groups=c1`，每个通道独立卷积 → MobileNet 风格的轻量化。

**参数量对比**：标准 3×3 Conv 是 `c1×c2×9` 参数；DWConv 只有 `c1×9`（约 1/c2）。

### 10. `GhostConv` — Ghost 模块

**位置**：`conv.py:311`

**思想**："一半通道走正经 Conv，一半通道用便宜的线性操作（DWConv）复制生成"。在 `yolov8-ghost.yaml` 里用，**YOLO26 主线没用**。

### 11. `C3` — YOLOv5 时代的 block

**位置**：`block.py:322`

```python
def forward(self, x):
    return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
```

老 CSP 块。和 C2f 的区别：C3 用 `cv2` 加工"另一半"分支，C2f 直接保留切片 + 收集所有 Bottleneck 中间结果。**C2f 是 C3 的改进版**。

#### C 系列演进史（彩蛋）

| 模块 | 出现时代 | 关键改进 |
|---|---|---|
| `C3`   | YOLOv5  | 经典 CSP 双分支 |
| `C2f`  | YOLOv8  | 多中间结果 concat → 更丰富特征融合 |
| `C3k`  | YOLOv11 | C3 + 可调 kernel size |
| `C3k2` | YOLO26  | C2f + 可选 attention/c3k 分支 |

掌握 C2f 和 C3k2，再扫一眼这张表，整个演进脉络就通了。

---

## 整体回顾

### 一张图串起所有类

```
                       ┌─────────────────────────────────┐
                       │      YOLO26 模块层次图           │
                       └─────────────────────────────────┘

   原子 (conv.py)           复合 (block.py)              注意力 (block.py)
   ─────────────            ────────────────             ─────────────────
   autopad ──┐
             │
   Conv ─────┼──► Bottleneck ──► C3 ────► C3k             PSA  ◄── Attention
             │          │           │                       │       (transformer.py)
   DWConv    │          └──► C2f ──┴──► C3k2 ──┐           │
             │                                  │           │
   GhostConv │                                  │           ▼
             │                                  └────► C2PSA (内含 PSABlock = Bot+PSA)
   Concat ───┘
                                                SPP ──► SPPF (多尺度池化)


   YAML 里出现的: Conv, C3k2, SPPF, C2PSA, Concat, nn.Upsample, Detect (Step 3 讲)
```

### 你应该掌握的 5 件事

读完 Step 2，你应该能：

1. **看到 YAML 一行 `- [-1, 2, C3k2, [256, False, 0.25]]`，能说出**：
   - 输入来自上一层（-1），重复 2 次
   - 输出通道 256，不用 c3k，expansion ratio 0.25
   - 隐藏通道 = 256 * 0.25 = 64
   - 中间用朴素 Bottleneck（因为 c3k=False, attn=False）

2. **在白纸上画出 `Conv.forward()` 的三步张量流**，并能算出输出形状。

3. **解释 C2f 为什么比 C3 表达力更强**（更多中间特征参与最终融合）。

4. **指出 SPPF 在哪个位置、为什么需要它**（Backbone 倒数第二层，扩大感受野融合多尺度信息）。

5. **理解 YOLO26 已经内置 PSA 注意力**，且只在分辨率最低的 P5/32 用 —— 这为你看 YOLO-pineapple 论文换 CBAM 的动机打下了基础。

---

## 下一步

完成 Step 2 之后，进入：

- **Step 3** —— 读 `Detect` 检测头（`head.py`）。看三个尺度的特征图怎么变成最终的"框 + 类别"输出。
- **Step 4** —— 读 `parse_model()`（`tasks.py`）。看 YAML 是如何被解析成 PyTorch 模型对象的 —— 这是你以后**添加自定义模块**时必须修改的地方。
- **Step 5** —— 跑通 `YOLO("yolo26.yaml").train(...)` 的高层 API（skim）。

#### 衔接 YOLO-pineapple 论文的两个早期动作

在进入 Step 3 之前，建议你做两件事，**把代码和论文的桥梁先架起来**：

1. **打开 `YOLO-pineapple.html`，搜索 "SPP"**：看论文怎么改 SPP 模块。把你看到的论文图和 SPPF 的源码做比较。
2. **打开 `YOLO-pineapple.html`，搜索 "CBAM"**：看论文里的 CBAM 在网络的哪个位置插入。对比 YOLO26 把 C2PSA 放在层 10 的选择。

带着这两个对比，再去 Step 3 读检测头，整个网络的"为什么这样设计"会越来越清晰。

---

> 📝 **使用建议**：阅读时如有任何不理解的概念（CSP / 自注意力 / DFL / BN 数学推导 …），随时回到对话里问我。这份文档作为脱机阅读材料 + 检索词典使用。
