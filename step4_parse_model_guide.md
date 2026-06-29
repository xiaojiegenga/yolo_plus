# Step 4 — parse_model：YAML 蓝图如何变成 PyTorch 模型

> **前置**：已完成 Step 1–3。你已经知道 YAML 每一行长什么样、每个模块怎么 forward，以及 Detect 头的输出。
>
> **核心文件**：`ultralytics-main/ultralytics/nn/tasks.py`
> - `parse_model()` 函数（第 1539 行）—— YAML → PyTorch 的翻译器
> - `BaseModel` 类（第 102 行）—— 推理时如何按层顺序跑 forward
> - `DetectionModel` 类（第 342 行）—— 把 parse_model + stride 计算 + loss 初始化串起来
>
> **为什么这一步重要**：以后你要往 YOLO26 里**加自定义模块**（如 CBAM），需要修改的就是这个文件。Step 4 是"读代码"和"改代码"的分界点。

---

## 目录

- [1. parse_model 全景](#1-parse_model-全景)
- [2. 逐段拆解](#2-逐段拆解)
  - [2.1 读取全局参数 + scales](#21-读取全局参数--scales)
  - [2.2 核心循环：逐行翻译 YAML](#22-核心循环逐行翻译-yaml)
  - [2.3 模块名 → Python 类](#23-模块名--python-类)
  - [2.4 base_modules 分支：通道计算 + 深度缩放](#24-base_modules-分支通道计算--深度缩放)
  - [2.5 Detect 等 Head 的特殊处理](#25-detect-等-head-的特殊处理)
  - [2.6 实例化 + 组装](#26-实例化--组装)
- [3. BaseModel._predict_once：推理时的执行引擎](#3-basemodel_predict_once推理时的执行引擎)
- [4. DetectionModel.__init__：完整的组装流程](#4-detectionmodel__init__完整的组装流程)
- [5. 如何添加自定义模块（预告）](#5-如何添加自定义模块预告)
- [6. 关键问答](#6-关键问答)
- [7. 下一步](#7-下一步)

---

## 1. parse_model 全景

**一句话**：`parse_model(d, ch)` 接收一个 YAML 字典 `d` 和初始通道数 `ch=3`，返回 `(nn.Sequential, savelist)`。

```
输入                                输出
─────                               ─────
d = {                               nn.Sequential(
  "nc": 80,                           Conv(3, 64, 3, 2),      # 层 0
  "scales": {"n": [0.5, 0.25, 1024]}, Conv(64, 128, 3, 2),    # 层 1
  "backbone": [...],                   C3k2(128, 256, ...),    # 层 2
  "head": [...]                        ...
}                                      Detect(nc=80, ...)      # 层 23
ch = 3  (RGB)                        )
                                    savelist = [4, 6, 10, 13, 16, 19, 22]
                                      ↑ 这些层的输出需要被保存，因为后面有层引用它们
```

---

## 2. 逐段拆解

### 2.1 读取全局参数 + scales

```python
nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
reg_max = d.get("reg_max", 16)
depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
scale = d.get("scale")
if scales:
    depth, width, max_channels = scales[scale]
```

**这里发生了什么**：
1. 从 YAML 字典里取出 `nc=80`、`end2end=True`、`reg_max=1`、`scales` 表
2. 根据 `scale`（比如 `"n"`）从 scales 表里取出 `[depth=0.50, width=0.25, max_channels=1024]`
3. 这三个值会作用到后面每一层的通道数和重复次数

### 2.2 核心循环：逐行翻译 YAML

```python
for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
```

把 backbone 和 head 的所有层拼成一个列表，挨个处理。每一行是 `[from, repeats, module_name, args]`。

### 2.3 模块名 → Python 类

```python
m = (
    getattr(torch.nn, m[3:])           # "nn.Upsample" → torch.nn.Upsample
    if "nn." in m
    else globals()[m]                   # "Conv" → 从 import 的 Conv 类取
)
```

YAML 里写 `"Conv"` 这个字符串 → 通过 `globals()["Conv"]` 找到文件开头 `from ... import Conv` 导入的那个 Python 类。

**这就是为什么你添加新模块后必须在 `tasks.py` 开头 import 它**。

### 2.4 base_modules 分支：通道计算 + 深度缩放

```python
base_modules = frozenset({
    Conv, ConvTranspose, GhostConv, Bottleneck, SPP, SPPF,
    C2PSA, DWConv, C1, C2, C2f, C3k2, C3, ...
})

n = n_ = max(round(n * depth), 1) if n > 1 else n  # 深度缩放

if m in base_modules:
    c1, c2 = ch[f], args[0]                         # 输入通道自动推断, 输出通道从 args 读
    if c2 != nc:
        c2 = make_divisible(min(c2, max_channels) * width, 8)  # 宽度缩放 + 封顶 + 对齐到 8
    args = [c1, c2, *args[1:]]                       # 把 c1 插到 args 开头
    if m in repeat_modules:
        args.insert(2, n)                            # 把 repeats 也插进 args
        n = 1
```

**完整追踪一个例子**：YAML 第 24 行 `- [-1, 2, C3k2, [256, False, 0.25]]`，scale=n

| 步骤 | 变量 | 值 | 说明 |
|---|---|---|---|
| 读取 | f, n, m, args | -1, 2, "C3k2", [256, False, 0.25] | 原始 YAML 行 |
| 查类 | m | C3k2 类 | globals()["C3k2"] |
| 深度缩放 | n | max(round(2×0.5), 1) = 1 | 重复次数从 2 变 1 |
| 输入通道 | c1 | ch[-1] = 128×0.25=32 | 上一层的输出通道 |
| 输出通道 | c2 | min(256, 1024)×0.25 = 64, 对齐到 8 | width 缩放 |
| 拼 args | args | [32, 64, 1, False, 0.25] | [c1, c2, n, c3k, e] |
| 实例化 | m_ | C3k2(32, 64, n=1, c3k=False, e=0.25) | 调用构造函数 |

### 2.5 Detect 等 Head 的特殊处理

```python
elif m in frozenset({Detect, Segment, Pose, OBB, ...}):
    args.extend([reg_max, end2end, [ch[x] for x in f]])
```

Detect 的 `from=[16, 19, 22]`，所以 `[ch[x] for x in f]` = `[层16 输出通道, 层19 输出通道, 层22 输出通道]`。
这三个通道数传给 `Detect.__init__` 的 `ch` 参数 → 用来创建 3 套 cv2/cv3 分支。

### 2.6 实例化 + 组装

```python
m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 创建模块
m_.np = sum(x.numel() for x in m_.parameters())  # 统计参数量
m_.i, m_.f, m_.type = i, f, t   # 附加元信息：层号、from、类型名

save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
layers.append(m_)
ch.append(c2)                   # 把输出通道数追加到 ch 列表，给后面的层做 c1
```

关键：每个模块被实例化后，附加了 `.i`（层索引）、`.f`（from 索引）、`.type`（类名字符串）三个属性。推理时 `_predict_once` 会用到它们。

**`save` 列表**：记录哪些层的输出需要被"存起来"（因为后面有层会通过 `from` 引用它们）。比如层 4 的输出被 Neck 的 Concat 引用了，所以 4 会进入 save 列表。

---

## 3. BaseModel._predict_once：推理时的执行引擎

**位置**：`tasks.py:161`

```python
def _predict_once(self, x, profile=False, visualize=False, embed=None):
    y, dt, embeddings = [], [], []
    for m in self.model:
        if m.f != -1:  # 如果输入不是"上一层"
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)       # 跑这一层
        y.append(x if m.i in self.save else None)  # 需要保存的层存起来，否则存 None 省内存
    return x
```

**这 10 行就是整个 YOLO 模型跑推理的核心**。

逐行图解：

```
层 0 (Conv):   m.f = -1   → x = 输入图   → x = Conv(x)   → y = [x₀]
层 1 (Conv):   m.f = -1   → x = x₀       → x = Conv(x)   → y = [x₀, None]  (层1不在save里)
层 2 (C3k2):   m.f = -1   → x = x₁       → x = C3k2(x)   → y = [..., None]
...
层 12 (Concat): m.f = [-1, 6] → x = [x₁₁, y[6]]   # 取上一层输出 + 层6的存储输出
                              → x = Concat(x)         # 拼接
...
层 23 (Detect): m.f = [16, 19, 22] → x = [y[16], y[19], y[22]]   # 取3个尺度
                                   → x = Detect(x)                 # 检测
```

**关键洞察**：`m.f` 和 `self.save` 配合，实现了 YAML 里声明的任意跳接拓扑。只要 YAML 写了 `from: [a, b]`，parse_model 就会把 a、b 加入 save，推理时就会从 y 数组里取出来。

---

## 4. DetectionModel.__init__：完整的组装流程

**位置**：`tasks.py:370`

```python
class DetectionModel(BaseModel):
    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)

        # 1. 调用 parse_model 构建网络
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)

        # 2. 计算 stride
        m = self.model[-1]  # 最后一层 = Detect
        m.stride = torch.tensor(
            [s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))]
        )
        # 做一次"假前向"（256×256 的零图），看三个输出特征图的空间尺寸
        # s=256, 输出分别是 32×32, 16×16, 8×8 → stride = 256/32, 256/16, 256/8 = [8, 16, 32]

        self.stride = m.stride
        m.bias_init()         # 初始化 Detect 的 bias
        initialize_weights(self)  # 初始化其他层的权重
```

**整个流程的完整图**：

```
yaml_model_load("yolo26n.yaml")
   │
   ▼
  YAML 字典 d = {nc, scales, backbone, head, ...}
   │
   ├── guess_model_scale("yolo26n") → scale = "n"
   │
   ▼
parse_model(d, ch=3)
   │
   ├── 遍历 backbone + head 每一行
   ├── 查 base_modules → 缩放通道数、深度
   ├── 实例化每个模块 → Conv, C3k2, SPPF, ..., Detect
   │
   ▼
(nn.Sequential, savelist)
   │
   ▼
DetectionModel
   ├── self.model = nn.Sequential(...)
   ├── self.save = savelist
   ├── self.stride = [8, 16, 32]
   └── ready for .train() / .predict()
```

---

## 5. 如何添加自定义模块（预告）

> 这里只是 **Step 4 的"路线预告"**，不是要你现在就改代码。等你读完 Step 5 后如果开始做论文复现，再回来看这部分。

假设你要添加一个叫 `CBAM` 的注意力模块（YOLO-pineapple 论文的核心改进之一）：

### 步骤概览（4 步）

```
                  ┌─────────────────────────────────────────────────────┐
                  │         添加自定义模块的 4 步流程                     │
                  │                                                     │
                  │  ① 定义 Python 类 (block.py 或新文件)               │
                  │     class CBAM(nn.Module):                          │
                  │         def __init__(self, c1, c2, ...): ...        │
                  │         def forward(self, x): ...                   │
                  │                                                     │
                  │  ② 注册到 tasks.py                                  │
                  │     a. 文件开头 import: from ... import CBAM        │
                  │     b. 加入 base_modules 集合                       │
                  │     c. 如果有 repeat 参数，还要加入 repeat_modules   │
                  │                                                     │
                  │  ③ 在 __init__.py 中 export                         │
                  │     ultralytics/nn/modules/__init__.py              │
                  │                                                     │
                  │  ④ 在 YAML 中使用                                   │
                  │     - [-1, 1, CBAM, [256]]                          │
                  └─────────────────────────────────────────────────────┘
```

### 涉及的文件（按修改顺序）

| 步骤 | 文件 | 做什么 |
|---|---|---|
| ① | `ultralytics/nn/modules/block.py` | 添加 `class CBAM(nn.Module)` |
| ② | `ultralytics/nn/modules/__init__.py` | 在 `__all__` 和 import 里加 `CBAM` |
| ③ | `ultralytics/nn/tasks.py` 开头 | `from ... import CBAM` |
| ③ | `ultralytics/nn/tasks.py` `base_modules` | 把 `CBAM` 加进去 |
| ④ | 你自己的 YAML 文件 | 在 backbone 或 head 里写 `- [-1, 1, CBAM, [256]]` |

**核心原理**：parse_model 通过 `globals()[m]` 查找类名 → 只要你的类在 tasks.py 的命名空间里能被找到，YAML 里就能用。

---

## 6. 关键问答

### Q1: `ch` 列表是怎么追踪每一层输出通道的？

**A**: `ch` 从 `[3]` 开始（RGB 3 通道），每处理完一层就 `ch.append(c2)`。后面的层用 `ch[f]` 取输入通道：

```
层 0: Conv(3→64)    → ch = [3, 64]          ← ch[-1] = 64
层 1: Conv(64→128)  → ch = [3, 64, 128]     ← ch[-1] = 128
...
层 12: Concat from [-1, 6] → c2 = ch[-1] + ch[6] = 拼接后通道数
```

`from=-1` 取 `ch[-1]`（上一层），`from=6` 取 `ch[6]`（第 6 层的输出通道）。

### Q2: `make_divisible(c2, 8)` 是干嘛的？

**A**: 把通道数向上取整到 8 的倍数。GPU 计算在通道数是 8/16/32 的倍数时效率最高（与 Tensor Core 对齐），这是一个性能优化。

例：`256 × 0.25 = 64` → `make_divisible(64, 8) = 64` ✓  
例：`512 × 0.25 = 128` → `make_divisible(128, 8) = 128` ✓  
例：`假设得到 100` → `make_divisible(100, 8) = 104`（向上取到 8 的倍数）

### Q3: `save` 列表里的层号是怎么算出来的？

**A**: 每处理一层时检查 `from` 字段 —— 如果 from 不是 -1（不是简单的"取上一层"），就把 from 指向的层号加入 save：

```python
save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
```

比如层 12 的 `from=[-1, 6]`：`-1` 被排除，`6` 被加入 save → 运行时层 6 的输出会被保存在 `y[6]`。

### Q4: depth 和 width 到底怎么作用？举个具体例子。

**A**: 以 `yolo26n` (scale="n", depth=0.50, width=0.25, max_channels=1024) 为例：

YAML 蓝图写的：`C3k2 [1024, True]` (层 8)

| 缩放项 | 蓝图值 | 缩放后 |
|---|---|---|
| c2 通道数 | 1024 | min(1024, 1024) × 0.25 = 256, 对齐 8 → **256** |
| n 重复次数 | 2 | max(round(2 × 0.50), 1) = **1** |

换成 `yolo26x` (depth=1.00, width=1.50, max_channels=512)：

| 缩放项 | 蓝图值 | 缩放后 |
|---|---|---|
| c2 通道数 | 1024 | min(1024, 512) × 1.50 = 768, 对齐 8 → **768** |
| n 重复次数 | 2 | max(round(2 × 1.00), 1) = **2** |

→ `yolo26x` 更深（n=2）更宽（768 通道），参数量和计算量远大于 `yolo26n`。

### Q5: YOLO26 的 C3k2 有特殊处理？

**A**: 有！`tasks.py:1657-1660`：

```python
if m is C3k2:
    legacy = False
    if scale in "mlx":
        args[3] = True   # 把 c3k 参数强制设为 True
```

对于 m/l/x 尺寸的模型，即使 YAML 写了 `c3k=False`，parse_model 也会覆盖为 `True` —— 大模型用更复杂的 C3k 块，小模型保持朴素 Bottleneck。这是一种**隐式的缩放策略**。

### Q6: `legacy` 变量控制什么？

**A**: `legacy=True` 表示旧版模型（v3/v5/v8/v9/v11），`legacy=False` 表示新版（YOLO26）。区别在 Detect Head：
- `legacy=True`：分类分支用标准 Conv
- `legacy=False`：分类分支用 DWConv（YOLO26 的优化）

一旦 parse_model 遇到 C3k2 或 A2C2f 模块，就把 `legacy` 设为 False → Detect 头自动切换到新模式。

---

## 7. 下一步

Step 4 是整个 Reading Roadmap 的**枢纽**。你现在已经理解了：
- 一行 YAML 怎么变成一个 PyTorch 模块（parse_model 的循环体）
- 整个模型怎么按拓扑顺序跑推理（_predict_once 的 for 循环 + save 列表）
- 以后加自定义模块改哪 4 个地方（import → base_modules → __init__.py → YAML）

最后一步 **Step 5** 只需要 skim：看 `YOLO` 类怎么把上面所有东西串到 `.train()` / `.val()` / `.predict()` 的高层 API 上。
