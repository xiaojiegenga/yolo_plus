# Step 3 — Detect 检测头：特征图如何变成"框 + 类别"

> **前置**：已完成 Step 1（YAML 蓝图）和 Step 2（Conv / C2f / SPPF 等模块）。
>
> **核心文件**：`ultralytics-main/ultralytics/nn/modules/head.py` —— `Detect` 类（第 26 行起）。
>
> **辅助文件**：`ultralytics-main/ultralytics/nn/modules/block.py` —— `DFL` 类（第 58 行起）。

---

## 目录

- [1. Detect 在整个网络中的位置](#1-detect-在整个网络中的位置)
- [2. __init__ 拆解：两组平行的卷积分支](#2-__init__-拆解两组平行的卷积分支)
- [3. forward() 完整流程](#3-forward-完整流程)
- [4. DFL —— 分布焦点损失回归头](#4-dfl--分布焦点损失回归头)
- [5. End-to-End 模式（YOLO26 新特性）](#5-end-to-end-模式yolo26-新特性)
- [6. 其他 Head 变体（Segment / Pose / OBB）](#6-其他-head-变体segment--pose--obb)
- [7. 关键问答](#7-关键问答)
- [8. 下一步](#8-下一步)

---

## 1. Detect 在整个网络中的位置

```
输入图 640×640×3
   │
┌──▼──────────────────────────────────────────┐
│  Backbone (层 0–10)                          │
│  提取多尺度特征: P3/8, P4/16, P5/32          │
└──┬──────────┬──────────┬────────────────────┘
   │P3        │P4        │P5
   ▼          ▼          ▼
┌──────────────────────────────────────────────┐
│  Neck (层 11–22)                              │
│  FPN + PAN 融合三个尺度                       │
└──┬──────────┬──────────┬────────────────────┘
   │层16      │层19      │层22
   ▼          ▼          ▼
┌──────────────────────────────────────────────┐
│  Detect Head (层 23)    ◄── 就是这一节讲的！   │
│                                              │
│  输入: 3 个特征图                             │
│  输出: 每个位置的 [bbox 坐标 + 类别概率]      │
└──────────────────────────────────────────────┘
```

**YAML 中的声明**（`yolo26.yaml` 最后一行）：

```yaml
- [[16, 19, 22], 1, Detect, [nc]]   # Detect(P3, P4, P5)
```

`from=[16, 19, 22]` 表示收集 Neck 三个尺度的输出。`args=[nc]` 传入类别数（如 COCO 的 80 类）。

---

## 2. `__init__` 拆解：两组平行的卷积分支

```python
def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
    self.nc = nc        # 类别数
    self.nl = len(ch)   # 检测层数 = 3（P3, P4, P5）
    self.reg_max = reg_max  # DFL 的 bin 数
    self.no = nc + self.reg_max * 4  # 每个锚点的输出数

    # 两组独立的卷积分支：
    self.cv2 = ...  # 回归分支（预测 bbox）
    self.cv3 = ...  # 分类分支（预测类别）
    self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
```

### 2.1 回归分支 `cv2`：预测边界框

```python
self.cv2 = nn.ModuleList(
    nn.Sequential(
        Conv(x, c2, 3),        # 3×3 卷积压缩通道
        Conv(c2, c2, 3),       # 再来一次 3×3 卷积
        nn.Conv2d(c2, 4 * self.reg_max, 1)  # 1×1 卷积 → 输出 4×reg_max 通道
    )
    for x in ch   # 对每个尺度（P3, P4, P5）各一套
)
```

为什么输出 `4 × reg_max` 通道？
- `4` = 上、下、左、右四个方向的距离
- `reg_max` = 每个方向用一个**概率分布**来预测（不是直接输出一个数字），这就是 DFL 的核心思想
- **YOLO26 中 `reg_max=1`，所以回归输出 = 4 通道**（退化为直接输出 4 个距离值，不再用 DFL 分布）

### 2.2 分类分支 `cv3`：预测类别

```python
# YOLO26 模式（非 legacy）:
self.cv3 = nn.ModuleList(
    nn.Sequential(
        nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),   # 深度可分离卷积 + 1×1 压缩
        nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)), # 再来一层
        nn.Conv2d(c3, self.nc, 1),                          # 1×1 → 输出 nc 通道（每类一个得分）
    )
    for x in ch
)
```

**回归用标准 Conv，分类用 DWConv**：分类需要的通道交互更少，DWConv 更轻量。这是 YOLO26 相对 YOLOv8 的优化。

### 2.3 全局数据流图

```
                       Detect Head
                    ┌───────────────────────────────────────────────┐
   P3 [1,256,80,80] │  cv2[0]: Conv→Conv→Conv1×1 → [1, 4, 80,80]  │  回归
                    │  cv3[0]: DWConv→DWConv→Conv1×1→[1,nc,80,80]  │  分类
                    ├───────────────────────────────────────────────┤
   P4 [1,512,40,40] │  cv2[1]: Conv→Conv→Conv1×1 → [1, 4, 40,40]  │  回归
                    │  cv3[1]: DWConv→DWConv→Conv1×1→[1,nc,40,40]  │  分类
                    ├───────────────────────────────────────────────┤
   P5 [1,1024,20,20]│  cv2[2]: Conv→Conv→Conv1×1 → [1, 4, 20,20]  │  回归
                    │  cv3[2]: DWConv→DWConv→Conv1×1→[1,nc,20,20]  │  分类
                    └───────────────────────────────────────────────┘
                          │                              │
                          ▼                              ▼
                    合并 3 个尺度的回归结果         合并 3 个尺度的分类结果
                    [1, 4, 8400]                  [1, nc, 8400]
                                                  (80×80 + 40×40 + 20×20 = 8400 个锚点)
```

**8400 的来历**：`80×80 + 40×40 + 20×20 = 6400 + 1600 + 400 = 8400`。网络在每个特征图的每个位置各放一个"锚点"，一共 8400 个候选检测位置。

---

## 3. `forward()` 完整流程

### 3.1 `forward_head()` —— 核心计算

```python
def forward_head(self, x, box_head=None, cls_head=None):
    bs = x[0].shape[0]  # batch size
    boxes = torch.cat(
        [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
        dim=-1
    )
    scores = torch.cat(
        [cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)],
        dim=-1
    )
    return dict(boxes=boxes, scores=scores, feats=x)
```

**逐行解读**：

1. `box_head[i](x[i])` —— 对第 i 个尺度的特征图跑 cv2 分支，得到 `[bs, 4*reg_max, Hi, Wi]`
2. `.view(bs, 4*reg_max, -1)` —— 把空间维（Hi×Wi）拉平成一维，变成 `[bs, 4*reg_max, Hi*Wi]`
3. `torch.cat(..., dim=-1)` —— 把 3 个尺度拼起来：`[bs, 4*reg_max, 8400]`
4. 分类分支同理：`[bs, nc, 8400]`

### 3.2 `forward()` —— 训练 vs 推理

```python
def forward(self, x):
    preds = self.forward_head(x, **self.one2many)   # 主预测
    if self.end2end:
        x_detach = [xi.detach() for xi in x]
        one2one = self.forward_head(x_detach, **self.one2one)
        preds = {"one2many": preds, "one2one": one2one}
    if self.training:
        return preds                    # 训练时直接返回原始预测，由 loss 函数处理
    y = self._inference(preds["one2one"] if self.end2end else preds)
    ...
    return y                            # 推理时返回解码后的 bbox + 类别
```

**两个关键分支**：
- **训练模式** (`self.training=True`)：返回原始的 boxes 和 scores 张量，交给损失函数计算 loss。
- **推理模式**：先 DFL 解码 boxes → 再乘 stride 还原到输入图尺度 → sigmoid 归一化分类得分 → 后处理（Top-K 筛选）。

### 3.3 `_inference()` —— 推理解码

```python
def _inference(self, x):
    dbox = self._get_decode_boxes(x)
    return torch.cat((dbox, x["scores"].sigmoid()), 1)
```

- `dbox`：解码后的边界框坐标，形状 `[bs, 4, 8400]`
- `x["scores"].sigmoid()`：对分类 logits 做 sigmoid → 变成 0~1 的概率，形状 `[bs, nc, 8400]`
- 拼接后：`[bs, 4+nc, 8400]` = 每个锚点有 4 个坐标 + nc 个类别概率

### 3.4 `postprocess()` —— 最终筛选

```python
def postprocess(self, preds):
    boxes, scores = preds.split([4, self.nc], dim=-1)
    scores, conf, idx = self.get_topk_index(scores, self.max_det)  # Top-K
    boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
    return torch.cat([boxes, scores, conf], dim=-1)
```

从 8400 个候选中取 **Top-300**（`max_det=300`）个最高分的预测，输出 `[bs, 300, 6]`：

```
最终输出格式：[x1, y1, x2, y2, 最大类别概率, 类别编号]
              ────bbox坐标──── ──── 分类信息 ────
```

---

## 4. DFL — 分布焦点损失回归头

**位置**：`block.py:58`

```python
class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
```

### DFL 做了什么？（用最直白的方式讲）

**传统回归**：网络直接输出 1 个数字表示"这个方向的距离"。

**DFL 回归**：把距离分成 `reg_max` 个"格子"（比如 0, 1, 2, …, 15），网络输出每个格子的**概率**，然后加权求和得到最终距离。

```
传统: 网络直接输出 → 7.3（一个数）

DFL:  网络输出 → [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.15, 0.05, 0.0, ...]
                  概率分布在 0~15 上

      最终距离 = 0×0.0 + 1×0.0 + ... + 5×0.3 + 6×0.2 + 7×0.15 + ...
              = 加权求和 ≈ 5.6
```

**好处**：模型可以表达"不太确定是 5 还是 6"这种模糊性，训练更稳定。

### YOLO26 为什么 `reg_max=1`？

当 `reg_max=1` 时，每个方向只有 1 个"格子"，概率分布退化为确定值 → **DFL 变成 `nn.Identity()`**（什么也不做）。
- 这是 YOLO26 的设计简化：实验发现 `reg_max=1` 性能不掉但推理更快
- YOLOv8 用的是 `reg_max=16`
- YOLO-pineapple 论文基于 YOLOv8，所以 `reg_max=16`，会用到完整的 DFL

---

## 5. End-to-End 模式（YOLO26 新特性）

**YAML 中的设置**：`end2end: True`

### 什么是 End-to-End？

传统 YOLO 推理流程：
```
模型输出 8400 个预测 → NMS（非极大值抑制）去重复框 → 最终结果
```

End-to-End 模式：
```
模型内部直接学会去重复 → 输出就是最终结果，不再需要 NMS 后处理
```

### 怎么实现的？双头架构

```python
if end2end:
    self.one2one_cv2 = copy.deepcopy(self.cv2)  # 额外一套 bbox 回归头
    self.one2one_cv3 = copy.deepcopy(self.cv3)  # 额外一套分类头
```

- **one2many 头**（`self.cv2 / cv3`）：训练用，一个 GT 可以匹配多个正样本 → 提供丰富的训练信号
- **one2one 头**（`self.one2one_cv2 / cv3`）：推理用，一个 GT 只匹配一个正样本 → 天然无需 NMS

```
训练时：
  特征图 → one2many 头 → 计算 loss（正样本多，梯度信号丰富）
        → one2one 头  → 计算 loss（正样本少但精准）

推理时：
  特征图 → one2one 头 → 直接输出 → postprocess (Top-K) → 完成
         （跳过 NMS！推理更快）
```

`forward()` 里 `x_detach = [xi.detach() for xi in x]`：one2one 头的梯度**不传回** Backbone —— 只有 one2many 头负责驱动特征学习。

---

## 6. 其他 Head 变体（Segment / Pose / OBB）

它们都**继承自 `Detect`**，只是在 bbox + 类别之外增加了额外输出：

| Head | 额外输出 | 额外分支 | 用途 |
|---|---|---|---|
| `Segment` | 实例掩码系数 + Proto | `cv4` + `Proto` | 实例分割 |
| `Pose` | 关键点坐标 | `cv4`（输出 kpt_shape） | 人体姿态估计 |
| `OBB` | 旋转角度 | `cv4`（输出 1 通道角度） | 旋转框检测 |

```python
class Segment(Detect):    # 继承 Detect，额外加 mask 分支
class Pose(Detect):       # 继承 Detect，额外加关键点分支
class OBB(Detect):        # 继承 Detect，额外加角度分支
```

**你做水稻病虫害识别，直接用 `Detect` 就够了**。如果以后要做"分割出每只虫子的精确轮廓"，再换 `Segment`。

---

## 7. 关键问答

### Q1: 8400 这个数字怎么来的？换成不同 imgsz 会变吗？

**A**: 8400 = `(640/8)² + (640/16)² + (640/32)² = 80² + 40² + 20² = 6400 + 1600 + 400`。

如果换成 `imgsz=320`：`40² + 20² + 10² = 1600 + 400 + 100 = 2100`。

锚点总数和输入图尺寸成正比的平方关系。

### Q2: 回归分支输出的 4 个值分别是什么？

**A**: `[dist_left, dist_top, dist_right, dist_bottom]` —— 从锚点中心到框的左/上/右/下四条边的**距离**。

```
         dist_top
           ┆
    ┌──────┼──────┐
    │      ┆      │
    │      ●──────┤ dist_right
dist_left  │      │
    │      ┆      │
    └──────┼──────┘
         dist_bottom

  ● = 锚点中心（特征图上的一个网格位置）
```

最终通过 `dist2bbox()` 工具函数把距离转换成 `[x1, y1, x2, y2]` 或 `[cx, cy, w, h]` 格式。

### Q3: 分类分支为什么用 sigmoid 而不是 softmax？

**A**: YOLO 使用**多标签分类**（每个锚点可以同时属于多个类别）。
- `softmax`：所有类别概率之和 = 1 → **互斥**，只能选一个类
- `sigmoid`：每个类别独立计算 0~1 概率 → **非互斥**，可以同时匹配多个类

对于水稻病虫害场景，如果某个位置同时有"稻飞虱"和"褐斑"，sigmoid 允许两个类都有高概率。

### Q4: YOLOv8 和 YOLO26 的 Detect Head 主要区别？

| 特性 | YOLOv8 | YOLO26 |
|---|---|---|
| `reg_max` | 16（完整 DFL） | 1（退化为直接回归） |
| `end2end` | False（需要 NMS） | True（免 NMS） |
| 分类分支 | 标准 Conv | DWConv（更轻量） |
| 双头 | 无 | one2many + one2one |

### Q5: `stride` 是什么？为什么重要？

**A**: `stride = [8, 16, 32]` 表示每个尺度的特征图相对于输入图的缩放倍数。

作用：把特征图上的坐标**还原**到原图尺度。
```
特征图上预测 (5, 3) → 原图坐标 (5×8, 3×8) = (40, 24)    # P3, stride=8
特征图上预测 (5, 3) → 原图坐标 (5×16, 3×16) = (80, 48)   # P4, stride=16
```

在 `_get_decode_boxes` 中：`dbox = decode_bboxes(...) * self.strides` —— 乘 stride 把距离从"特征图格子数"转回"像素数"。

### Q6: `make_anchors()` 做了什么？

**A**: 在每个特征图的每个网格位置生成一个"锚点"（中心坐标）。不像 YOLOv5 有预设框尺寸，YOLO26 是 **anchor-free** 的 —— 锚点只是位置标记，框的大小完全由网络学习。

```
P3 (80×80): 生成 6400 个锚点，坐标 = (0.5, 0.5), (1.5, 0.5), ..., (79.5, 79.5)
P4 (40×40): 生成 1600 个锚点
P5 (20×20): 生成 400 个锚点
总计: 8400 个锚点
```

---

## 8. 下一步

读完 Step 3，你已经理解了**特征如何变成检测结果**。结合 Step 1 和 Step 2：

```
完整推理链路：
  图像 → Conv 下采样（Step 2）→ C3k2 + SPPF + C2PSA 提特征（Step 2）
       → FPN + PAN 融合多尺度（Step 1）→ Detect 输出 bbox + class（Step 3）
```

接下来：
- **Step 4** 看 `parse_model()` 是如何把 YAML 蓝图"翻译"成上面这些 PyTorch 层的。
- **Step 5** 看 `YOLO().train()` 入口怎么把一切串起来。

#### 与论文的衔接

YOLO-pineapple 论文也可能改了检测头（如修改 DFL 参数、增加额外的分类分支、或改用不同的 loss）。你现在已经看懂了 Detect 的结构，去论文里找 "detection head" 或 "loss function" 相关的章节，看看它和标准 Detect 有什么不同。
