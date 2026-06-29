# Step 5 — 高层 API 与训练循环（Skim）

> **前置**：已完成 Step 1–4。你现在清楚：YAML 蓝图（Step 1）→ 模块实现（Step 2）→ Detect 输出预测（Step 3）→ parse_model 装配（Step 4）。
>
> **本节目标**：不要求精读每一行，只需搞清楚**三个入口**在哪、它们之间怎么串联。这是你以后 `model.train(data="rice.yaml")` 时需要知道的。
>
> **涉及文件**：
> - `ultralytics/models/yolo/model.py` — `YOLO` 类（用户直接用的入口）
> - `ultralytics/engine/model.py` — `Model` 基类（.train / .val / .predict）
> - `ultralytics/engine/trainer.py` — `BaseTrainer`（训练循环）

---

## 目录

- [1. 三层架构全景图](#1-三层架构全景图)
- [2. YOLO 类：用户入口](#2-yolo-类用户入口)
- [3. Model 基类：统一 API](#3-model-基类统一-api)
  - [3.1 _new vs _load](#31-_new-vs-_load)
  - [3.2 predict()](#32-predict)
  - [3.3 val()](#33-val)
  - [3.4 train()](#34-train)
- [4. BaseTrainer：训练循环](#4-basetrainer训练循环)
- [5. 常用命令速查](#5-常用命令速查)
- [6. 关键问答](#6-关键问答)
- [7. 总结：五步读完之后的全景图](#7-总结五步读完之后的全景图)

---

## 1. 三层架构全景图

```
用户写的 Python 代码 / CLI 命令
─────────────────────────────
   from ultralytics import YOLO
   model = YOLO("yolo26n.yaml")
   model.train(data="rice.yaml", epochs=100)

           │
           ▼
┌──────────────────────────────────────────────┐
│  第 1 层: YOLO 类                             │  models/yolo/model.py
│  (智能分发: YOLO / YOLOWorld / YOLOE / RTDETR) │
│                                              │
│  .__init__("yolo26n.yaml")                    │
│     → 判断模型类型 → 调父类 Model.__init__    │
│  .task_map → 根据 task 查找对应的              │
│     Trainer / Validator / Predictor 类        │
└──────────┬───────────────────────────────────┘
           │ 继承
           ▼
┌──────────────────────────────────────────────┐
│  第 2 层: Model 基类                          │  engine/model.py
│  (统一的 .train / .val / .predict / .export)  │
│                                              │
│  ._new(cfg) → 加载 YAML + parse_model         │
│  ._load(weights) → 加载 .pt 权重文件          │
│  .train(**kwargs) → 创建 Trainer 实例并训练   │
│  .val(**kwargs) → 创建 Validator 实例并验证   │
│  .predict(source) → 创建 Predictor 实例并推理 │
└──────────┬───────────────────────────────────┘
           │ 委托
           ▼
┌──────────────────────────────────────────────┐
│  第 3 层: BaseTrainer / BaseValidator /       │  engine/trainer.py
│          BasePredictor                        │  engine/validator.py
│  (实际执行训练循环 / 验证 / 推理)             │  engine/predictor.py
│                                              │
│  各 task 有子类:                              │
│  DetectionTrainer → models/yolo/detect/train.py │
│  SegmentationTrainer → models/yolo/segment/train.py │
└──────────────────────────────────────────────┘
```

---

## 2. YOLO 类：用户入口

**位置**：`ultralytics/models/yolo/model.py:26`

```python
class YOLO(Model):
    def __init__(self, model="yolo26n.pt", task=None, verbose=False):
        path = Path(model)
        if "-world" in path.stem:     # YOLOWorld 模型
            ...  # 切换到 YOLOWorld 类
        elif "yoloe" in path.stem:    # YOLOE 模型
            ...  # 切换到 YOLOE 类
        else:
            super().__init__(model=model, task=task, verbose=verbose)
```

**做的事情很简单**：根据文件名智能分发到正确的子类。你用 `YOLO("yolo26n.yaml")` 会走 `super().__init__` → 进入 `Model` 基类。

### task_map：任务路由表

```python
@property
def task_map(self):
    return {
        "detect":   {"model": DetectionModel,   "trainer": DetectionTrainer,   ...},
        "segment":  {"model": SegmentationModel, "trainer": SegmentationTrainer, ...},
        "classify": {"model": ClassificationModel, ...},
        "pose":     {"model": PoseModel, ...},
        "obb":      {"model": OBBModel, ...},
    }
```

当你调用 `model.train()` 时，框架通过 `self.task`（如 "detect"）查这张表，找到对应的 `DetectionTrainer` 来执行训练。

**你做水稻病虫害检测**，task = "detect"，走 `DetectionModel` + `DetectionTrainer`。

---

## 3. Model 基类：统一 API

**位置**：`ultralytics/engine/model.py:29`

### 3.1 `_new` vs `_load`

```python
# 从 YAML 新建模型（随机权重）
model = YOLO("yolo26n.yaml")
#  → 内部调 _new("yolo26n.yaml")
#  → yaml_model_load → parse_model → DetectionModel → 随机初始化

# 从 .pt 加载模型（预训练权重）
model = YOLO("yolo26n.pt")
#  → 内部调 _load("yolo26n.pt")
#  → load_checkpoint → 已训练好的权重 + 模型结构
```

**区别**：
| | `.yaml` | `.pt` |
|---|---|---|
| 模型结构 | 从 YAML 新建 | 从 checkpoint 恢复 |
| 权重 | 随机初始化 | 预训练权重 |
| 用途 | 从零开始训练 / 验证结构 | 微调 / 推理 / 迁移学习 |

**你以后的使用方式**：
```python
# 用预训练权重微调到水稻数据集
model = YOLO("yolo26n.pt")          # 加载预训练模型
model.train(data="rice.yaml", epochs=100)  # 迁移学习
```

### 3.2 `predict()`

**位置**：`engine/model.py:479`

```python
def predict(self, source=None, stream=False, predictor=None, **kwargs):
    ...
    self.predictor = (predictor or self._smart_load("predictor"))(overrides=args)
    self.predictor.setup_model(model=self.model)
    return self.predictor(source=source, stream=stream)
```

**调用链**：

```
model.predict("cat.jpg")
   │
   ▼
Model.predict()
   │
   ├── self._smart_load("predictor") → 查 task_map → DetectionPredictor
   │
   ▼
DetectionPredictor(source="cat.jpg")
   │
   ├── 图像预处理（resize, normalize, pad）
   ├── model.forward(img_tensor)           ← 走 BaseModel._predict_once（Step 4 讲的）
   ├── Detect.postprocess()                ← Step 3 讲的 Top-K 筛选
   ├── NMS / End2End 后处理
   │
   ▼
Results 对象列表 → 包含 boxes, scores, classes, 可视化
```

### 3.3 `val()`

**位置**：`engine/model.py:582`

```python
def val(self, validator=None, **kwargs):
    ...
    self.validator = (validator or self._smart_load("validator"))(args=args)
    self.validator(model=self.model)
    self.metrics = self.validator.metrics
```

验证流程：遍历验证集 → 逐 batch 推理 → 和 GT 对比 → 计算 mAP 等指标。

### 3.4 `train()`

**位置**：`engine/model.py:715`

```python
def train(self, trainer=None, **kwargs):
    self._check_is_pytorch_model()
    ...
    # 合并参数：默认值 + 用户配置
    args = {**overrides, **custom, **kwargs, "mode": "train"}

    # 创建 Trainer 并启动
    self.trainer = (trainer or self._smart_load("trainer"))(overrides=args)
    self.trainer.train()

    # 训练完后加载 best.pt
    self.model, self.ckpt = load_checkpoint(self.trainer.best)
    self.metrics = self.trainer.validator.metrics
```

**重点理解**：`model.train()` 本身不直接跑训练循环，它只是创建一个 `BaseTrainer` 子类实例，然后调 `trainer.train()`。训练循环的具体逻辑在 BaseTrainer 里。

---

## 4. BaseTrainer：训练循环

**位置**：`ultralytics/engine/trainer.py:67`

> 这个类有 500+ 行，不需要精读。只需理解它做了什么，以及你以后可能需要改什么。

### 训练流程的 5 个阶段

```
trainer.train()
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│ 阶段 1: _setup_train()                                          │
│   ├── setup_model() → 加载/创建模型、移到 GPU                   │
│   ├── freeze 指定层（如冻结 backbone 做迁移学习）                │
│   ├── check_amp() → 检查是否支持混合精度训练                     │
│   ├── build_optimizer() → 创建优化器（SGD/Adam/AdamW）           │
│   ├── build_dataloader() → 创建训练数据加载器                    │
│   └── ModelEMA → 指数移动平均（让模型更稳定）                    │
├──────────────────────────────────────────────────────────────────┤
│ 阶段 2: 主循环 while True:                                       │
│   for epoch in range(epochs):                                    │
│     ├── 学习率 warmup（前几个 epoch 从小 lr 慢慢升高）           │
│     │                                                            │
│     ├── for batch in train_loader:         ← 内层循环            │
│     │   ├── preprocess_batch(batch)        预处理                │
│     │   ├── model(batch)                   前向传播              │
│     │   ├── loss.backward()                反向传播              │
│     │   └── optimizer.step()               更新权重              │
│     │                                                            │
│     ├── 每 epoch 结束后验证 → self.validate()                    │
│     ├── Early Stopping 检查                                      │
│     └── 保存 checkpoint（last.pt / best.pt）                     │
├──────────────────────────────────────────────────────────────────┤
│ 阶段 3: 训练结束                                                 │
│   ├── final_eval() → 用 best.pt 做最终验证                       │
│   ├── plot_metrics() → 画 loss 和 mAP 曲线                      │
│   └── 保存结果到 runs/detect/train/                              │
└──────────────────────────────────────────────────────────────────┘
```

### 训练核心 4 行代码（`_do_train` 内部）

```python
# trainer.py 第 431-448 行（简化版）
with autocast(self.amp):              # 自动混合精度
    batch = self.preprocess_batch(batch)
    loss, self.loss_items = self.model(batch)    # 前向 + 算 loss
    self.loss = loss.sum()

self.scaler.scale(self.loss).backward()           # 反向传播
self.optimizer_step()                              # 更新权重
```

**理解这 4 行就够了**：
1. `model(batch)` —— 把图像喂进模型 → 得到预测 → 算 loss
2. `loss.backward()` —— 反向传播计算梯度
3. `optimizer.step()` —— 用梯度更新模型权重

这就是深度学习训练的核心："前向 → 算 loss → 反向 → 更新"，不停循环。

### 你以后可能需要改的地方

| 需求 | 改哪里 |
|---|---|
| 冻结 Backbone 做迁移学习 | `model.train(freeze=10)` 冻结前 10 层 |
| 换优化器 | `model.train(optimizer="AdamW")` |
| 修改学习率 | `model.train(lr0=0.001)` |
| 修改 loss 函数 | `DetectionModel.init_criterion()` 在 tasks.py |
| 加数据增强 | 配置 `data.yaml` 或修改 `augment.py` |

---

## 5. 常用命令速查

### Python API

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo26n.pt")               # 预训练权重
model = YOLO("yolo26n.yaml")             # 从零开始

# 训练
model.train(data="coco8.yaml", epochs=100, imgsz=640, batch=16, device=0)

# 验证
metrics = model.val(data="coco8.yaml")
print(metrics.box.map)                   # mAP@0.5:0.95
print(metrics.box.map50)                 # mAP@0.5

# 推理
results = model.predict("cat.jpg", conf=0.25)
results[0].boxes.xyxy                    # 边界框坐标
results[0].boxes.cls                     # 类别编号
results[0].boxes.conf                    # 置信度
results[0].plot()                        # 画框可视化

# 导出
model.export(format="onnx")              # 导出 ONNX

# 模型信息
model.info()                             # 打印层数、参数量、GFLOPs
```

### CLI 命令

```bash
# 安装（在 ultralytics-main 目录）
pip install -e .

# 训练
yolo train model=yolo26n.pt data=coco8.yaml epochs=100 imgsz=640

# 验证
yolo val model=yolo26n.pt data=coco8.yaml

# 推理
yolo predict model=yolo26n.pt source=cat.jpg conf=0.25

# 导出
yolo export model=yolo26n.pt format=onnx

# 基准测试
yolo benchmark model=yolo26n.pt imgsz=640
```

### 测试

```bash
# 运行全部测试
pytest tests/

# 运行单个测试文件
pytest tests/test_cli.py -v

# 包含慢速测试
pytest --slow tests/
```

---

## 6. 关键问答

### Q1: `model.train()` 和 `model(batch)` 的 `model` 是同一个对象吗？

**A**: 不是！要区分两层 model：

```
YOLO (engine/model.py)              ← 你写 model = YOLO("yolo26n.pt") 的这个
  └── self.model (nn.Module)        ← DetectionModel 实例，真正的 PyTorch 网络
        └── self.model (nn.Sequential) ← parse_model 生成的层序列
```

- `model.train(data=...)` —— 调的是 `YOLO` 类的 `.train()` 方法 → 启动训练流程
- `model(batch)` 在 trainer 内部 —— 调的是 `DetectionModel` 的 `forward()` → 前向传播算 loss

### Q2: 训练输出保存在哪里？

**A**: 默认保存在 `runs/detect/train/` 目录下：

```
runs/detect/train/
  ├── weights/
  │   ├── best.pt        ← mAP 最高的模型（你要用的）
  │   └── last.pt        ← 最后一个 epoch 的模型
  ├── results.csv        ← 每 epoch 的 loss 和指标
  ├── confusion_matrix.png
  ├── results.png        ← loss 和 mAP 曲线图
  ├── val_batch0_pred.jpg
  └── args.yaml          ← 本次训练的完整配置
```

### Q3: 迁移学习 vs 从零训练，怎么选？

**A**:

| 场景 | 做法 | 原因 |
|---|---|---|
| 数据集小（< 5000 张）| `YOLO("yolo26n.pt").train(data=...)` | 预训练权重提供通用特征，小数据集从零训练会过拟合 |
| 数据集大（> 50000 张）| `YOLO("yolo26n.yaml").train(data=...)` | 数据量够大，从零学习也能学好，且不受预训练分布限制 |
| 想微调 + 冻结 Backbone | `model.train(data=..., freeze=10)` | 冻结前 10 层，只训练 Neck + Head，适合和预训练域差异不大的情况 |

**对于你的水稻病虫害**：图片来自 UAV 航拍，和 COCO 的日常图片差异较大。推荐**先用预训练权重微调全网络**（不冻结），如果过拟合再尝试冻结 Backbone。

### Q4: EMA 是什么？为什么 trainer 里到处都有？

**A**: **EMA（Exponential Moving Average）**—— 指数移动平均。训练过程中维护一个"权重的滑动平均版本"：

```
EMA_weight = 0.999 × EMA_weight_old + 0.001 × current_weight
```

好处：EMA 模型比当前模型更"稳定"，验证和推理时用 EMA 版本通常比直接用训练模型效果好 0.5~1 mAP。

你在 `trainer.py` 里看到的 `self.ema.update_attr(self.model)` 就是每 epoch 更新一次 EMA。保存的 `best.pt` 存的是 EMA 版本的权重。

### Q5: callbacks 系统是什么？

**A**: 一个**钩子机制**，让你在训练的各个阶段插入自定义逻辑。常见钩子：

```
on_train_start       → 训练开始前
on_train_epoch_start → 每个 epoch 开始
on_train_batch_end   → 每个 batch 结束
on_train_epoch_end   → 每个 epoch 结束
on_train_end         → 训练结束后
on_model_save        → 保存 checkpoint 时
```

用法（举例：每个 epoch 结束打印学习率）：

```python
def my_callback(trainer):
    print(f"Epoch {trainer.epoch}, LR: {trainer.lr}")

model = YOLO("yolo26n.pt")
model.add_callback("on_train_epoch_end", my_callback)
model.train(data="rice.yaml", epochs=50)
```

---

## 7. 总结：五步读完之后的全景图

```
                        完整调用链（从用户代码到 GPU 计算）
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  用户:  model = YOLO("yolo26n.yaml")                                         │
│           │                                                                  │
│           ▼                                                                  │
│  Step 5: YOLO.__init__                                                       │
│           │                                                                  │
│           ├── Model._new("yolo26n.yaml")                                     │
│           │     │                                                            │
│           │     ▼                                                            │
│  Step 4:  │   yaml_model_load → parse_model()                                │
│           │     │  遍历 YAML 每一行                                          │
│           │     │  查 base_modules → 实例化模块 → nn.Sequential              │
│           │     │                                                            │
│  Step 1:  │     │  ◄── yolo26.yaml 提供层拓扑                                │
│  Step 2:  │     │  ◄── Conv, C3k2, SPPF, C2PSA 提供实现                     │
│  Step 3:  │     │  ◄── Detect 提供检测头                                     │
│           │     │                                                            │
│           │     ▼                                                            │
│           │   DetectionModel(model=nn.Sequential, save=[...], stride=[8,16,32])│
│           │                                                                  │
│  用户:  model.train(data="rice.yaml", epochs=100)                            │
│           │                                                                  │
│           ▼                                                                  │
│  Step 5: Model.train()                                                       │
│           │  查 task_map → DetectionTrainer                                  │
│           ▼                                                                  │
│         BaseTrainer._do_train()                                              │
│           │                                                                  │
│           ├── for epoch in range(100):                                        │
│           │     for batch in dataloader:                                      │
│           │       ├── preds = model(images)         ◄── _predict_once (Step 4)│
│           │       ├── loss = criterion(preds, targets)                        │
│           │       ├── loss.backward()                                         │
│           │       └── optimizer.step()                                        │
│           │     validate() → mAP                                             │
│           │     save best.pt                                                 │
│           │                                                                  │
│           ▼                                                                  │
│         runs/detect/train/weights/best.pt   ← 你最终要用的模型               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Reading Roadmap 完成后你具备的能力

| 能力 | 对应 Step |
|---|---|
| 看 YAML 知道网络长什么样 | Step 1 |
| 看模块类名知道它在做什么 | Step 2 |
| 理解检测头怎么输出 bbox + class | Step 3 |
| 知道加自定义模块改哪 4 个文件 | Step 4 |
| 会用 Python API / CLI 训练和推理 | Step 5 |

### 下一阶段预告

Reading Roadmap 读完后，你可以进入**论文复现阶段**：

1. **精读 YOLO-pineapple 论文**，提取 3 个改进点（CBAM 注意力、SPP 增强、其他）
2. **参考 Step 4 的自定义模块流程**，在 YOLO26 上实现每个改进
3. **准备水稻病虫害数据集**（YOLO 格式：images/ + labels/）
4. **训练 baseline**（原版 YOLO26）→ 记录 mAP
5. **逐个加入改进** → 对比 mAP 提升 → 写入毕设论文

到那个阶段再来对话窗口，我们一起做代码修改。现在先把 5 个 Step 读扎实。
