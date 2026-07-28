# YOLO26 源码改进工作流与实验记录

> 最后更新：2026-07-28
> 当前阶段：项目结构整理完成，旧 V2 已归档；新 V2 分支保持空白，尚未改代码、尚未训练
> 核心原则：每个单项改进都从同一 Baseline 出发，最终统一使用 `val`

---

## 1. 项目目的

本项目的目标是围绕水稻虫害无人机影像实例分割任务，对 YOLO26 源码进行有针对性的改进实验，为毕业论文和后续文章提供：

1. 可复现的源码改动；
2. 公平的实验对比；
3. 完整的训练与验证记录；
4. 能够解释“为什么这样改”的学习笔记；
5. 可回溯的 Git 分支和提交。

`YOLO-pineapple.html` 是农业无人机小目标检测方向的参考资料。项目借鉴它对小目标、复杂背景、多尺度特征和注意力机制的分析，但**不以复现该论文为目的**。

当前计划中的 P2、Dice 等内容，是结合本课题实例分割问题制定的定制改进，不是 YOLO-Pineapple 的原始模块。

---

## 2. 源码如何在训练时生效

### 2.1 两个工作区域

| 路径 | 用途 |
|---|---|
| `E:\Study\DeepCNN\yolo26\yolo_plus\` | Git 仓库、源码改进、文档和实验记录 |
| `E:\Study\DeepCNN\yolo26\code\` | 数据集、预训练权重、旧训练脚本和原始训练输出 |

真正被修改的 Ultralytics 源码位于：

`E:\Study\DeepCNN\yolo26\yolo_plus\ultralytics-main`

### 2.2 可编辑安装

`yolo26` Conda 环境使用了可编辑安装：

```powershell
cd E:\Study\DeepCNN\yolo26\yolo_plus\ultralytics-main
pip install -e .
```

可编辑安装后：

```text
训练脚本
  └─ from ultralytics import YOLO
       └─ 加载 yolo_plus\ultralytics-main\ultralytics\
```

因此，当前 Git 分支中的源码修改会直接影响训练。

训练前必须验证：

```powershell
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
```

路径应指向：

`E:\Study\DeepCNN\yolo26\yolo_plus\ultralytics-main\ultralytics\__init__.py`

### 2.3 路径与 Git 分支的区别

`ultralytics-main` 是物理文件夹，不等于 Git 的 `main` 分支。

- 切换分支不会改变源码物理路径；
- 切换分支会改变这个路径中显示的文件内容；
- `pip install -e` 仍指向同一路径；
- 所以训练实际使用的是**当前分支**中的源码。

---

## 3. Git 实验原则

### 3.1 分支职责

| 分支 | 内容 | 与谁比较 |
|---|---|---|
| `main` | Baseline 源码、公共文档、公共训练工具 | 对照组 |
| `v1-cbam` | 只包含 CBAM 改进 | Baseline |
| `v2-p2` | 只包含 P2 小目标尺度改进 | Baseline |
| `v3-dice` | 只包含 Mask Loss 改进 | Baseline |
| `v4-combined` | 只组合已经证明有效的模块 | Baseline 和单项版本 |
| `archive/v2-p2-failed` | 旧 V2 失败实现，仅供复盘 | 不参与实验 |

### 3.2 单项实验规则

V1、V2、V3 必须分别从同一个 Baseline 提交建立。

这类实验称为：

> 单因素独立对比实验

它和论文中的累积消融不同。后续正式论文阶段如有需要，再补：

```text
Baseline
  → +改进 A
  → +改进 A+B
  → +改进 A+B+C
```

### 3.3 公平比较

原则上保持以下设置一致：

- 数据划分；
- 输入尺寸；
- epoch；
- seed；
- optimizer；
- 数据增强；
- patience；
- 最终验证流程。

若因显存限制修改 batch，必须如实记录，不能写成“所有设置完全一致”。

---

## 4. 改进路线

| 版本 | 改进 | 位置 | 主要目标 | 状态 |
|---|---|---|---|---|
| Baseline | 原始 yolo26m-seg | Backbone + Neck + Head | 对照组 | 已完成 |
| V1 | CBAM | Backbone | 抑制复杂背景、增强虫害特征 | 已完成 |
| V2 | P2 小目标尺度 | Neck + Segment Head | 提高卷叶螟小目标召回率 | 新分支已建立，未实现 |
| V3 | BCE + Dice 等 Mask Loss | Loss | 改善前景学习和边缘精度 | 未开始 |
| V4 | 有效模块组合 | 综合 | 验证模块互补性 | 未开始 |

### 4.1 V1 与参考论文的关系

YOLO-Pineapple 的 GMSC 内部使用 CBAM，并比较了多种注意力机制。V1 借鉴了这部分结论，但当前 V1 是在 YOLO26 Backbone 中直接加入 CBAM，不是完整复现 GMSC。

### 4.2 V2 的新边界

新版 V2 只验证：

> 增加 P2 小目标预测尺度，是否能改善卷叶螟召回率和 Mask AP。

新版 V2 暂时不做：

- 不继承 CBAM；
- 不加入 Dice；
- 不提高 Mask Proto 分辨率；
- 不修改 SegmentationValidator；
- 不全局修改 Proto26；
- 不混入训练超参数优化。

这样可以保证 V2 的变量尽可能单一。

---

## 5. 已完成实验

### 5.1 Baseline

- 模型：`yolo26m-seg.pt`
- 数据：768 train / 95 val / 95 test
- epoch：400
- best epoch：246
- imgsz：640
- batch：8

历史记录中的 Val Mask 指标：

| 指标 | 数值 |
|---|---:|
| Precision(M) | 0.717 |
| Recall(M) | 0.620 |
| mAP50(M) | 0.683 |
| mAP50-95(M) | 0.329 |
| 卷叶螟 AP50(M) | 0.604 |
| 钻心虫 AP50(M) | 0.763 |

### 5.2 V1 CBAM

- 分支：`v1-cbam`
- best epoch：250
- early stop：epoch 350
- 训练时间：约 3.51 小时

历史记录中的 Val Mask 指标：

| 指标 | Baseline | V1 CBAM | 变化 |
|---|---:|---:|---:|
| 整体 P(M) | 0.717 | 0.753 | +0.036 |
| 整体 R(M) | 0.620 | 0.633 | +0.013 |
| mAP50(M) | 0.683 | 0.700 | +0.017 |
| mAP50-95(M) | 0.329 | 0.343 | +0.014 |
| 卷叶螟 P(M) | 0.679 | 0.662 | -0.017 |
| 卷叶螟 R(M) | 0.482 | 0.552 | +0.070 |
| 卷叶螟 AP50(M) | 0.604 | 0.607 | +0.003 |
| 钻心虫 P(M) | 0.753 | 0.844 | +0.091 |
| 钻心虫 R(M) | 0.755 | 0.714 | -0.041 |
| 钻心虫 AP50(M) | 0.763 | 0.793 | +0.030 |

这些数值保留为历史结果。正式论文表格前，应按第 7 节重新用 `best.pt` 执行统一独立 `val`。

---

## 6. 旧 V2 归档

旧 V2 曾同时修改：

- P2 Neck；
- Segment26 输入尺度；
- Proto26 基础特征和输出分辨率；
- SegmentationValidator；
- 预训练权重索引映射；
- batch。

训练中出现：

- Box mAP 能够学习；
- Mask mAP 长期接近 0；
- 后期 `seg_loss` 出现 NaN/Inf。

由于同时改变的变量过多，无法确认单一根因。旧记录提出的“近单位初始化解决 BCE Collapse”没有经过重新训练验证，因此不能作为已证明结论。

处理方式：

- 旧代码保存在 `archive/v2-p2-failed`；
- 旧训练不进入正式对比表；
- 新 `v2-p2` 从 `main` 独立建立；
- 新 V2 不继承旧 Proto26 和 Validator 改动。

详细历史可查：

- `PROJECT_TRANSFER.md`
- `node.md`
- `CODEX_PROJECT_CONTEXT.md`

---

## 7. 统一验证和指标口径

用户已经确定：最终比较统一使用 `val`。

### 7.1 正式结果来源

每次训练完成后，使用该模型的 `best.pt` 单独执行：

```text
task=segment
split=val
imgsz=640
相同 data、conf、iou、max_det
```

正式对比表使用这次独立验证结果。

### 7.2 `results.csv` 的用途

`results.csv` 用于：

- 查看收敛；
- 定位 best epoch；
- 检查过拟合；
- 检查 loss 是否出现 NaN；
- 生成训练曲线。

不要把 `results.csv` 某一行与独立验证结果混合到同一组指标中。

### 7.3 每个版本必须记录

| 类型 | 指标 |
|---|---|
| 整体 Mask | P、R、mAP50、mAP50-95、F1 |
| 卷叶螟 Mask | P、R、AP50、AP50-95 |
| 钻心虫 Mask | P、R、AP50、AP50-95 |
| Box 辅助 | mAP50、mAP50-95 |
| 复杂度 | Params、FLOPs、best.pt 大小 |
| 训练 | best epoch、耗时、batch、seed、commit |

F1 需要注明：

- 根据 P/R 计算的 F1；
- 或 F1 曲线上的 `Best F1@conf`。

二者不能混写。

### 7.4 复杂度必须用同类模型比较

Baseline、V1、V2 都是实例分割模型，因此 Params 和 FLOPs 必须全部来自分割模型。

不能把：

- YOLO26 Detect 的 Params/FLOPs；
- YOLO26 Segment 的 Params/FLOPs；

混在同一张表中。

---

## 8. 文件保存规则

### 8.1 本地原始结果：`results/`

`results/` 由用户管理，保存成功实验的完整原始输出：

- weights；
- results.csv；
- args.yaml；
- 曲线图；
- confusion matrix；
- 预测样例。

规则：

1. 不移动现有 Baseline/V1 文件；
2. 不由助手自动覆盖；
3. 不上传 Git；
4. 后续成功实验由用户手动复制进去备份。

`.gitignore` 已忽略整个 `/results/`。

### 8.2 Git 中的实验记录：`experiment_records/`

`experiment_records/` 由助手维护，保存轻量、可审查的实验记录：

```text
experiment_records/
├─ README.md
├─ comparison.csv
├─ baseline/run_record.md
├─ v1-cbam/run_record.md
├─ v2-p2/run_record.md
├─ v3-dice/run_record.md
└─ v4-combined/run_record.md
```

这里不存放 `.pt` 权重和完整图片集，只记录：

- 原始结果路径；
- Git commit；
- 训练配置；
- 独立 val 指标；
- Params/FLOPs/Size；
- 结论和异常。

---

## 9. 训练脚本管理

旧脚本位于：

`E:\Study\DeepCNN\yolo26\code\train_yolov26_seg.py`

旧脚本同时包含 Baseline、CBAM 和失败 V2 的构建逻辑，不适合直接作为新 V2 基础。

后续规则：

1. 公共训练脚本放入仓库 `scripts/`；
2. 脚本通过 `experiments/` 中的配置切换实验；
3. 不再通过反复修改 `MODEL_VERSION` 运行不同版本；
4. 训练时自动记录 Git commit 和配置；
5. 旧脚本保留在原路径作为历史参考，不直接复制旧 V2 逻辑。

目前只建立目录规范，新的公共训练脚本将在正式开始新 V2 前单独编写和检查。

---

## 10. 正式训练前门禁

开始数小时训练前必须通过：

- [ ] 当前分支正确；
- [ ] 工作区无意外改动；
- [ ] 与 `main` 的 diff 只包含本次模块；
- [ ] editable install 指向当前源码；
- [ ] 模型任务是 `segment`；
- [ ] 模型能正常构建；
- [ ] P2/P3/P4/P5 尺寸和 stride 正确；
- [ ] 权重迁移报告正确；
- [ ] 单次 forward 输出 Box、Mask coefficient 和 Proto；
- [ ] 单 batch loss 全部有限；
- [ ] 1 epoch train+val 正常；
- [ ] 预测 Mask 不全空；
- [ ] 5～10 epoch 短跑中 Box 和 Mask 都有学习趋势。

未通过以上检查时，不启动 400 epoch 正式训练。

---

## 11. 当前状态记录

2026-07-28 项目整理：

- [x] 重新确认项目目标不是复现 YOLO-Pineapple；
- [x] `results/` 改为纯本地原始结果目录；
- [x] 建立 `experiment_records/` 规范；
- [x] 旧 V2 归档为失败分支；
- [x] 公共文档回到 `main`；
- [x] 新 V2 规定为 Baseline 独立分支；
- [ ] 新 V2 源码实现；
- [ ] 新公共训练脚本；
- [ ] V2 短跑验证；
- [ ] V2 正式训练。
