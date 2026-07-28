# YOLO26 水稻害虫实例分割项目 — Codex 接管与执行规范

> 创建日期：2026-07-28
> 维护者：Codex
> 当前状态：项目结构已整理；旧 V2 已归档；正在建立**不含任何模型改动**的干净 `v2-p2` 起点
> 当前原则：保留历史记录，但旧 V2 的代码、结论和未验证修复不得直接继承到新版 V2

---

## 1. 本文档的用途

本文档是 Codex 接管项目后的“当前事实与后续执行规则”，主要解决以下问题：

1. 明确项目目标、参考论文和现有实验之间的关系。
2. 区分已经验证的事实、尚未验证的推测和准备废弃的旧实现。
3. 规定后续 Git 分支、训练脚本、训练结果和验证指标的统一方式。
4. 为从 `main` 独立重做 V2 提供干净的执行起点。

相关文档的定位如下：

| 文档 | 定位 |
|---|---|
| `AGENTS.md` / `CLAUDE.md` | 项目背景、用户情况和总体目标 |
| `code_plus.md` | 原有改进流程与实验记录 |
| `node.md` | 原有技术学习笔记和旧 V2 调试过程 |
| `PROJECT_TRANSFER.md` | Claude 生成的项目交接记录，仅作为历史参考 |
| `CODEX_PROJECT_CONTEXT.md` | **Codex 后续工作的当前依据** |

若文档内容发生冲突，后续采用以下优先级：

1. 用户最新明确说明；
2. Git 中的实际代码和提交历史；
3. 原始 `results.csv`、`args.yaml` 和模型验证输出；
4. 本文档；
5. 其他历史说明文档。

---

## 2. 课题与核心目标

课题是基于无人机 RGB 影像，对两类水稻虫害进行实例分割：

- Rice leaffolder（卷叶螟）
- Rice stemborers（钻心虫）

基础模型为 `yolo26m-seg`。当前最重要的问题是：

1. 卷叶螟目标小、密集、容易与叶片背景混合，漏检较多；
2. Mask 高 IoU 指标偏低，说明轮廓和边缘精度仍有提升空间；
3. 改进后仍需考虑参数量、FLOPs、模型大小和无人机部署成本。

### 2.1 已有 Baseline

- 训练目录：`yolo26m_seg_20260628_172809/`
- 数据划分：768 train / 95 val / 95 test
- 输入尺寸：640
- 最佳 epoch：246
- 主要 Val Mask 指标：

| 指标 | 数值 |
|---|---:|
| Precision(M) | 0.717 |
| Recall(M) | 0.620 |
| mAP50(M) | 0.683 |
| mAP50-95(M) | 0.329 |

后续正式表格中的 Baseline 数值必须重新按照第 8 节的统一 `val` 流程确认一次，避免把：

- `results.csv` 某一 epoch 的值；
- 训练结束时自动验证的值；
- 单独执行 `best.pt` 验证的值；
- test 集结果；

混在同一张对比表中。

### 2.2 已有 V1

V1 为 CBAM 注意力实验，已经完成训练。当前记录显示：

| 指标 | Baseline | V1 CBAM |
|---|---:|---:|
| Mask P | 0.717 | 约 0.753 |
| Mask R | 0.620 | 0.633 |
| Mask mAP50 | 0.683 | 0.700 |
| Mask mAP50-95 | 0.329 | 0.343 |

V1 可以保留为有效的独立实验，但正式写论文前仍需按统一 `val` 规范重新生成一次最终指标记录。

---

## 3. 对 YOLO-Pineapple 论文的重新理解

参考文件：`YOLO-pineapple.html`

论文研究的是无人机图像中的菠萝**目标检测**，不是实例分割。它针对小目标、尺度变化、叶片遮挡和复杂背景提出四项主要改进：

| 论文模块 | 所在位置 | 主要作用 |
|---|---|---|
| DITAH | Head | 缓解分类与定位任务的特征冲突和任务错位 |
| GMSC | Backbone/Bottleneck | 分组多尺度卷积，并在模块内部使用 CBAM |
| SCSA | SPPF 后 | 空间与通道协同注意力，增强语义交互 |
| Focaler-SIoU | Box Loss | 强调小目标、遮挡目标等困难样本的边界框回归 |

论文中的关键实验结论：

- CBAM 在 GMSC 的注意力机制比较中取得最高 mAP50 和 F1；
- DITAH、GMSC、SCSA、Focaler-SIoU 采用累积方式进行消融；
- GMSC 主要证明多尺度特征对小目标检测有效；
- Focaler-SIoU 优化的是**边界框回归**，不是 Mask Loss；
- 论文同时关注精度、参数量、模型大小、FLOPs/FPS。

### 3.1 当前项目和论文之间的真实对应关系

| 当前项目版本 | 与论文的关系 | 正确表述 |
|---|---|---|
| V1 CBAM | 受论文注意力机制比较和 GMSC 内部 CBAM 启发 | 论文启发的轻量注意力实验，不是完整 GMSC 复现 |
| V2 P2 | 论文未使用 P2 | 根据本课题小目标漏检问题进行的定制改进 |
| V3 Dice | 论文未使用 Dice | 根据实例分割 Mask 精度问题进行的定制改进 |
| V4 Combined | 项目自定义集成 | 整合本课题中验证有效的模块 |

因此，论文中应使用类似以下表述：

> 本研究参考 YOLO-Pineapple 对无人机农业小目标、复杂背景和多尺度特征问题的分析，并结合水稻虫害实例分割任务，设计 CBAM、P2 和分割损失改进。

不应写成：

> 完整复现了 YOLO-Pineapple 的四个模块。

因为当前路线并未实现完整 GMSC、DITAH、SCSA 和 Focaler-SIoU。

---

## 4. 对当前改进计划的判断与调整

现有的 `CBAM → P2 → Dice → Combined` 路线总体合理，原因是三项改进分别对应三个不同问题：

| 版本 | 主要变量 | 主要目标 |
|---|---|---|
| V1 CBAM | 注意力 | 抑制叶片背景干扰，增强虫害特征 |
| V2 P2 | 检测尺度 | 提高卷叶螟等小目标的召回率 |
| V3 Dice | Mask Loss | 提升前景区域学习和轮廓质量 |
| V4 Combined | 有效模块组合 | 验证模块能否互补 |

### 4.1 调整后的实验原则

1. V1、V2、V3 都从同一个 Baseline 代码提交建立，进行单因素独立对比。
2. V2 不再继承 V1 CBAM 的源码改动。
3. V4 只组合已经完成且结果合理的改进，不机械地把所有模块相加。
4. 独立实验应称为“单因素对比实验”；论文最终需要时，再补充累积消融：

   `Baseline → +A → +A+B → +A+B+C`

5. 所有版本使用相同数据划分、随机种子、输入尺寸、训练轮数和数据增强。
6. 若因显存降低 batch，必须在实验记录中注明；不得写“所有参数完全相同”。

### 4.2 可选的论文对齐扩展

若后续时间充足，可从以下项目中选择一项追加，而不是一次全部实现：

1. GMSC：与论文关系最直接，也能继续解决多尺度和复杂背景问题；
2. Focaler-SIoU：针对小目标框回归，但需确认 YOLO26 当前 Box Loss 结构；
3. SCSA：可能提高特征表达，但计算量和实现复杂度较高；
4. DITAH：涉及 YOLO26 end-to-end Head，改动风险最高，放在最后。

当前阶段不建议直接实现 DITAH，因为 YOLO26 的 Head 与论文使用的 YOLOv8 Head 不同，迁移成本和风险都较高。

---

## 5. Git 与源码路径的正确关系

源码物理路径始终是：

`E:\Study\DeepCNN\yolo26\yolo_plus\ultralytics-main`

但“源码路径”和“Git main 分支”不是同一个概念：

- 路径表示文件存放在哪里；
- 分支表示这个路径当前显示哪一套代码版本；
- `pip install -e` 始终指向同一个物理路径；
- 切换 Git 分支后，这个路径里的文件内容随分支变化，训练时会自动使用当前分支代码。

推荐规则：

- `main`：只保存可复现的 Baseline 和通用训练工具；
- `v1-cbam`：只包含 V1 改动；
- 新 V2 分支：从 `main` 建立，只包含 P2 改动；
- `v3-dice`：从 `main` 建立，只包含 Dice 改动；
- `v4-combined`：从 `main` 建立，再有选择地整合有效模块。

### 5.1 当前 Git 事实

2026-07-28 结构整理后的状态：

- 旧 V2 的提交、失败实现和未提交排错内容已统一保存在 `archive/v2-p2-failed`；
- 旧 V2 曾继承 V1，但归档分支不再作为任何新实验的代码基础；
- 新 `v2-p2` 从整理后的 `main` 创建，创建时与 `main` 的源码差异必须为 0；
- `results/` 改为用户管理的本地原始结果档案，并由 `.gitignore` 排除；
- `experiment_records/` 保存可提交 Git 的实验说明、指标摘要和来源信息；
- 新 V2 尚未实施 P2，也尚未开始训练。

### 5.2 舍弃旧 V2 时的安全规则

用户已经决定舍弃旧 V2 并从 `main` 重做。本次整理按以下规则执行：

1. 旧 V2 的 HEAD、源码和失败排错记录保存在 `archive/v2-p2-failed`；
2. Baseline、V1 原始结果保留在本地 `results/`，不删除、不上传 Git；
3. 新 `v2-p2` 从整理后的 `main` 创建；
4. 创建后检查 `git diff main...v2-p2`，确认不存在旧 CBAM、Proto26、Validator 或 P2 改动；
5. 在用户后续明确要求开始 V2 前，不修改模型源码，也不启动训练。

---

## 6. 旧 V2 的问题与结论边界

旧 V2 不应继续修补，原因不是 P2 思路本身错误，而是一次实验同时改变了太多内容：

1. Neck 新增 P2 路径；
2. Segment26 从 3 个尺度变成 4 个尺度；
3. Mask Proto 基础特征从 P3 改为 P2；
4. Proto 分辨率从约 160×160 改到约 320×320；
5. 修改 Proto26；
6. 修改 SegmentationValidator；
7. 增加复杂的预训练权重索引重映射；
8. batch 从 8 降到 4。

因此 Mask mAP≈0 时，无法通过一个实验判断到底是哪项变化造成的。

### 6.1 已确认的现象

- 旧 V2 能进入训练；
- Box 指标能够逐渐恢复，Box mAP50 曾达到约 0.666；
- Mask mAP 长期接近 0；
- 后期 `seg_loss` 出现 NaN/Inf；
- 这次训练结果不能作为有效 V2 对比结果。

### 6.2 尚未被证明的推测

历史文档将问题归因于“Proto26 随机初始化导致 BCE Collapse”，并提出近单位初始化修复。该修复尚未通过新的完整训练验证，因此只能视为假设，不能标记为“已经修复”。

此外，旧验证代码中的 `_proto_factor` 自动识别存在调用对象层级不一致的风险，宽泛的 `except Exception: pass` 可能吞掉错误。这说明旧 V2 的 Mask 验证值也可能受到验证逻辑影响。

结论：

> 旧 V2 的排错记录应保留用于复盘，但新版 V2 不继承其 Proto26、validator 和初始化改动。

---

## 7. 新版 V2 的推荐边界

新版 V2 只回答一个问题：

> 增加 P2 小目标检测尺度，能否提高卷叶螟的 Mask Recall 和 AP？

### 7.1 必须改变

- 在 Neck 中增加 P2 高分辨率特征融合路径；
- 为 Segment26 增加一个 P2 预测分支；
- 创建独立的 `yolo26-p2-seg.yaml`；
- 正确迁移 Baseline 中能够复用的预训练权重。

### 7.2 暂时不改变

- 不加入 CBAM；
- 不加入 Dice；
- 不提高 Mask Proto 分辨率；
- 不修改 Mask Loss；
- 不修改 SegmentationValidator；
- 不修改全局 Proto26 行为；
- 不混入其他训练超参数优化。

### 7.3 推荐的 Head 输入策略

为了让 P2 只影响小目标预测，同时保持原有 Mask Proto 走 P3 路径，优先评估以下最小改法：

`Segment26(P3, P4, P5, P2)`

设计含义：

- P3 仍是第一个输入，Proto26 继续用 P3 生成标准分辨率 Mask Proto；
- 原 P3/P4/P5 Head 分支通道结构保持不变，便于迁移预训练权重；
- P2 作为新增的第 4 个预测分支，随机初始化；
- 不再需要为 P2 Proto 修改 `block.py` 和 `segment/val.py`。

注意：该输入顺序不是常见的 P2→P5递增顺序，因此实施前必须检查 YOLO26 的 stride 构建、anchor 生成、loss 和导出逻辑是否依赖尺度顺序。只有静态检查和最小运行验证全部通过后才能采用。

若顺序依赖无法排除，则应新增一个作用范围明确的 P2 分割 Head，显式区分：

- 检测尺度输入；
- Proto 使用的特征层。

不得再次通过全局修改 Proto26/Validator 来“顺带适配”。

---

## 8. 统一指标口径

用户已经确定最终都使用 `val`。

### 8.1 正式对比指标的唯一来源

每个模型训练结束后，使用该次训练的 `best.pt` 单独运行一次：

- `task=segment`
- `split=val`
- 相同 `data`
- 相同 `imgsz=640`
- 相同 batch、device、conf、iou、max_det
- 保存完整验证输出

正式对比表采用这次独立 `val` 的结果，而不是直接抄训练中的某一行。

### 8.2 `results.csv` 的用途

`results.csv` 用于：

- 查看收敛趋势；
- 定位 best epoch；
- 判断过拟合、NaN 和训练稳定性；
- 绘制训练曲线。

它不直接替代最终独立 `val` 记录。

### 8.3 正式记录字段

每个模型至少记录：

| 范围 | 字段 |
|---|---|
| 整体 Mask | P、R、mAP50、mAP50-95、F1 |
| 卷叶螟 Mask | P、R、AP50、AP50-95 |
| 钻心虫 Mask | P、R、AP50、AP50-95 |
| Box 辅助指标 | mAP50、mAP50-95 |
| 模型复杂度 | Params(M)、FLOPs(G)、best.pt Size(MB) |
| 训练信息 | best epoch、总耗时、batch、seed、Git commit |

F1 必须注明来源：

- `F1(P,R)`：由本次 Val 的 P、R 计算；
- `Best F1@conf`：来自 F1 曲线的最佳阈值。

二者不能混写。

### 8.4 当前数据质量风险

历史表格中存在以下风险：

1. 检测模型 Params/FLOPs 与分割模型 Params/FLOPs 混用；
2. `results.csv` best epoch 与单独 `best.pt val` 的 P/R 混用；
3. test 与 val 的每类指标来源未完全区分；
4. “漏检率=1-Recall”与混淆矩阵漏检比例混写；
5. batch=4 的 P2 被描述为与 batch=8 Baseline “配置完全相同”。

这些历史数值在正式论文表格中使用前必须按本节重新核对。

---

## 9. 训练结果保存结构

采用“原始运行结果”和“Git 可追踪实验摘要”分离的方式。

### 9.1 推荐目录

```text
yolo_plus/
├─ scripts/
│  ├─ train_yolo26_seg.py
│  ├─ val_yolo26_seg.py
│  └─ archive_experiment.py          # 后续可选
├─ experiments/
│  ├─ baseline.yaml
│  ├─ v1_cbam.yaml
│  ├─ v2_p2.yaml
│  ├─ v3_dice.yaml
│  └─ v4_combined.yaml
├─ runs/                             # 原始输出，不提交 Git
│  └─ segment/
│     └─ <run_id>/
├─ results/                          # 用户本地保存的完整原始记录，不提交 Git
│  ├─ baseline/
│  ├─ v1-cbam/
│  └─ <future-successful-run>/       # 成功后由用户手动复制备份
└─ experiment_records/               # 轻量、可复查、可提交 Git
   ├─ README.md
   ├─ comparison.csv
   ├─ baseline/run_record.md
   ├─ v1-cbam/run_record.md
   ├─ v2-p2/run_record.md
   ├─ v3-dice/run_record.md
   └─ v4-combined/run_record.md
```

### 9.2 本地 `results/<experiment>/` 的结构

```text
<experiment>/
├─ train/
│  ├─ args.yaml
│  ├─ results.csv
│  └─ results.png
├─ val/
│  ├─ metrics.json
│  ├─ metrics.txt
│  ├─ MaskPR_curve.png
│  ├─ MaskF1_curve.png
│  ├─ confusion_matrix.png
│  ├─ confusion_matrix_normalized.png
│  └─ sample_predictions/
└─ model/
   ├─ best.pt
   └─ last.pt
```

已有 `results/baseline/` 和 `results/v1-cbam/` 保持原样。未来实验成功后，由用户将原始 run 按新实验名手动复制到这里；Codex 不自动移动或覆盖该目录。整个 `results/`（包括 `best.pt`、`last.pt`、CSV 和图像）仅作本地档案，不提交 Git。

### 9.3 `experiment_records/<experiment>/run_record.md` 必填信息

- 实验名称和目的；
- 分支、Git commit；
- 模型 YAML；
- 预训练权重文件和哈希；
- 数据集配置和数据版本；
- 完整训练参数；
- 原始 run 路径；
- 对应本地 `results/<experiment>/` 路径（若用户已完成备份）；
- best.pt 路径；
- best epoch；
- 独立 val 命令；
- 整体和每类 Mask 指标；
- Params、FLOPs、Size；
- 是否出现 OOM、NaN、early stop 或人工中止；
- 与 Baseline 的结论。

### 9.4 Run 命名

推荐：

`<version>_<module>_seed<seed>_<YYYYMMDD-HHMMSS>`

示例：

`v2_p2_seed42_20260730-013000`

禁止只使用 `val`、`val2`、`exp3` 这类无法追溯的名称。

---

## 10. 训练脚本纳入 Git

当前训练脚本位于：

`E:\Study\DeepCNN\yolo26\code\train_yolov26_seg.py`

它不断通过修改 `MODEL_VERSION` 切换实验，因此难以追踪某次训练究竟使用了哪一版脚本。

后续推荐：

1. 将一份规范化训练脚本放入仓库 `scripts/`；
2. 训练脚本本身保持通用，不再每次手动改 `MODEL_VERSION`；
3. 每个实验使用 `experiments/*.yaml` 保存模型、batch、seed、数据路径等配置；
4. 运行时将配置副本自动保存到 run 目录；
5. 自动记录当前 Git commit 和工作区是否干净；
6. 若工作区不干净，训练前给出明显警告；
7. 数据集、权重和原始 runs 继续通过 `.gitignore` 排除。

理想调用方式：

```powershell
python scripts/train_yolo26_seg.py --config experiments/v2_p2.yaml
```

这样训练脚本可以在 `main` 中统一维护，各实验分支只改变模型源码/YAML和自己的实验配置。

---

## 11. 完整训练前的门禁检查

任何需要数小时的训练开始前，必须依次通过：

1. **分支检查**：确认从正确 Baseline commit 建立；
2. **差异检查**：确认只包含本次实验目标相关改动；
3. **导入检查**：`ultralytics.__file__` 指向本项目源码；
4. **模型构建检查**：模型成功创建，Task 为 segment；
5. **结构检查**：打印 Backbone / Neck / Head 和 P2 输出尺寸；
6. **权重迁移检查**：列出成功迁移、随机初始化和跳过的层；
7. **单次前向检查**：Box、mask coefficient、Proto 形状正确；
8. **单 batch loss 检查**：所有 loss 为有限值；
9. **最小训练检查**：用小数据或 1 个 epoch 完成 train+val；
10. **Mask 有效性检查**：预测 Mask 不是全空，Mask mAP 不是纯噪声；
11. **短跑检查**：建议先运行 5～10 epoch，确认 Box/Mask 指标均有学习趋势；
12. 通过后再开始 400 epoch 正式训练。

若第 8～10 项失败，不得通过增加训练轮数或直接加入 Dice 来掩盖代码问题。

---

## 12. 下一步执行顺序

项目结构整理、旧 V2 归档和干净 V2 分支建立完成后，先停在“零模型改动”状态。等待用户明确开始 V2 后，按以下顺序操作：

1. 再次确认 `git diff main...v2-p2` 的模型源码差异为 0；
2. 将一份新的、通用且配置驱动的训练脚本纳入 Git；
3. 只实现 P2 检测尺度，保持 P3 Proto，不加入 CBAM、Dice 或 Validator 改动；
4. 完成模型构建、stride、通道、权重迁移和前向输出检查；
5. 完成 1 epoch train+val 门禁检查；
6. 完成 5～10 epoch 短跑，确认 Box 与 Mask 都正常学习；
7. 再开始正式训练；
8. 使用 `best.pt` 统一进行独立 `split=val`；
9. 更新 `experiment_records/`、`code_plus.md` 和 `node.md`；
10. 实验确认成功后，由用户自行将完整原始结果复制到 `results/<new-name>/`。

---

## 13. 当前结论

1. 现有改进方向适合课题，但属于“论文启发 + 任务定制”，不是 YOLO-Pineapple 原样复现。
2. V2 P2 的研究动机合理，旧 V2 失败不能说明 P2 无效。
3. 旧 V2 的根本问题是同时改动 Neck、Head、Proto、Validator 和权重迁移，无法隔离变量。
4. 新 V2 应从 `main` 独立建立，只增加 P2 小目标预测能力，并保持原 Mask Proto 逻辑。
5. 所有最终对比统一使用 `best.pt` 的独立 `val` 结果。
6. 原始 runs、本地完整档案 `results/` 与 Git 追踪摘要 `experiment_records/` 应分开保存。
7. 通用训练脚本应纳入 Git，并通过配置文件切换实验。
8. `PROJECT_TRANSFER.md` 和 `node.md` 中的旧 V2 修复结论只能作为排错线索，不能当作已验证事实。
