# V2 P2 实验记录

## 当前状态

- Git 分支：`v2-p2`
- 起点：`main` commit `d32a73f`
- 状态：模型源码实现完成，所有无训练门禁通过；尚未启动任何 epoch
- 对比对象：Baseline
- 正式指标来源：未来 400 epoch 训练所得 `best.pt` 的独立 `split=val`

> 截至本记录更新时，没有执行 1 epoch、10 epoch 或 400 epoch 训练。所有训练命令必须由用户本人手动输入。

## 唯一实验变量

在保留 Baseline Backbone、P3/P4/P5 Neck、P3-based Proto26、Mask Loss、Validator 和全部训练参数的前提下：

1. 从 Neck P3 上采样到 P2；
2. 与 Backbone P2 拼接并通过 C3k2 融合；
3. 增加 P2 的 Box、Class 和 Mask coefficient 预测分支；
4. 最终预测尺度由 P3/P4/P5 变为 P2/P3/P4/P5；
5. Proto26 仍只接收 P3/P4/P5。

本实验验证新增 P2 小目标预测能力能否提高卷叶螟的 Mask Recall 和 AP。

## 明确不包含

- CBAM；
- Dice Loss；
- P2 Mask Proto；
- `Proto26` 源码改动；
- `SegmentationValidator` 改动；
- Mask Loss 改动；
- 额外训练参数优化；
- batch、imgsz、增强策略或数据划分变化。

## 实现文件

| 文件 | 作用 |
|---|---|
| `ultralytics/cfg/models/26/yolo26m-p2-seg.yaml` | 保留 Baseline 0～22 层，新增最小 P2 Neck 分支 |
| `ultralytics/nn/modules/head.py` | 新增 `Segment26P2`，四尺度预测但保持三尺度 Proto |
| `ultralytics/nn/modules/__init__.py` | 导出 `Segment26P2` |
| `ultralytics/nn/tasks.py` | 注册新 Head，并提供受控的语义化预训练 Head 迁移入口 |
| `scripts/check_v2_p2.py` | 只做结构、权重、前向和损失检查，绝不训练 |
| `scripts/train_yolo26_seg.py` | 公共训练入口增加迁移报告及 1/10 epoch 非正式预检模式 |

## 预训练权重迁移

- Backbone 与 Baseline Neck 0～22 层按原键精确迁移；
- Baseline P3/P4/P5 的 Box、Class、Mask coefficient 分支按语义尺度迁移到 V2 P3/P4/P5；
- Baseline Proto26 按 P3/P4/P5 精确迁移；
- 新增 P2 Neck 和 P2 Head 分支保持随机初始化；
- 公共脚本在创建自定义模型前显式设置 Baseline `seed=42`，保证新 P2 初始权重可复现；
- COCO 80 类与课题 2 类之间，仅允许类别输出层张量因形状不同而跳过；其他不匹配均直接报错。

无训练检查中：

- Baseline 初始化总迁移：`904 / 1066` 个 state-dict 张量；
- 其中专用 Head 语义迁移：`376` 个张量；
- 已逐张量核对的 Baseline 参数：`904` 个；
- 新 P2 权重在加载 Baseline 后保持原初始化：通过。

## 无训练门禁结果

检查命令：`python scripts/check_v2_p2.py`

| 检查项 | 结果 |
|---|---|
| Task | `segment` |
| Head | `Segment26P2` |
| 模型规模 | `m` |
| 预测 stride | `4 / 8 / 16 / 32` |
| Head 输入 | `P2 / P3 / P4 / P5` |
| Proto 输入 | 仅 `P3 / P4 / P5` |
| Proto 输出（256 输入） | `1 × 32 × 64 × 64`，仍为输入的 1/4 |
| Mask coefficient（256 输入） | `1 × 32 × 5440` |
| NaN/Inf 前向检查 | 通过 |
| 2 类训练器重建 | 通过 |
| 单批次 loss 前向 | 通过；未创建优化器、未 backward |
| Loss 项 | Box、Seg、Cls、DFL、Semantic 均为有限正数 |

固定随机种子的单批次损失检查值仅用于排错，不是实验指标：

```text
box=3.383751
seg=8.126676
cls=7.006578
dfl=0.083516
semantic=8.866293
```

## 复杂度（静态统计，imgsz=640）

| 模型 | Params | FLOPs |
|---|---:|---:|
| Baseline YOLO26m-seg | 27,112,072 | 132.543 G |
| V2 minimal-P2 | 27,944,560 | 175.456 G |
| 增量 | +832,488（+3.07%） | +42.913 G（+32.38%） |

FLOPs 增幅来自 P2 的 160×160 高分辨率特征计算，是否值得必须由卷叶螟 Recall/AP 的正式结果判断。

## 训练前检查清单

- [x] 确认新 `v2-p2` 从 `main` 独立开始；
- [x] Backbone 与 Baseline P3/P4/P5 Neck 保持不变；
- [x] 确认 Head 尺度顺序为 P2/P3/P4/P5；
- [x] Proto26 仍为 P3/P4/P5；
- [x] 设计并逐张量验证预训练权重迁移；
- [x] 随机张量前向输出形状正确且无 NaN/Inf；
- [x] 2 类训练器重建成功；
- [x] 无优化器单批次 loss 全部有限；
- [x] Params/FLOPs 统计完成；
- [ ] 用户手动完成 1 epoch train+val；
- [ ] 确认 Box 与 Mask 均有非零、合理输出；
- [ ] 用户手动完成 10 epoch 短跑；
- [ ] 确认 Box/Mask loss 与指标有正常学习趋势；
- [ ] 用户手动启动 400 epoch 正式训练；
- [ ] 使用 `best.pt` 独立执行统一 `split=val`；
- [ ] 记录整体与分类别 Mask P/R/AP。

## 结论边界

目前只能得出“V2 代码通过静态和单批次损失门禁”，不能得出“P2 有效”或“可以直接用于论文结果”。是否进入 400 epoch 正式训练，需先由用户手动完成 1 epoch 和 10 epoch 预检并返回输出。
