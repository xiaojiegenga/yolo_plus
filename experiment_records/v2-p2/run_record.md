# V2 P2 实验记录

## 当前状态

- Git 分支：`v2-p2`
- 起点：`main` commit `d32a73f`
- 状态：原始 P2 Head 已完成 1 epoch 功能预检，但资源门禁失败；P2 分类支路已轻量化，等待重新预检
- 对比对象：Baseline
- 正式指标来源：未来 400 epoch 训练所得 `best.pt` 的独立 `split=val`

> 已由用户手动执行过一次 1 epoch 预检。该结果只用于检查功能和资源，不进入论文正式对比表。10 epoch 与 400 epoch 均未启动；所有训练命令仍必须由用户本人手动输入。

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
| V2 minimal-P2（P2 分类宽度 64） | 27,736,048 | 164.662 G |
| 增量 | +623,976（+2.30%） | +32.119 G（+24.23%） |

FLOPs 增幅来自 P2 的 160×160 高分辨率特征计算，是否值得必须由卷叶螟 Recall/AP 的正式结果判断。

## 2026-07-29：第一次 1 epoch 功能预检

运行目录：

```text
runs/segment/runs_seg/yolo26m_v2_p2_preflight1_seg_20260729_195807
```

对应提交：`7a7c417`。该次运行使用轻量化之前的 P2 Head，batch 仍为 Baseline 的 `8`。

| 项目 | 结果 |
|---|---:|
| Epoch 时间 | 1342 秒（约 22 分 22 秒） |
| GPU_mem | 10.2 G |
| Box mAP50 | 0.296 |
| Mask mAP50 | 0.276 |
| Mask P / R | 0.391 / 0.392 |
| train seg_loss | 2.624 |

结论分成两部分：

1. **功能门禁通过**：Box 与 Mask 都有非零指标，Seg loss 有限，说明四尺度预测、Mask coefficient、Proto、Loss 与 Validator 能正常连通；
2. **资源门禁失败**：RTX 4060 Ti 只有 8 GB 独立显存，日志却显示 `GPU_mem=10.2G`，并且一轮耗时约为 Baseline 首轮的 37 倍。P2 高分辨率激活叠加 Windows WDDM 共享内存分页，导致速度异常，不能直接进入 10/400 epoch。

这次预检不用于比较模型性能，也不复制进 `results/`。

## 2026-07-30：P2 分类支路轻量化

性能分析发现，原 P2 分类支路继承了 P3 的 256 中间通道。P2 特征图为 160×160，又同时存在 one-to-many 与 one-to-one 两套 Head；对只有 2 个类别的本课题而言，这个分类宽度计算和显存开销过大。

本次只修改新增 P2 的分类支路：

```text
原始：P2 128 → 256 → 256 → nc
现在：P2 128 →  64 →  64 → nc
```

- P2 Box 分支：不变；
- P2 Mask coefficient 分支：不变；
- P3/P4/P5 两套 Head：不变并继续精确迁移 Baseline 权重；
- Proto26、Loss、Validator、batch=8 和其余训练参数：不变；
- one-to-many 与 one-to-one 的 P2 分类支路同步轻量化。

无训练门禁重新通过：

| 检查项 | 结果 |
|---|---|
| P2 Box / Class / coefficient 隐藏宽度 | `64 / 64 / 64` |
| P3 分类隐藏宽度 | 仍为 `256` |
| P2/P3/P4/P5 stride | `4 / 8 / 16 / 32` |
| Proto 输入与输出 | 仍为 P3/P4/P5，输出 1/4 分辨率 |
| 预训练权重逐张量验证 | 904 个，通过 |
| 单批次五项 loss | 全部有限 |
| 2 类实际训练模型 | 27,585,586 Params，163.480 GFLOPs |

与轻量化前的静态模型相比，参数减少 208,512，FLOPs 减少 10.794 G。是否真正解决显存分页和异常慢速，仍必须由下一次用户手动 1 epoch 预检确认。

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
- [x] 用户手动完成原始 P2 Head 的 1 epoch train+val；
- [x] 确认原始 P2 Head 的 Box 与 Mask 均有非零、合理输出；
- [ ] 用户手动重新预检轻量 P2 Head 的 1 epoch 资源占用与速度；
- [ ] 确认轻量 P2 Head 不再发生严重显存分页；
- [ ] 用户手动完成 10 epoch 短跑；
- [ ] 确认 Box/Mask loss 与指标有正常学习趋势；
- [ ] 用户手动启动 400 epoch 正式训练；
- [ ] 使用 `best.pt` 独立执行统一 `split=val`；
- [ ] 记录整体与分类别 Mask P/R/AP。

## 结论边界

目前只能得出“V2 功能链路正常，轻量 P2 Head 通过无训练门禁”，不能得出“P2 有效”或“可以直接用于论文结果”。下一步必须先由用户手动重新完成 1 epoch 资源预检；只有速度和显存通过后，才进入 10 epoch 趋势检查。
