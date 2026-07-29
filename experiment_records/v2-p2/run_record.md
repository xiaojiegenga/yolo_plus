# V2 P2 实验记录

## 当前状态

- Git 分支：`v2-p2`
- 起点：`main` commit `d32a73f`
- 状态：两次 batch=8 资源门禁均失败；P2 分类宽度已确定为自然的 128，并使用显式 batch=4，等待用户手动重新预检
- 对比对象：Baseline
- 正式指标来源：未来 400 epoch 训练所得 `best.pt` 的独立 `split=val`

> 已由用户手动执行过一次 1 epoch 预检。该结果只用于检查功能和资源，不进入论文正式对比表。10 epoch 与 400 epoch 均未启动；所有训练命令仍必须由用户本人手动输入。

## 模型变量与资源例外

模型结构实验仍只研究 P2。在保留 Baseline Backbone、P3/P4/P5 Neck、P3-based Proto26、Mask Loss 和 Validator 的前提下：

1. 从 Neck P3 上采样到 P2；
2. 与 Backbone P2 拼接并通过 C3k2 融合；
3. 增加 P2 的 Box、Class 和 Mask coefficient 预测分支；
4. 最终预测尺度由 P3/P4/P5 变为 P2/P3/P4/P5；
5. Proto26 仍只接收 P3/P4/P5。

本实验验证新增 P2 小目标预测能力能否提高卷叶螟的 Mask Recall 和 AP。

由于两次 batch=8 预检都超过 8 GB 物理显存并触发严重分页，后续运行增加唯一资源例外：

```text
batch: 8 → 4
```

其余训练参数不变。该例外会写入运行名、参数指纹和 manifest；它不能再被表述为与原 batch=8 Baseline “训练参数完全相同”。若最终用于严谨论文对比，应补跑同为 batch=4 的 Baseline。

## 明确不包含

- CBAM；
- Dice Loss；
- P2 Mask Proto；
- `Proto26` 源码改动；
- `SegmentationValidator` 改动；
- Mask Loss 改动；
- batch 以外的训练参数优化；
- imgsz、增强策略或数据划分变化。

## 实现文件

| 文件 | 作用 |
|---|---|
| `ultralytics/cfg/models/26/yolo26m-p2-seg.yaml` | 保留 Baseline 0～22 层，新增最小 P2 Neck 分支 |
| `ultralytics/nn/modules/head.py` | 新增 `Segment26P2`，四尺度预测但保持三尺度 Proto |
| `ultralytics/nn/modules/__init__.py` | 导出 `Segment26P2` |
| `ultralytics/nn/tasks.py` | 注册新 Head，并提供受控的语义化预训练 Head 迁移入口 |
| `scripts/check_v2_p2.py` | 只做结构、权重、前向和损失检查，绝不训练 |
| `scripts/train_yolo26_seg.py` | 公共训练入口增加迁移报告、1/10 epoch 预检，以及显式记录的 `--batch 4` 资源覆盖 |

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
| V2 minimal-P2（P2 分类宽度 128） | 27,789,168 | 167.422 G |
| 增量 | +677,096（+2.50%） | +34.878 G（+26.31%） |

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

## 2026-07-30：轻量 Head 的 batch=8 资源预检仍失败

运行目录：

```text
runs/segment/runs_seg/yolo26m_v2_p2_preflight1_seg_20260730_013613
```

对应提交：`7f24bd4`。用户在训练进度约 `12/96` 时主动终止，因此没有完整 epoch、验证指标或可用于比较的 `results.csv`。

| 项目 | 观察结果 |
|---|---:|
| GPU_mem | 从 8.48 G 上升并稳定在约 8.56 G |
| 稳态速度 | 约 12～13 秒/it |
| 预计单 epoch | 约 20 分钟 |
| 物理显存 | 8188 MiB |

轻量分类支路将峰值记录从约 10.2 G 降至 8.56 G，说明优化确实节省了约 1.64 G，但仍超过独立显存并发生 Windows WDDM 共享内存分页。资源门禁仍失败，不继续 batch=8。

## 2026-07-30：采用显式 batch=4

公共脚本新增受控参数：

```text
--batch 4
```

保护规则：

- 默认值仍从 Baseline profile 读取为 8；
- 目前只允许显式覆盖为 4；
- 运行目录自动带 `_b4`；
- 控制台同时打印 Baseline profile SHA256 与实际参数 SHA256；
- `experiment_manifest.json` 记录 `profile_batch=8`、`effective_batch=4` 和 `runtime_overrides`；
- 400 epoch 若使用 batch=4，会标记为 `formal-resource-adjusted`，不会错误标记为与原 Baseline 完全同参数；
- Ultralytics 的 `nbs=64` 使稳定阶段梯度累积由约 8 次变为约 16 次，有效 batch 仍接近 64，但 BatchNorm 的真实 micro-batch 不同，因此不能声称完全无性能影响。

`--batch 4 --preflight-epochs 1 --dry-run` 已通过，未启动训练。预检实际参数指纹为：

```text
C4013DF9A005BDE98882BCEACF7B570DEB2261D7F9800B08E39FAD19195D33DF
```

## 2026-07-30：P2 分类宽度由 64 调整为自然宽度 128

batch=4 已承担训练显存控制后，不再需要为了显存把 P2 分类支路过度压缩到 64。对照官方 `yolo26-p2.yaml` 和 `Detect` 的通道规则：

- YOLO26m 的 P2 输出通道为 128；
- 官方分类支路中间宽度由首个输入尺度通道决定；
- 因而 YOLO26m P2 的自然分类宽度是 128；
- 原先的 256 是专用 Head 为保留 P3/P4/P5 预训练结构而先按 P3 构建时产生的继承宽度，不适合直接套在 P2 上；
- 64 则是额外的轻量化设计，会在“增加 P2”之外引入更明显的容量压缩。

最终结构确定为：

```text
P2 Box hidden:              64
P2 Class hidden:           128
P2 Mask coefficient hidden: 64
P3/P4/P5 Class hidden:     256（保持 Baseline）
batch:                       4（显式资源覆盖）
```

64→128 后的无训练门禁结果：

| 检查项 | 结果 |
|---|---:|
| 静态模型 | 27,789,168 Params / 167.422 GFLOPs |
| 实际 2 类训练模型 | 27,628,722 Params / 165.728 GFLOPs |
| 相对 64 通道实际模型 | +43,136 Params / +2.248 GFLOPs |
| Head stride | 4 / 8 / 16 / 32 |
| 预训练张量逐项验证 | 904 个，通过 |
| 五项 loss 前向 | 全部有限 |
| batch=4 公共入口 dry-run | 通过 |

没有启动过“64 通道 + batch=4”的任何 epoch，因此本次调整发生在正式训练和短跑之前，不存在挑选训练结果的问题。

## 2026-07-30：P2-128 + batch=4 的 1 epoch 预检通过

用户手动执行了 P2 最终候选结构的 1 epoch 资源与功能预检：

```text
runs/segment/runs_seg/yolo26m_v2_p2_b4_preflight1_seg_20260730_015604
```

### 可追溯信息

| 项目 | 记录 |
|---|---|
| Git branch / commit | `v2-p2` / `d80d2f50752e59e3665302ac417bc4c36d74f90e` |
| 模型 | `yolo26m-p2-seg.yaml`，P2 Class hidden=128 |
| 运行类型 | `preflight-1-epoch`，非正式对比 |
| Profile / Effective batch | `8 / 4` |
| Effective SHA256 | `C4013DF9A005BDE98882BCEACF7B570DEB2261D7F9800B08E39FAD19195D33DF` |
| `best.pt` SHA256 | `2D8CABE8C8AFA3F34941347B2AA58C7A98F24EBD2BC88F48E57FEE27810A2557` |
| `results.csv` SHA256 | `B94A3D00D7D0BDACB588749F09F6988EA4280B1056FF677E0DE4972C166810BF` |
| `experiment_manifest.json` SHA256 | `0DE0E11FD05CF7030E36DE463217D18FD7BEB2A1AB2633700ADBF56C061B3C69` |
| 正式对比资格 | `false` |

### 资源门禁

| 项目 | 结果 |
|---|---:|
| 物理显存 | 8188 MiB |
| 训练日志最高 `GPU_mem` | 5.09 G |
| 训练 batch 数 | 192 |
| 训练循环耗时 | 51.2 s |
| `results.csv` epoch 时间 | 55.2019 s |
| 稳态速度 | 约 3.7～4.2 it/s |
| OOM / NaN / Inf | 均未出现 |

相对轻量 Head、batch=8 的中止预检，显存记录由约 8.56 G 降至 5.09 G，减少约 40.5%；相对最初 Head、batch=8 的完整预检，显存由 10.2 G 降至 5.09 G，单 epoch 由 1342 s 降至 55.2 s。由于模型宽度和 batch 同时发生过变化，这些数字只用于判断当前配置是否适配 8GB 显卡，不用于比较模型精度。

当前显存低于独立显存上限，迭代速度稳定，未再出现 12～13 s/it 的分页卡顿。因此 **batch=4 资源门禁通过**。

### 功能门禁

`results.csv` 中五项训练损失均为有限值：

| Box | Seg | Cls | DFL | Semantic |
|---:|---:|---:|---:|---:|
| 2.03203 | 2.53701 | 4.32607 | 0.00894 | 4.96386 |

训练结束后使用 `best.pt` 自动执行的独立 `split=val` 结果：

| 类别 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.546 | 0.461 | 0.431 | 0.175 |
| Rice leaffolder | 0.550 | 0.295 | 0.374 | 0.151 |
| Rice stemborers | 0.541 | 0.626 | 0.488 | 0.199 |

整体 Box 指标为 P=`0.507`、R=`0.434`、mAP50=`0.391`、mAP50-95=`0.195`。`val_batch0_pred.jpg` 中可以直接看到非空的实例 Mask，和标注图的目标位置处于同一空间坐标系，未出现旧 V2 的全空 Mask、尺寸错位或 Mask mAP≈0 问题。因此 **四尺度预测、Mask coefficient、三尺度 Proto、Loss 与 Validator 功能门禁通过**。

### 结论边界

本次只能证明：

1. P2-128 + batch=4 能在 RTX 4060 Ti 8GB 上稳定训练；
2. Box 与 Mask 链路均正常；
3. 可以进入 10 epoch 趋势预检。

本次只有 1 epoch，而且处于 5 epoch warmup 的第一轮，不能证明 P2 优于 Baseline，也不能写入论文正式指标表。后续必须通过 10 epoch 短跑观察多轮 loss 和 val 指标趋势，再决定是否启动 400 epoch。

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
- [x] 用户手动启动轻量 P2 Head 的 batch=8 资源预检；
- [x] 确认轻量 P2 Head 在 batch=8 下仍发生严重显存分页并终止；
- [x] 增加并验证显式记录的 batch=4 资源覆盖；
- [x] 将 P2 分类宽度由额外轻量的 64 调整为自然宽度 128；
- [x] P2-128 + batch=4 的无训练门禁与 dry-run 通过；
- [x] 用户手动完成 P2-128、batch=4 的 1 epoch 资源预检；
- [x] 确认 batch=4 不再发生严重显存分页；
- [ ] 用户手动完成 10 epoch 短跑；
- [ ] 确认 Box/Mask loss 与指标有正常学习趋势；
- [ ] 用户手动启动 400 epoch 正式训练；
- [ ] 使用 `best.pt` 独立执行统一 `split=val`；
- [ ] 记录整体与分类别 Mask P/R/AP。

## 结论边界

目前可以得出“V2 功能链路正常，P2-128 + batch=4 已通过 1 epoch 功能与资源门禁”，仍不能得出“P2 优于 Baseline”或“可以直接用于论文结果”。下一步由用户手动完成 10 epoch 趋势预检，确认多轮 Box/Mask loss 与 val 指标走势后，才能决定是否进入 400 epoch。
