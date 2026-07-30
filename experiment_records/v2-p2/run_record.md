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

## 2026-07-30：10 epoch 趋势预检通过

用户手动完成：

```text
runs/segment/runs_seg/yolo26m_v2_p2_b4_preflight10_seg_20260730_020557
```

### 可追溯信息与数据完整性

| 项目 | 记录 |
|---|---|
| Git branch / commit | `v2-p2` / `09bb133b7a841032147f6f95770fc15617df89b6` |
| 运行类型 | `preflight-10-epoch`，非正式对比 |
| Profile / Effective batch | `8 / 4` |
| Effective SHA256 | `F5BAF9E12F49E378AC9B0D3FE004A774BFC795F3EAFBA31A3D3B5D753F02B5BF` |
| `best.pt` SHA256 | `3F92CCF5671A01DADCEAB00B847952CAF9FDF6B3D337251F5A39F18B4EE1DDBB` |
| `last.pt` SHA256 | `7AB4D75A2EC524FD032B40966D2382EC39BFC81492C0491B1968C8FD0DCFF538` |
| `results.csv` SHA256 | `F62E9344077378F624171B6FC9924E3196B4CF62363709F246ED540CF0CEDD7F` |
| `experiment_manifest.json` SHA256 | `2CBE16086BC70298BDCBB2E69A5816E010063AE06C21BD51B8CB13B4C8C0966C` |
| 正式对比资格 | `false` |

数据质量检查：

- `results.csv` 完整包含 epoch 1～10，共 10 行，无缺失轮次；
- 全部数值字段中 NaN/Inf 数量为 0；
- 与独立 1 epoch 预检相比，`args.yaml` 只有 epochs、run name 和 save_dir 不同；
- 两次独立运行的 epoch 1 训练损失和 Box/Mask 指标逐项完全一致，说明固定 seed 与 P2 新层初始化可复现；
- manifest 中模型、预训练权重、数据集哈希、branch、commit 和 batch 覆盖均与实际运行一致。

### Loss 与指标趋势

| 指标 | Epoch 1 | Epoch 10 | 趋势 |
|---|---:|---:|---|
| train box loss | 2.03203 | 1.58874 | 下降 |
| train seg loss | 2.53701 | 1.43963 | 下降 |
| train cls loss | 4.32607 | 1.88303 | 下降 |
| train dfl loss | 0.00894 | 0.00631 | 下降 |
| train semantic loss | 4.96386 | 1.52205 | 下降 |
| val box loss | 1.63519 | 1.45319 | 总体下降 |
| val seg loss | 1.50690 | 1.32166 | 总体下降；epoch 9 最低 1.30310 |
| val cls loss | 3.39152 | 1.82821 | 下降 |
| Box mAP50 | 0.39143 | 0.56770 | 上升 |
| Box mAP50-95 | 0.19577 | 0.33371 | 上升 |
| Mask Recall | 0.45998 | 0.54419 | 上升 |
| Mask mAP50 | 0.43096 | 0.53652 | 上升 |
| Mask mAP50-95 | 0.17449 | 0.25827 | 上升；epoch 9 峰值 0.27593 |

epoch 2～3 的 mAP 暂时回落发生在 `warmup_epochs=5` 的预热阶段。epoch 6～10 的 Box/Mask 指标恢复并形成总体上升趋势，同时训练和验证损失没有发散，因此属于正常早期波动。

10 个 epoch 累计时间为 496.748 s，单 epoch 平均 49.675 s，各轮约 48.567～54.043 s，没有重新出现显存分页导致的数量级卡顿。按当前速度线性估计，400 epoch 约需 5.5 小时，实际时间会受验证、绘图、数据增强和 EarlyStopping 影响。

### `best.pt` 独立 Val

| 类别 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.548 | 0.532 | 0.535 | 0.257 |
| Rice leaffolder | 0.510 | 0.512 | 0.496 | 0.210 |
| Rice stemborers | 0.585 | 0.551 | 0.573 | 0.304 |

整体 Box P/R/mAP50/mAP50-95 为 `0.547/0.568/0.568/0.333`。预测图中的小目标 Box 与实例 Mask 均非空、位置合理，没有发现 Mask 尺寸错位、全背景输出或类别链路异常。

### 放行结论

10 epoch 趋势预检通过，可以由用户手动启动 400 epoch 正式训练。该结论表示“当前实现值得进行完整训练”，不表示“P2 已经优于 Baseline”。

由于正式 V2 必须使用 batch=4，它会被记录为 `formal-resource-adjusted`。最终论文若要求严格的单变量比较，需要补跑同一训练脚本、同一数据和同一 batch=4 的 Baseline，再比较 `Baseline-b4` 与 `V2-P2-b4`。

## 2026-07-30：正式训练完成

### 运行身份

```text
runs/segment/runs_seg/yolo26m_v2_p2_b4_seg_20260730_022727
```

| 项目 | 记录 |
|---|---|
| Git branch / commit | `v2-p2` / `3e00818505c191abbc6136eb38d9ed5dfa0da966` |
| Run kind | `formal-resource-adjusted` |
| Profile / Effective batch | `8 / 4` |
| 计划 epoch | 400 |
| 实际 epoch | 339，EarlyStopping |
| Best epoch | 239 |
| Patience | 100 |
| 总训练时间 | 4.553 h |
| Effective SHA256 | `5A90247FF46C3D0A38BC4A16714CA1A7C75BD038042CD37DD70FB63B7DD5F917` |
| `best.pt` SHA256 | `9734323191B866702544AF7516F4645A745BCCE49700834FC81C111004633AF7` |
| `last.pt` SHA256 | `16D8DD08F0873699E09496BBB62DD25900DFB4C3CB59AE76D38F39FB01814F83` |
| `results.csv` SHA256 | `E5068465E6CB3D824209856C284404ED156041AA6B4F9B045934AFCFD0B4B542` |
| Manifest SHA256 | `DDC07FB65B4680082345F4118C90FA6D5E5289ABF4A280E59F6C5D962A3F4599` |
| `best.pt` 文件大小 | 56.03 MB（十进制）/ 53.44 MiB |

模型复杂度记录：

| 统计阶段 | Params | FLOPs@640 |
|---|---:|---:|
| 未融合、实际 2 类训练结构 | 27,628,722 | 165.728 G |
| 最终加载 `best.pt` 的 fused inference summary | 23,904,312 | 141.4 G |

论文比较时必须对所有模型使用同一种统计方式，不能把 Baseline 的未融合统计与 V2 的 fused 统计混用。

### 数据质量与训练健康检查

- `results.csv` 含 epoch 1～339，共 339 行，无缺失 epoch；
- 全部数值字段中 NaN/Inf 数量为 0；
- manifest 的源码 commit、权重哈希、数据集哈希、batch 覆盖和训练 profile 均匹配；
- epoch 239 后连续 100 epoch 没有出现更高综合 fitness，因此在 epoch 339 正常 EarlyStopping；
- 平均 48.35 s/epoch，中位数 47.91 s/epoch，没有显存分页导致的异常慢速；
- train 五项 loss 持续下降；val loss 在约 180～270 epoch 后进入平台并轻微波动，属于轻度过拟合/收敛平台，不是发散；
- 预测图中的 Box 和 Mask 非空、空间位置正常；
- 归一化混淆矩阵未显示卷叶螟与钻心虫互相混淆，主要错误仍是目标与背景之间的漏检/误检。

EarlyStopping 使用的分割模型 fitness 为：

```text
Box mAP50-95 + Mask mAP50-95
```

epoch 239 的 fitness 为 `0.42187 + 0.33525 = 0.75712`，是全部 339 个 epoch 中最高值，因此 `best.pt` 选择正确。虽然 Mask mAP50 在 epoch 246 达到单项最高 `0.68365`，但该轮综合 fitness 为 `0.75178`，低于 epoch 239。正式使用必须选择 `best.pt`，不能使用 `last.pt`，也不需要恢复训练或增大 patience。

### `best.pt` 最终 Val

| 类别 | Images | Instances | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 95 | 330 | 0.636 | 0.616 | 0.678 | 0.421 | 0.650 | 0.624 | 0.682 | 0.335 |
| Rice leaffolder | 66 | 281 | 0.624 | 0.498 | 0.580 | 0.322 | 0.650 | 0.512 | 0.589 | 0.264 |
| Rice stemborers | 32 | 49 | 0.648 | 0.735 | 0.776 | 0.519 | 0.650 | 0.735 | 0.775 | 0.406 |

以上数值来自训练结束后重新加载 `best.pt` 执行的 `split=val`，不是 `last.pt` 或单纯读取最后一行 CSV。轻量结构化指标另存于 `val_metrics.csv`。

### 与历史 Baseline、V1 的暂定比较

> 注意：Baseline/V1 使用 batch=8，V2 使用 batch=4。下表只用于判断研究方向，不具备严格单变量因果解释；正式论文结论需等待 `Baseline-b4`。

| Mask 指标 | Baseline 历史值 | V1 CBAM 历史值 | V2 P2-b4 | V2 - Baseline |
|---|---:|---:|---:|---:|
| Overall P | 0.717 | 0.753 | 0.650 | -0.067 |
| Overall R | 0.620 | 0.633 | 0.624 | +0.004 |
| Overall mAP50 | 0.683 | 0.700 | 0.682 | -0.001 |
| Overall mAP50-95 | 0.329 | 0.343 | 0.335 | +0.006 |
| Leaffolder P | 0.679 | 0.662 | 0.650 | -0.029 |
| Leaffolder R | 0.482 | 0.552 | 0.512 | +0.030 |
| Leaffolder AP50 | 0.604 | 0.607 | 0.589 | -0.015 |
| Stemborers P | 0.753 | 0.844 | 0.650 | -0.103 |
| Stemborers R | 0.755 | 0.714 | 0.735 | -0.020 |
| Stemborers AP50 | 0.763 | 0.793 | 0.775 | +0.012 |

### 阶段结论

1. **代码和训练有效**：无 Mask 崩溃、NaN、坐标错位或类别混淆，V2 是可复现的有效实验。
2. **总体性能基本持平**：Mask mAP50 与历史 Baseline 相差 `-0.001`，mAP50-95 暂时高 `+0.006`，都属于很小差异。
3. **卷叶螟召回小幅改善但代价明显**：Recall 提高 `+0.030`，但 Precision 降低 `-0.029`、AP50 降低 `-0.015`。P2 产生了更多小目标候选，同时也增加了误检或低质量 Mask，未实现预期的大幅提升。
4. **当前 V1 仍优于 V2**：按历史值，V1 在整体 mAP、卷叶螟 Recall 和 AP50 上均更好。
5. **P2 单独使用收益有限**：它可以作为后续组合实验中的候选模块，但不能据当前结果声称“P2 显著提升模型”。
6. **下一项必要对照是 Baseline-b4**：V2 与历史 Baseline 的差异很小，batch 不一致足以影响结论。补跑 Baseline-b4 后，才能判断这些差异究竟来自 P2 还是 micro-batch。

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
- [x] 用户手动完成 10 epoch 短跑；
- [x] 确认 Box/Mask loss 与指标有正常学习趋势；
- [x] 用户手动启动 400 epoch 正式训练；
- [x] 使用 `best.pt` 执行训练结束后的独立 `split=val`；
- [x] 记录整体与分类别 Mask P/R/AP。

## 结论边界

V2-P2 正式训练已经完成，代码与训练链路有效，但与历史 batch=8 Baseline 相比只表现为“整体基本持平、卷叶螟 Recall 小幅提高、Precision/AP50 有所下降”。该结果应记录为收益有限的独立模块实验。严格结论仍需补跑 `Baseline-b4`，随后再决定 P2 是否进入 V4 组合。
