# V1b-SR-CBAM-b4 实验记录

## 当前状态

```text
阶段：1/10 epoch 预检均通过，允许用户手动运行400 epoch正式实验
分支：v1b-srcbam-b4
起点：codex/baseline-b4
正式对照：Baseline-b4
```

本实验不会覆盖 `v1-cbam-b4`。旧 V1 的正式负结果用于说明“四个 Backbone 阶段直接使用
CBAM”在当前数据上提高 Precision 但降低 Recall；V1b 是针对该失败机制建立的新独立实验。

## 研究假设

旧 V1 存在三项风险：

1. Ultralytics 简化 ChannelAttention 只使用 Global Average Pool，小而稀疏的局部强响应
   可能被背景平均值稀释；
2. `y=CBAM(x)` 没有恒等旁路，四次通道/空间乘法可能压低弱目标特征；
3. P2/P3/P4/P5 四处全部加入注意力，可能造成过度抑制和小数据集过拟合。

V1b 使用 Selective Residual CBAM（SR-CBAM）：

```text
channel cue = shared_MLP(avg_pool(x)) + shared_MLP(max_pool(x))
attended    = spatial_attention(channel_attention(x))
y           = x + mix * (attended - x)
mix_init    = 0.1
placement   = Backbone P3 and P4 only
```

预期是保留 CBAM 抑制背景的 Precision 优势，同时避免卷叶螟 Recall 再次明显下降。

## 与参考工作的关系

- 原始 CBAM：Woo et al., ECCV 2018，使用 Average/Max Pool 和共享 MLP；
  `https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper`
- 残差小权重思想参考 ReZero：Bachlechner et al., UAI 2021；
  `https://proceedings.mlr.press/v161/bachlechner21a.html`
- `YOLO-pineapple.html` 中 CBAM 位于带直通通道和多尺度卷积的 GMSC 内部；本实验只借鉴
  “保留/融合特征后再注意力精炼”的思想，不复现 GMSC。

## 源码改动

| 文件 | 改动 |
|---|---|
| `ultralytics/nn/modules/conv.py` | 新增 `ResidualCBAM` |
| `ultralytics/nn/modules/block.py` | 新增保持 C3k2 权重键的 `C3k2SRCBAM` |
| `ultralytics/nn/modules/__init__.py` | 导出新模块 |
| `ultralytics/nn/tasks.py` | 注册 base/repeat module，并沿用 m-scale C3k2 解析 |
| `ultralytics/cfg/models/26/yolo26m-srcbam-seg.yaml` | 只在 Backbone P3/P4 使用 SR-CBAM |
| `scripts/train_yolo26_seg.py` | 锁定实验身份、预训练迁移和1/10 epoch预检入口 |

Neck、Segment26、Loss、数据和训练 profile 均不修改。

## 静态审计结果

```text
Baseline source tensors:     904
V1b target tensors:          912
Transferred tensors:         904
Verified equal tensors:      904
New attention tensors:       8
SR-CBAM modules:             2
Initial mix values:          0.1, 0.1
Random forward:              passed
All model tensors finite:    yes
SR-CBAM unit backward:       passed (deterministic mode, finite gradients)
80-class Baseline params:    27,112,072
80-class static parameters:  27,177,806
Parameter increase:          65,734 (+0.24%)
80-class Baseline fused:     23,569,148
80-class V1b fused:          23,634,882
80-class static FLOPs:       132.7 G
2-class train-state params:  27,037,484
80-to-2-class transfer:      898/912 (only class-output tensors skipped)
```

以上检查没有调用 optimizer 或训练 epoch；只额外执行了一次独立 SR-CBAM 小张量反向传播，用来验证梯度为有限值。

## 预先定义的判断门槛

| Val Mask 指标 | Baseline-b4 | V1b 判断方向 |
|---|---:|---|
| Overall mAP50 | 0.656 | 必须高于 Baseline，最好至少 +0.005 |
| Overall mAP50-95 | 0.322 | 不低于 Baseline |
| Rice leaffolder Recall | 0.523 | 不应重现旧 V1 的明显下降，最好不低于 Baseline |
| Rice leaffolder mAP50 | 0.574 | 高于 Baseline |

如果只提高 Precision 而降低 Recall 和 Overall AP，则仍判定失败。

## 1 epoch 功能预检（2026-08-04）

```text
run:       yolo26m_v1b_srcbam_b4_preflight1_seg_20260804_012308
commit:    89db38b7192566f7f35bf68e4c6c482580b5d57b
split:     val
images:    95
instances: 330
status:    passed（非正式结果）
```

终端最终独立验证结果：

| 类别 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.414 | 0.446 | 0.412 | 0.181 |
| Rice leaffolder | 0.520 | 0.362 | 0.393 | 0.161 |
| Rice stemborers | 0.309 | 0.531 | 0.430 | 0.202 |

为了比较同一训练阶段，以下使用各 run 的 `results.csv` 第1行，而不是终端重新搜索工作点后的
P/R。三组实验使用同一数据、seed、batch=4和训练参数；V1b 相对 Baseline 的差值仅用于判断
早期趋势，不能作为正式性能结论。

| epoch 1 Mask 指标 | Baseline-b4 | 旧 V1-CBAM-b4 | V1b-SR-CBAM-b4 | V1b - Baseline |
|---|---:|---:|---:|---:|
| Precision | 0.50321 | 0.28878 | 0.43603 | -0.06718 |
| Recall | 0.40137 | 0.39778 | 0.41825 | +0.01688 |
| mAP50 | 0.40218 | 0.28410 | 0.41142 | +0.00924 |
| mAP50-95 | 0.16228 | 0.11140 | 0.18340 | +0.02112 |

健康检查：

```text
train losses: box=2.05376, seg=2.52859, cls=3.60348（均有限）
val losses:   box=1.71792, seg=1.53127, cls=2.60497（均有限）
best/last:    各912个状态张量，NaN/Inf=0
P3 mix:       0.100000 -> 0.098947
P4 mix:       0.100000 -> 0.099996
fused model:  23,574,744 Params / 121.3 GFLOPs
```

判断：模型、Mask 分支和梯度均健康；V1b 没有重现旧 V1 在第一轮出现的明显总体 AP 下降，
并且 Recall/mAP50/mAP50-95 相对同阶段 Baseline 为正。由于只有1 epoch，尚不能判断最终性能，
但已经满足进入10 epoch趋势预检的门槛。预检指标不得写入正式 Baseline/V1b 性能对比表。

## 10 epoch 趋势预检（2026-08-04）

```text
run:       yolo26m_v1b_srcbam_b4_preflight10_seg_20260804_012957
commit:    4bf3a7d6520a5eeaa86ce8523d4ab3e56da4ee33
split:     val
images:    95
instances: 330
status:    passed（非正式结果）
```

终端对 `best.pt` 的最终独立验证结果：

| 类别 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.601 | 0.595 | 0.582 | 0.289 |
| Rice leaffolder | 0.535 | 0.516 | 0.538 | 0.233 |
| Rice stemborers | 0.667 | 0.673 | 0.626 | 0.346 |

`results.csv` 第10行与同为10 epoch总日程的旧 V1 预检比较：

| epoch 10 Overall Mask | 旧 V1-CBAM-b4 | V1b-SR-CBAM-b4 | V1b - 旧 V1 |
|---|---:|---:|---:|
| Precision | 0.61259 | 0.60484 | -0.00775 |
| Recall | 0.53043 | 0.59474 | **+0.06431** |
| mAP50 | 0.58869 | 0.58274 | -0.00595 |
| mAP50-95 | 0.27240 | 0.28940 | **+0.01700** |

终端分类别结果进一步显示，V1b 相对旧 V1 的卷叶螟 Mask Recall 从 `0.310` 提升到
`0.516`（`+0.206`），mAP50从 `0.531` 提升到 `0.538`；说明残差软融合和选择性放置已经
明显缓解旧 V1 的“Precision高、Recall低”问题。

曲线与权重健康检查：

```text
train box loss: 2.05376 -> 1.58075
train seg loss: 2.52859 -> 1.45458
train cls loss: 3.60348 -> 1.81071
val box loss:   1.71792 -> 1.47087
val seg loss:   1.53127 -> 1.30233
val cls loss:   2.60497 -> 1.82709
Mask Recall:    0.41825 -> 0.59474
Mask mAP50:     0.41142 -> 0.58274
Mask mAP50-95:  0.18340 -> 0.28940
best fitness:   epoch 10, 0.62969
best/last:      各912个状态张量，NaN/Inf=0
P3 mix:         0.100000 -> 0.094845
P4 mix:         0.100000 -> 0.100172
```

第1行除运行时间外与独立1-epoch预检逐字段相同，确定性复现通过。需要注意：10-epoch预检会
把总 epochs 改成10，因此其学习率日程与400-epoch Baseline正式实验的前10行不同；相对
Baseline-b4正式run的早期差值只能看方向，不能作为严格性能证据。正式性能仍只比较各自
400-epoch训练的 `best.pt` 独立 `val` 结果。

判断：1/10 epoch健康门禁均通过，源码保持冻结，允许用户手动启动400 epoch正式实验。
预检结果只证明训练链路健康并显示合理趋势，不预先保证正式指标一定超过 Baseline-b4。

## 用户手动命令

> 以下带 `--preflight-epochs` 或不带 `--dry-run` 的命令会启动训练，只能由用户手动输入。

### 只检查，不训练

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment v1b-srcbam-b4 --batch 4 --dry-run
```

### 1 epoch 功能预检

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment v1b-srcbam-b4 --batch 4 --preflight-epochs 1
```

### 10 epoch 趋势预检

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment v1b-srcbam-b4 --batch 4 --preflight-epochs 10
```

### 400 epoch 正式训练

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment v1b-srcbam-b4 --batch 4
```

正式训练只能在1/10 epoch健康门禁通过后执行。
