# V1b-SR-CBAM-b4 实验记录

## 当前状态

```text
阶段：源码完成，静态检查通过，尚未训练
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
