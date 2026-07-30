# V3 BCE + Dice 实验记录

## 当前状态

- Git 分支：`v3-dice`
- 源码实现 commit：`48229707ad39596feb8bf96efd6e7fcd8c1c37e5`
- 状态：源码与无 epoch 检查完成，等待用户手动执行 1 epoch 预检
- 严格对比对象：`Baseline-b4`
- 最终指标来源：统一 `split=val`

## 实验目的

在不改变 Backbone、Neck、Segment Head、Proto 分辨率或推理结构的前提下，为主实例
Mask Loss 增加 Soft Dice 区域重叠监督，验证其是否提高 Mask mAP50-95 和卷叶螟
Mask 质量。

严格比较：

```text
Baseline-b4：Ultralytics 8.4.80 + BCE            + batch=4
V3-Dice-b4：Ultralytics 8.4.80 + BCE + 0.5×Dice + batch=4
```

历史 Baseline-b8 的 checkpoint 来自 Ultralytics 8.4.13，不能作为本实验的严格单变量对照。

## 唯一模型变量

主实例 Mask Loss 从：

```text
L_mask = L_BCE
```

改为：

```text
L_mask = L_BCE + 0.5 × L_Dice

Dice = (2 × intersection + 1.0) / (prediction + target + 1.0)
```

Dice 使用 Sigmoid 概率计算；预测和 GT 使用同一个目标框裁剪；求和使用 float32。
原 BCE 路径、目标分配、面积归一化和正样本归一化保持不变。

## 修改文件

| 文件 | 修改 |
|---|---|
| `ultralytics-main/ultralytics/utils/loss.py` | 为主实例 `seg_loss` 增加框内 Soft Dice |
| `ultralytics-main/tests/test_v3_instance_dice_loss.py` | 增加公式、兼容性和梯度测试 |
| `scripts/train_yolo26_seg.py` | 恢复通用 1/10 epoch 预检入口，并记录 batch=4 配对身份 |

## 明确不包含

- CBAM；
- P2 Neck 或 P2 Head；
- 自定义模型 YAML；
- Proto 或 Mask coefficient 结构修改；
- `mask_ratio` 修改；
- Validator 修改；
- 数据增强修改；
- 优化器、学习率或其他训练参数调整。

## 无 epoch 检查结果

### 1. 公式级纯张量测试

3 项全部通过：

1. 完美框内重叠的 Dice Loss 小于错误重叠；
2. 框外预测噪声不影响 Dice；
3. `dice_gain=0` 与旧面积归一化 BCE 完全一致；
4. coefficient 和 Proto 的反向梯度全部有限；
5. GT Mask 不会被原地修改。

检查中发现 `crop_mask()` 在 CPU、小实例数量时会原地修改输入。V3 Dice 已改用非原地
向量化框掩码，避免破坏 Sigmoid backward；GPU 与 CPU 使用同一计算路径。

### 2. 完整 YOLO26m 合成样本检查

```text
device=cuda:0
input=1×3×128×128
criterion=E2ELoss
one-to-many Dice gain=0.5
one-to-one  Dice gain=0.5
finite loss=True
finite gradients=True
optimizer step=False
weights saved=False
```

同一预测下对比旧 BCE 与 V3，只有 `seg_loss` 变化，Box/Cls/DFL/Semantic Loss
差值全部为 0。

### 3. 真实验证集单 batch 检查

使用 2 类 YOLO26m、1 张真实验证图、3 个实例、真实标注 Mask：

```text
legacy seg_loss = 3.254490
V3 seg_loss     = 3.893481
相对增加        = 19.63%
其他四项差值    = 0
finite loss     = True
finite gradients= True
epoch started   = False
optimizer step  = False
weights saved   = False
```

该数值只用于检查损失尺度，不是训练指标。约 19.6% 的单 batch 增幅说明
`dice_gain=0.5` 是较保守的起点，但仍需观察 1 epoch 的完整均值和稳定性。

### 4. Dry-run

2026-07-31 已通过：

```text
Run kind         : formal-resource-adjusted
Model mode       : baseline-pretrained-pt
Profile SHA256   : FA16F5C3748A9B978E62EDC50E85A5F1FA014CCBA1A3382AA1378030F4F26926
Effective SHA256 : 5A90247FF46C3D0A38BC4A16714CA1A7C75BD038042CD37DD70FB63B7DD5F917
Git branch       : v3-dice
Git commit       : 48229707ad39596feb8bf96efd6e7fcd8c1c37e5
batch            : 4
epochs           : 400
```

没有启动训练。

## 用户手动命令

### 只检查，不训练

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment v3-dice --batch 4 --dry-run
```

### 1 epoch 预检

> 会真正训练，但属于非正式排错 run，只能由用户手动执行。

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment v3-dice --batch 4 --preflight-epochs 1
```

预期运行名：

```text
yolo26m_v3_dice_b4_preflight1_seg_YYYYMMDD_HHMMSS
```

### 10 epoch 预检

> 只有 1 epoch 结果确认无异常后才执行。

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment v3-dice --batch 4 --preflight-epochs 10
```

### 400 epoch 正式训练

> 只有 10 epoch 趋势确认正常后才执行；只能由用户手动执行。

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment v3-dice --batch 4
```

## 预检门禁

1 epoch 必须同时满足：

- `seg_loss`、总 Loss 和梯度保持有限；
- 没有 NaN/Inf、CUDA OOM 或 EMA 警告；
- Box 与 Mask 验证流程正常完成；
- Mask mAP 不出现代码级接近 0 的异常；
- 运行名和 manifest 明确标记 `preflight-1-epoch`、batch=4。

10 epoch 还需要确认：

- `train/seg_loss` 总体下降；
- Mask mAP50 和 mAP50-95 开始形成合理趋势；
- Box 指标没有因损失尺度发生异常退化；
- GPU 显存和训练速度稳定。

## 正式结果

等待完成预检和正式训练后填写。预检结果不得写入正式指标对比表。
