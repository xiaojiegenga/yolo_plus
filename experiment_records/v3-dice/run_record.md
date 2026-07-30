# V3 BCE + Dice 实验记录

## 当前状态

- Git 分支：`v3-dice`
- 源码实现 commit：`48229707ad39596feb8bf96efd6e7fcd8c1c37e5`
- 1 epoch 预检 commit：`0ee29b9d7e51bc8f016b0e3069cfd9b20f70ccf8`
- 10 epoch 预检 commit：`525b304c64bf13d5a2e29f3ddc3be2095622fb6c`
- 状态：1/10 epoch 预检均通过，可以由用户手动执行 400 epoch 正式训练
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

## 1 epoch 预检结果

用户于 2026-07-31 手动完成：

```text
runs/segment/runs_seg/yolo26m_v3_dice_b4_preflight1_seg_20260731_012746
```

### 身份与配置复核

```text
run_kind                  = preflight-1-epoch
formal_comparison_eligible= false
paired_comparison_group   = batch4
model                     = yolo26m-seg.pt
Ultralytics               = 8.4.80 editable source
Python / Torch            = 3.10.19 / 2.10.0+cu130
GPU                       = RTX 4060 Ti 8 GB
epochs / batch / imgsz    = 1 / 4 / 640
profile SHA256            = FA16F5C3748A9B978E62EDC50E85A5F1FA014CCBA1A3382AA1378030F4F26926
effective SHA256          = C4013DF9A005BDE98882BCEACF7B570DEB2261D7F9800B08E39FAD19195D33DF
```

模型融合后为 23,509,010 参数、121.2 GFLOPs，与 Baseline-b4 的模型结构一致。
这说明 V3 没有意外加载 n 模型，也没有混入 P2 或 CBAM 结构。

### 数值健康检查

| 检查项 | 结果 |
|---|---:|
| 峰值日志显存 | 3.75 GB |
| 单轮训练时间 | 45.2 s |
| `train/seg_loss` | 3.56231 |
| `val/seg_loss` | 2.19572 |
| Mask mAP50 | 0.41760 |
| Mask mAP50-95 | 0.17292 |
| Box mAP50 | 0.39981 |
| Box mAP50-95 | 0.20717 |
| `results.csv` 数值字段 | 全部有限 |
| NaN/Inf、CUDA OOM、EMA 警告 | 均未出现 |
| `best.pt` / `last.pt` | 均正常生成，单个 54,456,881 bytes |

最终独立验证还得到：

| 类别 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.524 | 0.382 | 0.417 | 0.173 |
| Rice leaffolder | 0.561 | 0.295 | 0.365 | 0.147 |
| Rice stemborers | 0.487 | 0.469 | 0.470 | 0.198 |

这些数值只证明验证链路与 Mask 输出正常，不能作为论文性能结论。

### 与 Baseline-b4 第 1 epoch 的排错对齐

| 指标 | Baseline-b4 epoch 1 | V3 epoch 1 | 差值 |
|---|---:|---:|---:|
| train Box Loss | 2.01507 | 2.00836 | -0.00671 |
| train Seg Loss | 2.50830 | 3.56231 | +1.05401（+42.02%） |
| train Cls Loss | 4.07233 | 4.08889 | +0.01656 |
| train DFL Loss | 0.00892 | 0.00886 | -0.00006 |
| train Semantic Loss | 4.35159 | 4.36299 | +0.01140 |
| Mask mAP50 | 0.40218 | 0.41760 | +0.01542 |
| Mask mAP50-95 | 0.16228 | 0.17292 | +0.01064 |

V3 的 Seg Loss 定义增加了 Dice 项，因此 Seg Loss 数值不能直接与 BCE-only 的
Baseline 判断“越低越好”。首轮 Seg Loss 增大约 42% 表明 Dice 确实参与训练；其余四项
训练 Loss 只发生很小变化，符合“唯一修改 Mask Loss”的设计。单轮 mAP 差值受早期训练
波动影响，不用于宣称 V3 已经优于 Baseline。

### 预检结论

1 epoch 的全部门禁通过：

- Dice 确实进入主实例 `seg_loss`；
- Box 与 Mask 均产生非零、数量级合理的验证指标；
- 预测图中存在位置合理的非空实例 Mask；
- 未发生数值崩溃、显存溢出或验证错误；
- run 与 manifest 正确标记为非正式预检。

因此可以进入用户手动 10 epoch 趋势预检，但仍不能进入正式指标表。

## 10 epoch 趋势预检结果

用户于 2026-07-31 手动完成：

```text
runs/segment/runs_seg/yolo26m_v3_dice_b4_preflight10_seg_20260731_013539
```

### 身份与产物

```text
run_kind                  = preflight-10-epoch
formal_comparison_eligible= false
paired_comparison_group   = batch4
git commit                = 525b304c64bf13d5a2e29f3ddc3be2095622fb6c
effective SHA256          = F5BAF9E12F49E378AC9B0D3FE004A774BFC795F3EAFBA31A3D3B5D753F02B5BF
epochs / batch / imgsz    = 10 / 4 / 640
```

- `results.csv` 共 10 行，epoch 1～10 连续；
- 第 1 行与独立 1 epoch 预检逐值一致，证明固定 seed 下可以复现；
- `best.pt` 和 `last.pt` 均正常生成；
- `best.pt` SHA256：
  `F68EB5CDAF120FD3D1734F29054DC115A4463DA874BDF45BBFA9EA862C01B0B6`；
- `best.pt` 共 904 个状态张量，NaN/Inf 张量数为 0；
- 总耗时 0.124 h，约 7.44 min。

### 训练趋势

| 指标 | Epoch 1 | Epoch 10 | 变化 |
|---|---:|---:|---:|
| train Box Loss | 2.00836 | 1.55965 | -22.34% |
| train Seg Loss | 3.56231 | 2.02351 | -43.20% |
| train Cls Loss | 4.08889 | 1.79118 | -56.19% |
| train DFL Loss | 0.00886 | 0.00617 | -30.36% |
| train Semantic Loss | 4.36299 | 1.47296 | -66.24% |
| Box mAP50 | 0.39981 | 0.62141 | +0.22160 |
| Box mAP50-95 | 0.20717 | 0.37396 | +0.16679 |
| Mask mAP50 | 0.41760 | 0.61392 | +0.19632 |
| Mask mAP50-95 | 0.17292 | 0.29936 | +0.12644 |

训练 Loss 整体下降，Box/Mask 指标在早期波动后明显恢复并在 epoch 10 达到本次最高值。
这表明 BCE + Dice 没有造成梯度冲突、Mask 崩溃或 Box 分支异常退化。

最终 `best.pt` 独立验证：

| 类别 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.573 | 0.599 | 0.614 | 0.299 |
| Rice leaffolder | 0.646 | 0.463 | 0.547 | 0.229 |
| Rice stemborers | 0.499 | 0.735 | 0.680 | 0.369 |

这些仍然是预检指标，不得写入正式对比表。

### 与 Baseline-b4 第 10 epoch 的排错对齐

| 指标 | Baseline-b4 epoch 10 | V3 epoch 10 | 差值 |
|---|---:|---:|---:|
| Mask P | 0.42594 | 0.57893 | +0.15299 |
| Mask R | 0.50980 | 0.58415 | +0.07435 |
| Mask mAP50 | 0.43671 | 0.61392 | +0.17721 |
| Mask mAP50-95 | 0.18580 | 0.29936 | +0.11356 |
| Box mAP50 | 0.42445 | 0.62141 | +0.19696 |
| Box mAP50-95 | 0.23074 | 0.37396 | +0.14322 |

该早期差值说明 V3 的学习趋势值得继续，但 10 epoch 仍处于 warmup 后的早期阶段，
不能替代完整训练，也不能据此宣称最终性能提升。

### 训练期 Val Loss 的已知 NaN

epoch 3 出现：

```text
val/seg_loss = nan
val/cls_loss = nan
```

但以下关键数据全部有限：

- 10 轮 `train/*_loss`；
- 10 轮全部 Box/Mask P、R、mAP；
- epoch 3 的 Box/Mask 指标；
- `best.pt` 的 904 个状态张量；
- 最终独立验证结果。

Baseline-b4 在相同 YOLO26 8.4.80 验证损失路径中也出现该现象：370 轮中有 122 轮
存在一个或多个 `val/*_loss=nan`，但 `train/*`、`metrics/*` 和最终 checkpoint 均有限。
模型选优 fitness 为：

```text
Box mAP50-95 + Mask mAP50-95
```

并不使用 `val/seg_loss` 或 `val/cls_loss`。因此这是同版本 Baseline 已存在的训练期
Val Loss 诊断异常，不是 V3 Dice 独有的数值崩溃，不阻止正式训练。但论文中不得使用
这些含 NaN 的 Val Loss 曲线进行优劣比较。

### 趋势预检结论

10 epoch 门禁通过，可以放行用户手动执行 400 epoch 正式训练。正式训练前不再修改
Loss、模型结构或训练参数。

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
