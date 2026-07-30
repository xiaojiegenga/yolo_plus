# Baseline-b4 配对实验记录

## 实验目的

V2-P2 因显存限制使用了 `batch=4`，历史 Baseline 使用 `batch=8`。本实验重新训练纯
YOLO26m-seg Baseline，并且只把 batch 从 8 改成 4，用于回答：

> V2 与历史 Baseline 的差异来自 P2 结构，还是来自 batch 变化与随机波动？

正式比较对象：

```text
Baseline-b4  vs  V2-P2-b4
```

## 代码身份

| 项目 | 记录 |
|---|---|
| Git 分支 | `codex/baseline-b4` |
| 起点 | `main` |
| 起点 commit | `d32a73f1e0f84a1b0139b69adb910cecd37361b9` |
| 训练入口准备 commit | `30beed04d53c169f805eac37c7138f834f5a55a4` |
| 模型 | 官方 `ultralytics-main/yolo26m-seg.pt` |
| 模型 YAML | 不使用自定义 YAML |
| P2/CBAM/Dice 源码 | 全部不包含 |
| Baseline 权重 SHA256 | `16B636F04E8FB6A325B3370F22DC5E5535FF473E384F4D041FD28D788F6EE9F5` |

## 训练变量控制

锁定 profile：

```text
experiments/yolo26m_seg_baseline_train.yaml
```

| 参数 | 历史 Baseline | Baseline-b4 | 是否变化 |
|---|---:|---:|---|
| epochs | 400 | 400 | 否 |
| imgsz | 640 | 640 | 否 |
| seed | 42 | 42 | 否 |
| optimizer | auto | auto | 否 |
| patience | 100 | 100 | 否 |
| mask_ratio | 4 | 4 | 否 |
| mosaic / mixup / copy_paste | 1.0 / 0.1 / 0.3 | 1.0 / 0.1 / 0.3 | 否 |
| batch | 8 | 4 | **是，唯一变化** |

期望 profile SHA256：

```text
FA16F5C3748A9B978E62EDC50E85A5F1FA014CCBA1A3382AA1378030F4F26926
```

期望 batch=4 effective SHA256：

```text
5A90247FF46C3D0A38BC4A16714CA1A7C75BD038042CD37DD70FB63B7DD5F917
```

该 effective SHA256 应与 V2-P2-b4 正式训练一致。

## 用户手动命令

### 只检查，不训练

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment baseline-b4 --batch 4 --dry-run
```

### 正式训练

> 以下命令会真正开始训练，只能由用户手动执行。

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment baseline-b4 --batch 4
```

预期运行名：

```text
yolo26m_baseline_b4_seg_YYYYMMDD_HHMMSS
```

## 开始训练前检查

- [x] 当前分支为 `codex/baseline-b4`；
- [x] Git 工作树干净，训练入口准备 commit 已记录；
- [x] `ultralytics.__file__` 指向本项目 `ultralytics-main`；
- [x] 模型模式显示 `baseline-pretrained-pt`；
- [x] 模型路径为 `ultralytics-main/yolo26m-seg.pt`；
- [x] 控制台没有 `Segment26P2`、P2 YAML 或自定义 Head 迁移信息；
- [x] Profile batch 显示 8，实际 batch 显示 4；
- [x] Effective SHA256 与 V2-P2-b4 一致；
- [x] 正式训练命令由用户手动输入。

2026-07-30 已执行一次 `--dry-run`，未运行 epoch 或反向传播。检查结果：

```text
Model mode       : baseline-pretrained-pt
Profile SHA256   : FA16F5C3748A9B978E62EDC50E85A5F1FA014CCBA1A3382AA1378030F4F26926
Effective SHA256 : 5A90247FF46C3D0A38BC4A16714CA1A7C75BD038042CD37DD70FB63B7DD5F917
Git commit       : 30beed04d53c169f805eac37c7138f834f5a55a4
imgsz / epochs   : 640 / 400
batch            : 4
```

## 2026-07-30：正式训练完成

### 运行身份

```text
runs/segment/runs_seg/yolo26m_baseline_b4_seg_20260730_160726
```

| 项目 | 记录 |
|---|---|
| Git branch / commit | `codex/baseline-b4` / `e2ee84c28c70d5a4aaa3ddb8229aa061944388c4` |
| Run kind | `formal-resource-adjusted` |
| Paired comparison group | `batch4` |
| Profile / Effective batch | `8 / 4` |
| 计划 epoch | 400 |
| 实际 epoch | 370，EarlyStopping |
| Best epoch | 270 |
| Patience | 100 |
| 总训练时间 | 3.839 h |
| 平均 / 中位 epoch 时间 | 37.35 / 37.27 s |
| Profile SHA256 | `FA16F5C3748A9B978E62EDC50E85A5F1FA014CCBA1A3382AA1378030F4F26926` |
| Effective SHA256 | `5A90247FF46C3D0A38BC4A16714CA1A7C75BD038042CD37DD70FB63B7DD5F917` |
| `best.pt` SHA256 | `94DA8F4E26F9126A48769FF6126299156EF2ED07DC90639AB937F9524E6623A8` |
| `last.pt` SHA256 | `67F2FDAD9A4A9464EE903C479ED72119BA6852C9359051B7401E8E49EBEEC498` |
| `results.csv` SHA256 | `5513BD577C4999FBA986CA8277AF7AA264592400548DE633AA56471712E5D864` |
| Manifest SHA256 | `A884D986B21BF7332CF60DF9FDDE75299201FD8DCB76C945695B138B6281866B` |
| `args.yaml` SHA256 | `F22D80D1257CF83E40B1092D04758FF5ABB96E5E9C87777BA5F7975F542F7D4F` |
| `best.pt` 文件大小 | 54.54 MB（十进制）/ 52.01 MiB |

最终 fused inference summary：

```text
23,509,010 parameters
121.2 GFLOPs
7.9 ms/image inference
```

### 数据质量与 NaN 边界

可以用于正式比较的证据：

- `results.csv` 包含 epoch 1～370，共 370 行，无缺失 epoch；
- 所有 train loss、P/R、mAP、学习率和 fitness 均为有限值；
- `best.pt` 共检查 904 个状态张量，非有限张量数量为 0；
- 最终独立 Val 的 Box/Mask 指标、可视化和混淆矩阵均正常；
- manifest 与 V2-P2-b4 的数据集、profile、effective 参数、epochs、batch 和运行时覆盖完全一致。

需要保留的质量警告：

- `results.csv` 有 460 个 NaN，但只出现在训练过程中的4个诊断性 Val loss 字段：
  `val/box_loss=108`、`val/seg_loss=122`、`val/cls_loss=122`、`val/dfl_loss=108`；
- NaN 是间歇出现并在后期增多，不存在于 train loss、mAP、fitness 或 checkpoint 参数；
- Ultralytics 当前分割损失在“某个验证 batch 没有前景匹配”时会进入 `tensor * 0` 的占位分支，
  源码注释也明确说明该路径可能产生 NaN。这是最可能原因，但若要完全确认仍需额外逐 batch 插桩；
- 验证阶段不反向传播，EarlyStopping/`best.pt` 使用的是有限的 Box 与 Mask mAP50-95 fitness，
  因此该问题不污染本次正式 mAP 对比；
- 结论边界：本次 `best.pt` 和最终 Val 指标可信，但不能使用这些 NaN 的 Val loss 字段做精确曲线比较。

### EarlyStopping 与 `best.pt`

分割模型 fitness 为：

```text
Box mAP50-95 + Mask mAP50-95
```

epoch 270：

```text
Box mAP50-95  = 0.41812
Mask mAP50-95 = 0.32176
fitness       = 0.73988
```

这是370个 epoch 中最高值。之后100个 epoch 没有超过，所以在 epoch 370 正常停止。
`last.pt` 的 Mask mAP50/mAP50-95 已降至约 `0.637/0.294`，必须使用 `best.pt`，
不需要恢复训练或增大 patience。

### `best.pt` 最终 Val

| 类别 | Images | Instances | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 95 | 330 | 0.648 | 0.611 | 0.656 | 0.418 | 0.657 | 0.609 | 0.656 | 0.322 |
| Rice leaffolder | 66 | 281 | 0.581 | 0.528 | 0.589 | 0.335 | 0.595 | 0.523 | 0.574 | 0.268 |
| Rice stemborers | 32 | 49 | 0.715 | 0.694 | 0.723 | 0.502 | 0.719 | 0.694 | 0.737 | 0.377 |

轻量结构化结果另存于 `val_metrics.csv`。

## Baseline-b4 与 V2-P2-b4 严格配对结论

两次实验的 dataset SHA、profile SHA、effective SHA、epochs、batch、seed 和全部训练参数一致，
模型结构是主要实验变量。

### Overall

| 指标 | Baseline-b4 | V2-P2-b4 | V2 - Baseline |
|---|---:|---:|---:|
| Box P | 0.648 | 0.636 | -0.012 |
| Box R | 0.611 | 0.616 | +0.005 |
| Box mAP50 | 0.656 | 0.678 | **+0.022** |
| Box mAP50-95 | 0.418 | 0.421 | +0.003 |
| Mask P | 0.657 | 0.650 | -0.007 |
| Mask R | 0.609 | 0.624 | **+0.015** |
| Mask mAP50 | 0.656 | 0.682 | **+0.026** |
| Mask mAP50-95 | 0.322 | 0.335 | **+0.013** |

### 分类别 Mask

| 类别/指标 | Baseline-b4 | V2-P2-b4 | V2 - Baseline |
|---|---:|---:|---:|
| Leaffolder P | 0.595 | 0.650 | **+0.055** |
| Leaffolder R | 0.523 | 0.512 | **-0.011** |
| Leaffolder mAP50 | 0.574 | 0.589 | +0.015 |
| Leaffolder mAP50-95 | 0.268 | 0.264 | -0.004 |
| Stemborers P | 0.719 | 0.650 | -0.069 |
| Stemborers R | 0.694 | 0.735 | **+0.041** |
| Stemborers mAP50 | 0.737 | 0.775 | **+0.038** |
| Stemborers mAP50-95 | 0.377 | 0.406 | **+0.029** |

### 计算代价

| 项目 | Baseline-b4 | V2-P2-b4 | V2 变化 |
|---|---:|---:|---:|
| Fused Params | 23.509 M | 23.904 M | +1.7% |
| FLOPs | 121.2 G | 141.4 G | +16.7% |
| Inference | 7.9 ms/image | 9.3 ms/image | +17.7% |
| 平均 epoch 时间 | 37.35 s | 48.35 s | +29.5% |

### 最终判断

1. **P2 对总体性能有小幅、真实的正向作用**：同 batch 下 Overall Mask mAP50 `+0.026`、
   mAP50-95 `+0.013`，之前“与历史 Baseline 基本持平”的判断确实受到 batch 差异干扰。
2. **P2 没有解决最初的卷叶螟漏检目标**：卷叶螟 Recall 反而 `-0.011`，严格对照不支持
   “P2 提高卷叶螟小目标召回率”的论断。
3. **卷叶螟发生的是 Precision/Recall 权衡**：Precision `+0.055`、mAP50 `+0.015`，
   但 mAP50-95 `-0.004`，说明预测更保守、更准确一些，严格边缘质量没有提高。
4. **主要收益来自钻心虫**：Recall `+0.041`、mAP50 `+0.038`、mAP50-95 `+0.029`。
5. **收益伴随明显计算代价**：FLOPs 和推理时间约增加17%，每 epoch 时间约增加30%。
6. **论文表述建议**：P2 是“总体精度小幅提升、但未命中卷叶螟召回目标”的有效独立实验；
   它可以保留为候选模块，但不能作为解决小目标漏检的核心贡献。
