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
- [ ] Git 工作树干净，正式 commit 已记录；
- [x] `ultralytics.__file__` 指向本项目 `ultralytics-main`；
- [x] 模型模式显示 `baseline-pretrained-pt`；
- [x] 模型路径为 `ultralytics-main/yolo26m-seg.pt`；
- [x] 控制台没有 `Segment26P2`、P2 YAML 或自定义 Head 迁移信息；
- [x] Profile batch 显示 8，实际 batch 显示 4；
- [x] Effective SHA256 与 V2-P2-b4 一致；
- [ ] 正式训练命令由用户手动输入。

2026-07-30 已执行一次 `--dry-run`，未运行 epoch 或反向传播。检查结果：

```text
Model mode       : baseline-pretrained-pt
Profile SHA256   : FA16F5C3748A9B978E62EDC50E85A5F1FA014CCBA1A3382AA1378030F4F26926
Effective SHA256 : 5A90247FF46C3D0A38BC4A16714CA1A7C75BD038042CD37DD70FB63B7DD5F917
imgsz / epochs   : 640 / 400
batch            : 4
```

## 训练后待填写

- 实际运行目录：
- 训练 commit：
- EarlyStopping / best epoch：
- `best.pt` SHA256：
- Overall Mask P/R/mAP50/mAP50-95：
- Rice leaffolder Mask P/R/mAP50/mAP50-95：
- Rice stemborers Mask P/R/mAP50/mAP50-95：
- 与 V2-P2-b4 的同 batch 差值：
- 最终结论：
