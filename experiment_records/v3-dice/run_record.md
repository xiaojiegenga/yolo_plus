# V3 Dice-b4 实验记录

## 正式状态

```text
run:       yolo26m_v3_dice_b4_seg_20260731_015225
branch:    v3-dice
commit:    482fdcf9eaa3a5e6a0c1aed48a031b4c077c8dcf
split:     val
images:    95
instances: 330
schedule:  400 epochs, batch=4
status:    completed
```

V3 在 Baseline 分割 BCE 上叠加 Dice 项，不改变模型结构、参数量和 FLOPs。dataset、预训练权重、
训练 profile、effective 参数和 batch 与 Baseline-b4 严格配对。

## `best.pt` 独立 Val

| 类别 | Mask P | Mask R | Mask F1 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|
| all | 0.744 | 0.619 | 0.676 | 0.673 | 0.325 |
| Rice leaffolder | 0.631 | 0.523 | 0.572 | 0.601 | 0.278 |
| Rice stemborers | 0.856 | 0.714 | 0.779 | 0.744 | 0.372 |

## 相对 Baseline-b4

```text
Overall Mask mAP50:       0.656 -> 0.673  (+0.017)
Overall Mask mAP50-95:    0.322 -> 0.325  (+0.003)
Leaffolder Recall:        0.523 -> 0.523  (+0.000)
Leaffolder Mask mAP50:    0.574 -> 0.601  (+0.027)
Leaffolder Mask mAP50-95: 0.268 -> 0.278  (+0.010)
```

结论：V3 的增益不大，但方向稳定，且是当前独立模块中对卷叶螟 AP50/AP50-95 最有利、没有
额外推理开销的候选。它没有提高卷叶螟 Recall，因此不能表述为解决漏检，只能表述为改善
Mask 区域重叠与严格 IoU 指标。
