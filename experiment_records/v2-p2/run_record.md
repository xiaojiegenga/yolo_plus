# V2 P2-b4 实验记录

## 正式状态

```text
run:       yolo26m_v2_p2_b4_seg_20260730_022727
branch:    v2-p2
commit:    3e00818505c191abbc6136eb38d9ed5dfa0da966
split:     val
images:    95
instances: 330
schedule:  400 epochs, batch=4; early stop at epoch 339; best epoch 239
status:    completed
```

唯一主要结构变量是在 Neck/Head 中增加轻量 P2 小目标预测尺度。未叠加 CBAM 或 Dice Loss。
dataset、预训练权重、训练 profile、effective 参数和 batch 与 Baseline-b4 严格配对。

## `best.pt` 独立 Val

| 类别 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.650 | 0.624 | 0.682 | 0.335 |
| Rice leaffolder | 0.650 | 0.512 | 0.589 | 0.264 |
| Rice stemborers | 0.650 | 0.735 | 0.775 | 0.406 |

## 相对 Baseline-b4

```text
Overall Mask Recall:      0.609 -> 0.624  (+0.015)
Overall Mask mAP50:       0.656 -> 0.682  (+0.026)
Overall Mask mAP50-95:    0.322 -> 0.335  (+0.013)
Leaffolder Recall:        0.523 -> 0.512  (-0.011)
Leaffolder Mask mAP50:    0.574 -> 0.589  (+0.015)
Leaffolder Mask mAP50-95: 0.268 -> 0.264  (-0.004)
```

计算代价：23.904M fused Params、141.4G FLOPs、9.3ms/image；相对 Baseline 的 FLOPs 与推理
时间约增加17%。

结论：P2 对总体 AP 有小幅正收益，但没有命中“提高卷叶螟 Recall”的原始目标，收益主要来自
钻心虫，同时带来明显计算开销。可作为完整组合候选，但不宜作为解决卷叶螟漏检的核心模块。
