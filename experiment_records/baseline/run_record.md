# Baseline 实验记录

## 身份

- 版本：Baseline
- 模型：YOLO26m-seg
- Git 分支：`main`
- 状态：训练已完成
- 原始本地备份：`results/baseline/`
- 原始训练目录：`yolo26m_seg_20260628_172809/`

## 训练配置

| 项目 | 数值 |
|---|---|
| train / val / test | 768 / 95 / 95 |
| imgsz | 640 |
| epochs | 400 |
| batch | 8 |
| seed | 42 |
| best epoch | 246 |

## 历史 Val Mask 指标

| 指标 | 数值 |
|---|---:|
| P(M) | 0.717 |
| R(M) | 0.620 |
| mAP50(M) | 0.683 |
| mAP50-95(M) | 0.329 |
| 卷叶螟 P / R / AP50 | 0.679 / 0.482 / 0.604 |
| 钻心虫 P / R / AP50 | 0.753 / 0.755 / 0.763 |

## 数据来源状态

以上为历史记录。正式论文对比前，应使用本地 `best.pt` 按统一参数重新执行一次独立 `split=val`，并在本文件补充：

- 验证命令；
- Git commit；
- `val_metrics.json`；
- Params、FLOPs、Size；
- 权重 SHA256。
