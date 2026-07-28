# V1 CBAM 实验记录

## 身份

- 版本：V1 CBAM
- Git 分支：`v1-cbam`
- 状态：训练已完成
- 原始本地备份：`results/v1-cbam/`
- best epoch：250
- early stop：350
- 训练时间：约 3.51 小时

## 历史 Val Mask 指标

| 指标 | 数值 |
|---|---:|
| P(M) | 0.753 |
| R(M) | 0.633 |
| mAP50(M) | 0.700 |
| mAP50-95(M) | 0.343 |
| 卷叶螟 P / R / AP50 | 0.662 / 0.552 / 0.607 |
| 钻心虫 P / R / AP50 | 0.844 / 0.714 / 0.793 |

## 初步结论

- 整体 Mask mAP50 比 Baseline 提高约 0.017；
- 卷叶螟 Recall 从 0.482 提高到 0.552；
- 钻心虫 Precision 和 AP50 提升；
- 卷叶螟 AP50 提升较小，说明注意力不能完全解决极小目标问题。

## 数据来源状态

以上为历史记录。正式论文对比前，应使用本地 `best.pt` 按统一参数重新执行一次独立 `split=val`，并记录：

- 验证命令；
- Git commit；
- Params、FLOPs、Size；
- 权重 SHA256。
