# data-v2：A、B 改进完成总结

更新日期：2026-09-07。总体分析分支：`feature/data-v2-abl-a-attention`。

A1、A2 与 B 的正式训练和结果核验均已完成。A1 与本次 B 未通过预设门控，A2 相对
正式 Baseline 近似持平，尚无进入组合所需的明确收益。保留全部结果，下一步独立准备 C：P2Head。

## 正式结果

所有实验使用冻结 P2 配方、seed=42、batch=16、imgsz=640 和同一 Val 划分。
`best.pt` 按官方 fitness（Box mAP50-95 + Mask mAP50-95）选择；下表使用对应 epoch
的 CSV 精确值，论文主指标为 Val Mask mAP50-95。

| 实验 | 改进 | 实际 / Best epoch | Official fitness | Mask mAP50 | Mask mAP50-95 | ΔMask mAP50-95 对 000 | 阶段结论 |
|---|---|---:|---:|---:|---:|---:|---|
| 000 | 官方 YOLO26m-seg | 300 / 216 | 0.81977 | 0.71132 | 0.36348 | 0.00000 | 正式比较基准 |
| A1 | P3/P4 SR-CBAM | 300 / 235 | 0.80644 | 0.70884 | 0.35097 | -0.01251 | 未通过门控 |
| A2 | P3-only Zero-init Residual CBAM | 276 / 176 | 0.82146 | 0.71506 | 0.36109 | -0.00239 | 近似持平，暂不组合 |
| B | 实例 BCE + 0.5 × Soft Dice | 300 / 243 | 0.80359 | 0.68016 | 0.34625 | -0.01723 | 未通过门控 |

A2 相对 A1 恢复了总体表现，但相对 000 的主指标仍未提高；卷叶螟 Mask mAP50 从
0.677 降至 0.655。B 的总体 Mask mAP50 下降 0.03116，Recall 下降 0.07457，两个类别的
Mask AP 均下降；top-5 / top-10 fitness 也低于基准。当前证据不支持将这些实现加入组合训练。

## 分支与资料

- 总体分析：`feature/data-v2-abl-a-attention`，统一维护进展、知识文档、正式记录与汇总表，并保留 A 的源码。
- Baseline：`cloud/data-v2-5090`；正式训练提交 `1c63ee4`，结果记录提交 `c0f4f35`。
- A1 / A2 实现来源：`9d0c479` / `a38aabf`；A1 结果记录提交 `ac11686`。
- B：`feature/data-v2-abl-b-dice` 从 `c0f4f35` 独立分叉；实现来源 `1d1a71e`，知识文档来源 `cef6b0b`。B 的知识文档已原样收录到当前分支，源码与训练配置保留在 B 分支。
- [知识文档索引](../knowledge/README.md)、[项目进展](../PROGRESS.md)、[正式结果汇总 CSV](comparison.csv)、[实验总表](../云服务器实验设计与记录表.md)。

## 正式记录与证据

| 实验 | 正式分析 | 本地原始 Run |
|---|---|---|
| 000 | [Baseline](runs/data-v2-abl-000-y26m-b16-s42.md) | `runs/data-v2-abl-000-y26m-b16-s42/` |
| A1 | [SR-CBAM](runs/data-v2-abl-100-srcbam-b16-s42.md) | `runs/data-v2-abl-100-srcbam-b16-s42/` |
| A2 | [P3 ZR-CBAM](runs/data-v2-abl-a2-p3-zrcbam-b16-s42.md) | `runs/data-v2-abl-a2-p3-zrcbam-b16-s42/` |
| B | [Instance Dice](runs/data-v2-abl-010-dice-b16-s42.md) | `runs/data-v2-abl-010-dice-b16-s42/` |

原始 Run 路径相对于仓库根目录。A2、B 的 CSV 已与权重 `train_metrics`、日志和实际
参数核验。GitHub 保存轻量记录；完整 Run、权重、数据集及 ZIP 保存在本地或外部存储。

## 限制与后续

仅有 seed=42，结论用于本次实现的工程门控，不推广为所有注意力或 Dice 方案无效。
A2 在 epoch 276 正常早停；A2 的 epoch 3、B 的 epoch 4 和 6 出现部分 Val loss NaN，
训练指标与权重保持有限。分类别结果采用结束时 best.pt 验证日志的三位小数。
A2、B 日志未记录实际训练 Git commit，上述实现提交仅说明源码来源。

当前不安排 A+B、A+C、B+C 或 A+B+C。后续独立定义和预检 C：P2Head，再按其单模块
结果决定后续实验；如重新设计 A/B，使用新假设与新 Run ID。正式训练由用户手动启动，
Test 保留到最终方案与阈值冻结后统一评估。
