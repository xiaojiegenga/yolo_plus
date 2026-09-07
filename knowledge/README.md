# 改进原理与实现

本目录在总体分析分支 `feature/data-v2-abl-a-attention` 统一收录 A、B 的知识文档。

| 文档 | 源码位置 | 内容 |
|---|---|---|
| [改进 A：SR-CBAM](改进A-SR-CBAM注意力机制原理与实现.md) | `feature/data-v2-abl-a-attention` | A1 的设计与实现原理 |
| [改进 B：Instance Dice](改进B-Instance-Dice损失原理与实现.md) | `feature/data-v2-abl-b-dice`，实现 `1d1a71e` | 框内实例 BCE + Soft Dice 的设计与实现原理 |

B 文档从 `feature/data-v2-abl-b-dice` 的 `cef6b0b` 原样复制，其中的损失源码、测试和
训练配置路径对应 B 分支。总体分析分支保留 A 源码，B 的独立实现仍由 B 分支保存。

知识文档用于解释设计；实际训练结论以 [A、B 改进完成总结](../experiment_records/data-v2-ab-summary.md)
和其中链接的正式 Run 记录为准。A2 的结构、参数与结果见 [A2 正式记录](../experiment_records/runs/data-v2-abl-a2-p3-zrcbam-b16-s42.md)。
