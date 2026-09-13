# 改进原理与实现

本目录在总体分析分支 `feature/data-v2-abl-a-attention` 统一收录 A、B、C、D、E 的知识文档；各独立实现分支保留相关说明。

本分支 `feature/data-v2-abl-f2-localgate` 实现 F2 局部特征引导条带门控。源码、配置和本地验证已完成；[F2 教材](改进F2-局部引导条带门控原理与实现.md) 与 [云端实验步骤](../实验步骤.md) 是当前训练入口。正式精度待测。

| 文档 | 源码位置 | 内容 |
|---|---|---|
| [改进 A：SR-CBAM](改进A-SR-CBAM注意力机制原理与实现.md) | `feature/data-v2-abl-a-attention` | A1 的设计与实现原理 |
| [改进 B：Instance Dice](改进B-Instance-Dice损失原理与实现.md) | `feature/data-v2-abl-b-dice`，实现 `1d1a71e` | 框内实例 BCE + Soft Dice 的设计与实现原理 |
| [改进 C：轻量 P2Head](改进C-P2Head小目标分支原理与实现.md) | `feature/data-v2-abl-c-p2head` | P2 旁路、四尺度预测与标准 P3 Mask Proto |
| [改进 D1：P2 高分辨率掩码原型](改进D1-P2高分辨率掩码原型原理与实现.md) | `feature/data-v2-abl-d-p2proto` | 掩码原型从 stride 4 提升到 stride 2；正式主指标 +0.00979，保留候选 |
| [改进 E：DySample 动态上采样](改进E-DySample动态上采样原理与实现.md) | `feature/data-v2-abl-e-dysample` | 两处 Neck 上采样；正式实验主指标下降 0.01699，未通过门控 |
| [改进 F：P3 局部与条带上下文](改进F-P3局部与条带上下文原理与实现.md) | `feature/data-v2-abl-f-p3strip` | 数据几何、逐层计算、深度卷积与残差理论、源码映射、云端训练命令 |
| [局部分类复核头：文献方案](改进F-局部分类复核头的文献依据与验证方案.md) | 方案资料 | 历史实验复盘、标注几何与误检诊断、DCR 文献依据、结构及短验证方案 |

B 文档从 `feature/data-v2-abl-b-dice` 的 `cef6b0b` 原样复制，其中的损失源码、测试和
训练配置路径对应 B 分支。总体分析分支保留 A 源码，B 的独立实现仍由 B 分支保存。

知识文档用于解释设计；实际训练结论以 [A、B 改进完成总结](../experiment_records/data-v2-ab-summary.md)
和其中链接的正式 Run 记录为准。C 已完成，见 [C 正式记录](../experiment_records/runs/data-v2-abl-001-p2head-b16-s42.md) 与 [小目标专项评估](../experiment_records/evaluations/data-v2-c-small-val.md)。A2 的结构、参数与结果见 [A2 正式记录](../experiment_records/runs/data-v2-abl-a2-p3-zrcbam-b16-s42.md)。
