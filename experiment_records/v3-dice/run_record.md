# V3 Dice 实验记录

- 状态：未开始
- 计划分支：`v3-dice`
- 对比对象：Baseline
- 目标：在不改变网络结构的前提下，改善 Mask 前景学习和边缘精度
- 主要观察指标：Mask mAP50-95、每类 Mask AP50-95、Mask F1

开始前需重新确认：

1. 从最新统一 Baseline 建立分支；
2. Dice 与现有 BCE 的组合方式；
3. Loss 权重；
4. 对 Box 分支无意外影响。
