# V2 P2 实验记录

## 当前状态

- Git 分支：`v2-p2`
- 状态：新分支已建立，尚未修改模型源码
- 对比对象：Baseline
- 正式指标来源：独立 `split=val`

## 唯一实验变量

在 Neck 和 Segment Head 中增加 P2 小目标预测尺度，验证其对卷叶螟小目标召回率的影响。

## 明确不包含

- CBAM；
- Dice Loss；
- P2 Mask Proto；
- Proto26 全局改动；
- SegmentationValidator 改动；
- 额外训练参数优化。

## 开始实现前

- [ ] 确认 `git diff main...v2-p2` 没有模型源码差异；
- [ ] 编写公共训练脚本；
- [ ] 确认 P2 Head 的尺度顺序；
- [ ] 设计可解释的预训练权重迁移；
- [ ] 先完成 1 epoch 和 5～10 epoch 短跑；
- [ ] 确认 Box 和 Mask 均正常学习；
- [ ] 再开始完整训练。
