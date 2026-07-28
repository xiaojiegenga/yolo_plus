# 旧 V2 P2 失败实验归档

> 本记录只用于复盘，不参与正式结果比较。

## 身份

- 归档分支：`archive/v2-p2-failed`
- 旧分支提交：`2053450`，归档时另有未验证初始化提交
- 旧训练目录：`E:\Study\DeepCNN\yolo26\code\runs\segment\runs_seg\yolo26m_p2_seg_20260701_034948`
- 状态：失败、人工停止

## 已确认现象

- Box 分支能够学习；
- Box mAP50 曾达到约 0.666；
- Mask mAP 长期接近 0；
- 后期 `seg_loss` 出现 NaN/Inf；
- 该模型不能进入正式对比表。

## 为什么不继续修补

旧实现同时改动 P2 Neck、Segment Head、Mask Proto、Validator、权重映射和 batch，变量无法隔离。

历史文档提出的 “BCE Collapse + 近单位初始化” 属于未完成验证的推测，不标记为已解决。

新版 V2 将从 `main` 重新建立，并保持标准 P3 Mask Proto。
