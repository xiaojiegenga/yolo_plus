# 公共脚本目录

本目录将保存由 Git 统一管理的训练、验证和结果归档脚本。

计划文件：

```text
scripts/
├─ train_yolo26_seg.py
├─ val_yolo26_seg.py
└─ archive_experiment.py
```

当前旧训练脚本仍位于：

`E:\Study\DeepCNN\yolo26\code\train_yolov26_seg.py`

旧脚本包含失败 V2 的专用构建和权重重映射逻辑，因此不会直接复制为新公共脚本。

开始新版 V2 前，将编写一个配置驱动、与具体改进解耦的公共训练脚本，并先进行只读配置检查和最小运行测试。
