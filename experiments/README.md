# 实验配置目录

本目录用于保存可提交 Git 的实验配置。

目标调用方式：

```powershell
python scripts/train_yolo26_seg.py --config experiments/v2_p2.yaml
```

每份配置应记录：

- 实验名称；
- 模型 YAML；
- 预训练权重名称；
- 数据配置；
- imgsz；
- epochs；
- batch；
- seed；
- optimizer 与增强参数；
- 输出目录。

包含本机绝对数据路径、密钥或隐私信息的配置不得直接提交。可以提交 `.example.yaml`，本地配置使用 `.local.yaml` 并加入忽略规则。

当前尚未创建正式配置；将在编写新的公共训练脚本时一起完成。
