# 实验记录目录

本目录保存可提交 Git、可复查、可用于论文整理的轻量实验记录。

它与本地 `results/` 的职责不同：

| 目录 | 维护者 | 内容 | 上传 Git |
|---|---|---|---|
| `results/` | 用户 | 完整原始训练输出、权重、图片 | 否 |
| `experiment_records/` | 助手持续维护 | 指标、配置摘要、commit、结论 | 是 |

## 目录规范

每个实验使用：

```text
<experiment>/
└─ run_record.md
```

正式训练后，按需要增加：

```text
<experiment>/
├─ run_record.md
├─ val_metrics.json
├─ model_info.txt
└─ best_pt_sha256.txt
```

禁止在这里提交：

- `.pt`、`.pth`；
- 完整训练图片；
- 数据集；
- 包含本机隐私路径的未清理配置。

## 指标规则

正式对比统一使用：

- `best.pt`
- `task=segment`
- `split=val`
- 相同数据集和验证参数

`results.csv` 只用于分析训练过程，不和独立 `val` 指标混为一组。
