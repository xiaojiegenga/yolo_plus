# 实验记录目录

本目录保存可提交 Git、可复查、可用于论文整理的轻量实验记录。

它与本地 `results/` 的职责不同：

| 目录 | 维护者 | 内容 | 上传 Git |
|---|---|---|---|
| `results/` | 用户 | 完整原始训练输出、权重、图片 | 否 |
| `experiment_records/` | 助手持续维护 | 指标、配置摘要、commit、结论 | 是 |

`experiment_records/` 是项目级 Markdown 的唯一 Git 上传白名单。学习笔记、AI 上下文、排错草稿和日常过程记录不得放入本目录。

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

## 更新时机

只在关键实验节点更新并提交：

1. 源码改动已经提交；
2. 训练或短跑已经实际完成；
3. 独立 `val` 指标已经生成；
4. 指标、配置和权重来源能够追溯；
5. 结论已区分“确认事实”和“待验证推测”。

普通学习过程、AI 分析和频繁变化的计划只写入本地笔记，不上传 GitHub。

## 指标规则

正式对比统一使用：

- `best.pt`
- `task=segment`
- `split=val`
- 相同数据集和验证参数

`results.csv` 只用于分析训练过程，不和独立 `val` 指标混为一组。
