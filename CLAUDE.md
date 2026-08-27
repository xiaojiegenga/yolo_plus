# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言

始终用中文回复。

## 这是什么项目

无人机航拍水稻害虫实例分割实验仓库（继续使用 `xiaojiegenga/yolo_plus`，当前分支
`cloud/data-v2-5090`）。数据集 `rice-pest-data-v2`，2 类（`Rice leaffolder`、
`Rice stemborers`），主选择指标为 **Val Mask mAP50-95**。

开始任何工作前，先完整阅读（按顺序）：

1. `云服务器实验项目交接.md`
2. `云服务器实验设计与记录表.md`
3. `AGENTS.md`
4. 本次涉及目录里的 `README.md`

用户当前指令优先于以上文档；已有历史结果不得因整理目录被改写。

## 工作模式：三地分工

本地电脑是**唯一**开发与记录环境，GitHub 是中转站，云服务器是**一次性**训练节点：

```text
本地改代码/配置并 push → 云端 clone/pull → 云端训练并打包 → 本地接收、分析、记录
```

- 云端不维护独立配置、不填实验表，只拉取已提交版本训练并打包 Run。
- 云端 Run 回传后，`runs/<run-id>/` 是本地长期保存的原始结果。

## 常用命令

所有命令在仓库根目录执行。没有测试套件或 lint 配置。

```bash
# 本地：读取配置，只打印最终参数、不训练（先验证配置）
python scripts/train_yolo26_seg.py --config experiments/yolo26m_seg_5090.yaml --run-name <id> --dry-run

# 云端：10 epoch 短预检（只用于兼容性/成本判断，不进论文精度排名）
python scripts/cloud_train_data_v2.py --preflight10 --run-name <run-id>

# 云端：正式训练（须由用户明确启动，否则不要跑 400 epoch）
python scripts/cloud_train_data_v2.py --run-name <run-id>

# 云端：把 runs/<run-id> 打成 exports/<run-id>.zip
python scripts/transfer_run.py pack --run-id <run-id>

# 本地：解包 ZIP 回 runs/<run-id>/
python scripts/transfer_run.py unpack --archive exports/<run-id>.zip

# 本地：从 results.csv 回填 experiment_records/comparison.csv
python scripts/fill_results_table.py --run-dir runs/<run-id> --run-id <run-id>
```

`train_yolo26_seg.py` 支持 `--experiment`、`--model`、`--data`、`--pretrained`、
`--run-name`、`--preflight10`、`--dry-run` 覆盖配置。

## 架构：两类 YAML + 一个入口

训练入口是 `scripts/train_yolo26_seg.py`（云端由 `cloud_train_data_v2.py` 包一层：
保留镜像自带 CUDA PyTorch、缺依赖时才 `pip install -e ultralytics-main`）。它读取
一个**简化训练配置 YAML**，字段映射固定为：

| 顶层字段 | 用途 |
|---|---|
| `experiment` | 默认 Run 名称前缀 |
| `model` | 传给 `YOLO(...)`（本地路径或官方模型名，如 `yolo26m-seg.pt`） |
| `data` | 解析为数据 YAML，再传给 `model.train(data=...)` |
| `train` | 其余键原样传给 `model.train(**runtime)` |

注意区分两种 YAML：

- **数据 YAML**（如 `experiments/yolo_data_v2_cloud.yaml`）：`path`/`train`/`val`/
  `test`/`nc`/`names`，`nc` 必须为 2，由 `validate_data_yaml()` 校验。
- **训练配置 YAML**（如 `experiments/yolo26m_seg_5090.yaml`）：顶层 `experiment`/
  `model`/`data`/`train`。旧的完整配置（如 `yolo26m_seg_baseline_train.yaml`）里的
  `profile_id`、`*_sha256` 等字段不会被当前入口读取或校验，仅作历史参照。

`train_yolo26_seg.py` 还会强制写入 `runtime["data"]`、`project`、`name`、`exist_ok=False`，
并把顶层 `experiment/model/data` 元数据与 `train` 参数分离，避免误传给 `model.train()`。

`ultralytics-main/` 是 vendored 的 YOLO 源码：入口把该目录插入 `sys.path` 再
`from ultralytics import YOLO`。改进实验改的是这个目录里的模型源码，不改训练参数。

## 结果回填链条

`fill_results_table.py` 是唯一连接原始结果与汇总表的脚本：

- 读 `runs/<run-id>/results.csv`；
- 按 `metrics/mAP50-95(M)` 取最高的一行作为该 Run 的最佳结果；
- 写出 `mask_p/r/map50/map50_95` 及分类型 P/R/AP 到 `experiment_records/comparison.csv`（upsert，按 `version` 列）。

`comparison.csv` 只放表格数据；每个实验的分析文字写到 `experiment_records/runs/<run-id>.md`
（模板 `experiment_records/runs/_template.md`）。`云服务器实验设计与记录表.md` 只回填
计划、参数、状态和结果表格，不写原因/解释/结论。

## 目录约定

| 路径 | 内容 | Git |
|---|---|---|
| `experiments/` | 实验参数 YAML | 跟踪 |
| `scripts/` | 训练、传输、结果回填脚本 | 跟踪 |
| `experiment_records/` | Run 记录与 `comparison.csv` | 跟踪 |
| `ultralytics-main/` | 模型源码 | 跟踪 |
| `runs/` | 完整原始结果（`args.yaml`、`results.csv`、`weights/` 等） | 不跟踪 |
| `exports/` | 传输 ZIP | 不跟踪 |
| `data/` | 数据集/挂载点 | 不跟踪 |

## 必须遵守的操作边界

- 未经用户明确要求，不启动 400 epoch 等长时间训练；10 epoch 预检不进论文精度排名。
- 不覆盖已有 Run、ZIP、权重或历史记录；参数、模型代码或数据口径变化时必须换新 Run ID。
- Val 用于选方案；Test 只在全部方案冻结后统一评估，不用 Test 调参。
- batch=8 已锁定为 data-v2 配对实验基准，即使显存更大也不在配对组内改动。
- 不提交数据、模型权重、完整 `runs/`、`exports/`、凭据或 SSH 配置。
- 只处理当前任务相关修改，不重置用户工作。
