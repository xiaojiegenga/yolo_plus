# YOLO 水稻害虫实例分割实验

仓库：[xiaojiegenga/yolo_plus](https://github.com/xiaojiegenga/yolo_plus)。
当前分支：`feature/data-v2-abl-drb-scsa`，从正式 Baseline `c0f4f35` 建立。

## 当前任务

全部 8 个 C3k2 内部使用 DRB，在原第 4、6 层之后独立加入零初始化残差 SCSA。
训练配方与正式 `000 Baseline` 相同，使用官方 `yolo26m-seg.pt` 迁移权重。
源码与 7 项 CPU 检查已完成，RTX 5090 检查和正式训练由用户手动执行。

- [实验步骤](实验步骤.md)：云端拉取、配置检查、CUDA 检查、预检、正式训练、日志打包和 SCP 下载。
- [原理与源码教学](knowledge/全C3k2-DRB与P3P4-SCSA原理与实现.md)：模块公式、插入层号、权重加载和检查结果。
- [当前进展](PROGRESS.md)：本分支接续点。
- [实验总表](云服务器实验设计与记录表.md)：本轮方案与性能登记在表 18。

A–E 的完整记录保留在 `feature/data-v2-abl-a-attention`。本分支从 Baseline 隔离开发，
其历史结果表继承 Baseline 快照；本轮不叠加 D1 或其他改进。近期讨论见
[2026-09-09 交接](knowledge/结构改进方向与对话交接-2026-09-09.md)。

## 工作模式与目录

本地编写代码、配置并推送 GitHub；云端拉取、训练、打包；本地通过 SCP 接收 Run，分析并保存。
云端不修改独立版本的配置，不填写实验表。

| 目录 | 内容 | Git |
|---|---|---|
| `experiments/` | 训练参数与数据 YAML | 跟踪 |
| `scripts/` | 训练、检查、传输和结果回填 | 跟踪 |
| `ultralytics-main/` | YOLO 模型源码 | 跟踪 |
| `knowledge/` | 教学与讨论交接 | 跟踪 |
| `experiment_records/` | 分析、正式记录、comparison.csv | 跟踪 |
| `runs/` | 本地长期保存的完整原始结果 | 不跟踪 |
| `exports/` | 传输 ZIP | 不跟踪 |
| `data/` | 数据集或挂载点 | 不跟踪 |

本轮配置为 `experiments/data-v2-abl-drb-scsa-p34-b16-s42.yaml`，训练命令必须显式传入
`--config`；不使用历史调参配置作为默认入口。

云端镜像需要可用的 CUDA PyTorch，`/root/yolo_data` 已包含 `images/` 和 `labels/`。
缺少其他运行依赖时，云端入口安装仓库内 Ultralytics；首次使用官方权重需要下载。
具体命令统一维护在 [实验步骤](实验步骤.md)。

## 结果记录

- 10 epoch 预检：不写表格或单次分析。
- 参数优化：写 `experiment_records/parameter_tuning/`，只回填总表表 2。
- 冻结配方的正式结构实验：回传后写 `experiment_records/runs/<run-id>.md`，
  更新对应正式实验表和 `comparison.csv`，明确 `data=data-v2`。

本轮尚未训练，`comparison.csv` 不预填新数据。已有 Run、权重、ZIP 和历史记录不覆盖。

## 固定研究信息

- 无人机航拍水稻害虫实例分割；数据集 `rice-pest-data-v2`。
- train / val / test = 938 / 117 / 118 张。
- Rice leaffolder / Rice stemborers = 4027 / 1110 个实例。
- RTX 5090，batch=16，epochs=300，imgsz=640，seed=42；全部配方见本轮配置。
- 主指标 Val Mask mAP50-95，同时报告 mAP50；best.pt 维持官方分割 fitness。
- Val 选方案，Test 在最终方案冻结后统一评估。
