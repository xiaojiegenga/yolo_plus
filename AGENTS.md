# AGENTS.md

## 开工顺序

本文件适用于项目根目录及全部子目录。开始工作前依次完整阅读：

1. `云服务器实验项目交接.md`；
2. `云服务器实验设计与记录表.md`；
3. 本文件；
4. 本次涉及目录中的 `README.md`。

用户当前指令优先。已有历史结果不得因为整理目录而被改写。

## 固定研究信息

- 仓库：`xiaojiegenga/yolo_plus`，当前开发分支 `cloud/data-v2-5090`。
- 任务：无人机航拍水稻害虫实例分割。
- 数据集：`rice-pest-data-v2`。
- 类别：`Rice leaffolder`、`Rice stemborers`。
- 实例数：卷叶螟 4027、钻心虫 1110。
- 图片数：train / val / test = 938 / 117 / 118。
- 主指标：Val Mask mAP50-95。
- Val 用于选方案；Test 在方案冻结后统一评估。

## 本地、GitHub 与云端职责

| 位置 | 唯一职责 |
|---|---|
| 本地电脑 | 编写代码和配置、接收 Run、分析、维护记录和表格 |
| GitHub | 同步已提交的代码、参数和轻量记录 |
| 云服务器 | `git clone/pull`、训练、生成 Run、打包 ZIP |

云端不得维护独立版本的配置或实验表。若训练参数需要改变，先在本地修改并推送，
再让云端拉取。云端 Run 回传后，本地 `runs/<run-id>/` 是原始结果的长期保存位置。

## 目录约定

| 路径 | 内容 | Git |
|---|---|---|
| `experiments/` | 实验参数 YAML | 跟踪 |
| `scripts/` | 训练、传输和结果回填脚本 | 跟踪 |
| `experiment_records/` | Run 记录和汇总表 | 跟踪 |
| `runs/` | 云端生成、本地接收的完整原始结果 | 不跟踪 |
| `exports/` | 云端打包或本地接收的临时 ZIP | 不跟踪 |
| `data/` | 数据集或挂载点 | 不跟踪 |
| `ultralytics-main/` | 模型源码 | 跟踪 |

旧仓库的历史实验目录保持原位，不移动、不覆盖。

## 标准流程

### 本地训练前

1. 修改代码和 `experiments/<config>.yaml`。
2. 确定不重复的 Run ID。
3. 提交并推送到 GitHub。

### 云服务器

1. 克隆仓库或执行 `git pull --ff-only`。
2. 确认 `/root/yolo_data` 已包含 `images/` 和 `labels/`。
3. 用 `scripts/cloud_train_data_v2.py` 启动预检或训练。
4. 用 `scripts/transfer_run.py pack` 把完整 Run 打成 ZIP。
5. 把 ZIP 传回本地；不在云端填写 `comparison.csv`。

### 本地训练后

1. 用 `scripts/transfer_run.py unpack` 解包到 `runs/<run-id>/`。
2. 用 `scripts/fill_results_table.py` 更新 `experiment_records/comparison.csv`。
3. 从模板创建或更新 Run 记录。
4. 所有实验分析文字写入 `experiment_records/runs/<run-id>.md`；总表只填表格数据。
5. 提交轻量记录；不提交 `runs/`、`exports/`、数据或权重。

当前流程不生成或校验哈希、manifest 或备份证明。

## 操作边界

- 未经用户明确要求，不启动 400 epoch 等长时间训练。
- 10 epoch 预检不进入论文精度排名。
- 不覆盖已有 Run、ZIP、权重或历史记录；重跑使用新 Run ID。
- 不使用 Test 调参。
- 参数、模型代码或数据口径变化时使用新 Run ID。
- 不提交凭据、SSH 配置、数据集、模型权重或完整训练输出。
- 只处理当前任务相关修改，不重置用户工作。

## 完成检查

- 云端从干净克隆可以读取默认配置并启动训练。
- 配置顶层 `model`、`data`、`experiment` 与 `train` 参数映射无遗漏。
- 打包和解包保持 `runs/<run-id>/` 目录结构。
- 本地表格数值能追溯到回传 Run 的 `results.csv` 或独立验证输出。
- `云服务器实验设计与记录表.md` 不包含实验原因、现象解释或结论段落。
- 待运行、历史参照和当前实测状态没有混用。
