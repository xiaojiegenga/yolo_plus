# YOLO 水稻害虫实例分割实验

本项目继续使用 [xiaojiegenga/yolo_plus](https://github.com/xiaojiegenga/yolo_plus)，
正式 Baseline 分支为 `cloud/data-v2-5090`；当前改进 A 开发分支为
`feature/data-v2-abl-a-attention`。

## 工作模式

```text
本地编写代码和配置
        ↓ git push
      GitHub
        ↓ git clone / pull
云服务器训练 → runs/<run-id>/ → 打包为 exports/<run-id>.zip
                                      ↓ scp
参数优化 Run：本地解包 → 写 parameter_tuning 分析 → 更新总表表 2
期刊正式 Run：本地解包 → 写正式记录 → 更新 comparison.csv
```

本地电脑是代码、实验记录和分析结果的唯一工作台。云服务器只负责拉取代码、
读取已提交的参数、执行训练和打包 Run，不在云端修改配置或填写实验表格。

当前实验进展先看 `PROGRESS.md`；完整实验表看 `云服务器实验设计与记录表.md`。

## 项目结构

```text
.
├─ experiments/             提交到 GitHub 的实验参数
├─ scripts/                 训练、打包、解包和结果回填
├─ experiment_records/      本地维护的实验记录和 comparison.csv
├─ runs/                    云端生成、本地接收的完整原始 Run
├─ data/                    数据集或挂载点
├─ exports/                 临时传输 ZIP，运行时自动创建
└─ ultralytics-main/        YOLO 源码
```

`runs/`、`data/`、`exports/` 和模型权重均不提交 Git。参数优化分析写入
`experiment_records/parameter_tuning/` 并回填总表表 2；参数冻结后、可用于期刊对比
的正式实验才写入 `experiment_records/runs/` 和 `comparison.csv`。

## 一、本地开发

1. 修改 `ultralytics-main/`、`scripts/` 或 `experiments/`。
2. 为实验确定唯一 Run ID。
3. 在本地提交并推送到 GitHub。
4. 云端只拉取这个已提交版本。

当前正式消融配置：

- `000 Baseline`：`experiments/data-v2-abl-000-y26m-b16-s42.yaml`（已完成）
- `100 SR-CBAM`：`experiments/data-v2-abl-100-srcbam-b16-s42.yaml`（待云端预检）
- 云端数据：`experiments/yolo_data_v2_cloud.yaml`
- 默认数据根目录：`/root/yolo_data`

如果服务器使用其他长期固定的数据路径，应在本地修改数据 YAML 后推送，不在云端
临时改出一个未提交版本。

## 二、云服务器从 GitHub 开始训练

云服务器镜像需要预装可用的 CUDA 版 PyTorch，数据目录需要已经放好：

```text
/root/yolo_data/
├─ images/train
├─ images/val
├─ images/test
├─ labels/train
├─ labels/val
└─ labels/test
```

首次使用：

```bash
git clone --branch feature/data-v2-abl-a-attention https://github.com/xiaojiegenga/yolo_plus.git yolo_plus
cd yolo_plus
```

已有仓库切换到当前改进 A 分支：

```bash
git fetch origin
git switch --track origin/feature/data-v2-abl-a-attention
git pull --ff-only
```

RTX 5090 和表 1 训练参数均已冻结。当前改进 A 只改变 Backbone P3/P4 的 SR-CBAM；
正式 Run ID 为 `data-v2-abl-100-srcbam-b16-s42`。云端运行命令见
`knowledge/改进A-SR-CBAM注意力机制原理与实现.md`。

入口会保留镜像自带的 PyTorch，并在缺少其他依赖时安装仓库内
`ultralytics-main`。首次使用官方 `yolo26m-seg.pt` 时可能需要联网下载权重。

## 三、训练 Run 打包并传回本地

训练完成后，使用本次唯一 Run ID 在云端执行：

```bash
RUN_ID="replace-with-current-run-id"
python scripts/transfer_run.py pack --run-id "${RUN_ID}"
```

得到对应的 `exports/<run-id>.zip`。在本地电脑下载：

```powershell
$sshHost = 'replace-with-ssh-host'
$runId = 'replace-with-current-run-id'
scp "${sshHost}:~/yolo_plus/exports/${runId}.zip" .\exports\
```

`<ssh-host>` 可使用本机 `~/.ssh/config` 中配置的主机别名。

## 四、本地解包、分析和保存

```powershell
$runId = 'replace-with-current-run-id'
python scripts/transfer_run.py unpack --archive "exports/${runId}.zip"
```

参数优化 Run 解包后：

1. 在 `experiment_records/parameter_tuning/<run-id>.md` 分析参数与结果。
2. 把结果填入 `云服务器实验设计与记录表.md` 表 2。
3. 不运行 `fill_results_table.py`，不修改 `comparison.csv`。

参数冻结后、可用于期刊对比的正式 Run，才从 `experiment_records/runs/_template.md`
创建记录并运行：

```powershell
python scripts/fill_results_table.py --run-dir "runs/${runId}" --run-id "$runId" --data data-v2
```

只提交配置、代码、轻量记录和表格；完整 `runs/` 与传输 ZIP 留在本地或外部存储。

## 当前数据口径

- Dataset ID：`rice-pest-data-v2`
- 图片：train / val / test = 938 / 117 / 118
- `Rice leaffolder`：4027 个实例
- `Rice stemborers`：1110 个实例
- 主指标：Val Mask mAP50-95
- Test：方案冻结后统一评估，不用于调参
