# YOLO 水稻害虫实例分割实验

本项目继续使用 [xiaojiegenga/yolo_plus](https://github.com/xiaojiegenga/yolo_plus)，
当前开发分支为 `cloud/data-v2-5090`。

## 工作模式

```text
本地编写代码和配置
        ↓ git push
      GitHub
        ↓ git clone / pull
云服务器训练 → runs/<run-id>/ → 打包为 exports/<run-id>.zip
                                      ↓ scp
本地解包到 runs/<run-id>/ → 分析 → 更新记录和汇总表
```

本地电脑是代码、实验记录和分析结果的唯一工作台。云服务器只负责拉取代码、
读取已提交的参数、执行训练和打包 Run，不在云端修改配置或填写实验表格。

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

`runs/`、`data/`、`exports/` 和模型权重均不提交 Git。所有实验分析文字写在
`experiment_records/` 的对应 Run 记录中；`云服务器实验设计与记录表.md` 只保存
实验计划、参数、状态和结果表格。

## 一、本地开发

1. 修改 `ultralytics-main/`、`scripts/` 或 `experiments/`。
2. 为实验确定唯一 Run ID。
3. 在本地提交并推送到 GitHub。
4. 云端只拉取这个已提交版本。

当前默认配置：

- 训练参数：`experiments/yolo26m_seg_5090.yaml`
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
git clone --branch cloud/data-v2-5090 https://github.com/xiaojiegenga/yolo_plus.git yolo_plus
cd yolo_plus
python scripts/cloud_train_data_v2.py --preflight10 --run-name data-v2-5090-preflight
```

已有仓库时只需更新：

```bash
git pull --ff-only
python scripts/cloud_train_data_v2.py --preflight10 --run-name data-v2-5090-preflight
```

短预检确认无误后，由用户决定是否启动正式训练：

```bash
python scripts/cloud_train_data_v2.py --run-name data-v2-5090-baseline
```

入口会保留镜像自带的 PyTorch，并在缺少其他依赖时安装仓库内
`ultralytics-main`。首次使用官方 `yolo26m-seg.pt` 时可能需要联网下载权重。

## 三、云端打包并传回本地

训练完成后在云端执行：

```bash
python scripts/transfer_run.py pack --run-id data-v2-5090-baseline
```

得到 `exports/data-v2-5090-baseline.zip`。在本地电脑下载：

```powershell
scp <ssh-host>:~/yolo_plus/exports/data-v2-5090-baseline.zip .\exports\
```

`<ssh-host>` 可使用本机 `~/.ssh/config` 中配置的主机别名。

## 四、本地解包、分析和保存

```powershell
python scripts/transfer_run.py unpack --archive exports/data-v2-5090-baseline.zip
python scripts/fill_results_table.py --run-dir runs/data-v2-5090-baseline --run-id data-v2-5090-baseline
```

然后：

1. 从 `experiment_records/runs/_template.md` 创建对应 Run 记录。
2. 根据 `args.yaml`、`results.csv` 和验证输出填写参数与指标。
3. 在 Run 记录中写实验分析，只把参数、状态和指标回填到总表。
4. 只提交配置、代码、记录和表格；完整 `runs/` 与传输 ZIP 留在本地或外部存储。

## 当前数据口径

- Dataset ID：`rice-pest-data-v2`
- 图片：train / val / test = 938 / 117 / 118
- `Rice leaffolder`：4027 个实例
- `Rice stemborers`：1110 个实例
- 主指标：Val Mask mAP50-95
- Test：方案冻结后统一评估，不用于调参
