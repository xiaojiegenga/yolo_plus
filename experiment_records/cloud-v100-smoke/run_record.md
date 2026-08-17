# V100 云服务器试运行与后续 4090 迁移说明

> 记录日期：2026-08-17
>
> 记录性质：云端基础设施 smoke test（非正式精度实验）
>
> GitHub 仓库：`xiaojiegenga/yolo_plus`
>
> 使用分支：`codex/data-v1-baseline-b4`
>
> 试运行前源码提交：`d63ea86`
>
> 正式训练权限：所有训练命令均由用户手动执行，助手不得代为启动训练

## 1. 本文用途

本文用于在另一台电脑或新的对话中继承 2026-08-17 的云服务器试用过程，说明：

- GitHub 中实际保存了哪些训练代码；
- 临时 V100 服务器采用了什么环境；
- data-v1 如何传入 Linux 服务器；
- 云端训练如何调用 GitHub 克隆下来的 YOLO26 源码；
- 本次遇到的错误、修复和性能观察；
- 服务器释放前应保存什么；
- 正式租用 RTX 4090 后应如何重新建立可复现环境。

本文不包含云服务器 IP、SSH 端口、云盘账号、密码、实例 ID 或访问令牌。历史截图中曾出现云盘登录信息，正式使用前应重置相关密码。

## 2. 项目与实验边界

项目目标是针对无人机水稻虫害实例分割改进 YOLO26，并非复现 YOLO-Pineapple。YOLO-Pineapple 仅用于借鉴农业无人机目标、注意力、多尺度设计和论文对比表格。

当前严格的 data-v1 对照组为：

```text
experiment_records/data-v1-baseline-b4/
```

data-v1 已消除原图级 train/val/test 泄露，主要统计为：

| 项目 | Train | Val | Test |
|---|---:|---:|---:|
| 图片 | 742 | 93 | 93 |
| 标签 | 742 | 93 | 93 |
| 空标签 | 60 | 5 | 7 |
| 卷叶螟实例 | 2846 | 353 | 370 |
| 钻心虫实例 | 477 | 56 | 53 |

锁定的数据内容 SHA-256：

```text
BBC51AAA6E53471AA72C6AABE4D3D09DA7770C0445A9597320C4A318D79EE922
```

本次 V100 运行只验证“新建云服务器 → 获取代码 → 获取数据 → 调用 GPU → 启动训练 → 生成 run”的完整流程。由于 Python、PyTorch、GPU 和 batch 均与本地正式实验不同，本次产生的指标不得写入论文正式对比表。

## 3. 本次临时服务器环境

| 项目 | 实际值 |
|---|---|
| 操作系统 | Ubuntu 22.04.3 LTS |
| GPU | Tesla V100-SXM2-32GB |
| NVIDIA Driver | 550.127.05 |
| `nvidia-smi` CUDA | 12.4 |
| Python | 3.12.7（平台 base 环境） |
| PyTorch | 2.5.0+cu124 |
| Torch CUDA runtime | 12.4 |
| Ultralytics | 8.4.80 |
| CUDA 可用性 | `torch.cuda.is_available() == True` |
| 系统内存 | 15 GiB |
| Swap | 5.9 GiB |
| 系统盘 | 约 200GB，测试前可用约 149GB |
| Shell 用户 | `root` |

注意：用户最初计划租用 RTX 4090，但本次使用的是平台赠送一小时额度的 V100，仅用于熟悉流程。V100 不能代表未来 4090 的训练速度。

## 4. GitHub 如何参与云端训练

GitHub 只保存和分发代码，不提供本次 GPU 计算。流程如下：

```text
GitHub 仓库
  │ git clone / git pull
  ▼
云服务器 /root/yolo_plus
  │ pip install -e ./ultralytics-main
  ▼
Python 从 /root/yolo_plus/ultralytics-main/ultralytics 导入源码
  │
  ├─ 读取 /root/yolo_data 中的数据
  ├─ 加载 yolo26m-seg.pt
  └─ 通过 PyTorch/CUDA 调用 V100
       ▼
  /root/yolo_plus/runs 中生成结果
```

仓库中包含真正的训练代码：

- `ultralytics-main/ultralytics/engine/model.py`
- `ultralytics-main/ultralytics/engine/trainer.py`
- `ultralytics-main/ultralytics/nn/tasks.py`
- `ultralytics-main/ultralytics/utils/loss.py`
- `ultralytics-main/ultralytics/cfg/models/26/`
- `scripts/train_yolo26_seg.py`
- `scripts/dataset_integrity.py`
- `experiments/yolo26m_seg_data_v1_baseline_b4.yaml`

仓库明确不保存：数据集、`.pt` 权重、`runs/`、`results/`、根目录学习笔记和 Ultralytics 示例图片资源。

本次在云服务器执行了：

```bash
git clone https://github.com/xiaojiegenga/yolo_plus.git
cd /root/yolo_plus
git switch -c codex/data-v1-baseline-b4 --track origin/codex/data-v1-baseline-b4
python -m pip install -e ./ultralytics-main
```

可编辑安装完成后，`yolo` 启动器位于 base Python 环境，但实际导入路径为：

```text
/root/yolo_plus/ultralytics-main/ultralytics/__init__.py
```

因此当前训练确实使用 GitHub 分支中的 YOLO26 源码。切换源码分支后，新的 Python 进程会读取新分支代码；不要在训练运行期间切换分支或执行 `git pull`。

## 5. data-v1 云端目录

数据集压缩包预先保存在平台免费云盘中，并从云服务器终端通过 SFTP 直接下载。该过程使用“云盘 ↔ 云服务器”的网络，不经过本地电脑传输数据正文。

临时路径：

```text
/root/cloud_download/datasets.zip
```

解压后路径：

```text
/root/yolo_data/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ labels/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ yolo_data_cloud.yaml
```

Linux 临时数据 YAML：

```yaml
path: /root/yolo_data
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: Rice leaffolder
  1: Rice stemborers
```

正式项目脚本当前锁定的是 Windows 版数据 YAML 文件哈希。本次为快速试用，使用通用 `yolo segment train` CLI 绕过了正式 wrapper 的路径和指纹门禁。因此本次只是非正式 smoke test。正式迁移 4090 前必须增加 Linux 专用、可提交的 profile，而不是继续手工绕过门禁。

## 6. 本次启动命令

本次用户手动执行的核心命令为：

```bash
YOLO_CONFIG_DIR=/root/yolo_runtime/Ultralytics yolo segment train \
  model=yolo26m-seg.pt \
  data=/root/yolo_data/yolo_data_cloud.yaml \
  epochs=400 patience=100 batch=0.70 imgsz=640 \
  device=0 workers=4 optimizer=auto seed=42 deterministic=True \
  cache=False amp=True val=True plots=True close_mosaic=15 \
  overlap_mask=True mask_ratio=4 mosaic=1.0 mixup=0.1 \
  copy_paste=0.3 copy_paste_mode=flip degrees=15.0 \
  translate=0.1 scale=0.5 flipud=0.5 fliplr=0.5 \
  warmup_epochs=5.0 lr0=0.01 lrf=0.01 momentum=0.937 \
  weight_decay=0.0005 \
  project=/root/yolo_plus/runs/segment/runs_seg \
  name=yolo26m_data_v1_baseline_v100_trial
```

由于同名目录已存在，实际 run 保存为：

```text
/root/yolo_plus/runs/segment/runs_seg/yolo26m_data_v1_baseline_v100_trial-2
```

这条 CLI 明确加载 `yolo26m-seg.pt`。AMP 自检内部额外加载的 `yolo26n.pt` 只是数值一致性检查，不代表正式模型被换成 nano。

## 7. AutoBatch 与速度观察

命令使用 `batch=0.70`，表示让 Ultralytics 根据约 70% GPU 显存自动选 batch。AutoBatch 依次测试 1、2、4、8、16、32，并在测试 32 时触发可恢复的 OOM，最终选择：

```text
AutoBatch: Using batch-size 12 for CUDA:0 22.46G/31.73G (71%)
```

因此实际训练 batch 为 12，不是 0.70，也不是 32。

已观察到的单 epoch 示例：

```text
训练：62/62，约 19.4 秒
验证：约 1.1 秒
合计：约 20.5 秒/epoch
```

按此粗略估算，400 epochs 约需 2.3 小时，不可能在一小时免费额度内完整跑完。本地 data-v1 Baseline-b4 平均约 36.4 秒/epoch，因此本次 V100 约快 1.8 倍。V100 是较老的 Volta 架构，主要优势是 32GB 显存，速度不应直接类比 RTX 4090。

本次速度数字只是工程观察，不是严格 benchmark。

## 8. 遇到的问题与修复

### 8.1 `pi-heif` 自动安装

首次训练时 Ultralytics 自动安装了：

```text
pi-heif 1.4.0
Pillow 12.3.0
```

安装成功后需要重新启动训练命令，让新 Python 进程加载依赖。这不是最终报错原因。

### 8.2 AMP 自检缺少 `bus.jpg`

真正中断原因：

```text
FileNotFoundError:
/root/yolo_plus/ultralytics-main/ultralytics/assets/bus.jpg
```

原因是仓库的 `.gitignore` 排除了 `ultralytics/assets/`，云端 clone 不包含该示例图片。修复命令：

```bash
mkdir -p /root/yolo_plus/ultralytics-main/ultralytics/assets
wget -O /root/yolo_plus/ultralytics-main/ultralytics/assets/bus.jpg \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg
```

修复后 AMP 检查通过，训练正常启动。未来 Linux 初始化脚本应自动准备该资源，或采用更稳健的 AMP 检查处理方式。

### 8.3 Ultralytics 配置目录不可写

base 环境曾提示 `/root/.config/Ultralytics` 不可写，因此运行时使用：

```bash
mkdir -p /root/yolo_runtime/Ultralytics
export YOLO_CONFIG_DIR=/root/yolo_runtime/Ultralytics
```

## 9. SSH、VNC 与网络流量

推荐正式使用：

- MobaXterm SSH：运行训练、Git、日志查看、多终端和 SFTP；
- WebVNC：仅在需要完整图形桌面或查看图片时使用；
- `tmux`：正式长训练必须使用，防止 SSH 断开导致进程终止。

在 SSH 远程 shell 中执行以下命令时，正文流量使用云服务器网络：

```text
wget / pip install / git clone / 服务器内执行的 sftp get
```

本地电脑只传输命令和终端文字。使用 MobaXterm 左侧 SFTP、`scp` 或 FileZilla 在本地与服务器之间拖动文件时，才使用本地上传/下载流量。WebVNC还会持续传输桌面画面。

## 10. 磁盘与自定义镜像结论

平台界面显示系统盘和自定义镜像均按完整 200GB 容量收费：

```text
0.0006 元 / GB·小时
200GB × 0.0006 × 24 = 2.88 元/天
9.5 折后约 2.74 元/天
约 82.08 元/30天
```

因此当前项目不建议长期保留实例磁盘或自定义镜像。推荐：

```text
GitHub：源码、配置、轻量正式记录
免费云盘：datasets.zip、必要权重、压缩后的完整 run
本地硬盘：完整实验二次备份
临时 GPU 实例：只负责安装、解压、训练，完成后释放
```

当前 V100 镜像也不适合作为未来 4090 正式镜像，因为它使用临时 Python 3.12/PyTorch 2.5 环境，且包含测试缓存与 V100 环境状态。

## 11. 释放服务器前的备份清单

原始 `runs/` 不会自动上传 GitHub。释放实例前至少保存：

- `weights/best.pt`
- `weights/last.pt`（若需要 resume）
- `results.csv`
- `args.yaml`
- `results.png`
- Box/Mask P、R、PR、F1 曲线
- confusion matrix
- `labels.jpg` 和必要的 `train_batch*.jpg`
- 正式 wrapper 生成的 `experiment_manifest.json`（本次 CLI 不会生成项目专用 manifest）
- Git branch、commit、环境版本和完整训练命令

可将完整 run 压缩：

```bash
cd /root/yolo_plus/runs/segment/runs_seg
tar -czf /root/yolo26_run_name.tar.gz run目录名称
```

然后在服务器内通过 SFTP 上传永久云盘。确认云盘文件完整后再释放实例。

## 12. 下一台电脑如何继承

在新电脑上：

```bash
git clone https://github.com/xiaojiegenga/yolo_plus.git
cd yolo_plus
git fetch origin
git switch -c codex/data-v1-baseline-b4 --track origin/codex/data-v1-baseline-b4
```

如果本地分支已经存在：

```bash
git switch codex/data-v1-baseline-b4
git pull --ff-only
```

然后首先阅读：

```text
experiment_records/cloud-v100-smoke/run_record.md
experiment_records/data-v1-baseline-b4/run_record.md
experiment_records/v1-v3-comparison/
```

根目录 `CODEX_PROJECT_CONTEXT.md`、`code_plus.md` 和 `node.md` 仍是本地笔记，不会出现在另一台电脑的 GitHub clone 中；稳定的跨设备事实以 `experiment_records/` 为准。

## 13. 正式 RTX 4090 迁移待办

正式租用 4090 后不要直接沿用本次临时 CLI。推荐先完成：

1. 选择 Ubuntu 22.04 + CUDA 兼容公共镜像；
2. 建立独立 Python 3.10 Conda 环境，不直接使用平台 base Python 3.12；
3. 固定并记录 Python、PyTorch、TorchVision、CUDA runtime 和 Ultralytics 版本；
4. 在 GitHub 增加 `scripts/setup_cloud_linux.sh` 或等价初始化脚本；
5. 创建 Linux 专用 data-v1 YAML 和正式 profile，处理路径变化但保持数据内容 SHA 不变；
6. 让正式训练重新走 `scripts/train_yolo26_seg.py` 的分支、数据和参数门禁；
7. 使用固定 batch 或统一 effective batch，不能让每个对比模型随意使用不同 AutoBatch；
8. 所有正式对比模型在同一云端环境、data-v1、split=val、imgsz、seed 和评估规则下重跑；
9. 正式训练通过 `tmux` 启动，完成后上传 run 并生成 Git 可追踪摘要；
10. 架构与数据方案冻结后，独立 test 只进行一次最终评估。

计划中的分割模型对比应优先使用有官方实例分割实现的模型：

```text
YOLOv5m-seg
YOLOv6-S-seg（官方 yolov6-seg 分支）
YOLOv8m-seg
YOLOv9-C-SEG
YOLO11m-seg
YOLO26m-seg
```

官方 YOLOv10 主要是检测实现，不应把 Box 指标和上述 Mask 指标混入同一张实例分割主表；如必须比较，应另建 Box-only 表。

## 14. 最终结论

- GitHub 中确实存在可执行的完整 YOLO26 训练源码；
- 云服务器通过 clone + editable install 使用该源码，通过本地 PyTorch/CUDA 调用 GPU；
- 本次 V100 已成功跑通数据下载、源码安装、AMP、自适应 batch、训练和 val 流程；
- 本次运行只是基础设施 smoke test，任何精度指标都不能用于论文归因；
- 未来不长期保留 200GB 磁盘或镜像，采用 GitHub + 免费云盘 + 本地备份；
- 下一关键工程任务是把当前手工 Linux 流程固化成可复现的 4090 初始化脚本和正式 Linux profile。
