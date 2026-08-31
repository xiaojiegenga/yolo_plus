# 项目进展记录

> 用途：在不同模型 / 会话之间切换时，快速了解项目当前进展。这里只放**已完成的关键步骤**和**运行结果**；固定规则与架构见 `CLAUDE.md` 与 `AGENTS.md`，实验方案细节见 `云服务器实验设计与记录表.md`。
> 最后更新：2026-09-01

## 当前状态（一句话）

RTX 5090 已确定，data-v2 正在进行**表 1 正式训练参数优化**。第 1 轮组合优化 `data-v2-tune-e300-b16-s42` 已完成，best epoch 251 / official fitness 0.80993，相对参数基线 +0.02960。第 2 轮掩膜精调方案已写入 `experiments/yolo26m_seg_5090.yaml`：保持 P1 的 300 epoch、batch=16、imgsz=640 和航拍增强，将 `mask_ratio 4→2`、`mixup 0.1→0`；Run ID 为 `data-v2-tune-mr2-nomix-e300-b16-s42`。

## 已完成的关键步骤

- [x] 建立 data-v2 云训练工作流：`scripts/cloud_train_data_v2.py`、`train_yolo26_seg.py`、`transfer_run.py`、`fill_results_table.py`
- [x] 新增预检配置 `experiments/yolo26m_seg_5090.yaml` 与云端数据 `experiments/yolo_data_v2_cloud.yaml`
- [x] 已建立三因素消融、模型尺度、跨代对比、稳定性复验和 Test 评估表格框架；正式训练参数待优化
- [x] 本地代码与文档提交并推送：commit `60f070a`，分支 `cloud/data-v2-5090`（已跟踪 `origin/cloud/data-v2-5090`）
- [x] 数据集 `rice-pest-data-v2` 已上传云端 `/root/yolo_data`
- [x] RTX 5090 兼容性预检完成并回传；GPU 选定后该预检不再占用总表表 2
- [x] 5090 完整原始 Run 已解包到 `runs/data-v2-5090-preflight/`，传输归档位于 `exports/data-v2-5090-preflight.zip`
- [x] 正式训练 GPU 已确定为 RTX 5090，不再进行 4090 / 5090 成本选型
- [x] 首轮参数基线 `data-v2-scale-y26m-seg-b16-s42` 已训练 292 epoch；官方 fitness 最优权重位于 epoch 192（0.78033）
- [x] 原始 Run 已解包到 `runs/data-v2-scale-y26m-seg-b16-s42/`
- [x] 参数诊断已保存到 `experiment_records/parameter_tuning/data-v2-scale-y26m-seg-b16-s42.md`
- [x] 总表表 2 已改为表 1 参数优化训练结果对比表
- [x] 训练入口、EarlyStopping、best.pt 与结果回填已恢复 Ultralytics 官方分割 fitness；论文保留两项 Mask mAP
- [x] 参数优化第 1 轮组合方案确定：epochs 400→300、新增 degrees=15/flipud=0.5/scale=0.3，batch 保持 16（为后续消融单变量原则统一 batch=16），分辨率固定 640（不做 832）；Run ID `data-v2-tune-e300-b16-s42`，已写入 `experiments/yolo26m_seg_5090.yaml`
- [x] 参数优化第 1 轮 `data-v2-tune-e300-b16-s42` 已训练完成：300 epoch，best epoch 251，official fitness 0.80993，Mask mAP50 / mAP50-95 为 0.68965 / 0.35805
- [x] 第 1 轮分类型 Mask mAP50：Rice leaffolder 0.671（较 P0 +0.022），Rice stemborers 0.708（较 P0 -0.010）
- [x] 第 2 轮参数方案确定：`mask_ratio 4→2`、`mixup 0.1→0`，其余核心参数继承 P1；Run ID `data-v2-tune-mr2-nomix-e300-b16-s42`

## 已有参数优化结果

| Run ID | 环境 | 状态 | Official fitness | Mask mAP50 / mAP50-95 | 说明 |
|---|---|---|---:|---:|---|
| `data-v2-scale-y26m-seg-b16-s42` | RTX 5090 | 参数优化 P0 | 0.78033（best epoch 192） | 0.68506 / 0.34685 | 首轮基线，仅用于决定表 1，不进入 `comparison.csv` |
| `data-v2-tune-e300-b16-s42` | RTX 5090 | 参数优化 P1（已完成） | 0.80993（best epoch 251） | 0.68965 / 0.35805 | 第 1 轮组合：e300+方向增强+scale；相对 P0 +0.02960 |
| `data-v2-tune-mr2-nomix-e300-b16-s42` | RTX 5090 | 参数优化 P2（待训练） | 待填 | 待填 | 继承 P1；mask_ratio=2、mixup=0，显式锁定 P1 实际 AdamW 参数 |

## 参数优化第 2 轮改动清单（本次）

Run ID：`data-v2-tune-mr2-nomix-e300-b16-s42`，相对当前最优 P1 `data-v2-tune-e300-b16-s42`：

| 参数 | 基线值 | 新值 | 依据 |
|---|---|---:|---|
| mask_ratio | 4 | 2 | 标签掩膜监督由约 160×160 提高到约 320×320，重点观察严格 Mask IoU 与边界质量 |
| mixup | 0.1 | 0.0 | 去除两幅图像透明叠加造成的实例掩膜边界歧义 |
| optimizer | auto→实际 AdamW | AdamW | 显式复现 P1 的自动选择，不改变优化器类型 |
| lr0 / momentum / weight_decay | auto→0.001667 / 0.9 / 0.0005 | 0.001667 / 0.9 / 0.0005 | 锁定 P1 实际优化器参数，提高云端复现性 |

不变项：model=yolo26m-seg.pt、epochs=300、batch=16、imgsz=640、patience=100、seed=42、deterministic=true、degrees=15、flipud=0.5、fliplr=0.5、scale=0.3、mosaic=1.0、copy_paste=0.3、close_mosaic=15、warmup_epochs=3、cos_lr=false、lrf=0.01。分辨率固定 640，不做 imgsz 实验。

源码机制说明：当前分割损失在标签掩膜尺寸与 Proto 不一致时，会把 Proto 双线性插值到标签尺寸再计算损失。因此 `mask_ratio=2` 能增加损失计算网格密度，但不会增加 Proto 的原生特征分辨率，预期应是小幅边界改善而非结构级跃升。

数值稳定性说明：P1 的 300 个 epoch 中有 24 个 epoch 的部分 Val loss 字段为 NaN，但训练损失、mAP 指标、best epoch 251 和权重均保持有限。P1 指标仍可用于参数比较，但不得再表述为“所有数值全程有限”；P2 需继续记录该现象是否复现。

## 下一步

1. 云端 `git pull --ff-only` 后按 `实验步骤.md` 启动 `data-v2-tune-mr2-nomix-e300-b16-s42`
2. 回传后新增 `experiment_records/parameter_tuning/data-v2-tune-mr2-nomix-e300-b16-s42.md`，并把结果填入总表表 2
3. 若 P2 明显优于 P1，使用相同方案做多 seed 复验（seed 42/2/3）；若差异小于约 0.005 或退化，再拆分验证 mask_ratio 与 mixup
4. 参数方案冻结后填写总表表 1，再开始可用于期刊的正式对比实验和 `comparison.csv` 回填

## Git 与本地文件状态

- 已提交并推送：分支 `cloud/data-v2-5090`。真实 git 工作区已迁移到 `e:\Study\论文撰写\yolo_plus_git`；本夸克目录（`模型训练`）不含 `.git/`，只用于编辑与夸克同步。
- 本次提交内容：
  - `experiments/yolo26m_seg_5090.yaml`：第 2 轮掩膜精调参数（mask_ratio=2、mixup=0、显式 AdamW 参数）
  - `PROGRESS.md`：补记 P1 实测结果、NaN 说明与 P2 计划
  - `实验步骤.md`：切换到 P2 的云端训练、监控、打包与回传命令
- 云端下一步：`git pull --ff-only` 后运行 `data-v2-tune-mr2-nomix-e300-b16-s42`（步骤见 `实验步骤.md`）。
- `runs/` 与 `exports/` 按约定不进入 Git。

## 关键约束（快速提醒）

- `best.pt` 与 EarlyStopping 使用官方分割 fitness；论文同时报告 Mask mAP50、Mask mAP50-95；Val 选方案，Test 在方案冻结后统一评
- RTX 5090 已确定；batch、epochs、imgsz、optimizer 等正式训练参数仍待优化
- 分辨率固定 640（本次决定不做 imgsz 832 实验）
- 消融实验单变量原则：batch 统一为 16（含尺度对比的 l 模型与源码改动实验），不因 32GB 显存改用 batch=32
- 参数优化 Run 只写 `parameter_tuning/` 并填总表表 2，不进入 `comparison.csv`
- 只有参数冻结后、可用于期刊对比的正式 Run 才进入 `comparison.csv`
- 未经用户要求不启动下一轮长时间训练
- 不覆盖已有 Run / 权重 / 记录；任何参数、代码或数据口径变化换新 Run ID
