# 项目进展记录

> 用途：在不同模型 / 会话之间切换时，快速了解项目当前进展。这里只放**已完成的关键步骤**和**运行结果**；固定规则与架构见 `CLAUDE.md` 与 `AGENTS.md`，实验方案细节见 `云服务器实验设计与记录表.md`。
> 最后更新：2026-09-02

## 当前状态（一句话）

RTX 5090 与 data-v2 正式训练参数均已冻结，项目核心已切换到源码改进消融实验。阶段 0 的正式
`000 Baseline` 配置、云端操作步骤和消融主计划已经准备完成；下一步由用户在云服务器手动运行
10 epoch 预检，再决定是否启动 300 epoch 正式 Baseline。当前未启动任何阶段 0 训练。

## 已完成的关键步骤

- [x] 建立 data-v2 云训练工作流：`scripts/cloud_train_data_v2.py`、`train_yolo26_seg.py`、`transfer_run.py`、`fill_results_table.py`
- [x] 新增预检配置 `experiments/yolo26m_seg_5090.yaml` 与云端数据 `experiments/yolo_data_v2_cloud.yaml`
- [x] 已建立三因素消融、模型尺度、跨代对比、稳定性复验和 Test 评估表格框架；正式训练参数已由 P2 冻结
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
- [x] 参数优化第 2 轮已训练完成：300 epoch，best epoch 216，official fitness 0.81977，Mask mAP50 / mAP50-95 为 0.71132 / 0.36348
- [x] 第 2 轮分类型 Mask mAP50：Rice leaffolder 0.677（较 P1 +0.006），Rice stemborers 0.745（较 P1 +0.037）
- [x] 已确认第 2 轮相对 P1 还存在有效 `warmup_bias_lr 0.0→0.1` 的差异；本轮收益不能只归因于 `mask_ratio=2` 和 `mixup=0`
- [x] 第 2 轮正式诊断已保存到 `experiment_records/parameter_tuning/data-v2-tune-mr2-nomix-e300-b16-s42.md`
- [x] 用户决定不再进行参数优化且不补跑 seed=2、3；P2 已冻结为 data-v2 后续正式实验统一训练配方，固定 seed=42
- [x] 建立 `experiment_records/data-v2-source-ablation-plan.md`，冻结消融矩阵、门控规则、评估指标和执行顺序
- [x] 决定不直接复用参数优化 P2 的 `best.pt`；正式消融 Baseline 从官方 `yolo26m-seg.pt` 重新训练
- [x] 新建正式阶段 0 配置 `experiments/data-v2-abl-000-y26m-b16-s42.yaml`，显式锁定全部训练参数
- [x] 将 `实验步骤.md` 切换为阶段 0：dry-run → 10 epoch 预检 → 用户手动正式训练 → 打包回传 → 正式登记
- [x] 阶段 0 准备文件已纳入 `cloud/data-v2-5090` 的 Git 提交与推送流程；云端只需拉取后按步骤执行

## 已有参数优化结果

| Run ID | 环境 | 状态 | Official fitness | Mask mAP50 / mAP50-95 | 说明 |
|---|---|---|---:|---:|---|
| `data-v2-scale-y26m-seg-b16-s42` | RTX 5090 | 参数优化 P0 | 0.78033（best epoch 192） | 0.68506 / 0.34685 | 首轮基线，仅用于决定表 1，不进入 `comparison.csv` |
| `data-v2-tune-e300-b16-s42` | RTX 5090 | 参数优化 P1（已完成） | 0.80993（best epoch 251） | 0.68965 / 0.35805 | 第 1 轮组合：e300+方向增强+scale；相对 P0 +0.02960 |
| `data-v2-tune-mr2-nomix-e300-b16-s42` | RTX 5090 | 参数优化 P2（已完成并冻结） | 0.81977（best epoch 216） | 0.71132 / 0.36348 | mask_ratio=2、mixup=0、显式 AdamW；相对 P1 fitness +0.00984；不补多 seed |

## 参数优化第 2 轮改动清单（本次）

Run ID：`data-v2-tune-mr2-nomix-e300-b16-s42`，相对当前最优 P1 `data-v2-tune-e300-b16-s42`：

| 参数 | 基线值 | 新值 | 依据 |
|---|---|---:|---|
| mask_ratio | 4 | 2 | 标签掩膜监督由约 160×160 提高到约 320×320，重点观察严格 Mask IoU 与边界质量 |
| mixup | 0.1 | 0.0 | 去除两幅图像透明叠加造成的实例掩膜边界歧义 |
| optimizer | auto→实际 AdamW | AdamW | 显式复现 P1 的自动选择，不改变优化器类型 |
| lr0 / momentum / weight_decay | auto→0.001667 / 0.9 / 0.0005 | 0.001667 / 0.9 / 0.0005 | 锁定 P1 实际优化器参数，提高云端复现性 |

不变项：model=yolo26m-seg.pt、epochs=300、batch=16、imgsz=640、patience=100、seed=42、deterministic=true、degrees=15、flipud=0.5、fliplr=0.5、scale=0.3、mosaic=1.0、copy_paste=0.3、close_mosaic=15、warmup_epochs=3、cos_lr=false、lrf=0.01。分辨率固定 640，不做 imgsz 实验。

有效参数差异补充：P1 使用 `optimizer=auto`，虽然实际同样选择 AdamW（lr=0.001667、momentum=0.9），但 `trainer.py` 会把运行时 `warmup_bias_lr` 改为 0.0；P2 使用显式 AdamW，保留 `args.yaml` 中的 `warmup_bias_lr=0.1`。因此 P2 相对 P1 实际同时改变了 `mask_ratio`、`mixup` 和 warmup bias 学习率，不能进行单因素因果归因。

源码机制说明：当前分割损失在标签掩膜尺寸与 Proto 不一致时，会把 Proto 双线性插值到标签尺寸再计算损失。因此 `mask_ratio=2` 能增加损失计算网格密度，但不会增加 Proto 的原生特征分辨率，预期应是小幅边界改善而非结构级跃升。

数值稳定性说明：P1 的 300 个 epoch 中有 24 个 epoch 的部分 Val loss 字段为 NaN；P2 仅在 epoch 7、8、36 出现同类现象。两轮的训练损失、mAP 指标、best epoch 和权重均保持有限，因此指标可用于参数比较，但不得表述为“所有 Val loss 全程有限”。

## 参数优化第 2 轮实测结论

- P2 相对 P1：Box mAP50 +0.02049、Box mAP50-95 +0.00441、Mask mAP50 +0.02167、Mask mAP50-95 +0.00543。
- Mask P/R 从 P1 的 0.72069 / 0.57720 变为 0.64822 / 0.67252：召回率明显提高，精确率下降，但 Mask F1 从 0.64101 提高到 0.66015，综合表现为正收益。
- P2 top-5 / top-10 official fitness 均值较 P1 分别提高约 0.00896 / 0.00848，收益不是单个 epoch 的孤立尖峰。
- 分类型收益偏向 Rice stemborers；Rice leaffolder 相对 P1 仅 +0.006，不能把本轮表述为专门解决卷叶螟小目标问题。
- P2 最佳结果出现在 epoch 216，epoch 300 的 fitness 已回落到 0.78002，说明应使用 `best.pt`；当前没有继续增加 epochs 的依据。
- P2 仅在 epoch 7、8、36 出现部分 Val loss NaN，训练损失、全部 mAP 和 best epoch 指标均为有限值，不属于训练崩溃。
- P2 训练用时 4410.58 s（约 1.225 h），较 P1 约增加 15.3%；`mask_ratio=2` 增加了掩膜监督网格密度和训练成本。

## 下一步

1. 云服务器切换到 `cloud/data-v2-5090` 并拉取阶段 0 提交。
2. 用户按 `实验步骤.md` 先执行 dry-run，再手动运行独立的 10 epoch 预检；不运行 1 epoch。
3. 预检通过后，由用户手动启动 `data-v2-abl-000-y26m-b16-s42` 的 300 epoch 正式训练。
4. 正式 Run 回传后建立 `experiment_records/runs/data-v2-abl-000-y26m-b16-s42.md`，并更新 `comparison.csv`。
5. 只有 `000` 正式结果核验完成后，才依次冻结 Dice、Attention、P2Head 的唯一源码实现。
6. Val 用于模型和方案比较；Test 仍只在最终方案与推理阈值冻结后统一执行。

## Git 与本地文件状态

- 当前工作分支：`cloud/data-v2-5090`。真实 Git 工作区位于 `E:\Study\论文撰写\yolo_plus_git`；
  夸克同步目录 `模型训练` 不含 `.git/`，只用于编辑与跨电脑同步。
- 本次阶段 0 Git 内容：正式 `000` YAML、消融主计划、P2 冻结记录、更新后的 `PROGRESS.md`、
  `实验步骤.md`、实验目录说明及当前正式训练参数表。
- 云端下一步：拉取该分支并由用户手动执行阶段 0 命令；助手没有启动训练。
- `runs/` 与 `exports/` 按约定不进入 Git。

## 关键约束（快速提醒）

- `best.pt` 与 EarlyStopping 使用官方分割 fitness；论文同时报告 Mask mAP50、Mask mAP50-95；Val 选方案，Test 在方案冻结后统一评
- RTX 5090 与 P2 配方均已冻结：epochs=300、batch=16、imgsz=640、显式 AdamW、mask_ratio=2、mixup=0、seed=42；未做多 seed 复验
- 分辨率固定 640（本次决定不做 imgsz 832 实验）
- 消融实验单变量原则：batch 统一为 16（含尺度对比的 l 模型与源码改动实验），不因 32GB 显存改用 batch=32
- 参数优化 Run 只写 `parameter_tuning/` 并填总表表 2，不进入 `comparison.csv`
- 只有参数冻结后、可用于期刊对比的正式 Run 才进入 `comparison.csv`
- 未经用户要求不启动下一轮长时间训练
- 不覆盖已有 Run / 权重 / 记录；任何参数、代码或数据口径变化换新 Run ID
