# 项目进展记录

> 用途：在不同模型 / 会话之间切换时，快速了解项目当前进展。这里只放**已完成的关键步骤**和**运行结果**；固定规则与架构见 `CLAUDE.md` 与 `AGENTS.md`，实验方案细节见 `云服务器实验设计与记录表.md`。
> 最后更新：2026-09-08

## 当前状态（一句话）

RTX 5090 与 data-v2 正式训练参数均已冻结。A2 已完成并回传，epoch 276 正常早停，
best epoch 176；相对 `000` 的 Mask mAP50 / mAP50-95 为 +0.00374 / -0.00239，
近似持平，暂不进入组合。B 完整 Run 已回传并核验，完成 300 epoch，best epoch 243；
Mask mAP50 / mAP50-95 为 0.68016 / 0.34625，较 `000` 下降 0.03116 / 0.01723，
本次 BCE + 0.5 × Soft Dice 未通过门控，不进入组合。C：轻量 P2Head 已完成全部 300 epoch 并回传核验，best epoch 282；Mask mAP50 / mAP50-95 为 0.70130 / 0.35116，较 `000` 下降 0.01002 / 0.01232。本次 C 未通过门控，不进入组合。
总体分析继续在 A 分支维护；C 独立实现位于 `feature/data-v2-abl-c-p2head`，从正式 Baseline `c0f4f35` 创建。

## 改进 C 实现与完成状态

- 独立实现提交：`feature/data-v2-abl-c-p2head` / `b8001b6`，从 Baseline `c0f4f35` 创建；当前工作区已回到 A 总体分析分支。
- 结构：轻量 P2 旁路，Head 为 `Segment26P2Lite`，stride=4/8/16/32；保留原 P3/P4/P5 和 P3-based Proto。
- 正式配置：C 分支 `experiments/data-v2-abl-001-p2head-b16-s42.yaml`；所有 train 参数与 000 相同。
- 本地 5 项聚焦测试通过：语义迁移、80→2 类重建、四尺度/Proto 尺寸、checkpoint 融合、真实分割损失正样本与空样本反向、配方一致性。
- 官方 `yolo26m-seg.pt` 到 2 类 C 模型迁移已验证，匹配 890/1078 个状态张量；14 个类别相关张量因形状变化不迁移，174 个为新增 P2 参数/缓冲张量。
- 2 类模型 fused Params=23,757,752，较 000 +248,742；GFLOPs@640=133.213389，较 000 +12.042240（约 9.94%）。
- 知识文档：`knowledge/改进C-P2Head小目标分支原理与实现.md`；操作入口：`实验步骤.md`。
- 云端结构检查通过后，按用户决定跳过 10 epoch 预检，在 tmux 完成正式训练；完整 Run、正式结果行与小目标专项评估已登记。

## 已完成的关键步骤

- [x] 确定 `feature/data-v2-abl-a-attention` 为总体分析分支，汇总 A1、A2、B 的结果；原样收录 B 分支 `cef6b0b` 的 `knowledge` 文档，阶段总结见 `experiment_records/data-v2-ab-summary.md`
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
- [x] 正式 `000 Baseline` 已完成全部 300 epoch，未触发 EarlyStopping；Run 与训练日志均已回传本地
- [x] 正式 Run 已核验并写入 `experiment_records/runs/data-v2-abl-000-y26m-b16-s42.md`、`comparison.csv` 与实验总表
- [x] 正式 `000` 与冻结 P2 除时间外的全部 300 轮数值完全一致，确认 seed=42 确定性复现
- [x] 将正式 Baseline 结果和轻量记录单独提交到 `cloud/data-v2-5090`：commit `c0f4f35`
- [x] 从 `c0f4f35` 创建独立分支 `feature/data-v2-abl-a-attention`
- [x] 改进 A 冻结为 P3/P4 Selective Residual CBAM：reduction=16、kernel=7、残差混合初值 0.1
- [x] 新增 `ResidualCBAM`、`C3k2SRCBAM` 及模型解析注册，保持 Backbone/Head 层号不变
- [x] 新建模型 YAML 与正式配置 `experiments/data-v2-abl-100-srcbam-b16-s42.yaml`
- [x] 训练入口支持配置顶层 `pretrained: yolo26m-seg.pt`，自定义结构可迁移官方权重
- [x] 本地聚焦测试通过：2 passed；Baseline 全部参数键和形状保留，只新增 8 个注意力状态张量
- [x] SR-CBAM fused Params 为 23,574,744（+65,734），GFLOPs@640 为 121.286586（+0.115437）
- [x] 教学文档保存到 `knowledge/改进A-SR-CBAM注意力机制原理与实现.md`
- [x] `实验步骤.md` 已切换为改进 A 的拉取、预检、正式训练、回传和登记流程
- [x] Baseline 提交与改进 A 分支已推送到 GitHub
- [x] 改进 A 正式训练完成 300 epoch：best epoch 235，official fitness 0.80644，Mask mAP50 / mAP50-95 为 0.70884 / 0.35097
- [x] 改进 A 的 `best.pt` 中 α(P3/P4)=0.16722/0.09860；模块参与学习，但没有转化为总体分割收益
- [x] 改进 A 已写入 `experiment_records/runs/data-v2-abl-100-srcbam-b16-s42.md`、`comparison.csv` 与实验总表
- [x] 按预设门控淘汰 A1 的 P3/P4 SR-CBAM，不直接将 A1 放入组合长训
- [x] A1 实验结果和结论已单独提交：commit `ac11686`
- [x] A2 冻结为 P3-only Zero-init Residual CBAM：`Y=X+β×CBAM(X)`，`β=0` 初始化
- [x] 新增 `ZeroInitResidualCBAM`、`C3k2ZRCBAM`、A2 模型 YAML 和正式训练配置
- [x] A2 聚焦测试通过：4 passed；初始输出严格恒等，只在 Backbone P3 使用新模块

- [x] 本机项目迁移到 `E:\Study\论文撰写\yolo_plus`，拉取 A、B 远端分支并建立 B 本地跟踪
- [x] 建立本地 `.venv`，复用本机 `yolo26` Conda 依赖，当前仓库 Ultralytics 以 editable 模式安装；依赖检查、4 项注意力测试和 A2 dry-run 通过
- [x] A2 完整 Run 已核验：连续 276 epoch，best epoch 176，CSV、训练日志与 best.pt 的 train_metrics 一致
- [x] A2 已写入单次记录、`comparison.csv` 与实验总表；Mask mAP50 / mAP50-95 为 0.71506 / 0.36109
- [x] B 独立实现已从远端同步：`feature/data-v2-abl-b-dice`，源码 `1d1a71e`，文档 `cef6b0b`；实例 BCE + 0.5 × Soft Dice，smooth=1.0
- [x] B 完整 Run 已回传并核验：300 epoch，best 243；CSV 与两份权重 train_metrics 一致；实际参数只新增两个 Dice 参数
- [x] B 已写入正式记录、comparison.csv 与总表；Mask mAP50 / mAP50-95 为 0.68016 / 0.34625，按门控淘汰本次 B 实现

## 正式改进 C 结果

| Run ID | 实际 / Best epoch | Official fitness | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | GPU_mem 峰值 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `data-v2-abl-001-p2head-b16-s42` | 300 / 282 | 0.80954 | 0.72949 | 0.60020 | 0.70130 | 0.35116 | 19.3 GB | 1.370 h |

- 相对 `000`：Mask mAP50 / mAP50-95 下降 0.01002 / 0.01232，fitness 下降 0.01023；本次 C 未通过门控，不进入组合。
- 两类别日志 Mask AP 均下降。全部 300 轮 CSV 数值及两份权重张量有限；CSV、checkpoint 元数据与训练参数核验通过，实际训练 commit 为 `b8001b6`。
- 训练耗时 4933.27 s（82.22 分钟），较 000 增加 11.69%；峰值显存由 15.6 升至 19.3 GB；计算量增加约 9.94%。
- 固定权重的独立 Val 小目标评估：174 个小卷叶螟，Mask AP50 / AP50-95 从 0.55612 / 0.19311 降至 0.51168 / 0.17347。
- 专项 IoU=0.50、conf=0.25 时，Recall 从 0.61494 升至 0.64368，正确检出 107→112；误检 123→149。召回局部提高，但 Precision 和 AP 下降；专项口径不与正式训练日志混用。
- top-10 fitness 均值比 000 高 0.002009，不能概括为全程收敛更差；正式最佳权重主指标仍下降。
- 证据：[正式记录](experiment_records/runs/data-v2-abl-001-p2head-b16-s42.md)、[小目标配对评估](experiment_records/evaluations/data-v2-c-small-val.md)。本轮没有密集场景单独量化，也未使用 Test。

## 正式改进 A2 结果

| Run ID | 实际 / Best epoch | Official fitness | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | GPU_mem 峰值 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `data-v2-abl-a2-p3-zrcbam-b16-s42` | 276 / 176 | 0.82146 | 0.66346 | 0.67319 | 0.71506 | 0.36109 | 15.6 GB | 1.134 h |

- 相对 `000`：fitness +0.00169，Mask mAP50 +0.00374，Mask mAP50-95 -0.00239；近似持平，尚无进入组合所需的明确收益。
- best.pt 复核分类别 Mask P/R/mAP50/mAP50-95：卷叶螟 0.598 / 0.649 / 0.655 / 0.262；钻心虫 0.727 / 0.701 / 0.773 / 0.463。卷叶螟 AP 未改善。
- best.pt 的 β=0.481689453125；注意力分支参与学习，但不能据此判断有效。
- epoch 276 按 patience=100 正常早停；仅 epoch 3 的四项 Val loss 为 NaN，全部训练损失、P/R/mAP 与两份权重状态张量有限。
- 证据：`experiment_records/runs/data-v2-abl-a2-p3-zrcbam-b16-s42.md`；完整 Run 位于同名 `runs/` 目录。

## 正式改进 B 结果

| Run ID | 实际 / Best epoch | Official fitness | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | GPU_mem 峰值 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `data-v2-abl-010-dice-b16-s42` | 300 / 243 | 0.80359 | 0.70942 | 0.59795 | 0.68016 | 0.34625 | 16.3 GB | 1.331 h |

- 相对 `000`：fitness -0.01618；Mask mAP50 / mAP50-95 -0.03116 / -0.01723；Mask R -0.07457。当前 B 未通过门控，不进入组合。
- top-5 / top-10 fitness 为 0.795822 / 0.792326，较 `000` 低 0.012996 / 0.012302；负结果不只体现在单个最佳 epoch。
- 训练结束 best.pt 分类型 Mask P/R/mAP50/mAP50-95：卷叶螟 0.678 / 0.602 / 0.651 / 0.260；钻心虫 0.739 / 0.596 / 0.711 / 0.433，两类别 AP 均下降。
- 完成全部 300 epoch，4792.11 s，较 Baseline 增加 8.50%；权重大小 54.524209 MB，推理结构和参数量保持 Baseline。
- epoch 4、6 有部分 Val loss NaN；全部训练损失、P/R/mAP 和两份权重张量有限。B seg_loss 含额外 Dice 项，不能直接与 Baseline 的绝对值比较。
- 完整证据与结论：`experiment_records/runs/data-v2-abl-010-dice-b16-s42.md`；原始文件为同名 `runs/` 目录中的 args、CSV、权重和日志。
- 源码实现提交 `1d1a71e`；日志未记录实际训练 commit，只有 seed=42，不推广为所有 Dice 方案无效。

## 正式改进 A1 结果

| Run ID | Best epoch | Official fitness | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | GPU_mem 峰值 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `data-v2-abl-100-srcbam-b16-s42` | 235 | 0.80644 | 0.69962 | 0.64107 | 0.70884 | 0.35097 | 15.1 GB | 1.290 h |

- 相对正式 `000`：official fitness -0.01333，Mask mAP50 -0.00248，Mask mAP50-95 -0.01251；top-5 / top-10 fitness 均值也分别低 0.00656 / 0.00484。
- 分类别 Mask P/R/mAP50/mAP50-95：Rice leaffolder 为 0.676 / 0.582 / 0.662 / 0.262，Rice stemborers 为 0.724 / 0.690 / 0.753 / 0.438。
- 当前工作点的 Mask P 和 F1 提高，但 Mask R、总体 AP 和卷叶螟指标下降，不能作为 A 的保留依据。

## 正式消融 Baseline 结果

| Run ID | Best epoch | Official fitness | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | GPU_mem 峰值 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `data-v2-abl-000-y26m-b16-s42` | 216 | 0.81977 | 0.64822 | 0.67252 | 0.71132 | 0.36348 | 15.6 GB | 1.227 h |

- 训练结束 `best.pt` 复核的分类别 Mask P/R/mAP50/mAP50-95：Rice leaffolder 为 0.630 / 0.656 / 0.677 / 0.266，Rice stemborers 为 0.671 / 0.687 / 0.745 / 0.461。
- 训练完成全部 300 epoch；epoch 300 指标低于 epoch 216，正式比较使用 `best.pt`。
- 两个训练图像各有 1 个重复标签被加载器移除；train / Val 均为 0 corrupt，后续消融保持相同数据处理口径。

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

1. 000、A1、A2、B、C 五组正式结果均已登记；A1/B/C 未通过门控，A2 近似持平，当前均不进入组合。
2. 保留完整 Run 和原始负结果；先依据已完成的单模块证据提出下一项独立假设。
3. 不自动启动组合或重复训练；重新设计任一模块须使用新 Run ID。
4. 本机通过 `.venv/Scripts/python.exe` 开发和分析；环境入口见 `LOCAL_SETUP.local.md`。
5. Val 用于选方案；Test 保留到最终模型与阈值冻结后统一评估。

## Git 与本地文件状态

- 当前 Git 根目录：`E:\Study\论文撰写\yolo_plus`；当前总体分析分支为 `feature/data-v2-abl-a-attention`，A2 实现提交为 `a38aabf`。
- 正式 Baseline 分支为 `cloud/data-v2-5090`，记录提交为 `c0f4f35`；A1 源码为 `9d0c479`、结果记录为 `ac11686`；A2 实现为 `a38aabf`。
- B 本地分支 `feature/data-v2-abl-b-dice` 跟踪同名远端分支，当前为 `cef6b0b`，源码提交为 `1d1a71e`。B 从 `c0f4f35` 独立分叉，不含注意力改动。
- `feature/data-v2-abl-a-attention` 为总体分析分支，统一维护 A、B、C 知识文档、实验记录与阶段总结；B 配置与损失源码仍由 B 分支维护。
- `.venv` 基于本机 `D:\tool\Anaconda3\envs\yolo26`：Python 3.10.19、PyTorch 2.10.0+cu130、Ultralytics 8.4.80；本地环境用于开发与结果读取，云端正式环境仍以表 3 和 Run 日志为准。
- `runs/`、`exports/`、本机环境与本地配置按约定不进入 Git；GitHub 同步知识文档、轻量分析记录与汇总表。

## 关键约束（快速提醒）

- `best.pt` 与 EarlyStopping 使用官方分割 fitness；论文同时报告 Mask mAP50、Mask mAP50-95；Val 选方案，Test 在方案冻结后统一评
- RTX 5090 与 P2 配方均已冻结：epochs=300、batch=16、imgsz=640、显式 AdamW、mask_ratio=2、mixup=0、seed=42；未做多 seed 复验
- 分辨率固定 640（本次决定不做 imgsz 832 实验）
- 消融实验单变量原则：batch 统一为 16（含尺度对比的 l 模型与源码改动实验），不因 32GB 显存改用 batch=32
- 参数优化 Run 只写 `parameter_tuning/` 并填总表表 2，不进入 `comparison.csv`
- 只有参数冻结后、可用于期刊对比的正式 Run 才进入 `comparison.csv`
- A1、本次 B/C 已淘汰；A2 近似持平，暂不进入组合；五组正式结果均已登记
- 未经用户要求不启动下一轮长时间训练
- 不覆盖已有 Run / 权重 / 记录；任何参数、代码或数据口径变化换新 Run ID
