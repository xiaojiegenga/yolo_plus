# 项目进展记录

> 用途：在不同模型 / 会话之间切换时，快速了解项目当前进展。这里只放**已完成的关键步骤**和**运行结果**；固定规则与架构见 `CLAUDE.md` 与 `AGENTS.md`，实验方案细节见 `云服务器实验设计与记录表.md`。
> 最后更新：2026-08-31

## 当前状态（一句话）

RTX 5090 已确定，data-v2 正在进行**表 1 正式训练参数优化**。首轮参数基线 `best.pt` 为 epoch 192 / official fitness 0.78033；第 1 轮组合优化方案已确定（epochs=300、航拍方向增强 + scale、batch 保持 16，分辨率固定 640），配置已写入 `experiments/yolo26m_seg_5090.yaml`，Run ID `data-v2-tune-e300-b16-s42`，待提交推送后云端训练。

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

## 已有参数优化结果

| Run ID | 环境 | 状态 | Official fitness | Mask mAP50 / mAP50-95 | 说明 |
|---|---|---|---:|---:|---|
| `data-v2-scale-y26m-seg-b16-s42` | RTX 5090 | 参数优化 P0 | 0.78033（best epoch 192） | 0.68506 / 0.34685 | 首轮基线，仅用于决定表 1，不进入 `comparison.csv` |
| `data-v2-tune-e300-b16-s42` | RTX 5090 | 待训练 | 待填 | 待填 | 第 1 轮组合：e300+方向增强+scale（batch 保持 16）；待提交推送后云端训练 |

## 参数优化第 1 轮改动清单（本次，未提交）

Run ID：`data-v2-tune-e300-b16-s42`，相对首轮基线 `data-v2-scale-y26m-seg-b16-s42`：

| 参数 | 基线值 | 新值 | 依据 |
|---|---|---:|---|
| epochs | 400 | 300 | 基线 292 早停，400 日程未吃到 close_mosaic 与 LR 尾段 |
| degrees | 0 | 15 | 航拍目标无固定方向 |
| flipud | 0 | 0.5 | 上下翻转 |
| scale | 0.5 | 0.3 | 减少把小目标进一步缩小 |

不变项：model=yolo26m-seg.pt、batch=16（为后续消融实验单变量原则，含尺度对比的 l 模型与源码改动实验在内统一 batch=16）、imgsz=640（分辨率固定，不做 832）、patience=100、seed=42、deterministic=true、mask_ratio=4、mosaic/mixup/copy_paste、close_mosaic=15。optimizer=auto 在 batch=16/epochs=300 下 iterations≈4500，仍自动选 AdamW（lr≈0.001667、momentum=0.9），与基线一致。

## 下一步

1. 本地提交并推送配置（`experiments/yolo26m_seg_5090.yaml` 与 `PROGRESS.md`），云端 `git pull --ff-only` 后启动 `data-v2-tune-e300-b16-s42`
2. 回传后新增 `parameter_tuning/data-v2-tune-e300-b16-s42.md` 分析并填入总表表 2
3. 第 2 轮（掩膜精调组合）：`mask_ratio 4→2`、`warmup_epochs 3→5`（分辨率固定 640，不做 imgsz 实验）
4. 参数方案确定后做多 seed 复验（seed 42/2/3），冻结并填写总表表 1
5. 冻结后再开始可用于期刊的正式对比实验和 `comparison.csv` 回填

## Git 与本地文件状态

- 远端已同步：`cloud/data-v2-5090` @ `60f070a`。
- 本次改动（均尚未提交或推送）：
  - `experiments/yolo26m_seg_5090.yaml`：第 1 轮组合参数（epochs=300、degrees=15、flipud=0.5、scale=0.3，batch 保持 16）
  - `PROGRESS.md`：进展快照更新
  - `云服务器实验项目交接.md`：云端命令、当前进展、下一步同步
  - `实验步骤.md`：新增，云端训练操作手册（从 pull 到打包下载的步骤）
  - `data-v2实验设计与记录表.md`：删除，内容由 `实验步骤.md` 取代
- 当前夸克同步目录未包含 `.git/`；上述修改需同步到实际 Git clone 后才能提交、推送并供云端拉取。
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
