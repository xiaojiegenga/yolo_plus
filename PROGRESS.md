# 项目进展记录

> 用途：在不同模型 / 会话之间切换时，快速了解项目当前进展。这里只放**已完成的关键步骤**和**运行结果**；固定规则与架构见 `CLAUDE.md` 与 `AGENTS.md`，实验方案细节见 `云服务器实验设计与记录表.md`。
> 最后更新：2026-08-27

## 当前状态（一句话）

data-v2 已进入**表二：RTX 4090 / 5090 云端预检选型**阶段，代码与配置已推送云端，等待云端执行 10 epoch 预检。

## 已完成的关键步骤

- [x] 建立 data-v2 云训练工作流：`scripts/cloud_train_data_v2.py`、`train_yolo26_seg.py`、`transfer_run.py`、`fill_results_table.py`
- [x] 新增 5090 训练配置 `experiments/yolo26m_seg_5090.yaml` 与云端数据 `experiments/yolo_data_v2_cloud.yaml`
- [x] 实验方案设计完成：三因素消融（A=CBAM / B=Dice / C=待定）、模型尺度、跨代对比、稳定性复验、Test 冻结评估
- [x] 本地代码与文档提交并推送：commit `7329f32`，分支 `cloud/data-v2-5090`（已跟踪 `origin/cloud/data-v2-5090`）
- [x] 数据集 `rice-pest-data-v2` 已上传云端 `/root/yolo_data`

## 运行结果

| Run ID | 环境 | 状态 | Val Mask mAP50-95 | 说明 |
|---|---|---|---|---|
| `data-v2-baseline-b8` | 旧环境 | 历史参照 | 0.374（best epoch 259） | 不得当作 5090 新环境结果 |
| `data-v2-4090-preflight` | RTX 4090 | 待运行 | — | 表二预检，10 epoch |
| `data-v2-5090-preflight` | RTX 5090 | 待运行 | — | 表二预检，10 epoch |

## 下一步

1. 云端 `git pull --ff-only` 后跑 10 epoch 预检（4090 / 5090 各一次），记录时长、峰值显存、稳定性、环境版本
2. 打包回传本地：`transfer_run.py pack` → `unpack` → `fill_results_table.py`
3. 回填 `云服务器实验设计与记录表.md` 表二，按成本公式选定 GPU
4. 选定环境后重跑官方 `YOLO26m-seg` Baseline，建立新配对组（Run ID `data-v2-scale-y26m-seg-b8-s42`）

## 关键约束（快速提醒）

- 主指标 **Val Mask mAP50-95**；Val 选方案，Test 在方案冻结后统一评
- **batch=8** 已锁定为配对基准，预检与正式实验都不改
- 10 epoch 预检不进论文精度排名；未经用户要求不启动 400 epoch 长训
- 不覆盖已有 Run / 权重 / 记录；任何参数、代码或数据口径变化换新 Run ID
