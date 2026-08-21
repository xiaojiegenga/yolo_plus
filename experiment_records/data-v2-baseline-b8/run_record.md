# data-v2 Baseline-b8 实验记录

## 1. 实验身份

- 状态：正式训练已完成，可作为 `data-v2-batch8` 后续实验的严格 Baseline
- 准备日期：2026-08-20
- Git 分支：`data-v2/main`
- 实验标识：`data-v2-baseline-b8`
- 模型：官方预训练 `yolo26m-seg.pt`
- 任务：水稻害虫实例分割
- 类别：Rice leaffolder、Rice stemborers
- 对比组：`data-v2-batch8`

本实验是 data-v2 上的新 Baseline，不包含 CBAM、P2、Dice 或其他模型源码改进。后续源码改进必须使用同一 data-v2 指纹和同一训练 profile，才能与本结果严格比较。

## 2. 锁定数据

| 项目 | 值 |
|---|---:|
| Dataset ID | `rice-pest-data-v2` |
| 图片数量（train/val/test） | 938 / 117 / 118 |
| 标签数量（train/val/test） | 938 / 117 / 118 |
| 空标签（train/val/test） | 74 / 7 / 9 |
| 卷叶螟实例（train/val/test） | 3145 / 462 / 420 |
| 钻心虫实例（train/val/test） | 924 / 95 / 91 |
| 原图组数量 | 268 |
| 跨 split 原图组 | 0 |
| 跨 split 完全相同图片 | 0 |
| 标签结构问题 | 0 |

- 数据 YAML SHA-256：`5CA21A32CF66AA2EC4776069E2507839E4005AE3E1D47C2117CB96473007AD33`
- 数据内容 SHA-256：`02B9A2475D45CE5C88D933E0B7338235AD1622DFEB3266C2E7356EE874538C49`

## 3. 锁定训练配置

配置文件：`experiments/yolo26m_seg_data_v2_baseline_b8.yaml`

| 参数 | 值 |
|---|---:|
| epochs | 400 |
| patience | 100 |
| batch | 8 |
| imgsz | 640 |
| optimizer | auto |
| seed | 42 |
| deterministic | true |
| device | 0 |
| workers | 4 |
| mask_ratio | 4 |
| mosaic | 1.0 |
| mixup | 0.1 |
| copy_paste | 0.3 |

训练参数 SHA-256：`FA16F5C3748A9B978E62EDC50E85A5F1FA014CCBA1A3382AA1378030F4F26926`

## 4. 用户手动执行命令

只保留10 epoch预检和400 epoch正式训练，不再执行1 epoch实验。

Windows 本地10 epoch预检：

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment data-v2-baseline-b8 --preflight10
```

Windows 本地正式训练：

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment data-v2-baseline-b8
```

Linux 云服务器将数据解压到 `/root/yolo_data` 后，10 epoch预检和正式训练分别为：

```bash
python scripts/cloud_train_data_v2.py --preflight10
python scripts/cloud_train_data_v2.py
```

默认使用轻量检查。只有需要重新核验完整数据内容时，才额外添加 `--strict-checks`。

云端简化规则：

- 脚本直接优先导入 Git clone 中的 `ultralytics-main`，不要求重复执行 editable install；
- 云端入口会检查环境；除CUDA版PyTorch外的依赖缺失时，自动完成一次 editable install；
- Git 仓库不保存 `.pt`，首次缺少权重时自动下载指定的 `yolo26m-seg.pt`；
- Linux 默认数据 YAML 为 `experiments/yolo_data_v2_cloud.yaml`，数据根目录固定为 `/root/yolo_data`；
- 默认仅检查2个类别、train/val/test目录及图片/标签数量；
- 不提供1 epoch入口，也不要求正式训练前执行 dry-run；
- 10 epoch结果仅用于确认流程，正式论文指标只取400 epoch正式训练。

助手不得代替用户执行正式训练命令。

## 5. 2026-08-21：正式训练结果

### 结论先行

本次正式运行身份、参数和产物均正确，可以作为后续 **data-v2 / batch=8** 源码改进实验的
唯一严格 Baseline。训练在 epoch 259 取得最佳 fitness，并在 epoch 359 触发 patience=100 的
EarlyStopping；日志未出现 NaN/Inf，应使用 `best.pt`，不使用性能已经波动回落的 `last.pt`。

### 运行身份与资源

| 项目 | 记录 |
|---|---|
| Run 目录 | `runs/segment/runs_seg/yolo26m_data_v2_baseline_b8_seg_20260821_002538` |
| Git branch / commit | `data-v2/main` / `a1b30908dd4e26f7083e1a2c0917aa951abfaa1f` |
| Run kind | `formal` |
| Paired comparison group | `data-v2-batch8` |
| 模型 | 官方 `YOLO26m-seg`，不含 CBAM/P2/Dice |
| 计划 / 实际 epoch | 400 / 359（EarlyStopping） |
| Best epoch | 259 |
| 总训练时间 | 15,176.4 s，约 4 h 12 min 56 s（4.216 h） |
| Fused Params / FLOPs | 23,509,010（23.509 M）/ 121.2 G |
| 推理时间 | 7.8 ms/image |
| `best.pt` 大小 | 54.53 MB（十进制）/ 52.00 MiB |

### `best.pt` 正式 Val 指标

| 类别 | Images | Instances | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask F1¹ | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 117 | 557 | 0.695 | 0.629 | 0.723 | 0.451 | 0.694 | 0.605 | 0.646 | 0.708 | 0.374 |
| Rice leaffolder | 87 | 462 | 0.686 | 0.617 | 0.703 | 0.401 | 0.698 | 0.600 | 0.645 | 0.689 | 0.318 |
| Rice stemborers | 38 | 95 | 0.704 | 0.642 | 0.743 | 0.502 | 0.690 | 0.611 | 0.648 | 0.727 | 0.430 |

¹ Mask F1 使用表中 P/R 按 `2PR/(P+R)` 计算，仅作辅助读数。机器可读指标保存在同目录
`val_metrics.csv`。

### 训练过程检查

- `results.csv` 包含 epoch 1～359，共 359 行，无缺失轮次和 NaN/Inf；
- epoch 259 的 Box mAP50-95=`0.45172`、Mask mAP50-95=`0.37344`，二者之和
  fitness=`0.82516`，为全程最高；
- epoch 259 的 CSV 指标与训练结束后重新载入 `best.pt` 的最终 Val 输出一致（仅存在显示精度差异）；
- 10 epoch 预检的 Overall Mask mAP50/mAP50-95 为 `0.594/0.295`，正式最佳权重达到
  `0.708/0.374`，说明模型后续训练仍带来明显收益；
- 最后一轮 epoch 359 的 Overall Mask mAP50/mAP50-95 为 `0.702/0.360`，仍保持正常，
  但低于最佳轮次，因此正式报告统一使用 `best.pt`。

### 产物校验

| 产物 | SHA256 |
|---|---|
| `best.pt` | `254D3F4C2FF0CE71F92713FFAC8553F85665E6F1347A6E3551AC8B9F499076BE` |
| `last.pt` | `2B1B44B35DD21C800EA452C07B6693DC824C27564B8272376D729315280ED1CC` |
| `results.csv` | `7173A786B5BEC1AF9496223190AB41C0EA124D68D5428C356D90ECEDAA650216` |
| `args.yaml` | `6A91AD33103C9C46471AD09EB5C31761E9A180834212ECBCA9B5CDC87185AD37` |
| `experiment_manifest.json` | `15F50E73076CB4449F36B5E50CA846CA756033F227A47FD6F0334382CB1E9D97` |

`best.pt`、`last.pt` 和完整 run 仅保存在本地/用户备份中，不上传 Git。

### 与 data-v1 Baseline-b4 的描述性观察

| Mask 指标 | data-v1-b4 | data-v2-b8 | 差值 |
|---|---:|---:|---:|
| Overall P | 0.717 | 0.694 | -0.023 |
| Overall R | 0.586 | 0.605 | +0.019 |
| Overall mAP50 | 0.657 | 0.708 | +0.051 |
| Overall mAP50-95 | 0.310 | 0.374 | +0.064 |
| Leaffolder P / R | 0.671 / 0.600 | 0.698 / 0.600 | +0.027 / 0.000 |
| Leaffolder mAP50 / mAP50-95 | 0.670 / 0.302 | 0.689 / 0.318 | +0.019 / +0.016 |
| Stemborers P / R | 0.763 / 0.571 | 0.690 / 0.611 | -0.073 / +0.040 |
| Stemborers mAP50 / mAP50-95 | 0.644 / 0.319 | 0.727 / 0.430 | +0.083 / +0.111 |

> 比较边界：data-v1 与 data-v2 的数据内容、Val 样本数量以及 batch 分别不同，因此这里仅描述
> 结果变化，不能把差值归因于某一项数据处理或 batch。源码改进的收益必须在
> `data-v2-batch8` 组内与本 Baseline 配对比较。

### 后续实验边界

1. data-v2 后续模型必须使用相同数据内容指纹、训练 profile、`batch=8`、`imgsz=640` 和 Val；
2. 当前 Overall Mask mAP50=`0.708`、mAP50-95=`0.374` 是后续源码改进必须超过的主要基线；
3. 卷叶螟 Mask Recall=`0.600`，仍是下一阶段需要重点提升的指标；
4. 不在模型和训练方案冻结前使用 test 反复调参；最终方案确定后再统一执行一次独立 test；
5. 原始 run 由用户自行复制到本地 `results/` 或外部存储备份，Git 只保存本轻量记录。
