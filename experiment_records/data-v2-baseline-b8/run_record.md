# data-v2 Baseline-b8 实验记录

## 1. 实验身份

- 状态：代码与数据门禁已准备，尚未开始正式训练
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

## 5. 正式结果（待训练后填写）

| 指标 | All | Rice leaffolder | Rice stemborers |
|---|---:|---:|---:|
| Mask Precision | 待填写 | 待填写 | 待填写 |
| Mask Recall | 待填写 | 待填写 | 待填写 |
| Mask mAP50 | 待填写 | 待填写 | 待填写 |
| Mask mAP50-95 | 待填写 | 待填写 | 待填写 |

- 最佳 epoch：待填写
- 实际完成 epoch：待填写
- 训练耗时：待填写
- 正式 run 路径：待填写
- `best.pt`：仅本地/云端备份，不上传 Git
