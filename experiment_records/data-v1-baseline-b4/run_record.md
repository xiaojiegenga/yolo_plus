# data-v1 Baseline-b4 正式实验记录

## 实验目的

在已消除原图级 train/val/test 泄露的 `rice-pest-data-v1` 上，重新训练官方
YOLO26m-seg Baseline。本结果将成为后续 data-v1 源码改进实验的唯一严格对照组。

历史 data-v0 结果继续保留，但不得与本实验直接比较模型优劣，因为数据划分已经变化。

## 当前状态

- [x] data-v1 已建立并替换到本地数据目录；
- [x] 图片与标签配对检查通过；
- [x] YOLO 分割标签结构检查通过；
- [x] 原图级跨 split 数量为 0；
- [x] 完全相同图片跨 split 数量为 0；
- [x] 官方 YOLO26m-seg 模型静态构建检查通过；
- [x] 训练参数和数据内容指纹已锁定；
- [x] 不运行 1 epoch / 10 epoch 预检；
- [ ] 用户手动启动 400 epoch 正式训练；
- [ ] 使用 `best.pt` 完成独立 `split=val` 评估；
- [ ] 回填正式指标、权重哈希与训练结论。

## 代码与模型身份

| 项目 | 记录 |
|---|---|
| Git 分支 | `codex/data-v1-baseline-b4` |
| 分支起点 | `codex/baseline-b4` |
| 起点 commit | `55fb63d5347439044824348cd7e4db40fd80f4a6` |
| 模型 | 官方 `ultralytics-main/yolo26m-seg.pt` |
| 模型模式 | `baseline-pretrained-pt` |
| 自定义模型 YAML | 不使用 |
| CBAM / P2 / Dice | 均不包含 |
| Ultralytics | `8.4.80`，导入本项目 `ultralytics-main` |
| 预训练权重 SHA256 | `16B636F04E8FB6A325B3370F22DC5E5535FF473E384F4D041FD28D788F6EE9F5` |

## data-v1 身份与质量门禁

| 项目 | 结果 |
|---|---:|
| Dataset ID | `rice-pest-data-v1` |
| Dataset content SHA256 | `BBC51AAA6E53471AA72C6AABE4D3D09DA7770C0445A9597320C4A318D79EE922` |
| Dataset YAML SHA256 | `75996638EB9BBAED8B80D0413FFD57B374C0024B2C9F9EF5689AD90B5ADF78AF` |
| train / val / test 图片 | 742 / 93 / 93 |
| train / val / test 标签 | 742 / 93 / 93 |
| train / val / test 空标签负样本 | 60 / 5 / 7 |
| 原图组 | 191 |
| 原图组跨 split | **0** |
| 完全相同图片跨 split | **0** |
| 结构性标签问题 | **0** |

实例数量：

| Split | Rice leaffolder | Rice stemborers | 合计 |
|---|---:|---:|---:|
| train | 2846 | 477 | 3323 |
| val | 353 | 56 | 409 |
| test | 370 | 53 | 423 |
| total | 3569 | 586 | 4155 |

说明：训练入口会在每次启动前重新读取全部图片和标签，并核对上面的内容指纹；如果data-v1
被替换、增删或修改，将拒绝启动正式训练。

## 锁定训练参数

Profile：

```text
experiments/yolo26m_seg_data_v1_baseline_b4.yaml
```

| 参数 | 值 |
|---|---:|
| epochs | 400 |
| patience | 100 |
| batch | 4 |
| imgsz | 640 |
| optimizer | auto |
| seed | 42 |
| deterministic | True |
| workers | 4 |
| mask_ratio | 4 |
| mosaic / mixup / copy_paste | 1.0 / 0.1 / 0.3 |

训练参数 SHA256：

```text
5A90247FF46C3D0A38BC4A16714CA1A7C75BD038042CD37DD70FB63B7DD5F917
```

该指纹与历史 Baseline-b4 的有效训练参数一致；本次相对于历史 Baseline-b4 的主要变化是
数据集从 data-v0 改为无原图泄露的 data-v1。

## 2026-08-07 静态检查

已执行 `--dry-run`，只完成数据指纹、源码路径、权重和模型构建检查，没有启动训练、没有运行
epoch。确认输出：

```text
Experiment       : data-v1-baseline-b4
Model mode       : baseline-pretrained-pt
Profile          : yolo26m-seg-data-v1-baseline-b4-20260807
Profile SHA256   : 5A90247FF46C3D0A38BC4A16714CA1A7C75BD038042CD37DD70FB63B7DD5F917
Dataset ID       : rice-pest-data-v1
Dataset SHA256   : BBC51AAA6E53471AA72C6AABE4D3D09DA7770C0445A9597320C4A318D79EE922
Dataset images   : {'train': 742, 'val': 93, 'test': 93}
Parent leakage   : 0
imgsz / epochs   : 640 / 400
batch            : 4
```

## 用户手动正式训练命令

不执行 1 epoch 或 10 epoch 预检，直接进行正式训练：

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment data-v1-baseline-b4
```

预期输出目录名：

```text
runs/segment/runs_seg/yolo26m_data_v1_baseline_b4_seg_YYYYMMDD_HHMMSS
```

正式训练结束后，统一使用 `best.pt` 的独立 `split=val` 结果填写论文对比表；在最终模型和方案
冻结前，不使用独立 `test` 反复调参。

