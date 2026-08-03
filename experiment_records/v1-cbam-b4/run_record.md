# V1-CBAM-b4 严格配对实验记录

## 当前状态

- 状态：`ready_for_formal_training`
- 日期：2026-08-03
- 新分支：`v1-cbam-b4`
- 独立起点：`codex/baseline-b4` / `55fb63d5347439044824348cd7e4db40fd80f4a6`
- 源码准备提交：`80033bceb4df9652232b8ba0e6b0c015186e8fcb`
- 严格对照：`Baseline-b4`
- 旧 `v1-cbam`：保留为历史实验，不在其上继续修改
- 正式训练：尚未开始；必须由用户手动启动

## 为什么重新补跑

历史 V1-CBAM 使用 batch=8 和较早的训练入口，其结果可以说明 CBAM 具有研究价值，但不能与
当前 batch=4 的 V2、V3 做严格配对比较。本次从纯 Baseline-b4 代码状态重新建立独立分支，
只引入 CBAM，并将其余模型结构、损失、数据和训练参数保持不变。

## 唯一模型变量

在 Backbone 的四个 C3k2 阶段后加入 CBAM：

```text
P2: C3k2 -> CBAM
P3: C3k2 -> CBAM
P4: C3k2 -> CBAM
P5: C3k2 -> CBAM
```

当前实现使用 `C3k2CBAM(C3k2)` 同层包装：先执行原 C3k2，再执行 CBAM。它与旧版在
C3k2 后单独插入 CBAM 的计算位置等价，但不会改变后续 YAML 层号。

明确不包含：

- P2 检测分支；
- Dice Loss；
- Neck 或 Segment26 Head 改造；
- 数据增强、优化器、学习率或其他训练参数改动。

## 修改文件

| 文件 | 作用 |
|---|---|
| `ultralytics/nn/modules/block.py` | 新增 `C3k2CBAM` 同层包装模块 |
| `ultralytics/nn/modules/__init__.py` | 导出新模块 |
| `ultralytics/nn/tasks.py` | 注册为 YAML 可解析、可重复缩放模块 |
| `ultralytics/cfg/models/26/yolo26m-cbam-seg.yaml` | 仅替换 Backbone 的 4 个 C3k2 |
| `scripts/train_yolo26_seg.py` | 锁定实验路径、batch=4、预检与迁移验证 |
| `tests/test_v1_cbam_model.py` | 结构、层号、Head 和前向检查 |

## 预训练权重迁移

2026-08-03 的干净工作树 dry-run：

```text
Transferred 904/916 items from pretrained weights
source tensors          = 904
transferred tensors     = 904
verified equal tensors  = 904
new CBAM tensors        = 12
CBAM modules            = 4
```

解释：Baseline 的 904 个状态张量全部按原键名迁移且数值逐一相等；新增的 12 个状态张量
全部属于四个 CBAM。没有随机初始化的新 Neck、Head 或 mask proto 层。

## 模型静态检查

在 `nc=2, imgsz=640` 下：

| 项目 | Baseline 构建态 | V1-CBAM 构建态 | 差值 |
|---|---:|---:|---:|
| Parameters | 26,971,750 | 27,825,902 | +854,152 |
| GFLOPs | 约 132.5 | 132.6 | 约 +0.1 |

最终 `best.pt` 的 fused Params、GFLOPs、推理时间和模型大小必须在正式训练后按同一验证流程记录，
不能直接用上述构建态数值替代。

结构测试结果：

```text
V1-CBAM structural tests passed.
```

测试包含：

- 原 C3k2 状态键完整保留；
- 新增状态键只属于 CBAM；
- CBAM 层号严格为 2/4/6/8；
- 最后一层仍为 `Segment26(P3,P4,P5)`；
- 64x64 无梯度前向传播成功。

## 公平训练配置

| 项目 | Baseline-b4 | V1-CBAM-b4 |
|---|---:|---:|
| epochs | 400 | 400 |
| batch | 4 | 4 |
| imgsz | 640 | 640 |
| seed | 42 | 42 |
| optimizer | auto | auto |
| mask_ratio | 4 | 4 |
| mosaic / mixup / copy_paste | 1.0 / 0.1 / 0.3 | 1.0 / 0.1 / 0.3 |
| 数据集 | 相同 | 相同 |
| 预训练权重 | 相同 | 相同 |

指纹：

```text
Profile SHA256   = FA16F5C3748A9B978E62EDC50E85A5F1FA014CCBA1A3382AA1378030F4F26926
Effective SHA256 = 5A90247FF46C3D0A38BC4A16714CA1A7C75BD038042CD37DD70FB63B7DD5F917
Weight SHA256    = 16B636F04E8FB6A325B3370F22DC5E5535FF473E384F4D041FD28D788F6EE9F5
```

## 用户手动执行顺序

以下会运行 epoch 的命令只能由用户手动输入。

### 1 epoch 功能预检

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment v1-cbam-b4 --batch 4 --preflight-epochs 1
```

### 10 epoch 趋势预检

仅在 1 epoch 无 OOM、NaN/Inf、Mask 崩溃和保存异常后运行：

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment v1-cbam-b4 --batch 4 --preflight-epochs 10
```

### 400 epoch 正式实验

仅在 10 epoch 训练 Loss 下降且 Box/Mask 指标趋势正常后运行：

```powershell
& "D:\tool\Anaconda3\envs\yolo26\python.exe" .\scripts\train_yolo26_seg.py --experiment v1-cbam-b4 --batch 4
```

预期正式运行名：

```text
yolo26m_v1_cbam_b4_seg_YYYYMMDD_HHMMSS
```

## 正式结果占位

正式训练结束后再填写：

- [ ] run 路径、run commit、best epoch、训练时间；
- [ ] best.pt 的 Overall 与分类别 Mask P/R/mAP50/mAP50-95；
- [ ] fused Params、GFLOPs、模型大小、推理时间；
- [ ] best.pt、results.csv、args.yaml、manifest 的 SHA256；
- [ ] 与 Baseline-b4 的逐项差值和是否进入 V4 的判断。

## 2026-08-03：1 epoch 功能预检通过

### 运行身份

```text
runs/segment/runs_seg/yolo26m_v1_cbam_b4_preflight1_seg_20260803_164907
```

| 项目 | 记录 |
|---|---|
| Git branch / commit | `v1-cbam-b4` / `0c9f7a3d70bb634bfdbc96e1927e363c92e484ea` |
| Run kind | `preflight-1-epoch` |
| epochs / batch / imgsz | 1 / 4 / 640 |
| Effective SHA256 | `C4013DF9A005BDE98882BCEACF7B570DEB2261D7F9800B08E39FAD19195D33DF` |
| 实际 epoch 时间 | 44.8154 s |
| best.pt / last.pt 状态张量 | 916 / 916 |
| best.pt / last.pt 非有限张量 | 0 / 0 |
| best.pt 大小 | 56,175,925 bytes |

权重迁移报告仍为：Baseline `904/904` 个张量逐值相等，新增 12 个状态张量全部属于4个CBAM。

### 第 1 epoch Loss

| 指标 | 数值 |
|---|---:|
| train/box_loss | 2.34179 |
| train/seg_loss | 2.88985 |
| train/cls_loss | 4.08065 |
| train/dfl_loss | 0.01098 |
| train/sem_loss | 3.88847 |
| val/box_loss | 1.83801 |
| val/seg_loss | 1.64626 |
| val/cls_loss | 2.58790 |
| val/dfl_loss | 0.01090 |

所有训练 Loss、验证 Loss、P/R、mAP 和学习率均为有限值。

### best.pt 最终预检 Val

| 类别 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.289 | 0.398 | 0.284 | 0.111 |
| Rice leaffolder | 0.389 | 0.530 | 0.417 | 0.156 |
| Rice stemborers | 0.191 | 0.265 | 0.153 | 0.067 |

对应 Overall Box mAP50/mAP50-95 为 `0.258/0.124`。Mask 与 Box 指标处于同一量级，
没有出现 mask coefficient/proto 数据流崩溃。

最终 fused summary：

```text
24,363,162 parameters
121.9 GFLOPs
8.6 ms/image inference
```

该 Params/GFLOPs 与历史 V1-CBAM 最终模型一致，进一步支持新同层包装与旧独立插层在模型
计算结构上等价。单次 1 epoch 的速度包含启动与短运行波动，不作为论文速度结论。

### 完整性哈希

```text
best.pt             C7168D6B3EEA9DCA0FF92FA4EFBAA72C1367914FFA663C96110331CE467F77EB
last.pt             D3836108F072FBDBCE0A4E09A99898C1BA5E9E5AE90A220EA64DEC409CDCFD65
results.csv         1CAF430AC05114EE4DAC6D0BAF580BD402FB0A59EDBD4C87F8C34F587079BA1C
args.yaml           2CB04EBF59B026A0B35906F7B9E35D5602E5122FEEFC9CE0E5992EF163EC5C9A
experiment_manifest B87B12DBF39D53D1BCA4E38981AE1536CB2A327891F479F6F723AC4C66498130
```

### 门禁判断

1. 模型、数据、batch、seed、profile 和预训练权重身份正确；
2. 无 OOM、NaN/Inf、EMA、保存或验证异常；
3. Mask 指标不为零且与 Box 指标同量级；
4. 第1轮处于 `warmup_epochs=5` 内，绝对 mAP 不能用于判断 CBAM 最终收益；
5. **允许用户手动进入 10 epoch 趋势预检，仍不得写入正式指标对比表。**

## 2026-08-03：10 epoch 趋势预检通过

### 运行身份

```text
runs/segment/runs_seg/yolo26m_v1_cbam_b4_preflight10_seg_20260803_165450
```

| 项目 | 记录 |
|---|---|
| Git branch / commit | `v1-cbam-b4` / `b11980b6bb2a7fafcfe3ac0ed3e2b9c06010abb5` |
| Run kind | `preflight-10-epoch` |
| epochs / batch / imgsz | 10 / 4 / 640 |
| Effective SHA256 | `F5BAF9E12F49E378AC9B0D3FE004A774BFC795F3EAFBA31A3D3B5D753F02B5BF` |
| 总运行时间（results.csv） | 404.905 s |
| 标准 fitness 最佳轮次 | epoch 9 |
| best fitness | 0.63329 |
| best.pt / last.pt 状态张量 | 916 / 916 |
| best.pt / last.pt 非有限张量 | 0 / 0 |

### 可复现性

独立 1 epoch run 与本次 10 epoch run 的第1行进行逐字段比较：

```text
除 time 外的字段差异数量 = 0
```

这证明相同 seed 下，新 CBAM 初始化、数据顺序、Loss 和指标完全复现。

### 训练趋势

| 指标 | Epoch 1 | Epoch 10 | 变化 |
|---|---:|---:|---:|
| train/box_loss | 2.34179 | 1.59425 | -31.92% |
| train/seg_loss | 2.88985 | 1.44928 | -49.85% |
| train/cls_loss | 4.08065 | 1.78878 | -56.16% |
| train/dfl_loss | 0.01098 | 0.00636 | -42.08% |
| train/sem_loss | 3.88847 | 1.52355 | -60.82% |
| Box mAP50 | 0.25785 | 0.57414 | +0.31629 |
| Box mAP50-95 | 0.12358 | 0.35070 | +0.22712 |
| Mask mAP50 | 0.28410 | 0.58869 | +0.30459 |
| Mask mAP50-95 | 0.11140 | 0.27240 | +0.16100 |

第3～4轮指标有波动，但仍处于5轮 warmup 以及较高学习率阶段；第7轮后 Box/Mask 指标明显
恢复并保持高位。全部10行的训练 Loss、验证 Loss、指标和学习率均为有限值。

### best.pt 最终预检 Val

| 类别 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.659 | 0.481 | 0.588 | 0.288 |
| Rice leaffolder | 0.691 | 0.310 | 0.531 | 0.233 |
| Rice stemborers | 0.628 | 0.653 | 0.645 | 0.342 |

对应 Overall Box P/R/mAP50/mAP50-95 为 `0.638/0.485/0.584/0.344`。

卷叶螟 Recall 在短期 best.pt 中偏低，但10轮预检的用途是检查训练健康与收敛趋势，不能据此
判定 CBAM 的正式效果；是否优于 Baseline-b4 只能等待完整训练的标准 fitness `best.pt`。

### 模型与速度

```text
24,363,162 fused parameters
121.9 GFLOPs
7.9 ms/image inference
```

速度来自单次短预检验证，只记录，不用于论文正式效率结论。

### 完整性哈希

```text
best.pt             7340161A693FC45E68A1D0DCD2CB710BF104C6CBBE02B50579B396FFE84AD924
last.pt             7258F2524929479D47FE2D18E6B7F4AE00980C660618FFA507F5D1FB0D58E9BD
results.csv         59BCC8BE6C0483161D7A2925BFC9501C9A885BCC55C1761D7BA3FA1905DAAC6D
args.yaml           8D2873C477E7DE7161983D115D5597DB883F66F4205F2C0FD8693BF565AB3723
experiment_manifest F4B2AB505CE7BE54A28FDB728882C4C36224599CEA98D2BDD7BF3F59A066EEA4
```

### 正式训练门禁判断

1. 1 epoch 与10 epoch 的首轮完全复现；
2. 全部主要训练 Loss 明显下降；
3. Box/Mask mAP 总体明显上升且彼此量级一致；
4. 无 OOM、NaN/Inf、EMA、保存、验证或 Mask 分支异常；
5. 模型源码与训练参数不再修改；
6. **允许用户手动启动400 epoch正式 V1-CBAM-b4 实验。**

正式 run 才能写入指标对比表，10 epoch 指标仍仅用于门禁。
