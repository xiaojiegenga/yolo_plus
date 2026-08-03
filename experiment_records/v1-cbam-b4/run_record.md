# V1-CBAM-b4 严格配对实验记录

## 当前状态

- 状态：`ready_for_user_preflight`
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

