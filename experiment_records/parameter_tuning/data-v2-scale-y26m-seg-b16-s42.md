# RTX 5090 参数优化基线分析：data-v2-scale-y26m-seg-b16-s42

## 技术摘要

- 本次 Run 训练过程正常：`results.csv` 包含连续的 epoch 1～292，所有数值均为有限值，未出现 NaN、Inf 或 OOM。
- 训练在 epoch 192 同时取得最高 Val Mask mAP50-95 `0.34685` 和最高综合 fitness `0.78033`，因此 Ultralytics 保存的 `best.pt` 与本项目主指标一致。
- `patience=100` 在 epoch 292 正常触发 EarlyStopping。后期没有性能崩溃，但主指标进入平台期；不建议增加 patience。
- 计划值 `epochs=400` 与实际早停位置不匹配：线性学习率在停止时仍为约 `0.000466`，并且 `close_mosaic=15` 原计划到 epoch 386 才生效，本次训练未进入该阶段。
- 本次 `0.34685` 低于历史 data-v2-b8 的 `0.37344`，差值为 `-0.02659`。两次训练同时改变了 batch、旋转/翻转增强、warmup 和运行环境，不能把差值单独归因于 batch。
- 下一轮优先做“训练日程对齐”，随后恢复历史有效的航拍方向增强，再测试分辨率和掩膜下采样；batch 32 放在精度方案基本稳定后做吞吐量对比。

本 Run 的用途是确定 `云服务器实验设计与记录表.md` 表 1 的正式训练参数，属于参数优化实验，不作为期刊对比实验，不写入 `experiment_records/comparison.csv`。

## 实验范围与证据

| 项目 | 值 |
|---|---|
| Run ID | `data-v2-scale-y26m-seg-b16-s42` |
| 任务 | 无人机航拍水稻害虫实例分割 |
| 数据 | `rice-pest-data-v2` |
| 选择 split | Val（117 images / 557 instances） |
| 主指标 | Val Mask mAP50-95 |
| Test | 未使用 |
| 原始 Run | [`runs/data-v2-scale-y26m-seg-b16-s42/`](../../runs/data-v2-scale-y26m-seg-b16-s42/) |
| 实际参数 | [`args.yaml`](../../runs/data-v2-scale-y26m-seg-b16-s42/args.yaml) |
| 逐 epoch 指标 | [`results.csv`](../../runs/data-v2-scale-y26m-seg-b16-s42/results.csv) |
| 训练曲线 | [`results.png`](../../runs/data-v2-scale-y26m-seg-b16-s42/results.png) |
| 最佳权重 | `runs/data-v2-scale-y26m-seg-b16-s42/weights/best.pt` |

## 本次实际训练参数

| 参数 | 实际值 | 说明 |
|---|---:|---|
| model | `yolo26m-seg.pt` | 官方预训练权重 |
| epochs | 400 | 最大轮次，实际未跑满 |
| patience | 100 | epoch 292 触发早停 |
| batch | 16 | 每轮 59 个物理 batch |
| nbs / accumulate | 64 / 4 | 有效名义 batch 约为 64 |
| imgsz | 640 | 当前空间分辨率基线 |
| optimizer | `auto` → AdamW | 源码按 6000 个计划 iteration 自动选择 AdamW |
| 实际初始 lr / momentum | 约 `0.001667` / `0.9` | `optimizer=auto` 会忽略 args 中显示的 `lr0=0.01`、`momentum=0.937` |
| lr schedule | linear，`lrf=0.01` | `cos_lr=false` |
| warmup_epochs | 3 | 历史 data-v2-b8 为 5 |
| workers / cache | 8 / false | 仅影响加载与吞吐量 |
| seed / deterministic | 42 / true | 保持 |
| amp | true | 保持 |
| mask_ratio | 4 | 后续小目标掩膜优化候选 |
| mosaic / mixup / copy_paste | 1.0 / 0.1 / 0.3 | 保持 |
| degrees / flipud / fliplr | 0 / 0 / 0.5 | 历史 data-v2-b8 为 15 / 0.5 / 0.5 |
| close_mosaic | 15 | 按 400 epoch 计划应从 epoch 386 起关闭，本次未触发 |

## 核心结果

| 指标 | best.pt / epoch 192 | last.pt / epoch 292 | 变化 |
|---|---:|---:|---:|
| Box mAP50-95 | 0.43348 | 0.42266 | -0.01082 |
| Mask P | 0.67711 | 0.62640 | -0.05071 |
| Mask R | 0.63382 | 0.62492 | -0.00890 |
| Mask mAP50 | 0.68506 | 0.69184 | +0.00678 |
| **Mask mAP50-95** | **0.34685** | **0.33548** | **-0.01137** |
| fitness（Box + Mask mAP50-95） | 0.78033 | 0.75813 | -0.02220 |

| 运行项 | 结果 |
|---|---:|
| 计划 / 实际 epoch | 400 / 292 |
| Best epoch | 192 |
| 总训练时间 | 4133.97 s（1.148 h） |
| epoch 10 后中位单轮时间 | 13.666 s |
| 训练显存 | 约 13.3 GB |
| 参数量 / GFLOPs | 23.509 M / 121.2 |

## 收敛与稳定性诊断

### 主指标在 epoch 192 达峰，之后进入波动平台

| epoch 区间 | Mask mAP50-95 均值 | 标准差 | 区间最高值 |
|---|---:|---:|---:|
| 1–50 | 0.21601 | 0.05120 | 0.28929 |
| 51–100 | 0.28341 | 0.02513 | 0.33805 |
| 101–150 | 0.30813 | 0.01574 | 0.34304 |
| 151–200 | 0.31992 | 0.01354 | **0.34685** |
| 201–250 | 0.31900 | 0.00905 | 0.33949 |
| 251–292 | 0.32555 | 0.00825 | 0.34433 |

训练后期的均值没有持续崩塌，epoch 290 仍达到 `0.34433`，因此证据更符合“验证指标平台期伴随小样本波动”，而不是严重过拟合。训练损失持续下降，而 Val segmentation loss 在约 `1.18～1.21` 附近波动，说明继续拟合训练集没有稳定转化为更高的 Val Mask mAP50-95。

![训练与验证曲线](../../runs/data-v2-scale-y26m-seg-b16-s42/results.png)

### best.pt 的选择口径正确

Ultralytics 分割任务的 fitness 为 Box mAP50-95 与 Mask mAP50-95 之和。本 Run 的最高 fitness 和最高 Mask mAP50-95 都出现在 epoch 192，不存在“综合 fitness 选出的 best.pt 偏离项目主指标”的问题。

### 主要误差是漏检和掩膜定位，而不是类别互相混淆

归一化混淆矩阵显示，两类真实实例约有 28%～29% 落入 background；两类之间几乎没有直接互相混淆。最终 Mask PR 曲线的 mAP@0.5 为：Rice leaffolder `0.649`、Rice stemborers `0.718`。这说明后续更值得优先验证输入分辨率、掩膜分辨率及航拍方向增强，而不是先调整类别损失权重。

![归一化混淆矩阵](../../runs/data-v2-scale-y26m-seg-b16-s42/confusion_matrix_normalized.png)

## 与历史 data-v2-b8 的描述性比较

| 指标 | 当前云端 b16 | 历史 data-v2-b8 | 当前 - 历史 |
|---|---:|---:|---:|
| Mask P | 0.677 | 0.694 | -0.017 |
| Mask R | 0.634 | 0.605 | +0.029 |
| Mask mAP50 | 0.685 | 0.708 | -0.023 |
| Mask mAP50-95 | 0.347 | 0.374 | -0.027 |
| Box mAP50-95 | 0.433 | 0.451 | -0.018 |
| 训练时间 / h | 1.148 | 4.216 | -3.068（约快 3.67 倍） |

该比较只能用于提出优化假设，不能视为严格消融。除云端硬件外，已确认存在以下训练参数差异：

| 参数 | 当前云端 Run | 历史 data-v2-b8 |
|---|---:|---:|
| batch | 16 | 8 |
| workers | 8 | 4 |
| warmup_epochs | 3 | 5 |
| degrees | 0 | 15 |
| flipud | 0 | 0.5 |

## 参数优化优先级

| 优先级 | 参数组 | 建议 | 依据与控制方式 |
|---:|---|---|---|
| 1 | `epochs` / LR 尾段 / `close_mosaic` | 下一轮先只把 `epochs` 改为 300，保持 `patience=100`、`close_mosaic=15` | 使线性 LR 在 300 epoch 内完整衰减，并让关闭 mosaic 从 epoch 286 左右开始；其余参数不变，便于判断日程是否是平台期原因 |
| 2 | 航拍方向增强 | 在日程方案确定后测试 `degrees=15`、`flipud=0.5` | 航拍图像通常允许旋转/上下翻转；历史 data-v2-b8 使用该组合且结果更高，但当前证据不能单独确认因果 |
| 3 | warmup | 在同一增强方案下比较 `warmup_epochs=3` 与 `5` | 历史配置为 5；只影响训练早期，优先级低于方向增强 |
| 4 | `imgsz` | 测试 832，保持 batch=16、mask_ratio=4 | 害虫属于航拍小目标，5090 仍有较大显存余量；属于待实验验证的高潜力假设 |
| 5 | `mask_ratio` | 在最佳 imgsz 下测试 4 → 2 | 更高的掩膜训练分辨率可能改善边界与小实例，但会增加显存和计算量 |
| 6 | batch / cache | 精度方案稳定后比较 batch 16 与 32；可单独测试 RAM cache | batch 16 仅占约 13.3 GB；batch 32 主要用于吞吐量优化，不应与分辨率或增强同时改动 |
| 7 | optimizer / LR | 将当前实际 AdamW 参数显式写入配置，再单独比较 cosine LR 或其他 optimizer | `optimizer=auto` 会覆盖 YAML 中的 lr0 和 momentum；显式化能避免误读并提高可复现性 |

### 推荐的下一轮单变量实验

| 项目 | 建议值 |
|---|---|
| Run ID | `data-v2-tune-e300-b16-s42` |
| 唯一实质变化 | `epochs: 400 → 300` |
| 保持 | model、data、batch=16、imgsz=640、patience=100、seed=42、全部增强参数 |
| optimizer 记录 | 可显式写成与本 Run 等效的 `AdamW`、`lr0≈0.001667`、`momentum=0.9`；若要保持配置文本也完全单变量，则暂时继续 `auto` |
| 选择指标 | Val Mask mAP50-95 |
| Test | 不运行 |

若下一轮与本 Run 的 Mask mAP50-95 差距小于约 `0.005`，先视为接近，优先选择耗时更短且收敛更稳定的方案；最终冻结前仍需使用相同配置做多 seed 复验，不能把单次波动当作稳定提升。

## 暂不调整的参数

- `patience`：当前值 100 已正确工作，不增加到 300，也不关闭 EarlyStopping。
- `seed=42`、`deterministic=true`：参数筛选阶段保持，避免额外随机差异。
- `amp=true`、`workers=8`：训练稳定，先保持。
- `mosaic=1.0`、`mixup=0.1`、`copy_paste=0.3`：在日程、方向增强和空间分辨率确定前不同时改动。
- Test split：正式方案冻结前不使用。

## 限制与待补证据

- 当前只有 seed 42 单次结果，无法估计随机波动范围。
- 历史 b8 与当前云端 b16 不是严格配对实验，比较只用于生成假设。
- 原始 Run 没有逐 epoch 的显存峰值和平均 GPU 利用率，只能使用训练终端观察到的约 13.3 GB。
- 本地没有重新执行独立 Val；当前 best.pt 指标来自 checkpoint 中保存的 `train_metrics`，与 epoch 192 的 `results.csv` 一致。
- 在提高 imgsz 或降低 mask_ratio 前，最好补充目标像素面积分布；当前“小目标受益”仍是任务背景支持的推断。

## 后续问题

1. 300-epoch 日程能否在关闭 mosaic 和更低学习率阶段超过 `0.34685`？
2. 恢复 `degrees=15`、`flipud=0.5` 后，历史与当前结果差距能否缩小？
3. imgsz=832 与 mask_ratio=2 对卷叶螟的 Mask mAP50-95 是否有稳定提升？
4. batch 32 能否缩短训练时间且不降低主指标？
