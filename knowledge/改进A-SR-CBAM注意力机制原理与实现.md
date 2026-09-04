# 改进 A：A1 SR-CBAM 与 A2 ZR-CBAM 原理和实现

## 1. A1 的初始选择

改进 A1 固定为项目自定义的 **Selective Residual CBAM（SR-CBAM，选择性残差轻量
CBAM）**。正式消融只测试这一种注意力结构，不把普通 CBAM 与 SR-CBAM 串联，也不安排
两个版本同时进入组合实验。

选择 SR-CBAM 的依据是：旧 data-v0 严格对照中，四阶段普通 CBAM 增加约 85.4 万参数，
Mask mAP50 反而下降 0.009；P3/P4 的 SR-CBAM 只增加 65,734 个参数，Mask mAP50 略升
0.007。旧结果只能用于选择候选，不能当作 SR-CBAM 在当前 data-v2 上有效的结论。最终是否
保留改进 A，必须由新的正式 `100` Run 与 `000 Baseline` 比较决定。

只有论文要专门证明“残差轻量设计优于普通 CBAM”时，才需要增加普通 CBAM 对照。那时应
建立两个独立 Run，并固定相同插入位置、通道降维率和空间卷积核，使唯一差别只有残差混合。

## 2. A1 插入 Backbone 的准确位置

输入尺寸为 640 时，YOLO26m Backbone 的主要特征层如下：

| 层号 | 阶段 | 输出形状 | 本次处理 |
|---:|---|---|---|
| 2 | P2/4 | `[B, 256, 160, 160]` | 保持原 `C3k2` |
| 4 | P3/8 | `[B, 512, 80, 80]` | 改为 `C3k2SRCBAM` |
| 6 | P4/16 | `[B, 512, 40, 40]` | 改为 `C3k2SRCBAM` |
| 8 | P5/32 | `[B, 512, 20, 20]` | 保持原 `C3k2` |
| 10 | P5 深层语义 | `[B, 512, 20, 20]` | 保持原 `C2PSA` |

数据流可以简化为：

```text
输入图像
  ↓
P2 C3k2（保留高分辨率细节）
  ↓
P3 C3k2 → SR-CBAM ① ──────────────→ Neck 的 P3 融合
  ↓
P4 C3k2 → SR-CBAM ② ───────→ Neck 的 P4 融合
  ↓
P5 C3k2 → SPPF → C2PSA ─→ Neck 的 P5 融合
```

“选择性”指只选择 P3 和 P4 两个阶段：P3 保留了较高空间分辨率，适合关注小目标局部；P4
提供更大的感受野和上下文。P2 不加注意力，避免过早筛减细粒度信息；P5 已有 SPPF 和
C2PSA，不再重复增加模块。这是结构假设，不是性能保证。

## 3. 先理解普通 CBAM

设输入特征为：

```text
X ∈ R^(B×C×H×W)
```

- `B`：一次送入模型的图片数；
- `C`：特征通道数，可以理解为模型学习到的不同特征类型；
- `H、W`：特征图的高和宽。

CBAM 依次回答两个问题：

1. **Channel Attention：哪些通道更重要？**
2. **Spatial Attention：特征图上的哪些位置更重要？**

### 3.1 通道注意力

先分别对空间维度做全局平均池化和全局最大池化：

```text
z_avg = AvgPool(X)
z_max = MaxPool(X)
```

两者都从 `[B,C,H,W]` 压缩为 `[B,C,1,1]`。平均池化描述总体响应，最大池化保留最强
局部响应；对画面中占比很小的虫害，最大池化可以补充平均值容易稀释的信息。

两个描述进入同一套共享 MLP：

```text
M_c(X) = sigmoid(MLP(z_avg) + MLP(z_max))
X_c = X ⊙ M_c(X)
```

本项目 `C=512`、`reduction=16`，所以共享 MLP 的通道变化为：

```text
512 → 32 → 512
```

这里的“共享”表示平均池化和最大池化使用同一组卷积权重，不是各自训练一套参数。

### 3.2 空间注意力

再沿通道维计算平均值和最大值，得到两张 `[B,1,H,W]` 的图，拼接后用 `7×7` 卷积生成
一张空间权重图：

```text
M_s(X_c) = sigmoid(Conv7×7([Mean_c(X_c), Max_c(X_c)]))
X_a = X_c ⊙ M_s(X_c)
```

最终 `X_a` 的形状仍与 `X` 完全相同，因此模块可以接在保持通道数和分辨率不变的
Backbone 阶段之后。

## 4. 本项目为什么增加残差软融合

普通 CBAM 直接输出 `X_a`。两个 Sigmoid 门连续相乘时，刚插入且尚未训练的注意力可能
立即改变预训练特征。SR-CBAM 保留原特征，并只混入一小部分注意力结果：

```text
α = sigmoid(mix_logit)
Y = X + α × (X_a - X)
  = (1 - α) × X + α × X_a
```

每个 SR-CBAM 有一个独立的可学习标量 `α`，初值为 `0.1`。初始化时可以直观理解为：

```text
Y = 90% 原特征 + 10% 注意力特征
```

这样既保留大部分官方预训练特征，又让通道和空间注意力参数从第一次反向传播就获得梯度。
训练过程中，模型自行学习 P3、P4 分别需要多强的注意力。

这一设计受可学习残差门控思想启发，但不等同于原始 ReZero：ReZero 的残差权重通常从 0
开始，而本项目采用 Sigmoid 约束并从 0.1 开始，目标是对预训练 CNN 做温和改动。

## 5. 源码具体改了什么

| 文件 | 改动 |
|---|---|
| `ultralytics-main/ultralytics/nn/modules/conv.py` | 新增 `ResidualCBAM`，实现标准通道/空间注意力与残差软融合 |
| `ultralytics-main/ultralytics/nn/modules/block.py` | 新增继承 `C3k2` 的 `C3k2SRCBAM` |
| `ultralytics-main/ultralytics/nn/modules/__init__.py` | 导出两个新模块 |
| `ultralytics-main/ultralytics/nn/tasks.py` | 注册 `C3k2SRCBAM`，使模型解析器能从 YAML 构建它 |
| `ultralytics-main/ultralytics/cfg/models/26/yolo26m-srcbam-seg.yaml` | 只把 Backbone 第 4、6 层替换为包装模块 |
| `experiments/data-v2-abl-100-srcbam-b16-s42.yaml` | 改进 A 的唯一正式训练配置 |
| `scripts/train_yolo26_seg.py` | 支持从配置顶层读取 `pretrained` |
| `ultralytics-main/tests/test_srcbam_model.py` | 检查反向传播、层位、参数键和模型前向 |

`C3k2SRCBAM` 的核心逻辑只有两步：

```python
features = original_c3k2(x)
output = srcbam(features)
```

它采用继承包装，而不是在 YAML 中插入新的独立层。这样第 4、6 层仍是第 4、6 层，Neck
对 Backbone 的引用仍为 `4` 和 `6`，最终 `Segment26` 仍读取 `[16,19,22]`。

## 6. 为什么预训练权重仍能正确迁移

Ultralytics 按 state-dict 的“参数名相同且形状相同”迁移权重。若在 YAML 中额外插入层，
后续层号会整体变化，即使张量形状碰巧相同，也可能迁移到错误语义的层。

本次包装保留了所有 Baseline 参数名和形状。自动测试确认：

- Baseline 的全部参数键都存在于 SR-CBAM 模型且形状一致；
- 只新增 8 个 `srcbam` 状态张量，即每个 P3/P4 模块 4 个；
- Head 层号和输入索引没有改变。

自定义结构不能直接用 `YOLO("yolo26m-seg.pt")` 构造，所以配置采用：

```yaml
model: ultralytics-main/ultralytics/cfg/models/26/yolo26m-srcbam-seg.yaml
pretrained: yolo26m-seg.pt
```

训练入口先按自定义 YAML 建模，再调用 `model.load("yolo26m-seg.pt")`。干净云服务器没有本地
权重时，Ultralytics 会解析官方模型名并下载；已有权重时直接复用。

## 7. 复杂度变化

一个 512 通道 SR-CBAM 的参数量为：

```text
通道 MLP：512×32 + 32×512 = 32,768
空间卷积：2×1×7×7 = 98
混合标量：1
单个模块：32,867
两个模块：65,734
```

本地构建实测：

| 指标 | 000 Baseline | 100 SR-CBAM | 变化 |
|---|---:|---:|---:|
| fused Params | 23,509,010 | 23,574,744 | +65,734（约 +0.28%） |
| GFLOPs@640 | 121.171149 | 121.286586 | +0.115437（约 +0.10%） |

这只能说明结构开销较小，不能代替云端的显存、训练时间和最终精度实测。

## 8. 分支与实验隔离

当前改进 A 分支为：

```text
feature/data-v2-abl-a-attention
```

A、B、C 在组合前应是从同一正式 Baseline 基点派生的兄弟分支，不应让 B 从 A 的源码继续
开发。这样单模块 `100/010/001` 才能分别只相对 `000` 改变一个因素。组合阶段再建立单独
分支，合入已经通过正式结果门控的模块。

本次正式 Run ID 固定为：

```text
data-v2-abl-100-srcbam-b16-s42
```

## 9. A1 本地验证与云端运行

### 9.1 本地只检查，不训练

```powershell
$env:PYTHONPATH = (Resolve-Path "ultralytics-main").Path
python -m pytest "ultralytics-main/tests/test_srcbam_model.py" -q

python scripts/train_yolo26_seg.py `
  --config experiments/data-v2-abl-100-srcbam-b16-s42.yaml `
  --run-name "data-v2-abl-100-srcbam-b16-s42" `
  --dry-run
```

dry-run 输出必须同时包含自定义模型 YAML 和：

```text
[PRETRAINED] yolo26m-seg.pt
```

### 9.2 推送分支后，云端拉取

```bash
cd /root/yolo_plus
git fetch origin
git switch --track origin/feature/data-v2-abl-a-attention
git pull --ff-only
git branch --show-current
```

### 9.3 用户手动运行 10 epoch 预检

```bash
python scripts/cloud_train_data_v2.py \
  --config experiments/data-v2-abl-100-srcbam-b16-s42.yaml \
  --run-name "data-v2-abl-100-srcbam-b16-s42-preflight10" \
  --preflight10
```

预检只确认预训练权重迁移、前向/反向、Val、显存和保存流程。它不进入正式表格，也不修改
`comparison.csv`。应在日志中确认模型包含两个 `C3k2SRCBAM`，并看到官方预训练权重成功
迁移，而不是从随机初始化直接训练。

### 9.4 预检通过后，由用户手动启动正式训练

```bash
nohup python scripts/cloud_train_data_v2.py \
  --config experiments/data-v2-abl-100-srcbam-b16-s42.yaml \
  --run-name "data-v2-abl-100-srcbam-b16-s42" \
  > train_data-v2-abl-100-srcbam-b16-s42.log 2>&1 &

tail -f train_data-v2-abl-100-srcbam-b16-s42.log
```

正式训练仍为 300 epoch、batch=16、seed=42，训练参数字典与 `000 Baseline` 完全一致。
训练完成后按脚本打印的打包和 SCP 命令回传，再建立正式 Run 记录并填写总表。

## 10. 结果应该怎样判断

必须与 `data-v2-abl-000-y26m-b16-s42` 的同一 Val 口径比较：

- best.pt 仍由 Ultralytics 官方分割 fitness 选择；
- 论文主指标按当前项目约定看 Val Mask mAP50-95，同时报告 Mask mAP50；
- 同时观察卷叶螟的 Mask Recall、mAP50 和 mAP50-95，判断注意力是否真正帮助目标类别；
- 对比 Params、GFLOPs、峰值显存和训练时间，确认收益是否值得增加结构；
- Test 在最终方案冻结前不使用。

如果总体与目标类别指标都下降，就把 A 记录为负结果并停止组合；如果总体接近持平但卷叶螟
专项指标有清晰改善，可以保留为组合候选。不能在看到正式结果后再修改 `reduction`、核大小、
插入位置或混合初值并仍称为同一个正式 A。

## 11. A1 正式结果与 A2 迭代

A1 的 data-v2 正式训练已经完成：

| 指标 | 000 Baseline | A1 P3/P4 SR-CBAM | 变化 |
|---|---:|---:|---:|
| Official fitness | 0.81977 | 0.80644 | -0.01333 |
| Mask mAP50 | 0.71132 | 0.70884 | -0.00248 |
| Mask mAP50-95 | 0.36348 | 0.35097 | -0.01251 |

A1 没有通过门控。其 `best.pt` 中 P3/P4 的 `alpha` 分别为 0.16722 和 0.09860：P3 的
混合强度发生了变化，P4 几乎停留在初值，但总体严格掩膜 AP 仍然下降。因此 A2 只保留
更有学习迹象、分辨率也更高的 P3 注意力，并改变残差形式。

A1 的软插值为：

```text
Y = X + alpha × (CBAM(X) - X)
```

因为 CBAM 输出已经经过两次 0～1 权重相乘，该公式只能在原特征 `X` 与被抑制后的特征之间
插值。A2 改为零初始化的加法残差：

```text
X_att = CBAM(X)
Y = X + beta × X_att
beta 初始值 = 0
```

`beta=0` 时，A2 的输出严格等于 `X`，因此刚开始训练时不会改变已经迁移的 Baseline 特征。
`beta` 不经过 sigmoid，训练后既能取正值以增强注意力特征，也能取负值进行反向修正。
这借鉴的是零初始化残差分支的优化思想，并不意味着 A2 尚未训练就必然优于 A1。

A2 只改变以下内容：

| 文件 | A2 改动 |
|---|---|
| `ultralytics-main/ultralytics/nn/modules/conv.py` | 新增 `ZeroInitResidualCBAM` |
| `ultralytics-main/ultralytics/nn/modules/block.py` | 新增包装器 `C3k2ZRCBAM` |
| `ultralytics-main/ultralytics/nn/tasks.py` | 注册新模块供模型 YAML 解析 |
| `ultralytics-main/ultralytics/cfg/models/26/yolo26m-p3-zrcbam-seg.yaml` | 仅在 Backbone 第 4 层 P3/8 使用 A2 |
| `experiments/data-v2-abl-a2-p3-zrcbam-b16-s42.yaml` | 新的正式 Run ID，训练参数继承 `000` |

A2 的 fused Params 为 23,541,877，较 Baseline 增加 32,867；GFLOPs@640 为
121.231795，增加 0.060646。A2 使用新的 Run ID
`data-v2-abl-a2-p3-zrcbam-b16-s42`，不会改写 A1 的实验身份或负结果。具体云端命令以
根目录 `实验步骤.md` 为准。

## 参考资料

- Woo 等，CBAM 原论文：[Convolutional Block Attention Module，ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper)
- Bachlechner 等，残差零初始化思想：[ReZero is All You Need](https://proceedings.mlr.press/v161/bachlechner21a.html)
