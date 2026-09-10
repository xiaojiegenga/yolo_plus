# 全 C3k2-DRB 与 P3/P4 SCSA：原理与实现

日期：2026-09-10。分支：`feature/data-v2-abl-drb-scsa`。
Run ID：`data-v2-abl-drb-scsa-p34-b16-s42`。训练命令见 [实验步骤](../实验步骤.md)。

## 1. 本次具体改什么

本轮是一个完整结构候选：**全部 C3k2 内部使用 DRB，在骨干 P3、P4 输出后分别加入独立 SCSA**。
DRB 负责改变卷积特征提取；SCSA 负责根据输入特征调整空间和通道响应。
SCSA 不写进 C3k2 内部，模型 YAML 中能直接看到两个独立模块。

```text
输入
 ↓
Backbone：P2 C3k2-DRB
 ↓
Backbone：P3 C3k2-DRB → 残差 SCSA → P3 跳连送到 Neck
 ↓
Backbone：P4 C3k2-DRB → 残差 SCSA → P4 跳连送到 Neck
 ↓
Backbone：P5 C3k2-DRB → SPPF → C2PSA
 ↓
Neck：上采样、拼接、C3k2-DRB、多尺度融合
 ↓
Head：原始 Segment26，原始 Proto
```

Backbone 是从图像提取特征的骨干；Neck 将不同尺度的特征融合；Head 生成检测框、类别和实例掩膜。
Ultralytics YAML 的 `head:` 字段同时包含 Neck 和最终输出头，不代表里面每一层都是预测层。

## 2. 第 4、6 层到底在哪

这里的“第 4、6 层”始终指 **Baseline 从 0 开始的层号**。加入新层以后，后面的数字会移动。

| 功能 | Baseline 层号 | 新模型层号 | 新模型操作 |
|---|---:|---:|---|
| Backbone P2 | 2 | 2 | C3k2 → C3k2DRB |
| Backbone P3 | 4 | 4 | C3k2 → C3k2DRB |
| P3 注意力 | 无 | 5 | ZeroInitResidualSCSA，输出 512×80×80 |
| Backbone P4 | 6 | 7 | C3k2 → C3k2DRB |
| P4 注意力 | 无 | 8 | ZeroInitResidualSCSA，输出 512×40×40 |
| Backbone P5 | 8 | 10 | C3k2 → C3k2DRB |
| SPPF / C2PSA | 9 / 10 | 11 / 12 | 保留 |
| Neck 四处 C3k2 | 13 / 16 / 19 / 22 | 15 / 18 / 21 / 24 | 全部使用 C3k2DRB，末级 PSA 保留 |
| Segment26 | 23 | 25 | 输入层改为 18 / 21 / 24，输出结构不变 |

空间尺寸按 640×640 输入给出。新第 14、17 层 Concat 分别接收注意力后的第 8、5 层，
因此 Neck 使用的是经过 SCSA 的 P4、P3 特征，而不是绕过注意力的旧特征。

## 3. C3k2 内部为什么能换成 DRB

C3k2 可以理解为“分流—一部分深入处理—拼接—融合”。外部的 1×1 卷积负责调整和混合通道，
内部 Bottleneck 负责局部特征提取。m 尺度还会使用嵌套的 C3k，不能只改表面的一层卷积。

`C3k2DRB` 继承原 C3k2，遍历内部模块，把 Bottleneck **整体替换**成输入输出通道相同的 DRB。
外部 CSP 分流与拼接、C3k 的投影卷积、最后一级原有 PSABlock 均保留。
本轮 8 个 C3k2DRB 内共有 15 个 DRB；DRB 本身不另外添加激活函数或 Bottleneck 的内部捷径。

普通卷积会混合输入通道。DRB 中用的是 depthwise convolution（逐通道卷积）：每个通道独立
观察空间邻域，通道间融合仍由外面的投影卷积完成。这样能扩大观察范围，同时控制参数量。

## 4. DRB：训练时多分支，推理时一个卷积

采用 UniRepLKNet 作者实现中的 K=9 分支设置：

| 分支 | 卷积核 k | 空洞率 d | 覆盖宽度 (k−1)d+1 |
|---|---:|---:|---:|
| 主分支 | 9 | 1 | 9 |
| 辅助 1 | 5 | 1 | 5 |
| 辅助 2 | 5 | 2 | 9 |
| 辅助 3 | 3 | 3 | 7 |
| 辅助 4 | 3 | 4 | 9 |

空洞率表示卷积采样点的间隔。3×3、d=4 的卷积覆盖 9×9 范围，但只有 9 个采样点；
它与密集 9×9 卷积并不相同。不同分支输出经过各自 BatchNorm，再相加：

`DRB(X) = BN₀(DWConv₉(X)) + Σ BNᵢ(DWConv(kᵢ,dᵢ)(X))`

推理时 BN 使用固定的运行均值和方差，可以合并到卷积中。对无 bias 的分支：

`W' = γW / sqrt(σ² + ε)`

`b' = β_BN − γμ / sqrt(σ² + ε)`

把空洞卷积的核按采样间隔填入 9×9 网格，其他位置补零，再将各分支 W'、b' 分别相加，
就得到一个等价的深度 9×9 卷积。这叫重参数化：训练与部署形式不同，融合前后的推理结果应相同。
该合并只适用于 BN 使用运行统计的推理状态，不是把训练状态的多分支随时删掉。

代码在 `DilatedReparamBlock.fuse()` 中完成合并，由模型正常 `fuse()` 流程调用。
理论和分支来源：[UniRepLKNet 作者源码](https://github.com/AILab-CVC/UniRepLKNet/blob/main/unireplknet.py)。

## 5. SCSA：先看空间，再看通道关系

SCSA 的顺序是空间注意力 → 通道自注意力。

### 5.1 空间注意力

输入为 `[B,C,H,W]`，分别沿宽、高求平均，得到高方向 `[B,C,H]` 和宽方向 `[B,C,W]` 描述。
将通道分成 4 组，各用 3、5、7、9 长度的一维逐通道卷积，随后拼接、GroupNorm、Sigmoid，
得到高、宽两个方向的权重。两条方向分支共享同一组卷积参数。

`X_s = X × A_h × A_w`

不同长度的卷积能表示不同范围的方向纹理。对细长叶片损伤，这是值得检验的机制假设，
但不能仅凭形状就认定最终分割精度会上升。

### 5.2 通道自注意力

将 X_s 用 7×7、stride=7 平均池化压缩，再做 GroupNorm 和逐通道 1×1 投影，得到 Q、K、V。
压缩后空间位置展平为 N，本轮采用单头，Q/K/V 形状为 `[B,C,N]`：

`A = softmax(QKᵀ / sqrt(C))`

`A_c = sigmoid(mean_spatial(AV))`

`SCSA(X) = X_s × A_c`

这里 QKᵀ 得到的是 C×C 通道关系，而非 H×W 个像素两两之间的关系。
缩放系数遵循作者实现的单头设置。矩阵乘法和归约使用 FP32，兼容外部 AMP 与半精度推理，
最后将门控值转回特征的数据类型。
机制来源：[SCSA 作者源码](https://github.com/HZAI-ZJNU/SCSA/blob/main/mmpretrain/models/attentions/SCSA.py)、
[发表版本](https://xinzhongzhu.github.io/document/1-s2.0-S0925231225005387-main.pdf)。

### 5.3 为什么外面加一个 β

两个插入位置各有自己的可学习标量 β，初始值均为 0：

`Y = X + β × SCSA(X)`

初始时 Y=X，SCSA 不会立刻改变刚输入该模块的特征。β 随训练学习，之后逐渐引入注意力。
第一次反向传播中，β 可以有梯度，而 SCSA 内部参数的梯度为 0；β 更新为非零后，内部参数
才开始获得有效梯度。这是接法本身的性质，不是梯度丢失。

该残差外壳是本项目的集成方式，不是 SCSA 论文已经验证的 YOLO26 结构。
此外，DRB 已改变卷积结构，因此“注意力初始为恒等”不等于“整个新模型初始等于 Baseline”。

## 6. 预训练权重怎样加载

新模型先由结构 YAML 构建，再加载官方 `yolo26m-seg.pt`。
插入两个独立模块后层号移动，不能直接按数字加载，否则同形状权重也可能对应错误的位置。
模型 YAML 的 `baseline_layer_map` 把原 0–23 层对应到新模型：

`[0,1,2,3,4,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]`

`BaseModel.load()` 先转换层号，再按名称与形状加载。CSP 投影、未改动的骨干层、C2PSA、
末级 PSA 和兼容的输出头权重可以复用；被替换的 Bottleneck 不强行塞入 DRB，新 DRB/SCSA
按初始化规则训练。训练器随后将 80 类模型重建为 2 类时，不再重复进行层号偏移。

本地使用官方权重检查，80 类目标模型匹配 724/1210 个状态张量；该计数包含 BN 状态，
不是参数量占比。2 类训练头还会因类别数不同而丢弃不兼容的分类张量。
新模块构建前固定随机种子，训练配置同样固定 seed=42。

## 7. 改动文件与阅读顺序

| 文件 | 作用 |
|---|---|
| `ultralytics-main/ultralytics/nn/modules/drb_scsa.py` | DRB、重参数化、C3k2DRB、SCSA 和零初始化残差外壳 |
| `ultralytics-main/ultralytics/nn/modules/__init__.py` | 导出新模块名称 |
| `ultralytics-main/ultralytics/nn/tasks.py` | YAML 解析、层号权重迁移、推理融合 |
| `ultralytics-main/ultralytics/cfg/models/26/yolo26m-drb-scsa-seg.yaml` | 网络连接与层号映射 |
| `experiments/data-v2-abl-drb-scsa-p34-b16-s42.yaml` | 数据、官方预训练来源、冻结训练配方 |
| `scripts/train_yolo26_seg.py` | 读取顶层 pretrained、构建前固定种子 |
| `scripts/check_drb_scsa.py` | 结构与数值检查，不运行数据集训练 |

初学时先对照第 2 节阅读模型 YAML，再读模块 forward，最后看权重迁移和 fuse。
上游 Apache-2.0 许可保存在 `ultralytics-main/licenses/DRB-SCSA-Apache-2.0.txt`。

## 8. 检查结果和实验判断

本地环境为 PyTorch 2.11.0+cpu，7 项测试通过：DRB 融合等价、残差初始化与梯度、640 输入
结构尺寸、官方预训练迁移及 2 类重建、含目标/空目标的分割反向、保存加载与整网融合、配方一致性。

2 类模型构建参数量为 23,955,496；融合后为 20,372,628。THOP 口径约 109.668 GFLOPs@640，
其中函数式注意力矩阵运算未被完整计入，该值不能当成全部计算量，也不能推导 5090 实际速度。

本轮只回答“完整 DRB+双位置 SCSA 候选是否比 Baseline 好”。按同一官方 best.pt 选择方式，
比较主指标 Mask mAP50-95，同时报告 mAP50、分类别结果、时长与显存。单次 seed=42 的正收益
可以支持继续验证该组合，但不能说明两个组成部分各自贡献，也不能证明跨随机种子的稳定提升。

文献中的 C3k2-DRB 为该结构替换提供参考，YOLO26m-seg 的本轮连接和残差策略是本项目方案：
[DSDC-YOLO 期刊原文](https://journal.scau.edu.cn/cn/article/doi/10.7671/j.issn.1001-411X.202512021)。

### 尚待云端完成

本地没有 CUDA，尚未验证 RTX 5090 上的 AMP、FP16、batch=16 显存与实际训练性能。
按 `实验步骤.md` 先完成 CUDA 检查和 10 epoch 预检，再启动正式训练；当前无精度结果。
