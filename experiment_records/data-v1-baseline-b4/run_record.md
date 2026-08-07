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
- [x] 用户手动启动 400 epoch 正式训练；
- [x] 训练结束后重新加载 `best.pt` 完成 `split=val` 评估；
- [x] 回填正式指标、权重哈希与训练结论。

## 代码与模型身份

| 项目 | 记录 |
|---|---|
| Git 分支 | `codex/data-v1-baseline-b4` |
| 分支起点 | `codex/baseline-b4` |
| 起点 commit | `55fb63d5347439044824348cd7e4db40fd80f4a6` |
| 正式训练准备 commit | `88532a5d5c2a7698730e939d9c8ad66069e2db52` |
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
Git commit       : 88532a5d5c2a7698730e939d9c8ad66069e2db52
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

## 2026-08-07：正式训练完成

### 结论先行

本次运行通过数据、参数、日志和权重完整性检查，可以作为后续 **data-v1 / batch=4** 源码改进实验的
唯一严格 Baseline。模型没有发生训练崩溃，EarlyStopping 在 387 epoch 正常结束，最佳结果出现在
epoch 287；无需继续训练，也不应使用性能已经回落的 `last.pt`。

与历史 data-v0 Baseline-b4 做描述性观察时，Overall Mask mAP50 从 `0.656` 变为 `0.657`，没有出现
此前担心的“大幅下降”。但是两类目标变化方向相反：卷叶螟明显提高，钻心虫明显下降。因此，Overall
均值接近不变不代表两个类别都没有变化。

> 重要边界：data-v0 与 data-v1 的划分、标注、负样本和 Val 构成均不同。下面的跨版本差值只能描述
> 数据版本变化后的结果，不能把差值单独归因于“消除泄漏”或某一个数据处理步骤。

### 运行身份

| 项目 | 记录 |
|---|---|
| Run 目录 | `runs/segment/runs_seg/yolo26m_data_v1_baseline_b4_seg_20260807_022722` |
| Git branch / commit | `codex/data-v1-baseline-b4` / `f01b155cd1ec06f7c042c38bb5368e4c3c91074c` |
| Run kind | `formal` |
| Paired comparison group | `data-v1-batch4` |
| 计划 / 实际 epoch | 400 / 387（EarlyStopping） |
| Best epoch | 287 |
| Patience | 100 |
| 总训练时间 | 3.918 h |
| 平均 / 中位 epoch 时间 | 36.45 / 36.31 s |
| Fused Params | 23,509,010（23.509 M） |
| FLOPs | 121.2 G |
| 推理时间 | 7.5 ms/image |
| `best.pt` 大小 | 54.53 MB（十进制）/ 52.01 MiB |

### 产物校验

| 产物 | SHA256 |
|---|---|
| `best.pt` | `1223C09F5D9D46321E296BE0A5402C3B3FCF49857E9E2056C80CAF906CB95448` |
| `last.pt` | `AB07BAB3B8FCAEA54116A9818D1544EC8FA418E5E05D0B5C9F64169F911D9BB5` |
| `results.csv` | `146F851D500E61677CCAD8327303930E5FE59257D4E99E81DBF6DC91C2C7168C` |
| `args.yaml` | `FAA352CB25E07428FD4CD773D079CDC2769091AAAA2EF20A61BF5C1AD3590630` |
| `experiment_manifest.json` | `E446B119E3DA6453072B4B9283AE99B471A13D05A2BB163CF8074C4AD6A63865` |

完整性检查结果：

- manifest 中的分支、commit、数据指纹、profile、batch 和 run kind 均与本记录一致；
- `results.csv` 包含 epoch 1～387，共 387 行，无缺失 epoch；
- `results.csv` 所有数值字段均无 NaN/Inf；
- `best.pt` 与 `last.pt` 各检查 904 个状态张量，非有限张量数均为 0；
- 训练和验证 loss 整体平稳下降，Box/Mask 指标正常上升并在约 250～300 epoch 进入平台；
- epoch 287 的 CSV fitness（Box mAP50-95 + Mask mAP50-95）最高，为 `0.70785`；
- epoch 387 的 Mask mAP50/mAP50-95 已回落到约 `0.581/0.287`，因此必须使用 `best.pt`。

### `best.pt` 正式 Val 指标

| 类别 | Images | Instances | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask F1¹ | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 93 | 409 | 0.722 | 0.594 | 0.661 | 0.399 | 0.717 | 0.586 | 0.645 | 0.657 | 0.310 |
| Rice leaffolder | 72 | 353 | 0.660 | 0.598 | 0.657 | 0.386 | 0.671 | 0.600 | 0.634 | 0.670 | 0.302 |
| Rice stemborers | 27 | 56 | 0.785 | 0.589 | 0.665 | 0.413 | 0.763 | 0.571 | 0.653 | 0.644 | 0.319 |

¹ Mask F1 由该表显示的 `2PR/(P+R)` 计算，因 P/R 已四舍五入，所以只作为辅助读数。Ultralytics
Mask F1 曲线显示全部类别的最大 F1 约为 `0.64 @ confidence=0.423`，与表中计算值一致。

机器可读指标另存为 `val_metrics.csv`。

### 可视化诊断

- Ultralytics 分割验证生成的归一化混淆矩阵主要依据**检测框与类别匹配**，不是 Mask IoU 混淆矩阵；
  在其绘图阈值下，卷叶螟和钻心虫对角项约为 `0.75/0.66`，被判为背景的比例约为 `0.25/0.30`。
  它说明检测层面的主要错误仍是漏检，而不是两个虫种彼此混淆，但不能把这两个比例直接写成 Mask 漏检率；
- 钻心虫被误判为卷叶螟约 `0.04`，卷叶螟被误判为钻心虫接近 0；
- 背景误检中大部分预测类别为卷叶螟。Val 预测拼图也能看到较多低置信度卷叶螟候选，因此卷叶螟仍存在
  “提高召回会带来更多假阳性”的阈值权衡；
- 钻心虫 PR 曲线比卷叶螟更呈阶梯状，和 Val 中只有 56 个钻心虫实例有关。少量样本即可使指标明显波动，
  所以不能仅凭一次运行断言钻心虫下降的唯一原因。

### 与历史 data-v0 Baseline-b4 的描述性对照

#### Mask 指标

| 类别 / 指标 | data-v0 | data-v1 | data-v1 - data-v0 |
|---|---:|---:|---:|
| Overall P | 0.657 | 0.717 | +0.060 |
| Overall R | 0.609 | 0.586 | -0.023 |
| Overall mAP50 | 0.656 | 0.657 | +0.001 |
| Overall mAP50-95 | 0.322 | 0.310 | -0.012 |
| Leaffolder P | 0.595 | 0.671 | +0.076 |
| Leaffolder R | 0.523 | 0.600 | **+0.077** |
| Leaffolder mAP50 | 0.574 | 0.670 | **+0.096** |
| Leaffolder mAP50-95 | 0.268 | 0.302 | **+0.034** |
| Stemborers P | 0.719 | 0.763 | +0.044 |
| Stemborers R | 0.694 | 0.571 | **-0.123** |
| Stemborers mAP50 | 0.737 | 0.644 | **-0.093** |
| Stemborers mAP50-95 | 0.377 | 0.319 | **-0.058** |

#### Box 指标

| 类别 / 指标 | data-v0 | data-v1 | data-v1 - data-v0 |
|---|---:|---:|---:|
| Overall P / R | 0.648 / 0.611 | 0.722 / 0.594 | +0.074 / -0.017 |
| Overall mAP50 / mAP50-95 | 0.656 / 0.418 | 0.661 / 0.399 | +0.005 / -0.019 |
| Leaffolder P / R | 0.581 / 0.528 | 0.660 / 0.598 | +0.079 / +0.070 |
| Leaffolder mAP50 / mAP50-95 | 0.589 / 0.335 | 0.657 / 0.386 | +0.068 / +0.051 |
| Stemborers P / R | 0.715 / 0.694 | 0.785 / 0.589 | +0.070 / -0.105 |
| Stemborers mAP50 / mAP50-95 | 0.723 / 0.502 | 0.665 / 0.413 | -0.058 / -0.089 |

两个 Val 集并非同一批样本：data-v0 为 95 张、330 个实例（卷叶螟 281、钻心虫 49），data-v1
为 93 张、409 个实例（卷叶螟 353、钻心虫 56）。此外，data-v1 同时改变了原图级划分、标注一致性、
负样本和样本组成。因此该表不能作为单变量消融实验，尤其不能写成“消除泄漏使卷叶螟提高 0.096”。

### 分析判断与后续边界

1. **data-v1 Baseline 已建立成功。** 后续 data-v1 的 V1/V2/V3 或新模块必须使用相同数据指纹、
   profile 和 `batch=4`，只和本 Baseline 严格比较；
2. **消除原图泄漏没有伴随 Overall Mask mAP50 崩塌。** 当前 `0.657` 与历史 `0.656` 基本相同，
   但这只是结果描述，不是泄漏影响的因果估计；
3. **卷叶螟是本次最明确的正向结果。** Mask Recall 达到 `0.600`，比历史 data-v0 Baseline 的
   `0.523` 更高；这支持继续在 data-v1 上做模型改进；
4. **钻心虫成为新的短板。** Mask Recall 为 `0.571`。可能原因包括无泄漏的原图级划分带来的场景
   泛化难度、Val 只有 9 个钻心虫原图组/56 个实例，以及 data-v1 的样本组成变化；这些目前是待验证
   假设，不是已确认原因；
5. 在改源码前，应先对本次 Val 中共同漏检和假阳性的钻心虫样本做一次错误清单，确认是否集中在特定
   原图、遮挡、尺度或标注边界；
6. 仍不使用 test 调参。结构与训练方案冻结后，才对独立 test 做一次最终评估；
7. 若按既定路线继续，优先在同一 data-v1 上复测 V3-Dice 是合理的，因为 Overall Mask mAP50-95
   仍只有 `0.310`；但正式训练仍必须由用户手动启动。

### 证据置信度

| 判断 | 置信度 |
|---|---|
| 本次运行身份正确、日志和权重完整 | 高 |
| 本结果可作为 data-v1 后续实验的严格 Baseline | 高 |
| data-v1 与 data-v0 的指标差值真实存在 | 高 |
| 差值由消除原图泄漏单独导致 | 不可判断（多项数据因素同时变化） |
| 钻心虫下降来自场景分布变化或样本量较小 | 中等，仅为待验证假设 |
