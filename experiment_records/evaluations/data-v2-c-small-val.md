# data-v2：C 与 Baseline 的小目标 Val 配对评估

正式权重固定为 000 best epoch 216 与 C best epoch 282。Val 有 117 图、557 实例；
其中卷叶螟 462 个，按冻结的 640 输入后 bbox 面积 <1024 像素口径分成小目标 174 个、非小目标 288 个。

## 结果

| 子集（实例数） | 指标 | 000 | C | C − 000 |
|---|---|---:|---:|---:|
| 小卷叶螟（174） | Mask AP50 | 0.55612 | 0.51168 | -0.04444 |
| 小卷叶螟（174） | Mask AP50-95 | 0.19311 | 0.17347 | -0.01964 |
| 小卷叶螟（174） | Mask Precision@IoU0.50/conf0.25 | 0.46522 | 0.42912 | -0.03610 |
| 小卷叶螟（174） | Mask Recall@IoU0.50/conf0.25 | 0.61494 | 0.64368 | +0.02874 |
| 小卷叶螟（174） | Mask AR50-95@300 | 0.39770 | 0.41897 | +0.02126 |
| 非小卷叶螟（288） | Mask AP50 | 0.78778 | 0.78584 | -0.00194 |
| 非小卷叶螟（288） | Mask AP50-95 | 0.30244 | 0.29892 | -0.00353 |
| 非小卷叶螟（288） | Mask Precision@IoU0.50/conf0.25 | 0.64820 | 0.58663 | -0.06157 |
| 非小卷叶螟（288） | Mask Recall@IoU0.50/conf0.25 | 0.81250 | 0.82292 | +0.01042 |
| 非小卷叶螟（288） | Mask AR50-95@300 | 0.46632 | 0.46215 | -0.00417 |

小卷叶螟固定阈值 Recall 从 0.61494 提高到 0.64368（+2.874 个百分点），
正确检出数从 107 增至 112；同时未被忽略的小尺寸卷叶螟误检数从 123 增至 149，
Precision 从 0.46522 降至 0.42912。小目标 Mask AP50 / AP50-95 分别下降
4.444 / 1.964 个百分点。C 的小目标召回有局部收益，但误检增多、AP 下降，
不满足小目标综合收益与总体表现的保留条件。
非小卷叶螟 AP50 / AP50-95 也略降，未显示通过牺牲大目标换取小目标 AP 的收益。

## 评估口径

本机 RTX 4060 Ti、Python 3.10.19、PyTorch 2.10.0+cu130、Ultralytics 8.4.80；
使用训练 commit `b8001b6` 的源码。两个模型均使用 imgsz=640、rect=false、batch=1、
FP32、conf=0.001、max_det=300、retina_masks=true。置信度 0.25 只用于报告固定工作点 P/R，
在预测前选定，不按结果调节。IoU 阈值 0.50:0.05:0.95；同时记录 IoU=0.50 的 AP 与 Recall。

GT 掩膜按原图多边形采用 COCO API 栅格化；预测掩膜映射回原图。Box 与 Mask IoU 均在原图
坐标计算，但 GT 和未匹配预测的大小分组都采用 bbox 面积乘以 `(640/max(H,W))²`。
保留完整场景中的其他尺寸 GT，作为组外忽略对象参与匹配，不把这些目标误记成背景。
AP 使用 COCO 的 101 点 Recall 插值；AR50-95@300 是多 IoU 平均最大召回，
Recall@conf0.25 是固定 IoU=0.50、置信度≥0.25 的 TP/GT。
匹配与忽略规则来自 [COCO 官方评估实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)。

这是独立补充评估，不能与训练 CSV 的 Ultralytics AP 或日志的最大 F1 工作点 P/R 直接相减。
组内比较使用相同模型后处理与指标协议。两组权重不再挑 epoch；没有使用 Test。
密集场景没有单独定义和量化，此表仅回答目标尺寸分组的表现。

## 复现与来源

```powershell
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) '.cache/ultralytics'
.\.venv\Scripts\python.exe scripts/evaluate_c_small_val.py `
  --source-root .cache/c-analysis-source/ultralytics-main `
  --data-root E:/Study/DeepCNN/yolo26/code/datasets
```

依赖 `pycocotools==2.0.11`、PyTorch 和仓库 Ultralytics。`--source-root` 指向包含 C 类定义的
源码目录；上述本地目录由 `git archive b8001b6 ultralytics-main/ultralytics` 解出，
也可指定独立 C checkout 的 `ultralytics-main`，总体分析工作区保持在 A 分支。

- 机器可读结果：[data-v2-c-small-val.json](data-v2-c-small-val.json)，包含两类、三个尺寸组、Box/Mask AP 与固定工作点计数。
- 评估脚本：[evaluate_c_small_val.py](../../scripts/evaluate_c_small_val.py)。
- 本地预测缓存：`.cache/c-small-000-predictions.json`、`.cache/c-small-C-predictions.json`；不提交 Git。
- 本机 Val 图像与标签规模、类别数量和小卷叶螟数量均与主计划一致；没有云端逐图内容版本证明，正式总体指标仍以云端 Run 为准。
