# 改进 H：局部候选分类精修原理与实现

## 1. 研究问题

D1 的 Val Mask mAP50-95 为 `0.37327`，高于 YOLO26m Baseline 的 `0.36348`，但 D1
原图尺寸诊断中，置信度 ≥ `0.5` 且与全部 GT 框 IoU < `0.1` 的候选比例为 `21.41%`。
H 针对这一分类置信度问题，不再改变检测框、类别、掩膜系数或原型分支，而是用独立
分类器复核已有候选。

设计借鉴 [DCR](https://openaccess.thecvf.com/content_ECCV_2018/html/Bowen_Cheng_Revisiting_RCNN_On_ECCV_2018_paper.html)
对基础检测器 hard false positives 进行独立分类复核的思路。H 是面向本项目 YOLO
实例分割输出的适配，不是对 DCR 网络或训练过程的逐项复现。

## 2. 候选与标签

D1 全部冻结，仅在 Train 图片上以 `conf=0.25` 生成 NMS 后候选。每个候选按以下固定规则
标注：

- 与同类 GT 框的 IoU ≥ `0.5`：标为对应虫类；
- 与全部 GT 框的最大 IoU < `0.1`：标为背景类；
- 其余候选：忽略，不进入分类器训练。

候选框内容保持纵横比，短边用 `114/255` 填充，再缩放到 `96×96`。分类训练每轮对
`Rice leaffolder`、`Rice stemborers`、背景三类按 `1:1:1` 过采样，避免背景数量支配
梯度。

## 3. 分类器与训练

`LocalCropRefiner` 深拷贝 D1 中 YOLO26m 的前五层作为独立特征提取器，随后使用全局
平均池化和三分类线性层。模型共 `1,224,515` 个参数。D1 本身始终冻结，分类器使用：

- 20 epochs，crop batch 128；
- AdamW，lr `1e-4`，weight decay `5e-4`；
- seed 42，确定性模式；
- 仅水平翻转和垂直翻转。

每一轮都在固定 Val 协议上评估，按 Ultralytics 官方分割 fitness
（Box mAP50-95 + Mask mAP50-95）保存组合 `best.pt`。最终指标重新加载该 `best.pt`
计算，不使用最后一轮替代最佳轮。

## 4. 推理公式与不变量

对 D1 预测类别为 `c`、原置信度为 `s_D1` 的候选，分类器输出三类 softmax 概率，最终
分数固定为：

```text
s_final = s_D1 × p_refiner(c)
```

H 只复制并更新 `conf` 字段。候选类别、框坐标、掩膜系数和最终掩膜不被修改；空候选
可以直接返回。Validator 和独立 Predictor 均调用同一 `apply_local_refinement`，避免
正式 Val 与原图尺寸评估出现两套评分逻辑。

## 5. 组合 checkpoint 与结果产物

H 的 `best.pt` 保留标准 `model` 字段作为 D1 基础模型，并增加 `refiner` 与
`h_metadata`。因此普通 `YOLO(best.pt)` 可以读取基础分割模型，H 的验证/预测入口则
额外加载同一文件中的 refiner。

正式 Run 同时保存：逐轮 `results.csv`、分类别指标、Mask AP75、原图尺寸 COCO 评估、
背景误检诊断、参数量、GFLOPs、batch-1 FP16 延迟和峰值显存。候选 crop 和 manifest
只属于对应 Run，不进入 Git。

## 6. 固定验收与能力边界

H 必须达到 Val Mask mAP50-95 `0.37827`，原图尺寸指标高于 `0.34814`，满足两个类别的
下降约束，并将原图尺寸低重叠高置信候选比例降到 `21.41%` 以下。门槛不随训练结果调整。

该模块只能重排或压低已有候选的置信度，不能找回 D1 漏检，也不能改善框或掩膜几何。
因此若主指标或背景误检机制指标未同时满足预注册门槛，下一步应进入独立 I 双分辨率
残差原型分支，而不是继续搜索 H 的阈值或以 D1 单独结题。

## 7. 源码位置

- 分类器：`ultralytics-main/ultralytics/nn/modules/refiner.py`
- 公共评分：`ultralytics-main/ultralytics/models/yolo/segment/refine.py`
- 候选、Val 与 Predictor：`ultralytics-main/ultralytics/models/yolo/segment/refine_val.py`
- 组合 checkpoint：`ultralytics-main/ultralytics/models/yolo/segment/refine_checkpoint.py`
- 训练入口：`scripts/train_local_refiner.py`
- 固定配置：`experiments/data-v2-abl-d1h-localrefine-rb128-s42.yaml`
- 云端步骤：`实验步骤.md`
