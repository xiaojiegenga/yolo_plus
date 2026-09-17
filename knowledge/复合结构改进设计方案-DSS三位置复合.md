# 复合结构改进设计方案 v2：DSS 三位置复合（DSEM + SFCM + DPRM）

撰写日期：2026-09-15。本版取代 v1（SDP-Proto 单位置方案）。工作区：
`E:\study\graduate_sec\论文撰写\Zcode_yolo\yolo_plus`（分支 `cloud/data-v2-5090`，基线提交 `c0f4f35`）。

**策略**：先构建各组件均有证据支撑、初始不扰动预训练的三位置复合改进，整体通过门控后再留一消融。

---

## 0. 本版扩展决定与设计立场

期刊体例（参考 DSDC-YOLO、改进 YOLOv8 水稻病害等同类论文）通常期待 Backbone/Neck/Head 多位置
改进，单一原型分支改动存在"改进量不足"的审稿风险。**因此扩展为三位置复合**，但必须遵守本项目
12 组正式实验得出的唯一有效范式：

- R1 官方预训练权重能迁移的张量一个不减（禁止整块替换、禁止在网络中段插入新 YAML 层——会改变
  后续所有层的权重键名，破坏迁移）；
- R2 新能力一律零初始化残差/门控，epoch 0 前向与基线一致；
- R3 每个组件必须针对一个**已测量的**失败模式或数据特征，不凭文献直觉加模块。

三位置共享一条叙事管线：**细节保留（Backbone）→ 前景校准（Neck）→ 高保真掩膜合成（Head）**。
复合代号 **DSS**（Detail–Foreground–Segmentation）。

## 1. 证据基础（来自旧项目 12 组正式实验，可查 `模型训练/experiment_records/`）

| # | 证据 | 数据 | 指向 |
|---|---|---|---|
| E1 | 掩膜空间保真度是唯一验证有效的杠杆 | D1（P2 注入 + 原型 stride 4→2）+0.98 pp，唯一正向 | 头部原型分支是复合主体 |
| E2 | 背景混淆是最大误差源 | 000 在 conf≥0.5 的 388 个预测中 66 个（17.0%）与全部 GT 框 IoU<0.1；GT 辅助降分可回收约 4.9 pp | 需要显式前景判别，且应作用于检测分支输入 |
| E3 | 破坏预训练复用的改动全部失败 | DRB/SCSA 整块替换 -4.63 pp；A1 标准 CBAM -1.25 pp；A2 零初始化残差 CBAM 持平（无害） | R1/R2 设计规则的来源 |
| E4 | Backbone/Neck/检测侧的"标准模块"全部失败 | A1/A2（注意力）、C（P2 检测头 -1.23）、E（DySample -1.70）、F/F2/G（Neck 条带上下文）、DRB | 扩展位置可以加组件，但禁止照搬这些形态 |

数据特性：卷叶螟旋转框短边中位 7.19 px（64.8% <8 px）、长宽比中位 10.92、多边形面积中位 392 px²
（96% <1024 px²）；稻叶稻穗与目标共享细长纹理。

## 2. 总体架构

```text
Backbone
  0 Conv ─ 1 Conv ─ 2 C3k2DSEM ←【DSEM：方向条带细节增强，零初始化残差】
                        │ 输出(256ch,160×160) 同时流向：
                        ├──→ 3 Conv(↓P3) … 10 C2PSA        （下游主干，官方路径）
                        └──→ ┐
                             │ 跨阶段跳连（新连线）
Neck                          │
  11-15 … 16 C3k2 (P3,256ch,80×80)
                        │ 输出 X16
Head (层23 Segment26DSS)│
  ┌─────────────────────┘
  │ SFCM：sem_trunk(X16)→sem_head(辅助监督) + sem_gate→前景门 g
  │ X16' = X16 × (1 + γ⊙g)                    【SFCM：语义前景校准，γ 零初始化】
  │      ├─→ 检测分支 Detect(P3',P4,P5)        （cv2/cv3/cv4 权重不动，输入被校准）
  │      └─→ DPRM 原型分支：
  │            官方多尺度融合(feat_refine/feat_fuse，P3'为基) → cv1 → ↑2 → cv2
  │            mid += p2_proj(P2')             【P2 跨阶段注入，零初始化】
  │            p_lo = cv3(mid)                 （官方权重，32×160×160）
  │            p_hi = hi_out(hi_cv(hi)+strip(hi))【stride-2 残差，零初始化，32×320×320】
  │            p = bilinear×2(p_lo) + p_hi     【DPRM：双分辨率原型】
  语义辅助损失槽位（loss[4]）由 SFCM 的 sem_head 提供，监督协议不变
```

| 论文位置 | 模块 | 作用点 | 回答的审稿问题 |
|---|---|---|---|
| Backbone | DSEM | 第 2 层 C3k2 输出（P2, 160×160） | 卷积级改动（方向条带核） |
| Neck | SFCM | 第 16 层（Neck P3）输出，检测/原型分支消费前 | 注意力（监督驱动的前景门） |
| Head | DPRM | 第 23 层原型分支 | 高分辨率掩膜合成 + 跨阶段注入 |
| 检测分支 | 不改权重 | 经 SFCM 校准的 P3 输入受益 | 检测侧收益的来源 |

## 3. 模块规格

### 3.1 DSEM（Backbone P2 方向细节增强，`C3k2DSEM(C3k2)`）

子类包裹（A2 的 `C3k2ZRCBAM` 已验证此模式可整体迁移官方权重），在 C3k2 标准输出后加零初始化
方向条带残差：

```python
class C3k2DSEM(C3k2):
    """C3k2 后接零初始化方向条带残差（DSEM）。

    继承全部子模块（键名不变，官方权重整体迁移）。α 逐通道零初始化，初始严格恒等。
    方向核 7×1/1×7 针对长宽比中位 10.92 的细长目标；输出同时服务下游主干与 DPRM 跨阶段注入。
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True, r=4):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        c_mid = max(c2 // r, 16)
        self.dw = nn.Sequential(  # 深度可分离方向条带核
            nn.Conv2d(c2, c2, (7, 1), padding=(3, 0), groups=c2, bias=False),
            nn.Conv2d(c2, c2, (1, 7), padding=(0, 3), groups=c2, bias=False),
        )
        self.pw = nn.Sequential(Conv(c2, c_mid, 1), Conv(c_mid, c2, 1))
        self.alpha = nn.Parameter(torch.zeros(c2))

    def forward(self, x):
        y = super().forward(x)
        return y + self.alpha.view(1, -1, 1, 1) * self.pw(self.dw(y))

    def forward_split(self, x):
        y = super().forward_split(x)
        return y + self.alpha.view(1, -1, 1, 1) * self.pw(self.dw(y))
```

- 参数 ≈ 36.7 k，FLOPs ≈ 1.9 G（@160²）；α=0 时初始恒等。
- 依据：短边 7.19 px 的细节在 stride-4 网格（约 1.8 格）最完整；条带核编码细长方向。
  与 DRB 失败的区别：DRB 整块替换 Bottleneck（破坏预训练），DSEM 是保留原块之上的零初始化残差。
- 风险：A2 先例（零初始化残差注意力持平）——DSEM 是最可能"学不起来"的组件，消融见分晓。

### 3.2 SFCM（Neck P3 语义前景校准，实现在 `Segment26DSS` 内）

基线 `Proto26` 的语义辅助头 `semseg` 只在训练期算损失、推理时删除，其前景/背景判别从未参与
预测。SFCM 把它重构为**参与推理的监督驱动前景门**：

```python
# Segment26DSS.__init__（ch 按 (P2, P3, P4, P5) 传入）：
super().__init__(nc, nm, npr, reg_max, end2end, ch[1:])      # 检测头吃 P3/P4/P5，权重同基线
c_sem = max(self.npr // 4, 32)
self.sem_trunk = Conv(ch[1], c_sem, k=3)                     # 256→64 @80×80
self.sem_head = nn.Conv2d(c_sem, nc, 1)                      # 语义 logits，接 loss[4]
self.sem_gate = nn.Conv2d(c_sem, 1, 1)                       # 前景门
self.gate_scale = nn.Parameter(torch.zeros(ch[1]))           # 逐通道零初始化
self.proto = Proto26DR(ch, self.npr, self.nm, nc)            # 见 3.3

# forward 核心逻辑：
p2, p3, p4, p5 = x
sem = self.sem_trunk(p3)
gate = torch.sigmoid(self.sem_gate(sem))
p3_cal = p3 * (1.0 + self.gate_scale.view(1, -1, 1, 1) * gate)
outputs = Detect.forward(self, [p3_cal, p4, p5])             # 检测分支消费校准后 P3
proto = self.proto((p2, p3_cal, p4, p5))                     # 原型分支以 P3' 为融合基
# 训练态把 (proto, sem_logits) 挂到 preds["proto"]（含 end2end one2many/one2one 分支），
# 与 v8SegmentationLoss 的 `isinstance(proto, tuple)` 解包协议对齐；推理态只挂 proto。
```

- 参数 ≈ 148 k，FLOPs ≈ 1.9 G（@80²）；γ=0 时初始恒等。
- 语义监督协议不变：仍走 loss[4]、BCE+Dice、权重 ×`hyp.box`，GT 网格与类别数不变。
- 依据（E2）：监督驱动的前景门直接压制与虫害共享细长纹理的背景响应；检测分支经校准输入受益。
  与 A1/A2 失败的区别：CBAM 的注意力要从零自学习权重，SFCM 的门有逐 epoch 的语义监督信号。
- 实现位置说明（论文写作需如实）：SFCM 物理上位于 Head 模块内，但作用于 Neck P3 输出、在检测与
  原型分支消费之前；Neck PAN 下采样路径（第 17 层起）消费的是未校准特征。结构图应画在第 16 层
  输出到 Head 的箭头上，文字表述为"Neck P3 输出处的前景校准"。
- 风险：乘性门改变检测头输入分布（零初始化保护起点）；若语义监督学不好，门退化。

### 3.3 DPRM（Head 双分辨率原型，`Proto26DR(Proto26)`）

```python
class Proto26DR(Proto26):
    """双分辨率原型：官方 stride-4 合成全迁移 + P2 零初始化注入 + stride-2 零初始化残差。

    相比 D1：cv3 保留在 stride-4 官方位置（权重迁移，D1 中被重定义丢失），stride-2 级只学残差。
    semseg 从本模块移除（由 SFCM 的语义分支取代），推理不再有仅训练组件。
    """

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80, narrow: int = 4):
        super().__init__(ch[1:], c_, c2, nc)          # 官方子模块键名全部保留
        del self.semseg                              # 语义分支移至 SFCM（省 ~1.20 M 参数）
        c_hi = max(c_ // narrow, c2)
        self.p2_proj = Conv(ch[0], c_, k=1)           # 组件：P2 跨阶段注入
        nn.init.zeros_(self.p2_proj.conv.weight); nn.init.zeros_(self.p2_proj.conv.bias)
        self.hi_up = nn.ConvTranspose2d(c_, c_hi, 2, 2, 0, bias=True)
        self.hi_cv = Conv(c_hi, c_hi, k=3)
        self.hi_strip = nn.Sequential(                # 方向条带子件（细长掩膜边界）
            nn.Conv2d(c_hi, c_hi, (7, 1), padding=(3, 0), groups=c_hi, bias=False),
            nn.Conv2d(c_hi, c_hi, (1, 7), padding=(0, 3), groups=c_hi, bias=False),
            nn.SiLU(),
        )
        self.hi_out = nn.Conv2d(c_hi, c2, 1, bias=True)  # 零初始化 → 初始 p_hi ≡ 0
        nn.init.zeros_(self.hi_out.weight); nn.init.zeros_(self.hi_out.bias)

    def forward(self, x, return_semantic: bool = True):
        p2, feats = x[0], x[1:]
        feat = feats[0]                                # 官方多尺度融合，逐行同 Proto26
        for i, f in enumerate(self.feat_refine):
            up = f(feats[i + 1])
            feat = feat + F.interpolate(up, size=feat.shape[2:], mode="nearest")
        mid = self.cv2(self.upsample(self.cv1(self.feat_fuse(feat))))  # 256ch @160×160
        mid = mid + self.p2_proj(p2)                   # P2 注入（P2 已经过 DSEM 增强）
        p_lo = self.cv3(mid)                           # 32×160×160，官方权重
        hi = self.hi_up(mid)
        p_hi = self.hi_out(self.hi_cv(hi) + self.hi_strip(hi))  # 32×320×320
        return F.interpolate(p_lo, scale_factor=2, mode="bilinear", align_corners=False) + p_hi

    def fuse(self):
        return self                                    # 无仅训练组件需要移除
```

- 新增参数 ≈ 171 k（cv3 约 8.2 k 官方迁移，不再像 D1 那样重定义）；FLOPs 增量与 D1 同档
  （D1 实测 +14 G）。
- 初始等价的关键事实：基线在 mask_ratio=2 下计算损失时本就把 160² 原型双线性插值到 320² 标签
  网格——**epoch 0 的实例掩膜损失与基线逐位一致**；检测路径（含 SFCM γ=0、DSEM α=0）同样一致。
  与基线的初始差异只剩语义辅助头为新建随机初始化（基线该头迁移自官方）。

## 4. 源码实现清单

| 步骤 | 内容 |
|---|---|
| 1 | 新建分支 `feature/data-v2-cmp-dss`（自 `cloud/data-v2-5090`） |
| 2 | `nn/modules/block.py`：`C3k2DSEM`、`Proto26DR` |
| 3 | `nn/modules/head.py`：`Segment26DSS`（含 SFCM；`fuse()` 无需覆盖语义分支，仅检测 one2many 照官方逻辑） |
| 4 | `nn/modules/__init__.py` + `nn/tasks.py` 注册：`C3k2DSEM` 加入 import、基础模块集合、repeat 集合、`if m in {C3k2, C3k2DSEM}`；`Segment26DSS` 加入 import、`args.extend` 头集合、`npr` 缩放集合、`m.legacy` 集合（四处，照 D1 分支 `Segment26P2` 的注册 diff） |
| 5 | `models/yolo/segment/val.py`：移植 D1 的验证对齐修复——官方验证器假设原型 stride 4（`imgsz = 4 × proto.shape`），stride-2 原型会把图像尺寸算成 1280 导致掩膜坐标错位；修复改用真实 batch 图像尺寸，并把原型双线性插回 stride-4 网格再合成掩膜，保证与基线同一评估网格（D1 r2 已在 300 epoch 正式训练中验证） |
| 6 | 模型 YAML `cfg/models/26/yolo26m-dss-seg.yaml`：复制 `yolo26-seg.yaml`，两处改动——第 2 层 `- [-1, 2, C3k2DSEM, [256, False, 0.25]]`；末行 `- [[2, 16, 19, 22], 1, Segment26DSS, [nc, 32, 256]]` |
| 7 | 训练配置 `experiments/data-v2-cmp1-dss-b16-s42.yaml`：train 段 33 项与 000 逐项一致 |
| 8 | 云端：`python scripts/cloud_train_data_v2.py --config experiments/data-v2-cmp1-dss-b16-s42.yaml --pretrained yolo26m-seg.pt`（入口已支持自定义 YAML + 官方权重按名字/形状交集迁移；`--pretrained` 要求该 .pt 已存在于云端仓库根目录或 cwd） |

## 5. 权重迁移与初始化审计表（预检必测）

| 张量组 | 来源 | 初始行为 |
|---|---|---|
| Backbone 第 0/1/3–10 层全部 | 官方 ckpt 迁移 | 与基线一致 |
| 第 2 层 `C3k2DSEM` 的 C3k2 部分 | 官方 ckpt 迁移（键名不变） | 与基线一致 |
| 第 2 层 `dw/pw/alpha` | 新增，α 零初始化 | 残差恒 0 |
| Neck 11–22 层全部（含第 16 层） | 官方 ckpt 迁移 | 与基线一致（层 16 本体未改） |
| 检测头 cv2/cv4 | 官方 ckpt 迁移 | 与基线一致 |
| 检测头 cv3（分类） | 80→2 形状重建 | 基线同款初始化 |
| `proto.feat_refine/feat_fuse/cv1/upsample/cv2/cv3` | **官方 ckpt 全量迁移** | 与基线一致 |
| `proto.p2_proj` | 新增，零初始化 | 注入恒 0 |
| `proto.hi_up/hi_cv/hi_strip` | 新增，标准初始化 | 被 hi_out 归零 |
| `proto.hi_out` | 新增，零初始化 | `p_hi ≡ 0` |
| `sem_trunk/sem_head/sem_gate/gate_scale` | 新增（γ 零初始化） | 门恒 1，sem logits 随机 |
| 基线 `proto.semseg.*`（ckpt 中存在） | 无对应目标，交集自动跳过 | — |

本地聚焦测试（CPU 即可）：(1) 形状：proto `(1,32,320,320)`，训练态 `preds["proto"]=(p, sem)`；
(2) 初始等价：α=γ=p2_proj=hi_out=0 时，检测输出与"官方权重加载的基线"逐位一致，掩膜损失路径
亦一致；(3) 迁移计数符合上表；(4) 真实 batch 上新增参数梯度非零（含 loss[4] 对 sem_trunk）；
(5) 配置 diff 为空。

## 6. 成本（m 尺度、640 输入）

| 项目 | 000 基线 | D1（实测） | DSS（本地构建实测） |
|---|---:|---:|---:|
| 参数量（nc=80 构建） | 27.11 M | — | 26.27 M（−0.84 M，与设计账目一致） |
| 参数量（nc=2 训练态） | 23.51 M | 23.67 M | ≈22.70 M（换算值，训练时以日志为准） |
| GFLOPs | 121.4 | 135.4 | ≈139（按 D1 实测口径外推，预检以 THOP 实测为准） |
| 峰值显存 | 15.6 GB | 15.7 GB | ≈15.7 GB |
| 训练时长 | 1.23 h | 1.15 h | ≈1.2 h |

本地构建实测：26,272,809 参数（nc=80），官方 `yolo26m-seg` 为 27,112,072——差值 −839,263
恰为"移除 semseg 约 1.20 M − 新增 DSEM/SFCM/DPRM 约 0.36 M"。参数量低于基线（移除了仅训练用
的 semseg 干线），GFLOPs 与 D1 同档。

## 7. 门控与"死组件剔除规则"（预注册，不随结果调整）

正式 Run：`data-v2-cmp1-dss-b16-s42`，300 epochs，冻结配方（seed=42、AdamW、mask_ratio=2、
mixup=0、batch=16、imgsz=640）。对照：000 的 Val Mask mAP50-95 = 0.36348。

| 条件 | 门槛 |
|---|---|
| 主门控 | Val Mask mAP50-95 ≥ **0.36848**（+0.5 pp） |
| 机制目标 | ≥ 0.37327（D1 水平）；理想 ≥ 0.37827 |
| 分类别 | 卷叶螟 Mask AP50-95 ≥ 0.266 且钻心虫 ≥ 0.461 |
| 小目标专项 | 174 个小卷叶螟 AP50-95 > 0.19311（沿用旧协议） |
| 成本 | 参数 ≤ 23.0 M、GFLOPs ≤ 142、显存 ≤ 16 GB、时长 ≤ 1.5 h |

**死组件剔除规则**（复合通过后消融阶段执行）：某组件被移除时主指标变化 < 0.005（噪声地板口径）
且机制指标（前景门：低重叠高置信候选比例；DSEM/DPRM：分类型 AP、小目标专项）无对应改善 →
该组件从最终模型剔除，论文如实报告为探索性负结果或直接删除。**最终模型不保留死组件**——
这是对"为凑位置而加模块"的风险对冲：位置覆盖先做，有效性由消融裁决。

训练后机制诊断：best.pt 读取 ‖α‖、‖γ‖、‖p2_proj‖、‖hi_out‖（全零=组件未激活，A2 教训：非零≠有效）；
复算 conf≥0.5 低 IoU 候选比例（000 为 66/388=17.0%），观察 SFCM 是否压低该值。

**结果分支**：通过主门控 → 消融；0.363–0.368 持平 → 先看组件激活诊断再定（组件未激活可加强
门结构换新 Run ID 重试一次）；< 0.363 → 判失败，回机制诊断，不无门槛重跑。

## 8. 消融矩阵（复合通过门控后执行，每个 Run 均完整 300 epochs）

| Run ID | 移除项 | 检验机制 |
|---|---|---|
| `data-v2-cmp1-dss-b16-s42` | —（主 Run） | 复合整体 |
| `data-v2-cmp2-nodsem-b16-s42` | DSEM（第 2 层换回 C3k2） | 方向细节增强的贡献 |
| `data-v2-cmp2-nosfcm-b16-s42` | SFCM 门（γ≡0 且 detach；**语义辅助监督保留**） | 前景校准超越纯辅助监督的增量 |
| `data-v2-cmp2-nop2inj-b16-s42` | p2_proj（P2 只经主干路径） | 跨阶段注入的贡献（与 DSEM 分离） |
| `data-v2-cmp2-nodual-b16-s42` | hi 级（p 退回 p_lo@160²，损失插值同基线） | stride-2 残差合成的贡献 |
| `data-v2-cmp2-nostrip-b16-s42`（可选） | hi_strip → 3×3 | 方向核子件的增量 |

参照行：000（已有）；旧项目 D1 的 0.37327 作跨项目参照（同一基线提交与配方），期刊要求独立成表
时复跑 D1 等价 Run。

## 9. 风险与如实评估

1. **DSEM/SFCM 学不起来**：A2 先例为零初始化残差持平。对冲=死组件剔除规则；论文最坏退路是
   DPRM 单模块 + 完整探索性消融叙事（届时与导师商定）。
2. **位置表述的诚实性**：SFCM 作用于 Neck P3 输出但物理实现在 Head 模块内，且 PAN 路径未校准；
   论文图表按第 2 节口径绘制，不夸大为"改造了整个 Neck"。
3. **与旧项目已证伪方向的区分**必须写清：DSEM≠DRB（零初始化残差 vs 整块替换）；SFCM≠CBAM
   （监督驱动 vs 自学习）；DPRM 条带子件≠F/F2/G（掩膜合成级 ~0.9 k 参数 vs Neck 检测特征数十万
   参数，且受零初始化输出保护）。
4. **门热启动性质（本地测试确认）**：γ=0 时梯度到 `sem_gate` 的贡献为 `p3·γ = 0`，门内权重在
   第一步更新前无梯度；γ 自身梯度非零，第一步后门打开、内部开始学习（ReZero/A2 同款性质）。
   `sem_trunk` 不受影响——它从第 0 步起就有语义辅助监督的梯度。这不是缺陷，但写作和诊断时
   要知道：SFCM 的生效晚于纯加性组件（DSEM 的 α、DPRM 的 p2_proj/hi_out 都是第 0 步即有梯度）。
5. **单 seed**：沿用 seed=42 口径；最终模型可补 seed 2/3（成本 2×1.2 h）。
5. epoch 0 等价性声明仅覆盖检测/实例掩膜损失路径；语义辅助头随机初始化导致语义损失列从第 0 步
   即不同，训练轨迹必然分叉（设计内行为）。

## 10. 实施步骤

1. [ ] 新建分支 `feature/data-v2-cmp-dss`；
2. [ ] 按第 3、4 节实现（`C3k2DSEM`、`Proto26DR`、`Segment26DSS`、注册、YAML、配置）；
3. [ ] 本地 5 项聚焦测试（第 5 节）；
4. [ ] 云端 1 epoch 预检（带时间戳 Run，不进正式记录）→ 核对迁移计数与初始等价；
5. [ ] tmux 正式 300 epochs → 回传 → 按第 7 节门控判定并登记；
6. [ ] 通过后执行第 8 节消融；死组件按规则剔除。
