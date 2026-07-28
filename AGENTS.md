# Project Context

## Environment
- The `ultralytics-main` folder contains the source code for YOLO's latest model (YOLO26).
- A related YOLO improvement paper is located at `/YOLO-pineapple.html`.

## Current Project State (2026-07-28)
- The project studies **task-specific YOLO26 improvements for UAV rice-pest instance segmentation**. It is not a reproduction of YOLO-Pineapple.
- YOLO-Pineapple is a source of research ideas and comparison conventions only; its modules must be adapted selectively to YOLO26 and this dataset.
- V1 CBAM is a completed independent experiment.
- The former V2 implementation failed and is preserved only on `archive/v2-p2-failed`.
- The new `v2-p2` starts independently from `main` and must initially contain no model-source changes.
- Do not implement or train the new V2 until the user explicitly starts that phase.

## About Me
- I am a beginner in deep learning. Please make explanations as easy to understand as possible, and clarify all technical terminology.
- I am a graduate student in Electronic Engineering. My graduation thesis focuses on rice pest and disease recognition based on UAV (drone) imagery.

## My Goals
1. Understand the YOLO model code structure — specifically, locate the model configuration files under `models/yolo/` (e.g., `yolov8.yaml`) and study each layer in terms of the Backbone / Neck / Head three-stage architecture. Ultralytics code has comprehensive comments and documentation.
2. Improve YOLO model performance for rice pest and disease recognition.
3. After understanding the code structure, selectively adapt useful ideas from `/YOLO-pineapple.html` and other relevant work to this rice-pest task, rather than reproducing that paper unchanged.

## Instructions for the Assistant
- When explaining code, always relate it back to the Backbone / Neck / Head architecture.
- Prioritize clarity and educational value over brevity.
- Help me bridge the gap between paper-level improvements and actual code implementation.
- **Current phase: project cleanup complete; new V2 implementation is paused.** Source changes are allowed only after the user explicitly starts the next implementation phase. All future changes must be tracked, explained, and tied to a specific target.

## Important Documents

- **[CODEX_PROJECT_CONTEXT.md](CODEX_PROJECT_CONTEXT.md)** — ★ 当前事实、实验边界与后续执行规范；与历史文档冲突时优先参考。
- **[code_plus.md](code_plus.md)** — ★ 改进工作流与修改记录。包含：
  - 代码生效机制说明（pip install -e 原理）
  - 环境搭建步骤（源码升级到 8.4.80 + 可编辑安装）
  - 改进追踪模板（每次改动记录版本号、改动文件、训练结果对比）
  - **这是改进阶段最重要的参考文件，每次改动后都要更新**
- **[node.md](node.md)** — 技术学习笔记；旧 V2 内容仅作失败复盘，新实验只在文件末尾追加。
- **[github_guide.md](github_guide.md)** — 当前 Git 分支、提交和远程同步流程。
- `results/` — 用户维护的本地完整训练档案，禁止上传 Git。
- `experiment_records/` — Git 追踪的轻量实验摘要、指标来源和结论。

## Repository Layout
- `ultralytics-main/` — upstream Ultralytics source (YOLOv3 → YOLO26 + RT-DETR). All model code, configs, and engine live here. **We will modify files under `ultralytics/nn/` and `ultralytics/cfg/`.**
- `YOLO-pineapple.html` — reference paper at workspace root. It provides ideas about UAV agricultural targets, attention and multi-scale design, but is not the implementation blueprint for this project.
- `ultralytics-main/yolo26s-seg.pt`, `ultralytics-main/yolo26m-seg.pt` — pre-trained YOLO26 segmentation checkpoints (small & medium).
- `ultralytics-main/cat.jpg` — test image for inference sanity checks.

## Baseline Model (Control Group)

**Training run**: `yolo26m_seg_20260628_172809/` (copied from `runs/segment/runs_seg/`)
- **Model**: yolo26m-seg.pt (pretrained, fine-tuned)
- **Task**: Instance segmentation — Rice leaffolder + Rice stemborers (2 classes)
- **Dataset split**: 768 train / 95 val / 95 test
- **Epochs**: 400 (best at **epoch 246**)
- **Image size**: 640×640
- **Optimizer**: Auto (AdamW), lr0=0.01, cos_lr=False, warmup 5 epochs
- **Augmentation**: mosaic=1.0, mixup=0.1, copy_paste=0.3, randaugment, erasing=0.4

### Key Baseline Metrics

| Metric | Value | Note |
|---|---|---|
| **Val Mask mAP50** | **0.683** | ★ Main metric, best epoch 246 |
| Val Mask mAP50-95 | 0.33 | Half of mAP50 → edge precision gap |
| Val Box mAP50 | 0.667 | Slightly below Mask (normal for seg) |
| Leaffolder Mask mAP50 | 0.604 | 52% miss rate — **biggest problem** |
| Stemborers Mask mAP50 | 0.763 | 22% miss rate — acceptable |
| Class confusion | 0% | Two classes never confuse each other |
| Leaffolder best F1 | 0.57 @ conf=0.35 | Low confidence on small/dense targets |

### Identified Improvement Targets (priority order)

1. **🔥 Leaffolder recall (漏检率 52% → target <30%)** — P0 critical
   - Cause: small targets, dense clustering, weak features on leaf backgrounds
   - Approaches: attention mechanism (CBAM), P2 high-res feature layer, small-object augmentation

2. **⚠️ Mask edge precision (mAP50-95: 0.33 → target >0.45)** — P1 important
   - Cause: segmentation head produces coarse masks
   - Approaches: Dice Loss, lightweight mask decoder, higher mask_ratio

3. **📈 Overall mAP50 (0.683 → target >0.75)** — P1 important
   - Approaches: better neck structure, CBAM attention, training recipe tuning

### Key Discovery: CBAM Already in Codebase

`ultralytics-main/ultralytics/nn/modules/conv.py` already has `ChannelAttention`, `SpatialAttention`, and `CBAM` classes (lines 512–613). **However**, CBAM is NOT registered in `tasks.py`'s `base_modules` set (line 1575–1612), so it cannot be used in a YAML config. To use CBAM in a model:
1. Register it in `tasks.py` `base_modules`
2. Add it to the model YAML (in backbone, before SPPF typically)
3. Or: wrap it inside an existing block like C3k2

## Reading Roadmap (Backbone / Neck / Head tour)

A 5-step ordered tour. Each step names the file(s), what to look at, and the mental model to build. Pitched at a beginner.

**Step 1 — YAML topology (start here):**
- `ultralytics-main/ultralytics/cfg/models/26/yolo26.yaml` (compare with `v8/yolov8.yaml`).
- Look at: the `backbone:` and `head:` blocks; the schema `[from, repeats, module, args]`; how `-1` means "previous layer" and `[a, b]` lists mean "concatenate inputs from layers a and b".
- Mental model: **YAML is the blueprint.** Every YOLO model is just a list of layers + connections. Backbone extracts features; the YAML's `head:` block actually contains both the Neck (Upsample + Concat for FPN/PAN) and the Detect head.

**Step 2 — Building blocks (Conv, C3k2, SPPF):**
- `ultralytics-main/ultralytics/nn/modules/conv.py` — `Conv`, `DWConv`, `GhostConv` (atomic operations).
- `ultralytics-main/ultralytics/nn/modules/block.py` — `C2f`, `C3k2`, `SPPF`, `PSA` (composed blocks that appear in the YAML).
- Look at: `forward()` of `Conv` and `C2f` first (simplest); then `SPPF` to see multi-scale pooling.
- Mental model: every string name in the YAML maps to a Python class here.

**Step 3 — Detection head:**
- `ultralytics-main/ultralytics/nn/modules/head.py` — `Detect` class.
- Look at: how the head takes three feature maps (P3/P4/P5 from the neck) and produces bbox + class scores; the `DFL` (Distribution Focal Loss) regression head.
- Mental model: this is where "features" become "predictions".

**Step 4 — YAML → PyTorch assembly:**
- `ultralytics-main/ultralytics/nn/tasks.py` — `parse_model()` function.
- Look at: how it loops through the YAML, looks up each module name in `base_modules`, applies depth/width multipliers, and stitches layers together.
- Mental model: **the bridge** between Step 1 (YAML) and Steps 2–3 (Python classes). When custom modules are added later, this is the file that needs to know about them.

**Step 5 — High-level API and training loop (skim only):**
- `ultralytics-main/ultralytics/models/yolo/model.py` — `YOLO` class.
- `ultralytics-main/ultralytics/engine/model.py` — base `Model` class with `.train()` / `.val()` / `.predict()`.
- `ultralytics-main/ultralytics/engine/trainer.py` — `BaseTrainer` (dense; skim only).
- Mental model: `YOLO("yolo26.yaml")` calls `parse_model()` (Step 4) to build the network; `.train()` hands off to `BaseTrainer`. Just know where the entry points live.

---

## Optimization Roadmap (Current Phase)

### Files We Will Modify

| File | What We Change | Why |
|---|---|---|
| `ultralytics/nn/modules/conv.py` | CBAM already exists here (lines 512–613) — no changes needed | CBAM module is ready to use |
| `ultralytics/nn/modules/block.py` | Possibly add CBAM-wrapped C3k2 variant | Cleaner YAML integration |
| **`ultralytics/nn/tasks.py`** | Register CBAM in `base_modules` (line 1575) | Required to use CBAM in YAML |
| **`ultralytics/cfg/models/26/yolo26.yaml`** | Insert CBAM after backbone stages, add P2 path | The actual model structure change |
| `ultralytics/cfg/default.yaml` | Adjust training hyperparams (lr, augmentation, loss weights) | Training recipe tuning |

### Improvement Attempts (tracked, one at a time)

Each attempt gets a short code name and a results subfolder. We compare against the baseline `yolo26m_seg_20260628_172809/`.

**Attempt A — CBAM in Backbone** (simplest, lowest risk)
- Insert `CBAM` after each C3k2 block in the backbone (or just before SPPF)
- Expected effect: better leaffolder feature attention → higher recall
- Files: `tasks.py` (register CBAM), `yolo26.yaml` (add CBAM layers), new `yolo26-cbam.yaml`

**Attempt B — P2 High-Resolution Feature Layer** (medium complexity)
- Add P2/4 feature path through the neck for small object detection
- Expected effect: leaffolder recall improvement for small/dense targets
- Files: new `yolo26-p2.yaml`

**Attempt C — Dice Loss for Segmentation** (loss function change)
- Replace or combine BCE mask loss with Dice Loss
- Expected effect: mask edge precision improvement (mAP50-95)
- Files: loss computation in `head.py` or trainer config

**Attempt D — Combined (CBAM + P2 + Dice)** (integration)
- Combine the best-performing individual changes
- Files: merged YAML + all prior changes

### Key Project Files (outside ultralytics)

| File | Purpose |
|---|---|
| `训练结果分析_yolo26m_混合数据.md` | Detailed baseline analysis with charts |
| `yolo26m_seg_20260628_172809/` | Baseline training results (weights, curves, CSV) |
| `YOLO-pineapple.html` | Reference paper (CBAM + SPP on YOLOv8 for UAV) |

### Training Config Reference (from baseline)

Extracted from `args.yaml`: batch=8, imgsz=640, optimizer=auto(AdamW), lr0=0.01, lrf=0.01, warmup=5, mosaic=1.0, mixup=0.1, copy_paste=0.3, auto_augment=randaugment, erasing=0.4, close_mosaic=15, patience=100. **All optimization experiments should use identical settings for fair comparison**, unless the specific experiment changes a training hyperparameter.
