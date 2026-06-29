# GitHub 代码管理指南

> **仓库**: yolo-plus | **平台**: GitHub | **用途**: YOLO26 改进实验的代码版本控制

---

## 1. 分支策略

```
main ────────────────────────────────────────────── (baseline 8.4.80 原始代码)
  │
  ├── v1-cbam ─────── (只加 CBAM 注意力模块)
  │
  ├── v2-p2 ───────── (只加 P2 高分辨率特征层)
  │
  ├── v3-dice ─────── (只改 Dice Loss)
  │
  └── v4-combined ─── (CBAM + P2 + Dice 全部集成)
```

### 分支用途

| 分支 | 改动范围 | 训练后对比对象 |
|---|---|---|
| `main` | ultralytics 8.4.80 原始代码（baseline） | 对照组 |
| `v1-cbam` | 仅注册 CBAM + 新增 yolo26-cbam.yaml | vs main |
| `v2-p2` | 仅新增 yolo26-p2.yaml（含 P2 层） | vs main |
| `v3-dice` | 仅修改 loss 计算（Dice Loss） | vs main |
| `v4-combined` | 合并 v1+v2+v3 三处改动 | vs main |

### 为什么不直接在 main 上改？

- 每个改进独立一个分支，互不干扰
- 可以随时切回 main 看原始代码
- 训练时激活对应分支，`pip install -e .` 自动生效
- GitHub 上可以直观看到每个分支改了什么

---

## 2. 操作流程

### 2.1 初始化（只做一次）

```bash
# 在 yolo_plus 目录下
cd e:/Study/DeepCNN/yolo26/yolo_plus

# 初始化 git
git init

# 创建 .gitignore（排除不需要版本控制的文件）
# （见第 3 节）

# 首次提交 — main 分支 = baseline 原始代码
git add .
git commit -m "init: baseline ultralytics 8.4.80, yolo26m-seg training results"
```

### 2.2 开始一个新改进（以 v1-cbam 为例）

```bash
# 从 main 创建新分支
git checkout -b v1-cbam

# ... 修改代码（编辑 tasks.py, 创建 yolo26-cbam.yaml 等）...

# 提交改动
git add ultralytics/nn/tasks.py
git add ultralytics/cfg/models/26/yolo26-cbam.yaml
git commit -m "v1: register CBAM in tasks.py, add yolo26-cbam.yaml"

# 推送分支到 GitHub
git push -u origin v1-cbam
```

### 2.3 改进做完后切回 main

```bash
# 先提交当前分支的所有改动
git add -A
git commit -m "完成 v1-cbam 实验"

# 切回 main
git checkout main
```

### 2.4 查看改了什么

```bash
# 看某个分支和 main 的差异
git diff main..v1-cbam

# 看某个分支有哪些提交
git log v1-cbam --oneline

# 看所有分支
git branch -a
```

---

## 3. .gitignore（排除规则）

以下文件/文件夹**不**提交到 git：

```gitignore
# Python 缓存
__pycache__/
*.pyc
*.pyo
*.egg-info/

# 虚拟环境
venv/
.venv/
.env

# 训练权重（太大，不适合 git）
*.pt
*.pth

# 训练结果 runs（太大）
runs/
runs_seg/

# IDE 配置
.vscode/
.idea/

# 系统文件
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Conda
conda-meta/

# 数据集
datasets/
*.yaml  # 数据配置文件（含路径信息）
```

> ⚠️ `yolo_data.yaml` 包含本地路径，不提交到 GitHub。训练脚本中的路径也需要改为相对路径或环境变量。

---

## 4. 训练时使用分支

```bash
# 1. 切到目标分支
git checkout v1-cbam

# 2. 重新 pip install -e（确保 Python 加载的是当前分支的代码）
pip install -e .

# 3. 在 yolo26 环境中运行训练
conda activate yolo26
cd E:/Study/DeepCNN/yolo26/code
python train_yolov26_seg.py

# 4. 训练完成后，把 results 复制到 yolo_plus 对应版本文件夹
# （如 yolo_plus/results/v1-cbam/）
```

---

## 5. 实验结果归档

每个版本的训练结果复制到：

```
yolo_plus/
  results/
    baseline/        ← yolo26m_seg_20260628_172809/
    v1-cbam/         ← 训练后复制 run 文件夹到这里
    v2-p2/
    v3-dice/
    v4-combined/
```

同时在 [code_plus.md](code_plus.md) 的对比表中填入指标。

---

## 6. GitHub 仓库信息

| 项目 | 值 |
|---|---|
| 仓库名 | yolo-plus |
| 默认分支 | main |
| 远程地址 | https://github.com/xiaojiegenga/yolo_plus.git |

### 创建远程仓库步骤

```bash
# 1. 在 GitHub 网页上创建新仓库 "yolo-plus"（不要勾选 README）

# 2. 关联远程
git remote add origin https://github.com/<你的用户名>/yolo-plus.git

# 3. 推送 main
git push -u origin main

# 4. 推送各分支
git push -u origin v1-cbam
git push -u origin v2-p2
# ...
```

---

*创建日期: 2026-06-29 | 更新: 2026-06-29 — git 仓库已初始化，5 个分支已创建*
