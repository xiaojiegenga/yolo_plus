# YOLO26 项目 Git/GitHub 管理指南

> 最后更新：2026-07-28
> 仓库：`https://github.com/xiaojiegenga/yolo_plus.git`

---

## 1. Git 在本项目中解决什么问题

Git 用于记录：

- 每个模型版本修改了哪些源码；
- 哪次训练对应哪个 commit；
- 如何回到 Baseline；
- 如何独立比较 CBAM、P2、Dice；
- 哪些失败尝试只用于复盘。

Git 不用于保存：

- `.pt` 权重；
- 完整训练输出；
- 数据集；
- 本机路径配置；
- `results/` 本地原始备份。

---

## 2. 物理路径与分支

源码始终位于：

`E:\Study\DeepCNN\yolo26\yolo_plus\ultralytics-main`

Git 分支改变的是这个路径中的文件内容。

```text
同一个 ultralytics-main 目录
        │
        ├─ 切到 main       → Baseline 源码
        ├─ 切到 v1-cbam    → CBAM 源码
        ├─ 切到 v2-p2      → P2 源码
        └─ 切到 v3-dice    → Dice 源码
```

由于使用 `pip install -e`，训练会读取当前分支中的源码。

---

## 3. 当前分支策略

```text
main                         Baseline + 公共工具和文档
├─ v1-cbam                   独立 CBAM 实验
├─ v2-p2                     独立 P2 实验（新建、尚未改模型）
├─ v3-dice                   独立 Dice 实验
└─ v4-combined               有效模块组合

archive/v2-p2-failed         旧 P2 失败实现，只供复盘
```

规则：

1. `main` 保持可复现、可构建；
2. V1/V2/V3 都从同一个 Baseline 建立；
3. 不从 V1 创建 V2；
4. 失败实现放入 `archive/`，不合并回正式分支；
5. V4 只整合已经验证有效的模块。

---

## 4. 开始一个改进版本

以 V2 为例：

```powershell
git switch main
git status
git switch -c v2-p2
```

开始改代码前检查：

```powershell
git diff main...v2-p2
```

新建空分支时，该命令应没有模型源码差异。

---

## 5. 每次源码改动后的记录

查看修改：

```powershell
git status
git diff
```

提交时只选择本次实验相关文件：

```powershell
git add <本次修改的文件>
git commit -m "feat(v2-p2): add P2 prediction scale for segmentation"
```

不要直接使用不加检查的全量提交。

提交信息建议：

| 前缀 | 用途 |
|---|---|
| `feat` | 新功能或新模块 |
| `fix` | 修复已确认的问题 |
| `docs` | 文档更新 |
| `test` | 测试或验证 |
| `archive` | 失败版本归档 |
| `chore` | 结构、配置等维护 |

---

## 6. 训练与 commit 的绑定

每次训练前记录：

```powershell
git branch --show-current
git rev-parse HEAD
git status --short
```

这些信息写入：

`experiment_records/<version>/run_record.md`

如果训练时工作区不干净，必须说明未提交改动，避免以后无法复现。

---

## 7. 结果文件策略

### 7.1 `results/`：本地原始备份

- 用户手动维护；
- 保存完整 Baseline、V1 和未来成功训练；
- 整个目录被 `.gitignore` 忽略；
- 不执行 `git add -f results/...`；
- 不由助手自动移动、覆盖或删除。

### 7.2 `experiment_records/`：Git 可跟踪记录

保存：

- `run_record.md`；
- 汇总 CSV；
- 配置摘要；
- val 指标；
- commit；
- 原始 run 路径和权重哈希。

不保存权重和大批图片。

---

## 8. 失败分支的处理

失败实验不要立刻永久删除。

推荐：

```powershell
git branch -m archive/<原分支名>-failed
```

归档分支必须在文档中注明：

- 为什么失败；
- 哪些结论已确认；
- 哪些只是推测；
- 是否允许将代码合并回正式分支。

旧 V2 的归档分支：

`archive/v2-p2-failed`

它不能作为新版 V2 的起点。

---

## 9. GitHub 同步

首次推送新分支：

```powershell
git push -u origin v2-p2
```

后续推送：

```powershell
git push
```

推送前检查：

```powershell
git status
git log --oneline -5
```

禁止在未确认目标分支时使用强制推送。

---

## 10. 常用恢复方法

查看某个文件的历史：

```powershell
git log --oneline -- <文件路径>
```

比较两个版本：

```powershell
git diff main...v2-p2
```

查看旧版本但不修改当前文件：

```powershell
git show <commit>:<文件路径>
```

切换分支前必须先检查：

```powershell
git status
```

如果存在未提交的用户文件，不进行覆盖、重置或清理。

---

## 11. 当前项目状态

- `main`：公共结构整理基线；
- `v1-cbam`：已完成；
- `archive/v2-p2-failed`：旧 V2 失败现场；
- `v2-p2`：从整理后的 `main` 独立建立，模型尚未改动；
- `v3-dice`、`v4-combined`：后续需要重新确认是否从最新 `main` 建立；
- `results/`：本地保存，不再上传；
- `experiment_records/`：后续由助手持续更新。
