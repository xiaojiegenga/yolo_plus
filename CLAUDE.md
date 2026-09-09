# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言

始终用中文回复。

## 这是什么项目

无人机航拍水稻害虫实例分割实验仓库（继续使用 `xiaojiegenga/yolo_plus`，正式 Baseline
分支为 `cloud/data-v2-5090`，总体分析与改进 A 源码分支为 `feature/data-v2-abl-a-attention`）。
数据集 `rice-pest-data-v2`，2 类（`Rice leaffolder`、
`Rice stemborers`），主选择指标为 **Val Mask mAP50-95**。

A1、A2、B、C、D1 均已完成正式训练。A1/B/C 未通过门控，A2 近似持平，D1 保留正向候选。
当前目标是验证两个有效的结构改进；E：Neck DySample 已实现，位于 `feature/data-v2-abl-e-dysample`，待云端训练后决定 D1+E。
数据集和训练配方固定，数据集排查不作为结构实验的前置步骤。实时状态与结果以 `PROGRESS.md` 为准。
本机环境见 `LOCAL_SETUP.local.md`。A、B、C 知识文档与结果统一在总体分析分支维护；
阶段总结见 `experiment_records/data-v2-ab-summary.md`。

开始任何工作前，先按顺序阅读（`PROGRESS.md` 是快速了解当前进展的入口，其余是规则与背景）：

1. `PROGRESS.md`（当前进展快照）
2. `云服务器实验项目交接.md`
3. `云服务器实验设计与记录表.md`
4. `AGENTS.md`
5. 本次涉及目录里的 `README.md`

用户当前指令优先于以上文档；已有历史结果不得因整理目录被改写。

## 工作模式：三地分工

本地电脑是**唯一**开发与记录环境，GitHub 是中转站，云服务器是**一次性**训练节点：

```text
本地改代码/配置并 push → 云端 clone/pull → 云端训练并打包 → 本地接收、分析、记录
```

- 云端不维护独立配置、不填实验表，只拉取已提交版本训练并打包 Run。
- 云端 Run 回传后，`runs/<run-id>/` 是本地长期保存的原始结果。

## 常用命令

所有命令在仓库根目录执行。项目没有全局 lint 配置。

```bash
RUN_ID="replace-with-run-id"
CONFIG="experiments/data-v2-abl-e-dysample-b16-s42.yaml"

# 本地：读取配置，只打印最终参数、不训练（先验证配置）
python scripts/train_yolo26_seg.py --config "${CONFIG}" --run-name "${RUN_ID}" --dry-run

# 云端：10 epoch 短预检（只用于兼容性/成本判断，不进论文精度排名）
python scripts/cloud_train_data_v2.py --config "${CONFIG}" --preflight10 --run-name "${RUN_ID}"

# 云端：参数优化或正式训练（使用唯一 Run ID，且由用户明确启动）
python scripts/cloud_train_data_v2.py --config "${CONFIG}" --run-name "${RUN_ID}"

# 云端：把 runs/<run-id> 打成 exports/<run-id>.zip
python scripts/transfer_run.py pack --run-id "${RUN_ID}"

# 本地：解包 ZIP 回 runs/<run-id>/
python scripts/transfer_run.py unpack --archive "exports/${RUN_ID}.zip"

# 本地：仅对参数冻结后、用于期刊对比的正式 Run 回填 comparison.csv
python scripts/fill_results_table.py --run-dir "runs/${RUN_ID}" --run-id "${RUN_ID}" --data data-v2

# 本地：改进模块的聚焦结构测试（在 ultralytics-main/ 目录内执行）
python -m pytest tests/test_dysample_model.py -q
# 根目录也可执行 python scripts/check_dysample.py --weights yolo26m-seg.pt --cuda，无需 pytest
```

`cloud_train_data_v2.py` 把全部参数原样转发给 `train_yolo26_seg.py`。不给 `--config`
时默认落到 `experiments/yolo26m_seg_5090.yaml`（参数优化配置），正式消融必须显式指定
对应的 `experiments/data-v2-abl-*.yaml`。

`train_yolo26_seg.py` 支持 `--config`、`--experiment`、`--model`、`--data`、
`--pretrained`、`--run-name`、`--preflight10`、`--dry-run` 覆盖配置。Run 名称只允许
英文字母、数字、点、下划线和连字符；`runs/<run-id>/` 已存在时入口直接报错退出，
这是"不覆盖历史 Run"规则的机制保障。

`fill_results_table.py` 还支持 `--dry-run`（只打印将写入的行）、`--status`、`--split`、
`--branch`、`--notes`、`--table`。

本地电脑不做任何模型训练：只改本目录下的源码与配置、跑 `--dry-run` 与结构测试、
接收并分析云端回传的 Run。全部训练都在云服务器上，由云端 `git pull` 拉取已推送的
版本执行。

本机为 Windows，README 中的本地 `.venv` 不随仓库分发；PowerShell 下执行脚本前设置
`$env:YOLO_CONFIG_DIR`（指向仓库内 `.cache\ultralytics`）与 `$env:PYTHONUTF8 = '1'`，
解释器细节见 `LOCAL_SETUP.local.md`（仅本机，不入 Git）。

## 架构：两类 YAML + 一个入口

训练入口是 `scripts/train_yolo26_seg.py`（云端由 `cloud_train_data_v2.py` 包一层：
保留镜像自带 CUDA PyTorch、缺依赖时才 `pip install -e ultralytics-main`）。它读取
一个**简化训练配置 YAML**，字段映射固定为：

| 顶层字段 | 用途 |
|---|---|
| `experiment` | 默认 Run 名称前缀 |
| `model` | 传给 `YOLO(...)`（本地路径或官方模型名，如 `yolo26m-seg.pt`） |
| `pretrained` | 可选；自定义 YAML 建模后传给 `model.load(...)` 的官方模型名或本地权重 |
| `data` | 解析为数据 YAML，再传给 `model.train(data=...)` |
| `train` | 其余键原样传给 `model.train(**runtime)` |

注意区分两种 YAML：

- **数据 YAML**（如 `experiments/yolo_data_v2_cloud.yaml`）：`path`/`train`/`val`/
  `test`/`nc`/`names`，`nc` 必须为 2，由 `validate_data_yaml()` 校验。
- **训练配置 YAML**（如 `experiments/data-v2-abl-a2-p3-zrcbam-b16-s42.yaml`）：顶层
  `experiment`/`model`/可选 `pretrained`/`data`/`train`。旧的完整配置（如
  `yolo26m_seg_baseline_train.yaml`）里的
  `profile_id`、`*_sha256` 等字段不会被当前入口读取或校验，仅作历史参照。

`train_yolo26_seg.py` 还会强制写入 `runtime["data"]`、`project`、`name`、`exist_ok=False`，
并把顶层 `experiment/model/data` 元数据与 `train` 参数分离，避免误传给 `model.train()`。

`ultralytics-main/` 是 vendored 的 YOLO 源码（基线 8.4.80）：入口把该目录插入
`sys.path` 再 `from ultralytics import YOLO`。改进实验改的是这个目录里的模型源码，
不改训练参数。本地对该目录的修改推送后，云端拉取的就是训练时实际执行的代码。

## 一个改进模块要落到的五个位置

A（SR-CBAM / ZR-CBAM）与 C（P2Head）都按同一套路径接入 vendored 源码，新增改进时
沿用即可：

1. `ultralytics/nn/modules/block.py`（或 `conv.py`）实现模块本体；
2. `ultralytics/nn/modules/__init__.py` 导出类名；
3. `ultralytics/nn/tasks.py` 的 `parse_model` 中注册：加入 import、通道推导分支，
   以及 M/L/X 缩放所用的 `{C3k2, ...}` 集合，否则宽度缩放不生效；
4. `ultralytics/cfg/models/26/<name>.yaml` 新建模型结构 YAML，保持 Backbone/Head
   层号与 Baseline 一致，改进才是单变量；
5. `ultralytics-main/tests/test_<module>_model.py` 补聚焦测试：形状与反向、相对
   Baseline 的层索引与权重键差异、恒等初始化等。

自定义结构靠训练配置顶层 `pretrained: yolo26m-seg.pt` 迁移官方权重——入口先按
`model` YAML 建结构，再 `model.load(pretrained)`，形状不匹配的张量自动跳过。

## 两类训练结果记录链条

`fill_results_table.py` 只用于正式训练，是连接原始结果与汇总表的脚本：

- 读 `runs/<run-id>/results.csv`；
- 按 Ultralytics 官方分割 fitness（`metrics/mAP50-95(B) + metrics/mAP50-95(M)`）取最高的一行作为该 Run 的最佳结果；
- 写出总体 `mask_p/r/map50/map50_95` 到 `experiment_records/comparison.csv`（upsert，按 `version` 列）；分类别指标从独立验证输出或曲线证据人工补录。

参数优化 Run 的分析写到 `experiment_records/parameter_tuning/<run-id>.md`，性能只填
`云服务器实验设计与记录表.md` 表 2，不运行 `fill_results_table.py`。参数冻结后、可用于
期刊对比的正式实验才写 `experiment_records/runs/<run-id>.md`（模板
`experiment_records/runs/_template.md`）并进入 `comparison.csv`。总表只回填计划、参数、
状态和结果数据，不写原因、解释或结论。

## 目录约定

| 路径 | 内容 | Git |
|---|---|---|
| `experiments/` | 实验参数 YAML | 跟踪 |
| `scripts/` | 训练、传输、结果回填脚本 | 跟踪 |
| `experiment_records/` | 参数优化分析、期刊正式 Run 记录与 `comparison.csv` | 跟踪 |
| `ultralytics-main/` | 模型源码 | 跟踪 |
| `runs/` | 完整原始结果（`args.yaml`、`results.csv`、`weights/` 等） | 不跟踪 |
| `exports/` | 传输 ZIP | 不跟踪 |
| `data/` | 数据集/挂载点 | 不跟踪 |

## 必须遵守的操作边界

- 未经用户明确要求，不启动正式长时间训练；10 epoch 预检不进论文精度排名。
- 总表表 2 只登记决定表 1 的参数优化训练；10 epoch 预检不进表 2 或 `comparison.csv`。
- 不覆盖已有 Run、ZIP、权重或历史记录；参数、模型代码或数据口径变化时必须换新 Run ID。
- Val 用于选方案；Test 只在全部方案冻结后统一评估，不用 Test 调参。
- RTX 5090 已确定；batch、workers、epochs 等正式参数仍待表 2 参数优化后冻结。
- 不提交数据、模型权重、完整 `runs/`、`exports/`、凭据或 SSH 配置。
- 只处理当前任务相关修改，不重置用户工作。


=== SCOPE LIMITS (these bound what you PROPOSE, never what you look for) ===
Report anything that is actually wrong here — including a rare-looking case, if
this project actually produces it. Then keep the fix in scope:
1. This is not a security paper. Verification is welcome; over-defense is not.
   Unless this project states otherwise, assume a cooperating operator on their
   own machine; if it has a real adversary, it will say so and that scope wins.
2. Do not add hashes, checksums or fingerprints unless the hash replaces a
   materially more expensive operation AND its result changes what happens next.
3. No defensive scaffolding: no feature flags, migration frameworks, compat
   layers or wrappers for cases that do not occur here.
4. No corner-case obsession: exotic encodings, symlink races, RTL text and
   millisecond races are out of scope unless the case is reachable through this
   project's supported use — its documented inputs, its published interface, its
   real data. Reachable is enough; you do not need a reproduction. Constructible
   in principle is not enough.
5. Where judgement is needed, judge. Do not replace it with a scoring table, a
   checklist, or a re-verification loop over something already settled.
6. None of this overrides security, migration, verification or review that the
   user, this project's own conventions, or a higher-priority rule asked for.
   Those were requested; they are the work, not scope creep.
7. Deliverable text is not a defense transcript. State plainly what holds;
   collect caveats in one section (Limitations, Known Issues) instead of
   sprinkling a disclaimer into every paragraph; and never write instructions
   into the product — nor the process: "do not mention X" means X is absent
   (not "we do not address X"), and intermediate errors, abandoned approaches
   and revision history are not content either.
8. Momentum is part of the deliverable. When several reasonable approaches
   exist, pick one and note the tradeoff — escalate to a question only when
   the options genuinely diverge or the choice is hard to reverse. A detail
   that does not block the goal gets recorded, not solved first. Stalling to
   avoid picking wrong is itself a wrong pick.
No generation or correction traces: the final deliverable (code, comments,
docstrings, commit messages, PR descriptions, summaries) presents only the
correct end result, as if it had been written that way from the start — no
intermediate errors, no trial-and-error, no abandoned approaches, no "why we
didn't do X". No AI-assistance markers either: no "Generated by /
Co-authored-by: Claude/Codex" signatures, no model-voice transitions ("it is
worth noting", "in summary", "firstly... secondly... finally"), no "here
is..." openers. Match the wording and level of detail to the project's
existing style.
Shapes already seen, for calibration. Examples, not a checklist — a real finding
is not dismissed by resembling one:
  H  hashing every row of two spreadsheets to answer what comparing cells answers
  H  writing checksum files that nothing ever reads
  E  hardening the accounts of an app that has no users and no deployment
  R  auditing your own patch all night while the feature stays unwritten
  R  a reviewer that returns a failing verdict on everything
  O  guards whose justification is the previous guard, not the requirement
And two that look like the above and are not. Report these:
  ✓  a digest that lets you skip re-reading a large file you already have
  ✓  a rare-looking input this project's own documentation example produces
Before running any check, answer: what specific failure would this detect, and
what would I do differently if it occurred? No answer means do not run it.
Say plainly when something is correct. Do not manufacture findings.
