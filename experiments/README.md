# experiments

本目录保存由本地电脑维护并提交 GitHub 的实验参数。云服务器只拉取已提交的 YAML，
不在云端临时改参数。

- `data-v2-abl-000-y26m-b16-s42.yaml`：data-v2 源码消融阶段 0 的正式 YOLO26m-seg
  Baseline 配置；不包含 Attention、Dice 或 P2Head 改动。
- `data-v2-abl-100-srcbam-b16-s42.yaml`：正式消融 A 配置；只在 Backbone P3/P4 使用
  SR-CBAM，并从官方 `yolo26m-seg.pt` 迁移 Baseline 权重；A1 已完成但未通过门控。
- `data-v2-abl-a2-p3-zrcbam-b16-s42.yaml`：改进 A2 配置；只在 Backbone P3 使用
  零初始化加法残差 CBAM，训练参数与正式 `000` 完全相同。
- C 分支的 `data-v2-abl-001-p2head-b16-s42.yaml`：轻量 P2Head 单因素配置；正式训练已完成，best epoch 282，未通过门控。该配置与 C 源码在 `feature/data-v2-abl-c-p2head` 维护。
- `yolo26m_seg_5090.yaml`：历史参数优化 P2 配置，用于说明冻结配方的来源，不再作为
  正式消融 Run 配置。
- `yolo_data_v2_cloud.yaml`：云端 data-v2 数据路径和类别定义。
- 其他 YAML：旧仓库的历史配置，不改写历史含义。

正式消融统一继承 `data-v2-abl-000-y26m-b16-s42.yaml` 的训练配方，后续模块实验只能
改变对应源码结构或损失因素，并使用新的 Run ID。000、A1、A2、B、C 均已完成并登记；A1/B/C 未通过门控，A2 暂不组合。
根目录 `实验步骤.md` 保留 C 操作参考，已有 Run ID 不重复运行。

训练入口的字段映射固定为：

| YAML 字段 | 用途 |
|---|---|
| `experiment` | 默认 Run 名称前缀 |
| `model` | 传给 `YOLO(...)` |
| `pretrained` | 可选；自定义模型 YAML 构建后由 `model.load(...)` 迁移的权重 |
| `data` | 解析为数据 YAML，再传给 `model.train(data=...)` |
| `train` | 其余键传给 `model.train(...)` |

旧配置中的哈希字段不会被当前入口读取或校验。
