# experiments

本目录保存由本地电脑维护并提交 GitHub 的实验参数。云服务器只拉取已提交的 YAML，
不在云端临时改参数。

- `data-v2-abl-000-y26m-b16-s42.yaml`：data-v2 源码消融阶段 0 的正式 YOLO26m-seg
  Baseline 配置；不包含 Attention、Dice 或 P2Head 改动。
- `data-v2-abl-001-p2head-b16-s42.yaml`：C 独立消融，轻量 P2Head；使用官方预训练权重，train 配方与 000 相同；本地检查已通过，待云端预检和正式训练。
- `yolo26m_seg_5090.yaml`：历史参数优化 P2 配置，用于说明冻结配方的来源，不再作为
  正式消融 Run 配置。
- `yolo_data_v2_cloud.yaml`：云端 data-v2 数据路径和类别定义。
- 其他 YAML：旧仓库的历史配置，不改写历史含义。

正式消融统一继承 `data-v2-abl-000-y26m-b16-s42.yaml` 的训练配方，后续模块实验只能
改变对应源码结构或损失因素，并使用新的 Run ID。000、A1、A2、B 已完成；本分支的 C
代码与配置已通过本地验证，云端预检与正式训练待用户执行。

训练入口的字段映射固定为：

| YAML 字段 | 用途 |
|---|---|
| `experiment` | 默认 Run 名称前缀 |
| `model` | 传给 `YOLO(...)` |
| `pretrained` | 自定义结构构建后加载的官方初始化权重 |
| `data` | 解析为数据 YAML，再传给 `model.train(data=...)` |
| `train` | 其余键传给 `model.train(...)` |

旧配置中的哈希字段不会被当前入口读取或校验。
