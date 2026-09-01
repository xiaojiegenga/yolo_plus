# experiments

本目录保存由本地电脑维护并提交 GitHub 的实验参数。云服务器只拉取已提交的 YAML，
不在云端临时改参数。

- `data-v2-abl-000-y26m-b16-s42.yaml`：data-v2 源码消融阶段 0 的正式 YOLO26m-seg
  Baseline 配置；不包含 Attention、Dice 或 P2Head 改动。
- `yolo26m_seg_5090.yaml`：历史参数优化 P2 配置，用于说明冻结配方的来源，不再作为
  正式消融 Run 配置。
- `yolo_data_v2_cloud.yaml`：云端 data-v2 数据路径和类别定义。
- 其他 YAML：旧仓库的历史配置，不改写历史含义。

正式消融统一继承 `data-v2-abl-000-y26m-b16-s42.yaml` 的训练配方，后续模块实验只能
改变对应源码结构或损失因素，并使用新的 Run ID。当前只完成 `000` 配置准备，训练仍由
用户在云服务器手动启动。

训练入口的字段映射固定为：

| YAML 字段 | 用途 |
|---|---|
| `experiment` | 默认 Run 名称前缀 |
| `model` | 传给 `YOLO(...)` |
| `data` | 解析为数据 YAML，再传给 `model.train(data=...)` |
| `train` | 其余键传给 `model.train(...)` |

旧配置中的哈希字段不会被当前入口读取或校验。
