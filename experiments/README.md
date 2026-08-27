# experiments

本目录保存由本地电脑维护并提交 GitHub 的实验参数。

- `yolo26m_seg_5090.yaml`：当前 5090 训练的默认配置。
- `yolo_data_v2_cloud.yaml`：云端数据路径和类别定义。
- 其他 YAML：旧仓库的历史配置，不改写历史含义。

新增实验时在本地复制配置、修改参数、确定 Run ID，然后提交并推送。云服务器只
拉取已提交的 YAML，不临时编辑参数。

训练入口的字段映射固定为：

| YAML 字段 | 用途 |
|---|---|
| `experiment` | 默认 Run 名称前缀 |
| `model` | 传给 `YOLO(...)` |
| `data` | 解析为数据 YAML，再传给 `model.train(data=...)` |
| `train` | 其余键传给 `model.train(...)` |

旧配置中的哈希字段不会被当前入口读取或校验。
