# data

这里放本地数据集、软链接或云盘挂载点，实际数据不提交 Git。云服务器默认把
数据放在 `/root/yolo_data`，其中直接包含 `images/` 和 `labels/`。

云端默认数据配置是 `experiments/yolo_data_v2_cloud.yaml`。如果使用其他固定
路径，应在本地修改该 YAML 并推送到 GitHub，不在云端临时修改。

当前数据口径：

- train / val / test 图片数：938 / 117 / 118
- `Rice leaffolder`：4027
- `Rice stemborers`：1110
