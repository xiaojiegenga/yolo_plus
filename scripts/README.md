# scripts

- `cloud_train_data_v2.py`：云服务器入口；保留镜像自带的 CUDA PyTorch，只在缺依赖时安装仓库内 Ultralytics。
- `train_yolo26_seg.py`：读取实验 YAML 并启动训练。
- `transfer_run.py`：云端打包完整 Run，本地解包回 `runs/`。
- `fill_results_table.py`：从原始 `results.csv` 更新 `experiment_records/comparison.csv`。

云端只运行训练和打包；`fill_results_table.py` 只在本地使用。当前入口不执行
哈希、manifest 或备份检查。长时间正式训练必须由用户明确启动。
