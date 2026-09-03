# scripts

- `cloud_train_data_v2.py`：云服务器入口；保留镜像自带的 CUDA PyTorch，只在缺依赖时安装仓库内 Ultralytics。
- `train_yolo26_seg.py`：读取实验 YAML 并启动训练。
- `transfer_run.py`：云端打包完整 Run，本地解包回 `runs/`。
- `fill_results_table.py`：仅从正式训练的 `results.csv` 按 Ultralytics 官方分割 fitness（Box mAP50-95 + Mask mAP50-95）选择最佳轮次并更新 `experiment_records/comparison.csv`；默认写入 `data-v2`，也可用 `--data` 指定数据版本。分类别指标仍从独立验证输出或曲线证据人工补录。

云端只运行训练和打包；`fill_results_table.py` 只在本地使用。当前入口不执行
哈希、manifest 或备份检查。预检不运行结果回填脚本，也不建立单次记录。参数优化
长训须由用户明确启动，只写 `experiment_records/parameter_tuning/` 和总表表 2；参数
冻结后的期刊正式训练才运行 `fill_results_table.py`。

训练成功后，`train_yolo26_seg.py` 会根据实际 Run ID 打印云端打包、本地 SCP 下载和
本地解包命令。SCP 命令中的 `SCP_PORT` 需要替换为当次实例页面显示的 SSH 端口。
