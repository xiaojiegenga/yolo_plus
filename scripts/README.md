# scripts

- `cloud_train_data_v2.py`：云服务器入口；保留镜像自带的 CUDA PyTorch，只在缺依赖时安装仓库内 Ultralytics。
- `train_yolo26_seg.py`：读取实验 YAML 并启动训练。
- `check_p2head.py`：不读取数据集，验证 C 的权重语义迁移、四尺度输出、Proto、分割损失反向和模型规模；`--weights yolo26m-seg.pt` 可额外核验官方权重。
- `transfer_run.py`：云端打包完整 Run，本地解包回 `runs/`。
- `fill_results_table.py`：仅从正式训练的 `results.csv` 按 Ultralytics 官方分割 fitness（Box mAP50-95 + Mask mAP50-95）选择最佳轮次并更新 `experiment_records/comparison.csv`；默认写入 `data-v2`，也可用 `--data` 指定数据版本。分类别指标仍从独立验证输出或曲线证据人工补录。

云端只运行训练和打包；`fill_results_table.py` 只在本地使用。当前入口不执行
哈希、manifest 或备份检查。预检不运行结果回填脚本，也不建立单次记录。参数优化
长训须由用户明确启动，只写 `experiment_records/parameter_tuning/` 和总表表 2；参数
冻结后的期刊正式训练才运行 `fill_results_table.py`。

训练成功后，`train_yolo26_seg.py` 会根据实际 Run ID 打印云端打包、本地 SCP 下载和
本地解包命令。SCP 命令中的 `SCP_PORT` 需要替换为当次实例页面显示的 SSH 端口。

C 配置顶层 `model` 为自定义 YAML，`pretrained: yolo26m-seg.pt` 为官方初始化来源。
入口在模型构建前固定 seed，加载时按 P3/P4/P5 语义迁移原分支；C checkpoint 复载使用同布局。
完整 C 云端流程和本地下载路径以根目录 `实验步骤.md` 为准。
