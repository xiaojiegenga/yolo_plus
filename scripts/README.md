# scripts

- `check_p2proto.py`：D1 的 CPU 结构、seed 初始化、checkpoint 融合及官方权重迁移检查。
- `cloud_train_data_v2.py`：云服务器入口；普通训练保持原路径，`--local-refiner` 调度 H，并为原图评估补齐 `pycocotools`。
- `train_yolo26_seg.py`：读取实验 YAML 并启动训练。
- `train_local_refiner.py`：H 的薄入口；冻结 D1、生成 Train 候选、训练 crop refiner，并按每轮正式 Val 选择组合 `best.pt`。
- `local_refiner_config.py`、`local_refiner_training.py`：H 的固定配置、1:1:1 采样、训练和 Val 支持。
- `evaluate_local_refiner.py`、`local_refiner_coco.py`：使用同一精修 Predictor 完成原图尺寸 Val 与 COCO 指标。
- `local_refiner_reporting.py`：执行预注册门槛、GFLOPs 和云端 batch-1 FP16 测速。
- `transfer_run.py`：云端打包完整 Run，本地解包回 `runs/`。
- `fill_results_table.py`：仅从正式训练的 `results.csv` 按 Ultralytics 官方分割 fitness（Box mAP50-95 + Mask mAP50-95）选择最佳轮次并更新 `experiment_records/comparison.csv`；默认写入 `data-v2`，也可用 `--data` 指定数据版本。分类别指标仍从独立验证输出或曲线证据人工补录。

- `evaluate_c_small_val.py`：本地对 000/C 的 best.pt 做相同协议的小目标 Val 评估；需要 C 源码目录与 `pycocotools==2.0.11`，复现命令及口径见 `experiment_records/evaluations/data-v2-c-small-val.md`。

云端只运行训练和打包；`fill_results_table.py` 只在本地使用。当前入口不执行
哈希、manifest 或备份检查。预检不运行结果回填脚本，也不建立单次记录。参数优化
长训须由用户明确启动，只写 `experiment_records/parameter_tuning/` 和总表表 2；参数
冻结后的期刊正式训练才运行 `fill_results_table.py`。

训练成功后，`train_yolo26_seg.py` 会根据实际 Run ID 打印云端打包、本地 SCP 下载和
本地解包命令。SCP 命令中的 `SCP_PORT` 需要替换为当次实例页面显示的 SSH 端口。

H 使用 `python scripts/cloud_train_data_v2.py --local-refiner --config <yaml>`。其 1 epoch
预检和正式训练都会把原始诊断保存在独立 Run 目录；预检不回填论文总表。

自定义模型结构可在实验配置顶层写 `pretrained: yolo26m-seg.pt`。训练入口会先按
`model` YAML 构建结构，再调用 `model.load(pretrained)` 迁移官方权重；命令行
`--pretrained` 仍可覆盖配置值。
