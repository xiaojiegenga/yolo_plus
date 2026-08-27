# runs

本目录保存 Ultralytics 生成的完整原始实验结果，包括：

- `args.yaml`；
- `results.csv`；
- 曲线和预测图；
- `weights/best.pt` 与 `weights/last.pt`。

云端训练和本地保存统一使用 `runs/<run-id>/`。云端打包后，本地通过
`scripts/transfer_run.py unpack` 恢复相同目录结构，后续分析无需修改路径。

目录内容不提交 Git，也不要覆盖已有 Run；需要重跑时换一个新的 Run ID。
