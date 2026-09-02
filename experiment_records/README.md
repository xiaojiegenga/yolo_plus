# experiment_records

本目录只在本地电脑维护，保存可提交 Git 的参数优化分析、期刊正式实验记录和汇总表。

- `parameter_tuning/<run-id>.md`：用于决定总表表 1 的参数优化 Run 分析；结果只回填总表表 2。
- `runs/_template.md`：正式 Run 记录模板。
- `runs/<run-id>.md`：参数冻结后、可用于期刊对比的正式实验记录。
- `comparison.csv`：仅汇总可用于期刊对比的正式实验；`data` 列记录数据版本。
- 其他已有子目录：旧仓库的历史实验记录，保持原位。

云服务器不修改本目录。参数优化 Run 回传并解包后，不运行
`fill_results_table.py`：在 `parameter_tuning/` 写单次分析，并把性能数据填入
`云服务器实验设计与记录表.md` 表 2。

正式参数冻结后，可用于期刊对比的 Run 才运行：

```powershell
$formalRunId = 'replace-after-freeze'
python scripts/fill_results_table.py --run-dir "runs/${formalRunId}" --run-id "$formalRunId" --data data-v2
```

随后根据 `args.yaml`、`results.csv` 和验证输出填写 `runs/<run-id>.md`。不需要附加
哈希、manifest 或备份证明。

10 epoch 预检不进入 `comparison.csv`，也不创建参数优化或正式 Run 分析。

`云服务器实验设计与记录表.md` 只保存所有实验的表格数据，不写实验原因、结果解释
或结论段落。
