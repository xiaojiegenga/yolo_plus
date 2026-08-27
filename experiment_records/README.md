# experiment_records

本目录只在本地电脑维护，保存可提交 Git 的轻量实验记录。

- `runs/_template.md`：新 Run 记录模板。
- `runs/<run-id>.md`：新实验的参数、结果及全部分析文字。
- `comparison.csv`：从 `results.csv` 回填的机器可读汇总表。
- 其他已有子目录：旧仓库的历史实验记录，保持原位。

云服务器不修改本目录。完整 Run 回传并解包到本地 `runs/<run-id>/` 后，再运行：

```powershell
python scripts/fill_results_table.py --run-dir runs/<run-id> --run-id <run-id>
```

随后根据 `args.yaml`、`results.csv` 和验证输出填写 Run 记录。不需要附加哈希、
manifest 或备份证明。

`云服务器实验设计与记录表.md` 只保存所有实验的表格数据，不写实验原因、结果解释
或结论段落。
