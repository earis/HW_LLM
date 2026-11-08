# HW7 提交说明

## 建议提交文件

- `hw_requirement`：作业要求；
- `report.md`：第一部分，LLM 后训练发展路径与方法综述；
- `report_submit_part2.md`：第二部分，开源项目代码分析与 GridWorld 实践报告；
- `GridWorld.py`：可执行的 GridWorld、Mini-RLHF、PPO 和 DPO 实验代码；
- `image/`：报告引用的实验图片；
- `hw7_requirement_no_submit.jpg`：题目图片版本（如课程要求提交）。

## 可不提交文件

以下文件是写作过程中的草稿或重复副本，除非课程要求保留过程记录，否则不建议放入最终提交：

- `report_draft.md`
- `report_part2.md`
- `report copy.md`

## 运行验证

在仓库根目录执行：

```powershell
python .\hw7\GridWorld.py
```

脚本会运行 GridWorld、Q-learning、Mini-RLHF、PPO 和 DPO 实验，并输出主要指标。

## Git 提交前检查

```powershell
git status --short
git add hw7\README.md hw7\report.md hw7\report_submit_part2.md hw7\GridWorld.py hw7\image hw7\hw_requirement hw7\hw7_requirement_no_submit.jpg
git diff --cached --stat
git commit -m "Add HW7 post-training reports and experiments"
```

如果课程不要求提交题目图片，可以从 `git add` 命令中去掉 `hw7\hw7_requirement_no_submit.jpg`。
