# hw-4 唐诗宋词英译提示消融实验

本实验使用 Kimi 的 OpenAI 兼容 API，比较直接翻译、任务标准、边界示例、术语约定和关系保持等提示改动。

## 安全配置

不要把 API Key 写入代码、数据、报告或日志。先在 PowerShell 当前窗口设置：

```powershell
$env:MOONSHOT_API_KEY = "你的新 API Key"
```

建议先在 Kimi 控制台撤销已经出现在工作区文本中的旧 Key，再生成新 Key。

## 安装与冒烟测试

本项目只使用 Python 标准库，不需要下载模型，也不需要安装第三方包。

```powershell
# 如果账号使用其他可用模型，请先覆盖默认模型：
$env:MOONSHOT_MODEL = "kimi-k3.1-pro"
python .\src\run_experiment.py --smoke
```

冒烟测试默认使用 2 首开发集诗词和 C0、C1、C2，每个条件 1 次。

如果仍收到 404，请根据官方模型列表确认账号可用的模型 ID，再设置
`MOONSHOT_MODEL`。模型列表接口需要有效 Key，脚本不会猜测或回退到其他模型。

也可以直接查询当前 Key 有权限使用的模型：

```powershell
python .\src\run_experiment.py --list-models
```

## 正式实验

```powershell
python .\src\run_experiment.py --repeats 3
python .\src\analyze_results.py
```

正式实验会把请求和响应保存到对应 provider 的结果目录，统计表保存到 `results/`。如果中途停止，可以用 `--resume` 跳过已经成功保存的样本。

## 百度千帆 DeepSeek

Kimi 和百度实现相互独立。将百度 API Key 保存在本地 `BAIDU_KEY`（不要提交），然后运行：

```powershell
.\run_baidu_smoke.ps1
```

百度结果保存到 `results/baidu_raw/`，默认模型 ID 为 `deepseek-v4.1-flash`。如果千帆控制台显示的模型 ID 不同，可设置 `BAIDU_MODEL` 后再运行。模型 ID、权限和配额由百度账号决定。

完整运行也可以直接执行：

```powershell
.\run_baidu_formal.ps1
```

如需进行模型辅助七维评测：

```powershell
$env:BAIDU_API_KEY = (Get-Content .\BAIDU_KEY -Raw).Trim()
& "C:\pytorch_m\hw\hw2\gensim_text_classification_lab\llm_text\Scripts\python.exe" `
  ".\src\evaluate_results.py" --resume
```

模型辅助评测结果只能作为辅助证据，正式报告应说明其评测者不是人工专家。

## 提交前检查

```powershell
git --no-pager status --short
git check-ignore .\BAIDU_KEY .\KIMI_KEY
```

确认两个密钥文件均被忽略后再提交。API Key 不应出现在代码、报告、CSV、JSON、日志或聊天记录中。

## 数据与实验条件

数据在 `data/poems.json`，提示条件在 `data/prompts.json`。开发集用于选择提示，验证集只能在提示冻结后运行。所有条件共用模型和 API 参数，条件差异只来自提示内容。
