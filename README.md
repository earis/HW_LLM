# HW3：BERT 中文断句与标点恢复

本项目使用预训练中文 BERT，通过序列标注方式为去掉标点的中文文本恢复标点，并实现长文本滑动窗口和 Batch 推理。

## 项目结构

```text
hw3/
├── src/
│   ├── preprocess.py       # 语料清洗、标签构造和统计
│   ├── train.py            # BERT 微调、验证和 checkpoint 保存
│   ├── predict.py          # 测试、样例、滑动窗口和 Batch 基准
│   └── smoke_test.py       # 最小 CUDA 训练链路验证
├── data/processed/
│   └── statistics.json     # 可提交的清洗统计摘要
├── results/
│   ├── inference.json      # 初始推理结果和 Batch 基准
│   └── benchmark.json      # 公平串行/Batch 对比基准
├── report.md               # 实验报告记录
├── requirements.txt        # Python 依赖
└── hw_requirements.txt     # 作业要求
```

原始数据、处理后的 JSONL、模型缓存和 checkpoint 被 `.gitignore` 忽略，避免提交大文件。运行前需要自行准备人民日报语料和模型。

## 环境

本次实验使用：

- Python 3.10
- PyTorch 2.3.1+cu121
- Transformers 4.51.3
- NVIDIA GeForce RTX 4060

建议使用已经配置好的 `medsam` 环境。PyTorch 的 CUDA 版本应根据本机驱动和硬件选择，不建议盲目执行 `pip install -r requirements.txt` 覆盖现有 CUDA 安装。

```powershell
$py = "C:\Users\zhangxins\anaconda3\envs\medsam\python.exe"
Set-Location "C:\pytorch_m\hw\hw3"
```

## 运行流程

### 1. 准备模型

下载 `google-bert/bert-base-chinese`，保存到：

```text
models/bert-base-chinese/
```

本次实验通过 Hugging Face 镜像下载。也可以使用 Transformers 已有的本地缓存，但需要保证 tokenizer 和模型文件可加载。

### 2. 预处理语料

将人民日报文本放在被忽略的目录：

```text
1998-上半年-带音(1)/1998-上半年-带音/*.txt
```

然后运行：

```powershell
& $py -m src.preprocess `
  --raw-dir ".\1998-上半年-带音(1)\1998-上半年-带音" `
  --output-dir ".\data\processed"
```

预处理会删除文章 ID、词间空格、花括号注音和非目标排版符号，保留：

```text
O、，、。、？、！、；、：
```

统计摘要写入 `data/processed/statistics.json`。

### 3. 小规模验证

```powershell
& $py -m src.smoke_test `
  --data ".\data\processed\train.jsonl" `
  --cache-dir ".\models\bert-base-chinese" `
  --limit 8
```

### 4. 正式训练

```powershell
& $py -m src.train `
  --data-dir ".\data\processed" `
  --cache-dir ".\models\bert-base-chinese" `
  --output-dir ".\checkpoints\full" `
  --epochs 3 `
  --batch-size 8 `
  --max-length 384
```

显存不足时将 `--batch-size 8` 改为 `4`。不指定数据量限制参数时使用完整训练集和验证集。

### 5. 测试与批量推理

推理只加载已经训练好的 checkpoint，不会重新训练：

```powershell
& $py -m src.predict `
  --model-dir ".\checkpoints\full" `
  --test-limit 1000 `
  --batch-sizes "1,4,8,16" `
  --window-size 384 `
  --stride 300 `
  --output ".\results\inference.json"
```

该命令会输出：

- 测试集 accuracy 和非 `O` 类 Macro-F1；
- `孔子曰学而时习之不亦说乎` 样例；
- 现代中文样例；
- 滑动窗口 Batch size 基准。

公平的串行策略与 Batch 策略对比：

```powershell
& $py -m src.benchmark `
  --model-dir ".\checkpoints\full" `
  --test-data ".\data\processed\test.jsonl" `
  --text-limit 100 `
  --data-repeats 3 `
  --repeats 3 `
  --window-size 400 `
  --commit-size 300 `
  --strides "300,350,400" `
  --batch-sizes "1,4,8,16" `
  --output ".\results\benchmark.json"
```

## 实验结果

完整过程、清洗统计、训练指标、测试指标、样例输出和 Batch 基准见 [report.md](report.md)。当前最终推理结果见 [results/inference.json](results/inference.json)。

## GitHub 提交说明

提交前确认以下内容未被 Git 跟踪：

```powershell
git status --short
git check-ignore -v models checkpoints data/processed/train.jsonl
```

如果需要复现实验，需另外准备原始语料、预训练模型和本地 checkpoint；报告中的统计摘要和最终推理结果可以直接查看。
