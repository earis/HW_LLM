# ModernBERT 组件消融实验

本目录包含 ModernBERT 风格 mini-BERT 的设计报告、训练脚本和正式实验结果。

## 文件说明

- [report.md](./report.md)：设计、消融协议、实验结果与限制；
- [train_ablation.py](./train_ablation.py)：数据预处理、模型训练和指标计算；
- [results/dataset.json](./results/dataset.json)：正式实验数据统计；
- [results/raw_results.json](./results/raw_results.json)：12 次实验的原始结果；
- [results/summary.json](./results/summary.json)：按变体聚合的均值和标准差；
- [hw_requirement](./hw_requirement)：课程作业要求。

模型 checkpoint 未纳入仓库，以避免提交大体积二进制文件。训练脚本支持通过
`--save-checkpoints` 在本地生成 checkpoint。

## 实验配置

正式实验使用 1998 年 1 月至 6 月人民日报语料，按月份划分：

- 1 月至 4 月：训练；
- 5 月：验证；
- 6 月：测试。

mini-BERT 配置：

- 6 层；
- hidden size 256；
- 8 个 attention heads；
- 8K 词表；
- 最大长度 512；
- 1200 个优化步；
- 有效 batch size 32；
- 每个变体约 19.66M tokens；
- 3 个随机种子：17、29、43。

比较的变体：

```text
baseline  : RoPE + window 128 + 周期性全局层
window256 : RoPE + window 256 + 周期性全局层
no_global : RoPE + window 128 + 无全局层
rope_abs  : 可学习绝对位置编码 + window 128 + 周期性全局层
```

## 主要结果

| 变体 | 测试 MLM loss | 版面探针准确率 |
|---|---:|---:|
| baseline | 4.3920 ± 0.0118 | 0.2341 ± 0.0181 |
| window256 | 4.4049 ± 0.0013 | 0.2186 ± 0.0097 |
| no_global | 4.3751 ± 0.0126 | 0.2485 ± 0.0096 |
| rope_abs | 6.3468 ± 0.0071 | 0.1259 ± 0.0143 |

在当前语料、模型规模和版面分类探针上，结果支持保留 RoPE 和较小窗口；
本轮实验没有证明周期性全局层的必要性。

## 运行

```powershell
python train_ablation.py --variant all --steps 1200 `
  --batch-size 8 --grad-accumulation 4 `
  --probe-epochs 10 --seeds 17 29 43 `
  --save-checkpoints --output-dir runs
```

当前实现的局部注意力是在完整 attention 矩阵上施加 mask，因此不能把本实验的
延迟直接解释为真正稀疏 attention kernel 的加速结果。mini-BERT 结果只用于方向性
验证，不能等价于完整 3 亿参数、8K 上下文模型的性能证明。
