# HW7 第二部分：开源大模型强化学习代码初步分析

## 1. 分析目标

本部分不追求立即运行完整的大模型训练，而是回答一个更基础的问题：

> 论文中的 DPO、PPO、GRPO 公式，如何在真实开源项目中变成数据处理、概率计算、优势估计和参数更新代码？

推荐的阅读顺序是：

```text
自己的 GridWorld
    ↓
TRL：先看 DPO 的单步损失
    ↓
OpenRLHF：再看完整 PPO/RLHF 训练流程
    ↓
verl：最后看 GRPO 和 RLVR 的组内优势
```

三个项目的定位不同：

| 项目             | 重点              | 适合回答的问题                              |
| ---------------- | ----------------- | ------------------------------------------- |
| Hugging Face TRL | 算法模块化        | DPO loss 如何计算？                         |
| OpenRLHF         | 完整 RLHF 工程    | rollout、奖励、critic、actor 如何连接？     |
| verl             | 推理后训练与 GRPO | 为什么可以用组内相对奖励代替部分 PPO 组件？ |

本文代码摘录均来自项目公开仓库，保留了原项目的简短核心逻辑；完整代码、许可证和最新版本请以项目原文为准。

---

## 2. 项目一：Hugging Face TRL 的 DPO

项目：[huggingface/trl](https://github.com/huggingface/trl)

目标文件：[trl/trainer/dpo_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py)

TRL 把 DPO 封装成类似普通 Hugging Face Trainer 的训练器。对初学者来说，最值得先看的是：

1. 数据整理：`chosen` 和 `rejected` 如何拼接；
2. 当前策略的 log probability 如何计算；
3. 参考策略的 log probability 如何计算；
4. 两者相减得到 log ratio；
5. 用 sigmoid loss 更新当前模型。

### 2.1 偏好数据的结构

DPO 样本可以抽象成：

```python
{
    "prompt": "...",
    "chosen": "较好的回答",
    "rejected": "较差的回答",
}
```

在代码中，chosen 和 rejected 会分别形成两条完整序列，然后放入同一个 batch。重要的概念是：

```text
prompt + chosen       → 正样本
prompt + rejected     → 负样本
```

这与当前 GridWorld 中的：

```text
chosen_steps          → 回报更高的轨迹
rejected_steps        → 回报更低的轨迹
```

是同一种数据结构，只是 LLM 的动作变成了 token。

### 2.2 当前策略和参考策略的 log probability

TRL 中的核心逻辑可以简化为下面的形式（变量名与原项目保持对应关系）：

```python
chosen_logps, rejected_logps = logps.chunk(2, dim=0)

ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

chosen_logratios = chosen_logps - ref_chosen_logps
rejected_logratios = rejected_logps - ref_rejected_logps

delta_score = chosen_logratios - rejected_logratios
loss = -F.logsigmoid(beta * delta_score)
```

这里要分清三层含义：

- `logps`：当前正在训练的模型对回答的 log probability；
- `ref_logps`：冻结参考模型对回答的 log probability；
- `logratio`：当前模型相对于参考模型改变了多少。

最后的 `delta_score` 同时比较了 chosen 和 rejected：

\[
\Delta =
(\log\pi_\theta(y_w|x)-\log\pi_{\mathrm{ref}}(y_w|x))
-
(\log\pi_\theta(y_l|x)-\log\pi_{\mathrm{ref}}(y_l|x)).
\]

如果当前模型相对于参考模型更偏向 chosen，`delta_score` 会变大，DPO 损失会变小。

### 2.3 与自己的 DPO 代码对应

你 notebook 中的函数：

```python
def _sequence_log_prob(logits, steps):
    probs = _softmax_rows(logits)
    return sum(np.log(max(probs[s, a], 1e-12))
               for s, a, _ in steps)
```

对应 TRL 中的“计算一条回答的 token log probability”。区别是：

| GridWorld 教学代码        | TRL                                   |
| ------------------------- | ------------------------------------- |
| 一个动作是一个时间步      | 一个动作是一个 token                  |
| 表格 softmax              | Transformer 输出 vocabulary logits    |
| 轨迹 log probability 求和 | completion token log probability 求和 |
| `Q2` 作为参考策略       | reference language model              |

因此，你的 DPO 代码不是完整 LLM DPO，但保留了 DPO 最核心的数学结构。

### 2.4 建议重点阅读

建议暂时跳过视觉模型、PEFT 和各种 f-divergence 变体，先追踪：

```text
DataCollatorForPreference
    ↓
chosen/rejected batch
    ↓
model forward
    ↓
chosen_logps / rejected_logps
    ↓
reference logps
    ↓
delta_score
    ↓
logsigmoid loss
```

---

## 3. 项目二：OpenRLHF 的 PPO

项目：[OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

目标文件：[openrlhf/trainer/ppo_trainer.py](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_trainer.py)

OpenRLHF 的代码更接近完整工程系统。不要一开始阅读 Ray 和分布式通信细节，先理解一个训练 step 的数据流。

### 3.1 PPO 训练步的核心结构

OpenRLHF 的 `train_step` 可以概括为：

```python
def train_step(self, rollout_samples, global_step):
    experiences = self.experience_maker.make_experience_batch(
        rollout_samples
    )

    self.actor_model_group.append(experiences)
    self.critic_model_group.append(experiences)

    status = self.ppo_train(global_step)
    self.broadcast_to_vllm()

    return status, global_step + 1
```

每一行对应不同阶段：

1. `rollout_samples`：当前策略生成的 prompt-response；
2. `make_experience_batch`：计算奖励、旧策略概率、价值和优势等 PPO 所需数据；
3. `actor_model_group.append`：把经验交给 Actor；
4. `critic_model_group.append`：把经验交给 Critic；
5. `ppo_train`：更新策略模型和价值模型；
6. `broadcast_to_vllm`：把新策略参数同步给生成服务。

### 3.2 OpenRLHF 中的模型角色

真实 RLHF 通常至少涉及以下角色：

| 模块               | 作用                             |
| ------------------ | -------------------------------- |
| Actor/Policy       | 生成回答并被 PPO 更新            |
| Reference Model    | 提供 KL 约束，防止策略漂移过大   |
| Reward Model       | 给回答提供偏好分数               |
| Critic/Value Model | 估计状态价值，帮助计算 advantage |
| vLLM rollout       | 高效生成新的回答                 |

你的 GridWorld 版本可以这样对应：

```text
Q-learning policy       ≈ Actor
r_hat                   ≈ Reward Model
Q value                 ≈ Value Model 的简化版本
greedy_traj             ≈ rollout
真实环境回报            ≈ 外部评价指标
```

### 3.3 训练循环应如何阅读

建议沿着下面的调用链阅读，不要从文件第一行一直读到最后：

```text
train_step
  ↓
make_experience_batch
  ↓
experience 中的 reward / value / old_log_prob
  ↓
ppo_train
  ↓
actor fit + critic fit
  ↓
新策略重新 rollout
```

你需要重点记录每个张量的含义：

| 张量              | 含义                             |
| ----------------- | -------------------------------- |
| `sequences`     | prompt 和 response 的 token 序列 |
| `old_log_probs` | rollout 时旧策略的 token 概率    |
| `rewards`       | Reward Model 或规则得到的奖励    |
| `values`        | Critic 对未来回报的估计          |
| `advantages`    | 当前动作比预期好多少             |
| `kl`            | 当前策略和参考策略的距离         |

### 3.4 与 GridWorld PPO 的区别

你当前的 PPO 使用：

```python
ratio = exp(current_logp - old_logp)
```

真实 LLM PPO 也使用同样的概率比思想，但通常是对 response 中每个 token 计算 log probability，再结合 response mask 聚合。

因此可以把 GridWorld 的一条轨迹：

```text
s0 → a0 → s1 → a1 → s2
```

看成 LLM 的一条回答：

```text
prompt → token1 → token2 → token3
```

核心区别不是 PPO 公式完全不同，而是：

- LLM 的动作空间是整个词表；
- 轨迹长度可变；
- 奖励通常在完整回答生成后才得到；
- 需要处理 padding、mask、KL 和显存。

---

## 4. 项目三：verl 的 GRPO/RLVR

项目：[verl-project/verl](https://github.com/verl-project/verl)

核心算法文件：[verl/trainer/ppo/core_algos.py](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py)

PPO 的核心 policy loss：[compute_policy_loss_vanilla](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py)

GRPO 优势函数：[compute_grpo_outcome_advantage](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py)

### 4.1 GRPO 为什么需要“分组”

GRPO 对同一个问题生成多条回答：

```text
问题 q
  ├─ 回答 y1，奖励 r1
  ├─ 回答 y2，奖励 r2
  ├─ 回答 y3，奖励 r3
  └─ 回答 y4，奖励 r4
```

然后在同一组内计算平均值和标准差：

\[
A_i=\frac{r_i-\mu_q}{\sigma_q+\epsilon}.
\]

这样不必单独训练一个完整的 Value Model，就能得到相对优势：

- 高于组平均值的回答：优势为正；
- 低于组平均值的回答：优势为负。

### 4.2 关键代码摘录

verl 中的核心逻辑可以简化为：

```python
scores = token_level_rewards.sum(dim=-1)

for i in range(batch_size):
    id2score[index[i]].append(scores[i])

for group_id in id2score:
    id2mean[group_id] = mean(id2score[group_id])
    id2std[group_id] = std(id2score[group_id])

for i in range(batch_size):
    scores[i] = (
        scores[i] - id2mean[index[i]]
    ) / (id2std[index[i]] + epsilon)

advantages = scores.unsqueeze(-1) * response_mask
```

这段代码值得重点理解：

- `index[i]` 表示第 `i` 条回答属于哪个问题；
- `scores` 是每条回答的最终结果奖励；
- `id2mean` 和 `id2std` 只在同一个问题的候选回答内部计算；
- 最终的标量优势会广播到回答中的 token；
- `response_mask` 防止 padding token 参与损失。

### 4.3 与 GridWorld 的类比

可以把一个问题的多条回答类比成同一个起点出发的多条轨迹：

```text
同一个起点 S
  ├─ 轨迹 1，回报 9.1
  ├─ 轨迹 2，回报 -0.4
  ├─ 轨迹 3，回报 9.1
  └─ 轨迹 4，回报 -3.0
```

组内平均回报可以作为简单基准。高于平均值的轨迹增加概率，低于平均值的轨迹降低概率。

这比当前 notebook 的 PPO 更接近现代推理训练，但仍是教学类比，因为真实 GRPO 还需要：

- 同一 prompt 的多次生成；
- 数学或代码 verifier；
- token-level mask；
- 大模型 rollout；
- 分布式训练。

### 4.4 verl 中的 PPO clipped loss

verl 的 PPO policy loss 核心形式可以概括为：

```python
ratio = torch.exp(log_prob - old_log_prob)

pg_loss_1 = -advantages * ratio
pg_loss_2 = -advantages * torch.clamp(
    ratio,
    1 - clip_ratio_low,
    1 + clip_ratio_high,
)

policy_loss = torch.maximum(pg_loss_1, pg_loss_2)
```

这与 notebook 中的：

```python
ratio = np.exp(current_logp - old_logp)
clipped = np.clip(
    ratio,
    1 - clip_eps,
    1 + clip_eps,
)
```

是同一个 PPO 思想。verl 的实现更加工程化，额外处理了：

- batch 和 token 维度；
- response mask；
- 多种 advantage estimator；
- KL 和 clip 统计；
- 分布式训练；
- 不同的 loss aggregation。

---

## 5. 三个项目的统一分析框架

阅读任何一个后训练项目，都可以填写下面这张表：

| 问题                   | 需要找到的代码                          |
| ---------------------- | --------------------------------------- |
| 数据从哪里来？         | dataset、collator、prompt loader        |
| 模型生成什么？         | rollout、generate、sample               |
| 奖励在哪里计算？       | reward model、verifier、reward function |
| 旧策略概率在哪里保存？ | old log probability                     |
| 参考策略在哪里使用？   | reference log probability、KL           |
| advantage 如何得到？   | GAE、group mean/std、return             |
| loss 如何计算？        | DPO loss、PPO clipped loss              |
| 参数什么时候更新？     | backward、optimizer.step、fit           |
| 如何判断训练有效？     | reward、accuracy、success rate、KL      |

建议你每读一个函数，都记录：

```text
输入张量：
输出张量：
每个维度代表什么：
它对应论文中的哪个公式：
它对应 GridWorld 中哪个变量：
```

---

## 6. 建议的学习任务

### 任务一：复述 TRL 的 DPO

不看论文，只根据代码回答：

1. 为什么需要 chosen 和 rejected？
2. 为什么要保存 reference log probability？
3. `chosen_logratios - rejected_logratios` 表示什么？
4. `beta` 增大后，模型更新会有什么变化？

### 任务二：画 OpenRLHF 的 PPO 数据流

画出以下节点：

```text
PromptDataset
  → Actor rollout
  → Reward Model
  → Critic / Value
  → Advantage
  → Actor PPO update
  → 新一轮 rollout
```

然后在每个箭头旁边写出传递的数据名称。

### 任务三：用 GridWorld 模拟 GRPO

可以把一个起点采样出的 4 条轨迹放在一组中：

```python
rewards = np.array([9.1, -0.4, 9.1, -3.0])
advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
```

观察：

1. 哪些轨迹的优势为正？
2. 如果四条轨迹奖励完全相同，会发生什么？
3. 如果 verifier 错误地只奖励“到达终点”而忽略陷阱，GRPO 会优化出什么？

---

## 7. 初步结论

三个项目体现了后训练代码的发展层次：

1. **TRL：** 把 DPO、GRPO 等算法封装成清晰的 Trainer，适合学习单个损失函数；
2. **OpenRLHF：** 展示完整 RLHF 工程链路，重点是 Actor、Reward Model、Critic 和 rollout 的协作；
3. **verl：** 将 PPO/GRPO 与大规模 rollout、验证器和推理训练结合，代表较新的 RLVR 工程方向。

与自己的 GridWorld 对照后，可以得到一个重要认识：

> 大模型强化学习的数学核心没有完全改变，仍然是轨迹、奖励、概率比、优势和策略更新；真正增加的复杂性主要来自 token 序列、模型规模、奖励来源、分布式生成和评估可靠性。

## 参考链接

1. [Hugging Face TRL](https://github.com/huggingface/trl)
2. [TRL DPO Trainer](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py)
3. [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
4. [OpenRLHF PPO Trainer](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_trainer.py)
5. [verl](https://github.com/verl-project/verl)
6. [verl core algorithms](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py)
7. Rafailov et al., *Direct Preference Optimization*, 2023.
8. Shao et al., *DeepSeekMath*, 2024.
