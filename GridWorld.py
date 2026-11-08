import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

rng = np.random.default_rng(42)
plt.rcParams['figure.dpi'] = 100
class GridWorld:
    """5x5 格子世界。接口仿照 Gymnasium：reset() 返回状态，step(a) 返回
    (下一状态, 奖励, 回合结束, 步数超限, 附加信息)。状态用 0～24 的整数表示。"""

    ROWS, COLS = 5, 5
    START = (0, 0)
    GOAL = (2, 4)
    TRAP = (2, 2)
    WALLS = {(0, 2), (1, 2), (3, 2)}
    R_STEP, R_TRAP, R_GOAL = -0.1, -10.0, 10.0
    ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上、右、下、左
    ACTION_SYMBOLS = ['↑', '→', '↓', '←']

    def __init__(self, slip=0.0, max_steps=50, seed=0):
        # slip 是打滑概率：以 slip 的概率执行一个随机动作，用于思考题的随机环境
        self.slip = slip
        self.max_steps = max_steps
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self.n_states = self.ROWS * self.COLS
        self.n_actions = 4
        self.P = self._build_P()  # 转移表：P[s][a] = [(概率, 下一状态, 奖励, 是否结束), ...]
        self.reset()

    @staticmethod
    def to_pos(s):
        return divmod(s, GridWorld.COLS)

    @staticmethod
    def to_state(pos):
        return pos[0] * GridWorld.COLS + pos[1]

    def _move(self, pos, a):
        if pos == self.GOAL:
            return pos
        dr, dc = self.ACTIONS[a]
        r, c = pos[0] + dr, pos[1] + dc
        if r < 0 or r >= self.ROWS or c < 0 or c >= self.COLS or (r, c) in self.WALLS:
            return pos
        return (r, c)

    def _outcome(self, pos, a):
        nxt = self._move(pos, a)
        if nxt == self.GOAL:
            return nxt, self.R_GOAL, True
        if nxt == self.TRAP:
            return nxt, self.R_TRAP, False
        return nxt, self.R_STEP, False

    def _build_P(self):
        P = {}
        for s in range(self.n_states):
            pos = self.to_pos(s)
            P[s] = {}
            for a in range(self.n_actions):
                if self.slip > 0:
                    merged = {}
                    for a2 in range(self.n_actions):
                        prob = (1 - self.slip) if a2 == a else 0.0
                        prob += self.slip / self.n_actions
                        nxt, r, done = self._outcome(pos, a2)
                        key = (self.to_state(nxt), r, done)
                        merged[key] = merged.get(key, 0.0) + prob
                    P[s][a] = [(p, ns, r, d) for (ns, r, d), p in merged.items()]
                else:
                    nxt, r, done = self._outcome(pos, a)
                    P[s][a] = [(1.0, self.to_state(nxt), r, done)]
        return P

    def reset(self):
        self.pos = self.START
        self.steps = 0
        return self.to_state(self.pos)

    def step(self, a):
        if self.slip > 0 and self.rng.random() < self.slip:
            a = int(self.rng.integers(self.n_actions))
        nxt, r, done = self._outcome(self.pos, a)
        self.pos = nxt
        self.steps += 1
        truncated = self.steps >= self.max_steps and not done
        return self.to_state(nxt), r, done, truncated, {}

def draw_world(ax, title=''):
    """画出格子世界的底图：墙、陷阱、起点和终点。"""
    color_map = {}
    for w in GridWorld.WALLS:
        color_map[w] = '#555555'
    color_map[GridWorld.TRAP] = '#f2b8b8'
    color_map[GridWorld.GOAL] = '#b8e6b8'
    color_map[GridWorld.START] = '#b8d4f0'
    for r in range(GridWorld.ROWS):
        for c in range(GridWorld.COLS):
            face = color_map.get((r, c), 'white')
            ax.add_patch(Rectangle((c, GridWorld.ROWS - 1 - r), 1, 1,
                                   facecolor=face, edgecolor='#999999'))
    def center(pos):
        return pos[1] + 0.5, GridWorld.ROWS - 1 - pos[0] + 0.5
    ax.text(*center(GridWorld.START), 'S', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(*center(GridWorld.GOAL), 'G', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(*center(GridWorld.TRAP), 'T', ha='center', va='center', fontsize=14, weight='bold')
    ax.set_xlim(0, GridWorld.COLS)
    ax.set_ylim(0, GridWorld.ROWS)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def plot_policy(policy, title='Policy'):
    """用箭头画出每个状态的策略。policy[s] 是动作编号 0～3。"""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    draw_world(ax, title)
    for s in range(GridWorld.ROWS * GridWorld.COLS):
        pos = GridWorld.to_pos(s)
        if pos in GridWorld.WALLS or pos == GridWorld.GOAL:
            continue
        dr, dc = GridWorld.ACTIONS[policy[s]]
        x, y = pos[1] + 0.5, GridWorld.ROWS - 1 - pos[0] + 0.5
        ax.arrow(x, y, dc * 0.3, -dr * 0.3, head_width=0.12,
                 head_length=0.1, fc='#1f77b4', ec='#1f77b4')
    plt.show()


def plot_values(V, title='State values'):
    """画出每个状态的价值热力图。"""
    grid = np.full((GridWorld.ROWS, GridWorld.COLS), np.nan)
    for s in range(GridWorld.ROWS * GridWorld.COLS):
        pos = GridWorld.to_pos(s)
        if pos not in GridWorld.WALLS:
            grid[pos] = V[s]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap='viridis')
    for r in range(GridWorld.ROWS):
        for c in range(GridWorld.COLS):
            if not np.isnan(grid[r, c]):
                ax.text(c, r, f'{grid[r, c]:.1f}', ha='center', va='center',
                        fontsize=9, color='white')
    for w in GridWorld.WALLS:
        ax.add_patch(Rectangle((w[1] - 0.5, w[0] - 0.5), 1, 1, facecolor='#555555'))
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def plot_trajectory(traj, title='Trajectory'):
    """在地图上画出一条轨迹。traj 是状态编号的列表。"""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    draw_world(ax, title)
    xs = [GridWorld.to_pos(s)[1] + 0.5 for s in traj]
    ys = [GridWorld.ROWS - 1 - GridWorld.to_pos(s)[0] + 0.5 for s in traj]
    ax.plot(xs, ys, 'o-', color='#d62728', lw=2, markersize=6)
    plt.show()


def plot_returns(returns, title='Learning curve', window=100):
    """画出学习曲线：浅色是每个回合的回报，深色是滑动平均。"""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(returns, alpha=0.25, color='#1f77b4', label='Episode return')
    if len(returns) >= window:
        avg = np.convolve(returns, np.ones(window) / window, mode='valid')
        ax.plot(np.arange(window - 1, len(returns)), avg,
                color='#d62728', lw=2, label=f'Moving average ({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title(title)
    ax.legend()
    plt.show()

env = GridWorld(seed=0)
fig, ax = plt.subplots(figsize=(4.5, 4.5))
draw_world(ax, 'GridWorld: two routes to the goal')
plt.show()

# 转移表的一个例子：在起点向右走，只有一种结果（确定性环境）
s_start = GridWorld.to_state(GridWorld.START)
print('P[start][right] =', env.P[s_start][1])

# 手动与环境交互几步
s = env.reset()
for a in [2, 1, 1]:  # 下、右、右
    s, r, done, truncated, _ = env.step(a)
    print(f'step -> state={GridWorld.to_pos(s)}, reward={r}, done={done}')

def eps_greedy(Q, s, epsilon, rng):
    import numpy as np
    # TODO 4：以 ε 的概率随机选一个动作，否则选 Q 值最大的动作。
    # 提示：rng.random() 生成 [0,1) 的随机数；
    # rng.integers(n) 生成 [0,n) 的随机整数。
    # 我的思路：
    # random_number = rng.random();
    # if random_number < epsilon:
    #   action_index = rng.integers(4);
    # else:
    #   action_index = np.argmax(Q[s]);
    # return action_index;
    random_number = rng.random();
    if random_number < epsilon:
        action = rng.integers(4);
    else:  
        action = np.argmax(Q[s]);
    return action;


def q_learning(env, episodes=3000, alpha=0.1, gamma=0.9,
               eps_start=0.3, eps_end=0.02, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = np.zeros(episodes)
    for ep in range(episodes):
        epsilon = eps_start + (eps_end - eps_start) * ep / episodes
        s = env.reset()
        total, done, truncated = 0.0, False, False
        while not (done or truncated):
            a = eps_greedy(Q, s, epsilon, rng)
            s_next, r, done, truncated, _ = env.step(a)
            # TODO 5：TD 更新。先算 TD 目标 r + γ·max Q(s′,·)，
            # 再按 Q(s,a) ← Q(s,a) + α·(目标 − Q(s,a)) 更新。
            # 提示：回合结束时未来价值按 0 计。
            # 我的思路：
            # if done == True:
            #     TD_target = r + 0;
            # else:
            #     TD_target = r + gamma * max(Q[s_next]);
            # Q[s][a] += alpha * (TD_target - Q[s][a]);
            if done == True:
                TD_target = r + 0;
            else:
                TD_target = r + gamma * np.max(Q[s_next]);
            Q[s][a] += alpha * (TD_target - Q[s][a]);
            s = s_next
            total += r
        returns[ep] = total
    return Q, returns
env2 = GridWorld(seed=1)
Q2, returns2 = q_learning(env2, episodes=3000, seed=1)
plot_returns(returns2, 'Q-learning on GridWorld')
pi_q = np.argmax(Q2, axis=1)
plot_policy(pi_q, 'Greedy policy learned by Q-learning')

s = env2.reset()
traj_q, G_q, done = [s], 0.0, False
while not done and len(traj_q) <= 30:
    a = pi_q[s]
    s, r, done, _, _ = env2.step(a)
    traj_q.append(s)
    G_q += r
print('Q-learning 学到的路径：', [GridWorld.to_pos(x) for x in traj_q])
print('该路径的真实回报 =', round(G_q, 2))

# Part 3：Mini-RLHF（未知奖励）
GOAL_POS = GridWorld.GOAL


def manhattan(pos):
    return abs(pos[0] - GOAL_POS[0]) + abs(pos[1] - GOAL_POS[1])


def phi(s, a, s_next):
    """单步特征：[到终点的距离变化, 是否到达终点, 步数指示]。"""
    p, p_next = GridWorld.to_pos(s), GridWorld.to_pos(s_next)
    import numpy as np
    return np.array([
        float(manhattan(p) - manhattan(p_next)),
        1.0 if p_next == GOAL_POS else 0.0,
        1.0,
    ])


def traj_features(steps):
    """整条轨迹的特征 = 各步特征之和。steps 是 (s, a, s_next) 的列表。"""
    import numpy as np
    return np.sum([phi(s, a, s2) for s, a, s2 in steps], axis=0)
def collect_trajectories(env, Q_behavior, n_traj=200, noise=0.1,
                         max_steps=20, seed=0):
    """用带噪声的行为策略采样轨迹，并剔除踩过陷阱的轨迹（覆盖盲区）。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    trap_s = GridWorld.to_state(GridWorld.TRAP)
    trajs = []
    while len(trajs) < n_traj:
        s = env.reset()
        steps, total, done = [], 0.0, False
        while not done and len(steps) < max_steps:
            a = eps_greedy(Q_behavior, s, noise, rng)
            s_next, r, done, _, _ = env.step(a)
            steps.append((s, a, s_next))
            s = s_next
            total += r
        if any(s2 == trap_s for _, _, s2 in steps):
            continue  # 覆盖盲区：标注数据不含灾难行为
        trajs.append((steps, total))
    return trajs


def make_preference_pairs(trajs, n_pairs=400, min_gap=0.5, seed=0):
    """随机配对，用真实回报标注哪条更好；差距太小的对子跳过。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        i, j = rng.integers(len(trajs), size=2)
        if i == j:
            continue
        (steps_i, G_i), (steps_j, G_j) = trajs[i], trajs[j]
        if abs(G_i - G_j) < min_gap:
            continue
        winner, loser = (steps_i, steps_j) if G_i > G_j else (steps_j, steps_i)
        pairs.append((traj_features(winner), traj_features(loser)))
    return pairs

trajs = collect_trajectories(env2, Q2, seed=10)
pairs = make_preference_pairs(trajs, seed=11)
print(f'收集到 {len(trajs)} 条轨迹，{len(pairs)} 对偏好')

def sigmoid(x):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-x))


def fit_reward_model(pairs, lr=0.05, iters=3000, l2=1e-4):
    import numpy as np
    theta = np.zeros(3)
    for _ in range(iters):
        grad = np.zeros(3)
        for F_win, F_lose in pairs:
            diff = F_win - F_lose
            # TODO 6：累加单个对子的梯度 −σ(−θ·diff)·diff。
            # 提示：sigmoid 函数已在上方定义。
            # grad += -sigmoid(-theta * diff) * diff;
            grad += -sigmoid(-theta * diff) * diff;
        grad = grad / len(pairs) + l2 * theta
        theta -= lr * grad
    return theta

theta = fit_reward_model(pairs)
print('学到的奖励模型参数 θ =', np.round(theta, 3))
print('（三个分量依次对应：靠近终点、到达终点、步数）')


def r_hat(s, a, s_next, r):
    """用学到的奖励模型给每一步打分，替换环境的真实奖励。"""
    return float(theta @ phi(s, a, s_next))

class RewardWrapper(GridWorld):
    """替换奖励函数的环境包装器。"""

    def __init__(self, reward_fn, **kwargs):
        self.reward_fn = reward_fn
        super().__init__(**kwargs)

    def step(self, a):
        s_old = self.to_state(self.pos)
        s_next, r, done, truncated, info = super().step(a)
        return s_next, self.reward_fn(s_old, a, s_next, r), done, truncated, info

env_rm = RewardWrapper(r_hat, seed=2)
Q_rlhf, returns_rlhf = q_learning(env_rm, episodes=3000, seed=2)
plot_returns(returns_rlhf, 'Q-learning on the LEARNED reward')

def greedy_traj(env, Q, max_steps=30):
    """按 Q 的贪心策略走一遍，返回 (s, a, s_next) 的列表。"""
    import numpy as np
    s = env.reset()
    steps, done = [], False
    while not done and len(steps) < max_steps:
        a = int(np.argmax(Q[s]))
        s_next, _, done, _, _ = env.step(a)
        steps.append((s, a, s_next))
        s = s_next
    return steps

env_eval = GridWorld(seed=3)
steps_base = greedy_traj(env_eval, Q2)
steps_rlhf = greedy_traj(env_eval, Q_rlhf)

true_base = sum(env_eval._outcome(GridWorld.to_pos(s), a)[1] for s, a, s2 in steps_base)
true_rlhf = sum(env_eval._outcome(GridWorld.to_pos(s), a)[1] for s, a, s2 in steps_rlhf)
proxy_base = sum(r_hat(s, a, s2, 0.0) for s, a, s2 in steps_base)
proxy_rlhf = sum(r_hat(s, a, s2, 0.0) for s, a, s2 in steps_rlhf)

print(f'基线策略：  奖励模型分数 = {proxy_base:.2f}，真实回报 = {true_base:.2f}')
print(f'RLHF 策略： 奖励模型分数 = {proxy_rlhf:.2f}，真实回报 = {true_rlhf:.2f}')

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].bar(['Baseline', 'RLHF'], [proxy_base, proxy_rlhf], color=['#1f77b4', '#d62728'])
axes[0].set_title('Score under the LEARNED reward model')
axes[1].bar(['Baseline', 'RLHF'], [true_base, true_rlhf], color=['#1f77b4', '#d62728'])
axes[1].set_title('TRUE return')
plt.tight_layout()
plt.show()

traj_rlhf_states = [steps_rlhf[0][0]] + [s2 for _, _, s2 in steps_rlhf]
plot_trajectory(traj_rlhf_states, 'RLHF policy: it goes THROUGH the trap')
# 拓展：PPO 与 DPO 的教学版实现
def _softmax_rows(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _sample_action(probs, state, rng):
    return int(rng.choice(len(probs[state]), p=probs[state]))


def ppo_learning(env, updates=400, episodes_per_update=16, gamma=0.99,
                 clip_eps=0.2, policy_lr=0.02, value_lr=0.05,
                 update_epochs=4, seed=0):

    rng = np.random.default_rng(seed)
    logits = np.zeros((env.n_states, env.n_actions))
    values = np.zeros(env.n_states)
    returns = []

    for update in range(updates):
        batch = []
        for _ in range(episodes_per_update):
            s = env.reset()
            episode = []
            done = truncated = False
            while not (done or truncated):
                probs = _softmax_rows(logits)
                a = _sample_action(probs, s, rng)
                s_next, r, done, truncated, _ = env.step(a)
                old_logp = np.log(max(probs[s, a], 1e-12))
                episode.append((s, a, r, s_next, done or truncated, old_logp))
                s = s_next

            G = 0.0
            for s, a, r, s_next, terminal, old_logp in reversed(episode):
                G = r + (0.0 if terminal else gamma * G)
                advantage = G - values[s]
                batch.append((s, a, G, advantage, old_logp))
            returns.append(sum(item[2] for item in episode))

        for _ in range(update_epochs):
            probs = _softmax_rows(logits)
            for s, a, target, advantage, old_logp in batch:
                current_logp = np.log(max(probs[s, a], 1e-12))
                ratio = np.exp(np.clip(current_logp - old_logp, -20.0, 20.0))
                clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                use_ratio = not (advantage >= 0 and ratio > clipped or
                                 advantage < 0 and ratio < clipped)
                if use_ratio:
                    grad = advantage * ratio * (np.eye(env.n_actions)[a] - probs[s])
                    logits[s] += policy_lr * grad
                values[s] += value_lr * (target - values[s])
                probs = _softmax_rows(logits)
    return logits, values, np.asarray(returns)


ppo_logits, ppo_values, ppo_returns = ppo_learning(GridWorld(seed=20), seed=20)
pi_ppo = np.argmax(ppo_logits, axis=1)
steps_ppo = greedy_traj(GridWorld(seed=21), np.exp(ppo_logits))
eval_env = GridWorld(seed=21)
ppo_true = sum(eval_env._outcome(GridWorld.to_pos(s), a)[1]
               for s, a, _ in steps_ppo)
print('PPO mean return over last 40 episodes =', round(float(np.mean(ppo_returns[-40:])), 2))
print('PPO greedy-policy true return =', round(ppo_true, 2))
plot_policy(pi_ppo, 'Greedy policy learned by tabular PPO')
def collect_dpo_pairs(env, Q_behavior, n_pairs=300, noise=0.2,
                      max_steps=20, seed=0):

    rng = np.random.default_rng(seed)
    trajectories = []
    while len(trajectories) < n_pairs * 3:
        s = env.reset()
        steps, total = [], 0.0
        done = truncated = False
        while not (done or truncated) and len(steps) < max_steps:
            a = eps_greedy(Q_behavior, s, noise, rng)
            s_next, r, done, truncated, _ = env.step(a)
            steps.append((s, a, s_next))
            total += r
            s = s_next
        trajectories.append((steps, total))

    pairs = []
    for _ in range(n_pairs):
        i, j = rng.integers(len(trajectories), size=2)
        if i == j or trajectories[i][1] == trajectories[j][1]:
            continue
        first, second = trajectories[i], trajectories[j]
        pairs.append((first[0], second[0]) if first[1] > second[1]
                     else (second[0], first[0]))
    return pairs


def _sequence_log_prob(logits, steps):
    probs = _softmax_rows(logits)
    return sum(np.log(max(probs[s, a], 1e-12)) for s, a, _ in steps)


def dpo_learning(preference_pairs, reference_logits, beta=0.5,
                 lr=0.03, epochs=20):

    logits = reference_logits.copy()
    for _ in range(epochs):
        for chosen, rejected in preference_pairs:
            ref_gap = (_sequence_log_prob(reference_logits, chosen)
                       - _sequence_log_prob(reference_logits, rejected))
            policy_gap = (_sequence_log_prob(logits, chosen)
                         - _sequence_log_prob(logits, rejected))
            x = beta * (policy_gap - ref_gap)
            coeff = 1.0 / (1.0 + np.exp(np.clip(x, -40.0, 40.0))) - 1.0
            probs = _softmax_rows(logits)
            for s, a, _ in chosen:
                logits[s] -= lr * beta * coeff * (np.eye(logits.shape[1])[a] - probs[s])
            for s, a, _ in rejected:
                logits[s] += lr * beta * coeff * (np.eye(logits.shape[1])[a] - probs[s])
    return logits


dpo_pairs = collect_dpo_pairs(GridWorld(seed=30), Q2, seed=30)
dpo_logits = dpo_learning(dpo_pairs, Q2)
pi_dpo = np.argmax(dpo_logits, axis=1)
steps_dpo = greedy_traj(GridWorld(seed=31), np.exp(dpo_logits))
eval_env = GridWorld(seed=31)
dpo_true = sum(eval_env._outcome(GridWorld.to_pos(s), a)[1]
               for s, a, _ in steps_dpo)
print('DPO number of preference pairs =', len(dpo_pairs))
print('DPO greedy-policy true return =', round(dpo_true, 2))
plot_policy(pi_dpo, 'Greedy policy learned by tabular DPO')
