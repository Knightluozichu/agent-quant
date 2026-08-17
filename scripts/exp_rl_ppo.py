"""RL(PPO) + CNN视觉特征 — 动态调仓 全量回测对比 V3 (14:50口径).

模块2 (RL 全链路第二步):
  状态: 7×64 CNN视觉特征(预计算) + 7×2 价格特征(mom5,vol20) + 持仓one-hot(8) = 470维
  动作: 8 离散 {0=持有, 1..7=切换至资产i}, t日收盘成交(14:50口径, 含成本与可交易检查)
  奖励: 持仓收益(%, t→t+1) − λ·vol20·100 − 换仓成本; λ=0.1
  算法: 自实现 PPO (clip ε=0.2, GAE λ=0.95, γ=0.99, lr=3e-4, 4epochs/更新), MPS GPU
  训练: 1000 轮, 每轮 rollout 训练段(2023-07~2024-12, 370步) → 一次 PPO 更新
  测试: 2025-01~2026-08 (OOS, 未参与任何训练), greedy rollout → 净值
  对比: V3 同测试段 (run_v3_r4_sameday thr=1.0, 14:50口径, 10万本金, 万五+千一)

防未来函数: 状态仅用 ≤t 数据; 动作 t 收盘执行; 奖励用 t→t+1 (标准 MDP, 无泄漏).

输出: data/v9_results/rl_ppo_ab.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR = PROJECT_ROOT / "data" / "v9_results" / "cnn_feats"

sys.path.insert(0, str(Path(__file__).parent))
from exp_v3_r4_sameday import run_v3_r4_sameday  # noqa: E402
from run_qixing_v3 import ETF_POOL, load_data  # noqa: E402

WARMUP = 130
INITIAL_CAPITAL = 100_000.0
N_ASSETS = len(ETF_POOL)  # 7
FEAT_DIM = 64
PRICE_FEATS = 2  # mom5, vol20
STATE_DIM = N_ASSETS * (FEAT_DIM + PRICE_FEATS) + N_ASSETS + 1  # 448+14+8=470
N_ACTIONS = N_ASSETS + 1  # 8: 0=持有, 1..7=切换
LAM = 0.0  # 波动惩罚(关闭: 年化vol≈3%远超日收益0.1%, 导致现金最优)
COST = 0.003  # 换仓双向成本 万五+千一 ×2 (收益单位%)

TRAIN_START, TRAIN_END = "2023-07-03", "2024-12-31"
TEST_START, TEST_END = "2025-01-02", "2026-08-03"

PPO_CLIP, GAE_LAMBDA, GAMMA = 0.2, 0.95, 0.99
PPO_LR, PPO_EPOCHS, ENTROPY = 3e-4, 4, 0.03
N_EPISODES = 1000
BATCH = 256
SEED = 42


# --------------------------------------------------------------------------- #
# RL 环境
# --------------------------------------------------------------------------- #
class ETFEnv:
    """ETF 轮动环境 (14:50 同日成交口径, 无未来函数)."""

    def __init__(
        self,
        data,
        dates,
        mat,
        feats_map,
        price_map,
        code_list,
        start_idx,
        end_idx,
        initial_capital: float = INITIAL_CAPITAL,
    ):
        self.data = data
        self.dates = dates
        self.mat = mat
        self.feats_map = feats_map  # {(date_str, code): 64维特征}
        self.price_map = price_map  # {(date_str, code): [mom5, vol20]}
        self.codes = code_list
        self.code_idx = {c: i for i, c in enumerate(code_list)}
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.initial_capital = float(initial_capital)
        self.t = start_idx
        self.holding = N_ACTIONS - 1  # 现金 = index 7 (现金用 onehot 第8位)
        self.cash = float(initial_capital)
        self.shares = 0.0

    def _state(self, t) -> np.ndarray:
        """t 日收盘后的状态 (无未来函数: 仅用 ≤t 数据)."""
        parts = []
        for code in self.codes:
            key = (str(self.dates[t]), code)
            f = self.feats_map.get(key)
            parts.append(f if f is not None else np.zeros(FEAT_DIM))
            p = self.price_map.get(key, [0.0, 0.0])
            parts.append(np.array(p, float))
        s = np.concatenate(parts)
        # 持仓 one-hot: [0..7]=资产, 8=现金
        hold_vec = np.zeros(N_ASSETS + 1)
        hold_vec[self.holding] = 1.0
        return np.concatenate([s, hold_vec]).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.t = self.start_idx
        self.holding = N_ACTIONS - 1  # 现金
        self.cash = self.initial_capital
        self.shares = 0.0
        return self._state(self.t)

    def _tradable(self, code: str, t) -> bool:
        """收盘可交易检查 (与 same_day 一致: 有数据+未涨跌停)."""
        td = self.dates[t]
        row = self.data[code][self.data[code]["trade_date"] == td]
        if row.empty:
            return False
        price = float(row.iloc[0]["close"])
        if price <= 0:
            return False
        hist = self.data[code][self.data[code]["trade_date"] < td]
        if not hist.empty:
            prev = float(hist.iloc[-1]["close"])
            if prev > 0 and abs(price / prev - 1) >= 0.099:
                return False
        return True

    def _price(self, code: str, t) -> float:
        row = self.data[code][self.data[code]["trade_date"] == self.dates[t]]
        return float(row.iloc[0]["close"]) if not row.empty else 0.0

    def step(self, action: int, eps: float = 0.0):
        """t 日收盘执行 action → 奖励 (t→t+1 收益) → 新状态."""
        t = self.t
        done = t + 1 >= self.end_idx
        cost_pen = 0.0
        # 执行动作 (14:50 口径: 今日收盘成交)
        if action != self.holding:
            if action < N_ASSETS:
                code = self.codes[action]
                if self._tradable(code, t):
                    # 卖出旧持仓
                    if self.holding < N_ASSETS and self.shares > 0:
                        old_code = self.codes[self.holding]
                        px = self._price(old_code, t)
                        if px > 0:
                            self.cash += self.shares * px * (1 - 0.0015)
                    elif self.holding == N_ACTIONS - 1:
                        pass  # 现金, 无需卖
                    # 买入新
                    price = self._price(code, t)
                    if price > 0 and self.cash > 0:
                        self.shares = self.cash * 0.99 / price
                        self.cash -= self.shares * price * 1.0015
                        self.holding = action
                        cost_pen = COST
            else:  # 切现金
                if self.holding < N_ASSETS and self.shares > 0:
                    old_code = self.codes[self.holding]
                    px = self._price(old_code, t)
                    if px > 0:
                        self.cash += self.shares * px * (1 - 0.0015)
                    self.shares = 0.0
                    self.holding = N_ACTIONS - 1
                    cost_pen = COST

        # 奖励: 持仓市值 t→t+1 收益
        t1 = t + 1
        value_t = self._value(t)
        value_t1 = self._value(t1)
        ret = (value_t1 / value_t - 1) * 100.0 if value_t > 0 else 0.0
        # 风险惩罚 (修正尺度: 日波动 = 年化vol/√252)
        vol = 0.0
        if self.holding < N_ASSETS:
            code = self.codes[self.holding]
            vol_ann = self.price_map.get((str(self.dates[t1]), code), [0, 0.3])[1]
            vol = vol_ann * 100.0 / np.sqrt(252) if vol_ann > 0 else 0.0
        reward = ret - LAM * vol - cost_pen
        self.t = t1
        return self._state(self.t), float(reward), done, {}

    def _value(self, t) -> float:
        if self.holding < N_ASSETS:
            px = self._price(self.codes[self.holding], t)
            return self.cash + self.shares * px
        return self.cash

    def equity(self, t) -> float:
        return self._value(t)


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.pi(h), self.v(h).squeeze(-1)


def ppo_train(env: ETFEnv, episodes: int = N_EPISODES, device="mps") -> ActorCritic:
    """PPO 训练 (每轮 rollout 整段 → 一次更新)."""
    torch.manual_seed(SEED)
    ac = ActorCritic(STATE_DIM, N_ACTIONS).to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=PPO_LR)

    def rollout(ac, env, device):
        s = env.reset()
        states, actions, rewards, dones, logps = [], [], [], [], []
        while True:
            st = torch.tensor(s, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = ac(st)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
            s2, r, done, _ = env.step(int(a.item()))
            states.append(s)
            actions.append(int(a.item()))
            rewards.append(r)
            dones.append(done)
            logps.append(logp.item())
            if done:
                break
            s = s2
        return (
            np.array(states, np.float32),
            np.array(actions),
            np.array(rewards),
            np.array(dones),
            np.array(logps),
        )

    for ep in range(episodes):
        states, actions, rewards, dones, logps = rollout(ac, env, device)
        # GAE
        t_len = len(rewards)
        with torch.no_grad():
            st_t = torch.tensor(states, device=device)
            _, vals = ac(st_t)
            vals = vals.cpu().numpy()
        gaes = np.zeros(t_len)
        adv = 0.0
        for t in reversed(range(t_len)):
            next_val = vals[t + 1] if t + 1 < t_len else 0.0
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - vals[t]
            adv = delta + GAMMA * GAE_LAMBDA * adv
            gaes[t] = adv
        returns = gaes + vals
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        # PPO 更新
        st_t = torch.tensor(states, dtype=torch.float32, device=device)
        a_t = torch.tensor(actions, dtype=torch.long, device=device)
        old_logp = torch.tensor(logps, dtype=torch.float32, device=device)
        adv_t = torch.tensor(gaes, dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
        n = t_len
        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, BATCH):
                idx = perm[i : i + BATCH]
                logits, v = ac(st_t[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a_t[idx])
                ratio = (logp - old_logp[idx]).exp()
                adv_b = adv_t[idx]
                loss_pi = -torch.min(
                    ratio * adv_b, torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_b
                ).mean()
                loss_v = functional.mse_loss(v, ret_t[idx])
                entropy = dist.entropy().mean()
                loss = loss_pi + 0.5 * loss_v - ENTROPY * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()
        if (ep + 1) % 200 == 0:
            print(
                f"    轮 {ep + 1}/{episodes} | 累计奖励 {rewards.sum():.1f} | "
                f"换手 {int((actions[1:] != actions[:-1]).sum())}"
            )
    return ac


def ppo_eval(env: ETFEnv, ac: ActorCritic, device="mps") -> dict:
    """greedy rollout → 净值曲线 → 指标 (与 run_dl_backtest 口径一致)."""
    s = env.reset()
    init = env.initial_capital
    eq_curve = [init]
    trades = 0
    last_hold = None
    while True:
        st = torch.tensor(s, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = ac(st)
            a = int(logits.argmax(dim=-1).item())
        if a != last_hold:
            trades += 1
            last_hold = a
        _s2, _r, done, _ = env.step(a)
        eq_curve.append(env.equity(env.t))
        if done:
            break
        s = _s2
    eq = np.array(eq_curve)
    total = eq[-1] / init - 1
    rets = np.diff(eq) / eq[:-1]
    ann_ret = (1 + total) ** (252 / max(len(rets), 1)) - 1
    ann_vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    cummax = np.maximum.accumulate(eq)
    max_dd = float(((eq - cummax) / cummax).min())
    return {
        "final_value": round(float(eq[-1]), 0),
        "total_return": round(float(total), 4),
        "ann_return": round(float(ann_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(max_dd, 4),
        "n_trades": trades,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("  RL(PPO) + CNN视觉特征 — 动态调仓 vs V3 (14:50口径)")
    print(
        f"  状态 {STATE_DIM}维 | 动作 {N_ACTIONS} | PPO {N_EPISODES}轮 | "
        f"device={'mps' if torch.backends.mps.is_available() else 'cpu'}"
    )
    print("=" * 74)

    data = load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    n = len(dates)
    codes = [c for c in ETF_POOL if c in data]
    mat = None  # 环境用 data 查询价格, mat 仅 V3 需要
    from exp_short_window_patterns import close_matrix

    mat = close_matrix(data, dates)

    # 加载 CNN 特征表
    z = np.load(FEAT_DIR / "feats.npz")
    feats_all = z["feats"]
    t_arr = z["t_arr"]
    code_arr = z["code_arr"]
    feats_map = {}
    for j in range(len(t_arr)):
        feats_map[(str(dates[int(t_arr[j])]), str(code_arr[j]))] = feats_all[j]
    print(f"  特征表: {feats_all.shape}")

    # 价格特征 (mom5, vol20) 每资产每日
    price_map = {}
    for c in codes:
        df = data[c].set_index("trade_date")
        close = df["close"].reindex(dates)
        for i in range(20, n):
            if i >= 5 and np.isfinite(close.iloc[i]) and np.isfinite(close.iloc[i - 5]):
                mom5 = close.iloc[i] / close.iloc[i - 5] - 1.0
            else:
                mom5 = 0.0
            seg = close.iloc[i - 19 : i + 1].astype(float)
            if seg.isna().any():
                vol20 = 0.3
            else:
                dr = seg.diff().dropna() / seg.shift(1).dropna()
                vol20 = float(dr.std() * np.sqrt(252)) if len(dr) > 1 else 0.3
            price_map[(str(dates[i]), c)] = [float(mom5), vol20]

    # 训练/测试段索引
    def seg_idx(s0, s1):
        a = next(i for i, d in enumerate(dates) if str(d) >= s0)
        b = next(i for i, d in enumerate(dates) if str(d) >= s1)
        return a, b

    tr_s, tr_e = seg_idx(TRAIN_START, TRAIN_END)
    te_s, te_e = seg_idx(TEST_START, TEST_END)
    print(f"  训练段: {dates[tr_s]}~{dates[tr_e - 1]} ({tr_e - tr_s}天)")
    print(f"  测试段: {dates[te_s]}~{dates[te_e - 1]} ({te_e - te_s}天)")

    # 环境 (每段新实例)
    env_tr = ETFEnv(data, dates, mat, feats_map, price_map, codes, tr_s, tr_e)
    env_te = ETFEnv(data, dates, mat, feats_map, price_map, codes, te_s, te_e)

    # 训练 PPO
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n  PPO 训练 {N_EPISODES} 轮...")
    ac = ppo_train(env_tr, N_EPISODES, device)

    # 测试段 rollout
    print("\n  测试段 (OOS) rollout...")
    rl = ppo_eval(env_te, ac, device)

    # V3 同段对比
    tb_start = max(te_s - WARMUP, 0)
    r_v3 = run_v3_r4_sameday(data, mat, thr=1.0, start_idx=tb_start)
    v3 = {
        k: r_v3[k]
        for k in ("final_value", "total_return", "ann_return", "sharpe", "max_drawdown", "n_trades")
    }

    print("\n" + "=" * 74)
    hdr = f"  {'配置':<14} {'期末金额':>10} {'总收益':>9} {'年化':>8} "
    hdr += f"{'夏普':>6} {'回撤':>8} {'换手':>4}"
    print(hdr)
    for name, r in (("V3基线", v3), ("RL+CNN", rl)):
        print(
            f"  {name:<14} {r['final_value']:>10,.0f} {r['total_return']:>+9.1%} "
            f"{r['ann_return']:>+8.1%} {r['sharpe']:>6.2f} {r['max_drawdown']:>8.1%} "
            f"{r['n_trades']:>4}"
        )
    beat = rl["final_value"] > v3["final_value"]
    print("=" * 74)
    print(f"  → RL+CNN {'跑赢' if beat else '跑输'} V3")

    out = {
        "meta": {
            "state_dim": STATE_DIM,
            "n_actions": N_ACTIONS,
            "lam": LAM,
            "ppo": {
                "clip": PPO_CLIP,
                "gamma": GAMMA,
                "gae": GAE_LAMBDA,
                "lr": PPO_LR,
                "epochs": PPO_EPOCHS,
                "episodes": N_EPISODES,
            },
            "train_seg": f"{TRAIN_START}~{TRAIN_END}",
            "test_seg": f"{TEST_START}~{TEST_END}",
            "cost": "万五+千一 单边",
            "device": device,
        },
        "rl": rl,
        "v3": v3,
        "beat": beat,
    }
    out_path = OUT_DIR / "rl_ppo_ab.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
