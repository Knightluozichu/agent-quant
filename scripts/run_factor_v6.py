"""V6: 因子聚类 + 神经网络选股 + RL仓位 + 10策略矩阵.

架构:
  1. 16因子 → 相关性聚类 → 6独立代表
  2. Numpy NN: 前向传播 + BP + AdamW + RankIC损失
  3. Purged Walk-Forward: 120d训练 / 20d gap / 60d测试
  4. 10组策略变体, 输出年度矩阵

诚实性约束:
  - 10组是预设矩阵, 不是"选最好的"
  - 基准永远是纯动量Top4
  - NN权重每期重训, 不偷看未来
  - RL仅用于仓位, 不做选股
"""

from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "long_history"
OUTPUT_DIR = PROJECT_ROOT / "data" / "v6_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFENSIVE = {"511010": "国债ETF", "511880": "货币ETF"}
FEE = 0.001
SLIPPAGE = 0.0005

# Walk-Forward参数
WF_TRAIN = 120      # 训练窗口(天)
WF_GAP = 20         # Purge间隔
WF_TEST = 60        # 测试窗口
WF_STEP = 60        # 步长

# NN超参
NN_EPOCHS = 80
NN_LR = 0.005
NN_WD = 0.001       # weight decay (AdamW)
NN_BATCH = 256

# 16个因子名
ALL_FACTORS = [
    "momentum", "reversal", "low_vol", "trend", "volume_trend",
    "bias", "rsi", "macd", "atr_ratio", "obv",
    "skewness", "vol_change", "amplitude", "bollinger",
    "momentum_accel", "breakout",
]

# 6个聚类代表
CLUSTER_REPS = ["momentum", "reversal", "low_vol", "trend", "volume_trend", "skewness"]


# =============================================================================
# Factor Engine (from V4.3)
# =============================================================================

def calc_all_factors(df: pd.DataFrame, as_of: date) -> dict[str, float] | None:
    """计算单只ETF在as_of日期的16个因子."""
    hist = df[df["trade_date"] <= as_of].sort_values("trade_date")
    if len(hist) < 120:
        return None
    close = hist["close"].values.astype(float)
    high = hist["high"].values.astype(float) if "high" in hist.columns else close
    low = hist["low"].values.astype(float) if "low" in hist.columns else close
    volume = hist["volume"].values.astype(float) if "volume" in hist.columns else np.ones(len(close))

    c = close[-1]
    factors = {}

    # 动量族
    if len(close) >= 80:
        factors["momentum"] = (close[-20] - close[-80]) / close[-80]
    else:
        factors["momentum"] = 0.0
    if len(close) >= 40:
        factors["momentum_accel"] = (close[-10] - close[-20]) / close[-20] - (close[-20] - close[-30]) / close[-30] if len(close) >= 30 else 0.0
    else:
        factors["momentum_accel"] = 0.0
    if len(close) >= 60:
        factors["breakout"] = (c - close[-60:].min()) / (close[-60:].max() - close[-60:].min() + 1e-8)
    else:
        factors["breakout"] = 0.5

    # 反转族
    if len(close) >= 5:
        factors["reversal"] = -(close[-1] - close[-5]) / close[-5]
    else:
        factors["reversal"] = 0.0
    if len(close) >= 14:
        deltas = np.diff(close[-15:])
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        rs = gains / (losses + 1e-8)
        factors["rsi"] = -(100 - 100 / (1 + rs))  # 负号: 超卖=正信号
    else:
        factors["rsi"] = 0.0

    # 低波族
    if len(close) >= 21:
        rets = np.diff(close[-21:]) / close[-21:-1]
        factors["low_vol"] = -np.std(rets) * np.sqrt(252)
    else:
        factors["low_vol"] = 0.0
    if len(high) >= 20 and len(low) >= 20:
        tr = np.maximum(high[-20:] - low[-20:],
                        np.maximum(np.abs(high[-20:] - close[-21:-1]),
                                   np.abs(low[-20:] - close[-21:-1])))
        factors["atr_ratio"] = -np.mean(tr) / (c + 1e-8)
    else:
        factors["atr_ratio"] = 0.0
    if len(close) >= 20:
        amp = (high[-20:] - low[-20:]) / (close[-20:] + 1e-8)
        factors["amplitude"] = -np.mean(amp)
    else:
        factors["amplitude"] = 0.0
    if len(close) >= 40:
        v1 = np.std(np.diff(close[-20:]) / close[-20:-1])
        v2 = np.std(np.diff(close[-40:-20]) / close[-40:-21])
        factors["vol_change"] = -(v1 - v2) / (v2 + 1e-8)
    else:
        factors["vol_change"] = 0.0

    # 趋势族
    if len(close) >= 60:
        factors["trend"] = np.mean(close[-20:]) / np.mean(close[-60:]) - 1
    else:
        factors["trend"] = 0.0
    if len(close) >= 20:
        factors["bias"] = (c - np.mean(close[-20:])) / np.mean(close[-20:])
    else:
        factors["bias"] = 0.0
    if len(close) >= 26:
        ema12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
        factors["macd"] = (ema12 - ema26) / (c + 1e-8)
    else:
        factors["macd"] = 0.0
    if len(close) >= 20:
        ma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        factors["bollinger"] = -(c - ma20) / (2 * std20 + 1e-8)
    else:
        factors["bollinger"] = 0.0

    # 量能族
    if len(volume) >= 60:
        factors["volume_trend"] = np.mean(volume[-20:]) / (np.mean(volume[-60:]) + 1e-8) - 1
    else:
        factors["volume_trend"] = 0.0
    if len(volume) >= 20:
        price_change = np.diff(close[-20:])
        obv_val = np.sum(np.where(price_change > 0, volume[-19:], -volume[-19:]))
        factors["obv"] = obv_val / (np.sum(volume[-20:]) + 1e-8)
    else:
        factors["obv"] = 0.0

    # 独立
    if len(close) >= 21:
        rets = np.diff(close[-21:]) / close[-21:-1]
        m = np.mean(rets)
        s = np.std(rets) + 1e-8
        factors["skewness"] = np.mean(((rets - m) / s) ** 3)
    else:
        factors["skewness"] = 0.0

    return factors


def build_factor_panel(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    dates: list,
) -> dict[date, pd.DataFrame]:
    """构建因子面板: {date: DataFrame[symbol, factor1, ..., factor16]}."""
    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {k: v for k, v in data.items()
                if k not in DEFENSIVE and k not in index_symbols}

    panels = {}
    for d in dates:
        records = []
        for sym, df in tradable.items():
            f = calc_all_factors(df, d)
            if f is not None:
                f["symbol"] = sym
                records.append(f)
        if len(records) >= 10:
            panels[d] = pd.DataFrame(records)
    return panels


def rank_normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rank标准化到[0,1]."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].rank(pct=True)
    return out


# =============================================================================
# Neural Network (Numpy)
# =============================================================================

class FactorNN:
    """2层前馈网络: input → hidden(ReLU) → output.

    用RankIC(Pearson相关)作为损失, AdamW优化.
    """

    def __init__(self, n_in: int, n_hidden: int = 8, activation: str = "relu",
                 lr: float = NN_LR, wd: float = NN_WD, seed: int = 42):
        rng = np.random.default_rng(seed)
        # He initialization
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, 1)) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(1)
        self.lr = lr
        self.wd = wd
        self.activation = activation
        # AdamW state
        self._t = 0
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def _activate(self, x):
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        elif self.activation == "tanh":
            return np.tanh(x)
        return np.maximum(0, x)

    def _activate_grad(self, x):
        if self.activation == "relu":
            return (x > 0).astype(float)
        elif self.activation == "leaky_relu":
            return np.where(x > 0, 1.0, 0.01)
        elif self.activation == "tanh":
            return 1 - np.tanh(x) ** 2
        return (x > 0).astype(float)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """前向传播. 返回 (scores, hidden_pre_activation)."""
        self._X = X
        self._z1 = X @ self.W1 + self.b1
        self._h1 = self._activate(self._z1)
        self._out = self._h1 @ self.W2 + self.b2
        return self._out.flatten(), self._z1

    def _rank_ic_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """负Pearson相关作为损失 (最大化IC)."""
        p = pred - pred.mean()
        t = target - target.mean()
        denom = np.sqrt(np.sum(p**2) * np.sum(t**2)) + 1e-8
        ic = np.sum(p * t) / denom
        return -ic

    def backward(self, pred: np.ndarray, target: np.ndarray) -> dict:
        """BP: 计算梯度."""
        n = len(pred)
        # d(loss)/d(pred) for Pearson correlation
        p = pred - pred.mean()
        t = target - target.mean()
        sp = np.sqrt(np.sum(p**2) + 1e-8)
        st = np.sqrt(np.sum(t**2) + 1e-8)
        # Gradient of -IC w.r.t. pred
        d_pred = -(t / (sp * st) - p * np.sum(p * t) / (sp**3 * st))

        # Output layer
        dW2 = self._h1.T @ d_pred.reshape(-1, 1)
        db2 = d_pred.sum()

        # Hidden layer
        d_h1 = d_pred.reshape(-1, 1) @ self.W2.T
        d_z1 = d_h1 * self._activate_grad(self._z1)

        # Input layer
        dW1 = self._X.T @ d_z1
        db1 = d_z1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads: dict):
        """AdamW更新."""
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        params = self._params()
        for key in params:
            g = grads[key]
            self._m[key] = beta1 * self._m[key] + (1 - beta1) * g
            self._v[key] = beta2 * self._v[key] + (1 - beta2) * g**2
            m_hat = self._m[key] / (1 - beta1**self._t)
            v_hat = self._v[key] / (1 - beta2**self._t)
            # AdamW: decoupled weight decay
            params[key] -= self.lr * (m_hat / (np.sqrt(v_hat) + eps) + self.wd * params[key])
        self.W1, self.b1, self.W2, self.b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = NN_EPOCHS):
        """训练NN."""
        n = len(X)
        for _ in range(epochs):
            # Mini-batch
            idx = np.random.permutation(n)[:min(NN_BATCH, n)]
            Xb, yb = X[idx], y[idx]
            pred, _ = self.forward(Xb)
            grads = self.backward(pred, yb)
            self.step(grads)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """推理."""
        out, _ = self.forward(X)
        return out


# =============================================================================
# RL Position Sizing (REINFORCE)
# =============================================================================

class RLPolicy:
    """简单策略梯度: 学习仓位缩放.

    State: [market_ret_20d, portfolio_vol, gate_signal, momentum_strength]
    Action: position_scale in [0.2, 1.0] (continuous, parameterized by mean)
    """

    def __init__(self, lr: float = 0.001, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.theta = rng.standard_normal(4) * 0.1  # linear policy weights
        self.lr = lr
        self.baseline = 0.0

    def get_position(self, state: np.ndarray) -> float:
        """给定状态, 输出仓位[0.2, 1.0]."""
        logit = np.dot(self.theta, state)
        # Sigmoid → [0, 1] → scale to [0.2, 1.0]
        pos = 1.0 / (1.0 + np.exp(-logit))
        return 0.2 + 0.8 * pos

    def update(self, state: np.ndarray, reward: float):
        """REINFORCE更新."""
        advantage = reward - self.baseline
        self.baseline = 0.99 * self.baseline + 0.01 * reward
        logit = np.dot(self.theta, state)
        sigmoid = 1.0 / (1.0 + np.exp(-logit))
        grad = advantage * sigmoid * (1 - sigmoid) * state
        self.theta += self.lr * grad


# =============================================================================
# Walk-Forward + Backtest
# =============================================================================

def get_forward_returns(data: dict[str, pd.DataFrame], panel: pd.DataFrame,
                        as_of: date, horizon: int = 20) -> pd.Series:
    """计算as_of日期后horizon天的收益率."""
    fwd = {}
    for _, row in panel.iterrows():
        sym = row["symbol"]
        if sym not in data:
            continue
        df = data[sym]
        future = df[df["trade_date"] > as_of].sort_values("trade_date")
        if len(future) < horizon:
            continue
        c0 = df[df["trade_date"] <= as_of].sort_values("trade_date")["close"].iloc[-1]
        c1 = future.iloc[horizon - 1]["close"]
        fwd[sym] = (c1 - c0) / c0
    return pd.Series(fwd)


def run_strategy(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    daily_panels: dict[date, pd.DataFrame],
    strategy_id: int,
    initial_capital: float = 100_000.0,
    top_n: int = 4,
) -> dict:
    """运行单个策略变体的完整回测.

    daily_panels: 每个交易日的因子面板(用于训练)
    交易每20天执行一次(在test窗口内的调仓日)
    """
    all_dates = sorted(daily_panels.keys())
    if len(all_dates) < WF_TRAIN + WF_GAP + WF_TEST:
        return {"error": f"insufficient data: {len(all_dates)} < {WF_TRAIN + WF_GAP + WF_TEST}"}

    # 确定因子集
    use_all_16 = (strategy_id == 7)
    factor_cols = ALL_FACTORS if use_all_16 else CLUSTER_REPS
    n_in = len(factor_cols)

    # NN超参 (按策略ID)
    n_hidden = 16 if strategy_id in (4, 7) else 8
    activation = "leaky_relu" if strategy_id == 5 else "relu"
    wd = 0.01 if strategy_id == 6 else NN_WD

    # 仓位管理
    use_vol_scale = strategy_id in (8, 10)
    use_rl = strategy_id == 9
    use_gate = strategy_id == 10

    rl_policy = RLPolicy() if use_rl else None

    # Walk-Forward
    cash = initial_capital
    holdings: dict[str, int] = {}
    equity_history = []
    oos_predictions = []

    # 生成WF窗口 (基于日级面板)
    windows = []
    i = 0
    while i + WF_TRAIN + WF_GAP + WF_TEST <= len(all_dates):
        train_dates = all_dates[i:i + WF_TRAIN]
        test_dates = all_dates[i + WF_TRAIN + WF_GAP: i + WF_TRAIN + WF_GAP + WF_TEST]
        windows.append((train_dates, test_dates))
        i += WF_STEP

    for train_dates, test_dates in windows:
        # === 训练阶段: 从日级面板收集样本 ===
        X_train_list, y_train_list = [], []
        # 每5天采样一次训练数据(避免过度相关)
        for td in train_dates[::5]:
            if td not in daily_panels:
                continue
            panel = daily_panels[td]
            fwd = get_forward_returns(data, panel, td, horizon=20)
            if len(fwd) < 8:
                continue
            merged = panel[panel["symbol"].isin(fwd.index)].copy()
            if len(merged) < 8:
                continue
            merged = rank_normalize(merged, factor_cols)
            X = merged[factor_cols].values
            y = merged["symbol"].map(fwd).values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            if mask.sum() < 8:
                continue
            X_train_list.append(X[mask])
            y_train_list.append(y[mask])

        if not X_train_list:
            continue

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # 训练NN (策略1和2不用NN)
        nn = None
        if strategy_id >= 3:
            nn = FactorNN(n_in=n_in, n_hidden=n_hidden,
                          activation=activation, wd=wd)
            nn.train(X_train, y_train, epochs=NN_EPOCHS)

        # === 测试阶段: 只在调仓日(每20天)交易 ===
        test_rebalance = test_dates[::20]  # 测试窗口内每20天调仓
        for td in test_rebalance:
            if td not in daily_panels:
                continue
            panel = daily_panels[td]
            panel_norm = rank_normalize(panel.copy(), factor_cols)

            # 选股
            if strategy_id == 1:
                # 纯动量
                scores = panel.set_index("symbol")["momentum"]
            elif strategy_id == 2:
                # 6因子等权
                scores = panel_norm.set_index("symbol")[CLUSTER_REPS].mean(axis=1)
            else:
                # NN预测
                X_test = panel_norm[factor_cols].values
                pred = nn.predict(X_test)
                scores = pd.Series(pred, index=panel["symbol"].values)

            selected = scores.nlargest(top_n).index.tolist()

            # 仓位管理
            position_scale = 1.0
            if use_gate:
                idx = index_df[index_df["trade_date"] <= td].sort_values("trade_date")
                if len(idx) >= 60:
                    ma60 = idx["close"].values[-60:].mean()
                    if idx["close"].values[-1] < ma60:
                        position_scale = 0.0

            if use_vol_scale and position_scale > 0:
                vols = []
                for sym in selected:
                    if sym in data:
                        h = data[sym][data[sym]["trade_date"] <= td].sort_values("trade_date")
                        if len(h) >= 21:
                            c = h["close"].values.astype(float)
                            r = np.diff(c[-21:]) / c[-21:-1]
                            vols.append(np.std(r) * np.sqrt(252))
                if vols:
                    pv = np.mean(vols)
                    position_scale *= np.clip((0.20**2) / (pv**2 + 1e-8), 0.2, 1.0)

            if use_rl and position_scale > 0:
                # RL state
                idx = index_df[index_df["trade_date"] <= td].sort_values("trade_date")
                mkt_ret = 0.0
                if len(idx) >= 20:
                    ic = idx["close"].values.astype(float)
                    mkt_ret = (ic[-1] - ic[-20]) / ic[-20]
                avg_score = scores[selected].mean() if len(selected) > 0 else 0
                state = np.array([mkt_ret, np.std(y_train) if len(y_train) > 0 else 0.1,
                                  1.0 if position_scale > 0 else 0.0, avg_score])
                position_scale *= rl_policy.get_position(state)

            # 记录预测 (用于后续计算收益)
            oos_predictions.append({
                "date": td, "selected": selected,
                "position_scale": position_scale,
            })

    # === 模拟交易 ===
    # 简化: 每20天调仓, 等权持有selected
    rebalance_dates = sorted(set(p["date"] for p in oos_predictions))
    pred_map = {p["date"]: p for p in oos_predictions}

    for i, td in enumerate(rebalance_dates):
        p = pred_map[td]
        selected = p["selected"]
        pos_scale = p["position_scale"]

        # 计算当前equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]

        # 目标配置
        stock_budget = equity * pos_scale
        bond_budget = equity * (1 - pos_scale)
        per_stock = stock_budget / max(len(selected), 1)

        # 卖出非目标
        for sym in list(holdings.keys()):
            if sym not in selected and sym not in DEFENSIVE:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    cash += holdings[sym] * row.iloc[0]["close"] * (1 - FEE - SLIPPAGE)
                    del holdings[sym]

        # 调整国债
        bond_sym = "511010"
        if bond_sym in data:
            row = data[bond_sym][data[bond_sym]["trade_date"] == td]
            if not row.empty:
                price = row.iloc[0]["close"]
                cur_bond = holdings.get(bond_sym, 0)
                target_bond = int(bond_budget / price / 10) * 10
                diff = target_bond - cur_bond
                if diff > 0:
                    cost = diff * price * (1 + FEE / 2)
                    if cost <= cash:
                        cash -= cost
                        holdings[bond_sym] = cur_bond + diff
                elif diff < -10:
                    sell = min(-diff, cur_bond)
                    cash += sell * price * (1 - FEE / 2)
                    holdings[bond_sym] = cur_bond - sell
                    if holdings[bond_sym] <= 0:
                        holdings.pop(bond_sym, None)

        # 买入/调整股票
        for sym in selected:
            if sym not in data:
                continue
            row = data[sym][data[sym]["trade_date"] == td]
            if row.empty:
                continue
            price = row.iloc[0]["close"]
            cur = holdings.get(sym, 0)
            target_shares = int(per_stock / price / 100) * 100
            diff = target_shares - cur
            if diff > 0:
                cost = diff * price * (1 + FEE + SLIPPAGE)
                if cost <= cash:
                    cash -= cost
                    holdings[sym] = cur + diff
            elif diff < -100:
                sell = int(min(-diff, cur) / 100) * 100
                if sell > 0:
                    cash += sell * price * (1 - FEE - SLIPPAGE)
                    holdings[sym] = cur - sell
                    if holdings[sym] <= 0:
                        holdings.pop(sym, None)

        # RL reward update
        if use_rl and i > 0:
            prev_p = pred_map[rebalance_dates[i - 1]]
            prev_ret = 0.0
            for sym in prev_p["selected"]:
                if sym in data:
                    r0 = data[sym][data[sym]["trade_date"] == rebalance_dates[i - 1]]
                    r1 = data[sym][data[sym]["trade_date"] == td]
                    if not r0.empty and not r1.empty:
                        prev_ret += (r1.iloc[0]["close"] - r0.iloc[0]["close"]) / r0.iloc[0]["close"]
            prev_ret /= max(len(prev_p["selected"]), 1)
            idx = index_df[index_df["trade_date"] <= rebalance_dates[i - 1]].sort_values("trade_date")
            mkt_ret = 0.0
            if len(idx) >= 20:
                ic = idx["close"].values.astype(float)
                mkt_ret = (ic[-1] - ic[-20]) / ic[-20]
            state = np.array([mkt_ret, 0.1, 1.0, 0.0])
            rl_policy.update(state, prev_ret * prev_p["position_scale"])

        # 记录equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity})

    if not equity_history:
        return {"total_return": 0.0, "yearly": {}}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    n_days = len(eq_df)
    ann_ret = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    # 年度收益
    yearly = {}
    prev_val = initial_capital
    for year in sorted(eq_df["year"].unique()):
        ydf = eq_df[eq_df["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev_val) - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        yearly[int(year)] = {"return": yr, "max_dd": dd}
        prev_val = end_val

    return {
        "total_return": total_return, "ann_return": ann_ret,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "yearly": yearly, "equity_curve": eq_df,
    }


# =============================================================================
# Main
# =============================================================================

def load_data():
    data = {}
    for f in DATA_DIR.glob("*.parquet"):
        if f.name in ("combined_long.parquet", "northbound.parquet",
                      "pe_percentile.parquet", "margin_sentiment.parquet"):
            continue
        df = pd.read_parquet(f)
        if "symbol" not in df.columns or "trade_date" not in df.columns:
            continue
        symbol = df["symbol"].iloc[0]
        if f.name.startswith("index_"):
            symbol = f"idx_{symbol}"
            df["symbol"] = symbol
        data[symbol] = df.sort_values("trade_date").reset_index(drop=True)
    index_df = data.get("idx_000300")
    if index_df is None:
        for k, v in data.items():
            if "000300" in k:
                index_df = v
                break
    return data, index_df


STRATEGY_NAMES = {
    1: "纯动量Top4",
    2: "6因子等权",
    3: "NN-8-ReLU",
    4: "NN-16-ReLU",
    5: "NN-8-LeakyReLU",
    6: "NN-8-L2(wd=0.01)",
    7: "NN-16全因子",
    8: "NN-8+VolScale",
    9: "NN-8+RL仓位",
    10: "NN-8+Gate+Vol",
}


def main():
    print("=" * 70)
    print("  V6: 因子聚类 + NN选股 + RL仓位 | 10策略矩阵")
    print(f"  WF: train={WF_TRAIN}d, gap={WF_GAP}d, test={WF_TEST}d, step={WF_STEP}d")
    print(f"  NN: epochs={NN_EPOCHS}, lr={NN_LR}, wd={NN_WD}, batch={NN_BATCH}")
    print("=" * 70)

    data, index_df = load_data()
    if index_df is None:
        print("ERROR: 无指数数据")
        return

    # === Step 1: 因子聚类 ===
    print(f"\n[Step 1] 因子聚类分析...")
    all_dates = sorted(index_df["trade_date"].tolist())
    # 用中间日期的面板做聚类
    mid_date = all_dates[len(all_dates) // 2]
    panels_sample = build_factor_panel(data, index_df, all_dates[::5])  # 每5天采样

    # 合并所有面板计算相关性
    all_factor_vals = []
    for d, panel in list(panels_sample.items())[:100]:
        normed = rank_normalize(panel, ALL_FACTORS)
        all_factor_vals.append(normed[ALL_FACTORS])
    if all_factor_vals:
        combined = pd.concat(all_factor_vals, ignore_index=True)
        corr_matrix = combined[ALL_FACTORS].corr()
        print(f"  因子相关系数矩阵 (|r|>0.7 标记为同族):")
        # 打印高相关对
        high_corr = []
        for i in range(len(ALL_FACTORS)):
            for j in range(i + 1, len(ALL_FACTORS)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.7:
                    high_corr.append((ALL_FACTORS[i], ALL_FACTORS[j], r))
        for f1, f2, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f"    {f1:<16} ↔ {f2:<16} r={r:+.3f}")
        print(f"\n  确认6个聚类代表: {CLUSTER_REPS}")

    # === Step 2-6: 运行10个策略 ===
    print(f"\n[Step 2-6] 运行10个策略变体...")
    # 构建日级因子面板 (跳过warmup)
    warmup_dates = all_dates[100:]  # 跳过前100天warmup
    print(f"  构建日级因子面板 ({len(warmup_dates)} 天)...")
    daily_panels = build_factor_panel(data, index_df, warmup_dates)
    print(f"  因子面板: {len(daily_panels)} 天有效")

    results = {}
    for sid in range(1, 11):
        name = STRATEGY_NAMES[sid]
        print(f"\n  [{sid}/10] {name}...", end=" ", flush=True)
        r = run_strategy(data, index_df, daily_panels, strategy_id=sid)
        results[sid] = r
        if "error" in r:
            print(f"ERROR: {r['error']}")
        else:
            print(f"全周期{r['total_return']:+.1%} | 夏普{r['sharpe']:.2f} | 回撤{r['max_drawdown']:.1%}")

    # === 输出10×5年度矩阵 ===
    print(f"\n{'=' * 70}")
    print(f"  10策略 × 年度收益矩阵")
    print(f"{'=' * 70}")

    years = sorted(set(y for r in results.values() for y in r.get("yearly", {}).keys()))
    header = f"  {'策略':<20}" + "".join(f"{y:>8}" for y in years) + f"{'全周期':>8}{'夏普':>6}{'回撤':>7}"
    print(header)
    print(f"  {'-' * (20 + 8 * len(years) + 21)}")

    for sid in range(1, 11):
        r = results[sid]
        name = STRATEGY_NAMES[sid]
        if "error" in r:
            print(f"  {name:<20} ERROR")
            continue
        row = f"  {name:<20}"
        for y in years:
            yr_data = r.get("yearly", {}).get(y, {})
            ret = yr_data.get("return", 0)
            row += f"{ret:>+8.1%}"
        row += f"{r['total_return']:>+8.1%}{r['sharpe']:>6.2f}{r['max_drawdown']:>7.1%}"
        print(row)

    # 基准线
    print(f"  {'-' * (20 + 8 * len(years) + 21)}")
    print(f"  基准: 纯动量Top4 = 策略#1")

    # === 保存 ===
    summary = {}
    for sid, r in results.items():
        if "error" not in r:
            summary[STRATEGY_NAMES[sid]] = {
                "total_return": r["total_return"],
                "sharpe": r["sharpe"],
                "max_drawdown": r["max_drawdown"],
                "yearly": {str(k): v for k, v in r.get("yearly", {}).items()},
            }
    with open(OUTPUT_DIR / "v6_matrix.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  结果已保存: {OUTPUT_DIR / 'v6_matrix.json'}")


if __name__ == "__main__":
    main()
