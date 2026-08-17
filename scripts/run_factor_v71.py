"""V7.1: 前2年训练 / 后2年测试 — 最简验证.

数据划分:
  训练集: 2022-01 ~ 2023-12 (2年)
  测试集: 2024-01 ~ 2026-04 (2.3年, 纯OOS)

策略:
  A: 纯动量Top4 (无需训练, 全周期直接跑)
  B: NN-8+Gate+Vol (2022-2023训练NN, 2024-2026固定权重跑)
  C: v4.2-AdamW (2022-2023优化因子权重, 2024-2026固定权重跑)
  D: v4.2-等权 (无需训练, 全周期直接跑)

基准: 沪深300, 21ETF等权
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "v71_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFENSIVE = {"511010": "国债ETF", "511880": "货币ETF"}
FEE = 0.001
SLIPPAGE = 0.0005

# 数据划分
TRAIN_END = date(2023, 12, 31)
TEST_START = date(2024, 1, 1)

# 调仓频率
REBALANCE = 20  # 月频(20天)

ALL_FACTORS = [
    "momentum",
    "reversal",
    "low_vol",
    "trend",
    "volume_trend",
    "bias",
    "rsi",
    "macd",
    "atr_ratio",
    "obv",
    "skewness",
    "vol_change",
    "amplitude",
    "bollinger",
    "momentum_accel",
    "breakout",
]
CLUSTER_REPS = ["momentum", "reversal", "low_vol", "trend", "volume_trend", "skewness"]


# =============================================================================
# Factor Engine
# =============================================================================


def calc_all_factors(df: pd.DataFrame, as_of: date) -> dict[str, float] | None:
    hist = df[df["trade_date"] <= as_of].sort_values("trade_date")
    if len(hist) < 120:
        return None
    close = hist["close"].values.astype(float)
    high = hist["high"].values.astype(float) if "high" in hist.columns else close
    low = hist["low"].values.astype(float) if "low" in hist.columns else close
    volume = (
        hist["volume"].values.astype(float) if "volume" in hist.columns else np.ones(len(close))
    )
    c = close[-1]
    f = {}

    f["momentum"] = (close[-20] - close[-80]) / close[-80] if len(close) >= 80 else 0.0
    f["momentum_accel"] = (
        ((close[-10] - close[-20]) / close[-20] - (close[-20] - close[-30]) / close[-30])
        if len(close) >= 30
        else 0.0
    )
    f["breakout"] = (
        (c - close[-60:].min()) / (close[-60:].max() - close[-60:].min() + 1e-8)
        if len(close) >= 60
        else 0.5
    )
    f["reversal"] = -(close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0.0
    if len(close) >= 14:
        deltas = np.diff(close[-15:])
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        f["rsi"] = -(100 - 100 / (1 + gains / (losses + 1e-8)))
    else:
        f["rsi"] = 0.0
    if len(close) >= 21:
        rets = np.diff(close[-21:]) / close[-21:-1]
        f["low_vol"] = -np.std(rets) * np.sqrt(252)
    else:
        f["low_vol"] = 0.0
    if len(high) >= 20:
        tr = np.maximum(
            high[-20:] - low[-20:],
            np.maximum(np.abs(high[-20:] - close[-21:-1]), np.abs(low[-20:] - close[-21:-1])),
        )
        f["atr_ratio"] = -np.mean(tr) / (c + 1e-8)
    else:
        f["atr_ratio"] = 0.0
    f["amplitude"] = (
        -np.mean((high[-20:] - low[-20:]) / (close[-20:] + 1e-8)) if len(close) >= 20 else 0.0
    )
    if len(close) >= 40:
        v1 = np.std(np.diff(close[-20:]) / close[-20:-1])
        v2 = np.std(np.diff(close[-40:-20]) / close[-40:-21])
        f["vol_change"] = -(v1 - v2) / (v2 + 1e-8)
    else:
        f["vol_change"] = 0.0
    f["trend"] = (np.mean(close[-20:]) / np.mean(close[-60:]) - 1) if len(close) >= 60 else 0.0
    f["bias"] = (c - np.mean(close[-20:])) / np.mean(close[-20:]) if len(close) >= 20 else 0.0
    if len(close) >= 26:
        ema12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
        f["macd"] = (ema12 - ema26) / (c + 1e-8)
    else:
        f["macd"] = 0.0
    if len(close) >= 20:
        ma20 = np.mean(close[-20:])
        f["bollinger"] = -(c - ma20) / (2 * np.std(close[-20:]) + 1e-8)
    else:
        f["bollinger"] = 0.0
    f["volume_trend"] = (
        (np.mean(volume[-20:]) / (np.mean(volume[-60:]) + 1e-8) - 1) if len(volume) >= 60 else 0.0
    )
    if len(volume) >= 20:
        pc = np.diff(close[-20:])
        f["obv"] = np.sum(np.where(pc > 0, volume[-19:], -volume[-19:])) / (
            np.sum(volume[-20:]) + 1e-8
        )
    else:
        f["obv"] = 0.0
    if len(close) >= 21:
        rets = np.diff(close[-21:]) / close[-21:-1]
        m, s = np.mean(rets), np.std(rets) + 1e-8
        f["skewness"] = np.mean(((rets - m) / s) ** 3)
    else:
        f["skewness"] = 0.0
    return f


def build_factor_panel(data: dict, dates: list) -> dict:
    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {k: v for k, v in data.items() if k not in DEFENSIVE and k not in index_symbols}
    panels = {}
    for d in dates:
        records = []
        for sym, df in tradable.items():
            fv = calc_all_factors(df, d)
            if fv is not None:
                fv["symbol"] = sym
                records.append(fv)
        if len(records) >= 10:
            panels[d] = pd.DataFrame(records)
    return panels


def rank_normalize(df, cols):
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].rank(pct=True)
    return out


def get_forward_returns(data, panel, as_of, horizon=20):
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


# =============================================================================
# NN
# =============================================================================


class FactorNN:
    def __init__(self, n_in, n_hidden=8, lr=0.005, wd=0.001, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, 1)) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(1)
        self.lr, self.wd = lr, wd
        self._t = 0
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X):
        self._X = X
        self._z1 = X @ self.W1 + self.b1
        self._h1 = np.maximum(0, self._z1)
        return (self._h1 @ self.W2 + self.b2).flatten()

    def train(self, X, y, epochs=100):
        n = len(X)
        for _ in range(epochs):
            idx = np.random.permutation(n)[: min(256, n)]
            Xb, yb = X[idx], y[idx]
            pred = self.forward(Xb)
            p, t = pred - pred.mean(), yb - yb.mean()
            sp, st = np.sqrt(np.sum(p**2) + 1e-8), np.sqrt(np.sum(t**2) + 1e-8)
            d_pred = -(t / (sp * st) - p * np.sum(p * t) / (sp**3 * st))
            dW2 = self._h1.T @ d_pred.reshape(-1, 1)
            db2 = d_pred.sum()
            d_h1 = d_pred.reshape(-1, 1) @ self.W2.T
            d_z1 = d_h1 * (self._z1 > 0).astype(float)
            dW1 = self._X.T @ d_z1
            db1 = d_z1.sum(axis=0)
            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
            self._t += 1
            params = self._params()
            for key in params:
                g = grads[key]
                self._m[key] = 0.9 * self._m[key] + 0.1 * g
                self._v[key] = 0.999 * self._v[key] + 0.001 * g**2
                m_hat = self._m[key] / (1 - 0.9**self._t)
                v_hat = self._v[key] / (1 - 0.999**self._t)
                params[key] -= self.lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + self.wd * params[key])
            self.W1, self.b1, self.W2, self.b2 = (
                params["W1"],
                params["b1"],
                params["W2"],
                params["b2"],
            )

    def predict(self, X):
        return self.forward(X)


# =============================================================================
# AdamW Factor Weight Optimizer (for v4.2-C)
# =============================================================================


def optimize_factor_weights(panels, data, train_dates, factors, lr=0.01, epochs=200):
    """在训练集上用AdamW优化因子权重, 最大化RankIC."""
    n_factors = len(factors)
    # 初始化权重
    rng = np.random.default_rng(42)
    w = rng.standard_normal(n_factors) * 0.1
    # AdamW state
    m = np.zeros(n_factors)
    v = np.zeros(n_factors)
    wd = 0.01

    for epoch in range(epochs):
        # 采样训练数据
        sample_dates = train_dates[::10]  # 每10天采样
        all_ic = []
        grad_accum = np.zeros(n_factors)
        n_samples = 0

        for td in sample_dates:
            if td not in panels:
                continue
            panel = panels[td]
            fwd = get_forward_returns(data, panel, td, horizon=20)
            if len(fwd) < 8:
                continue
            merged = panel[panel["symbol"].isin(fwd.index)].copy()
            if len(merged) < 8:
                continue
            merged = rank_normalize(merged, factors)
            X = merged[factors].values  # (N, F)
            y = merged["symbol"].map(fwd).values  # (N,)

            # Composite score = X @ softmax(w)
            exp_w = np.exp(w - w.max())
            sw = exp_w / exp_w.sum()
            scores = X @ sw  # (N,)

            # IC gradient: d(IC)/d(w) via chain rule
            s_norm = scores - scores.mean()
            y_norm = y - y.mean()
            ss = np.sqrt(np.sum(s_norm**2) + 1e-8)
            sy = np.sqrt(np.sum(y_norm**2) + 1e-8)
            ic = np.sum(s_norm * y_norm) / (ss * sy)
            all_ic.append(ic)

            # d(IC)/d(scores)
            d_scores = y_norm / (ss * sy) - s_norm * np.sum(s_norm * y_norm) / (ss**3 * sy)
            # d(scores)/d(sw) = X.T @ d_scores
            d_sw = X.T @ d_scores  # (F,)
            # d(sw)/d(w) = softmax jacobian (approx: diag(sw) - sw*sw.T)
            d_w = sw * (d_sw - np.sum(d_sw * sw))
            grad_accum += d_w
            n_samples += 1

        if n_samples == 0:
            continue
        grad = -grad_accum / n_samples  # negative because we minimize -IC

        # AdamW update
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        m_hat = m / (1 - 0.9 ** (epoch + 1))
        v_hat = v / (1 - 0.999 ** (epoch + 1))
        w -= lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + wd * w)

    # 返回softmax权重
    exp_w = np.exp(w - w.max())
    return exp_w / exp_w.sum()


# =============================================================================
# Gate & Vol
# =============================================================================


def check_gate(index_df, as_of, ma_period=60):
    idx = index_df[index_df["trade_date"] <= as_of].sort_values("trade_date")
    if len(idx) < ma_period:
        return True
    close = idx["close"].values.astype(float)
    return close[-1] > close[-ma_period:].mean()


def calc_vol_scale(data, selected, as_of, target_vol=0.20):
    vols = []
    for sym in selected:
        if sym not in data:
            continue
        h = data[sym][data[sym]["trade_date"] <= as_of].sort_values("trade_date")
        if len(h) >= 21:
            c = h["close"].values.astype(float)
            r = np.diff(c[-21:]) / c[-21:-1]
            vols.append(np.std(r) * np.sqrt(252))
    if not vols:
        return 1.0
    pv = np.mean(vols)
    return float(np.clip((target_vol**2) / (pv**2 + 1e-8), 0.2, 1.0))


# =============================================================================
# Backtest Engine
# =============================================================================


def run_backtest(
    data,
    index_df,
    panels,
    strategy,
    rebalance=20,
    start_date=None,
    end_date=None,
    nn=None,
    factor_weights=None,
    initial_capital=100_000.0,
    top_n=4,
):
    """统一回测. start_date/end_date控制测试区间."""
    all_dates = sorted(panels.keys())
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]

    rebalance_dates = all_dates[::rebalance]

    cash = initial_capital
    holdings = {}
    equity_history = []
    n_trades = 0

    for td in rebalance_dates:
        if td not in panels:
            continue
        panel = panels[td]

        # === 选股 ===
        if strategy == "momentum":
            scores = panel.set_index("symbol")["momentum"]
            selected = scores.nlargest(top_n).index.tolist()
            position_scale = 1.0

        elif strategy == "nn_gate_vol":
            gate_open = check_gate(index_df, td)
            if not gate_open:
                selected, position_scale = [], 0.0
            else:
                if nn is not None:
                    panel_norm = rank_normalize(panel.copy(), CLUSTER_REPS)
                    pred = nn.predict(panel_norm[CLUSTER_REPS].values)
                    scores = pd.Series(pred, index=panel["symbol"].values)
                else:
                    scores = panel.set_index("symbol")["momentum"]
                selected = scores.nlargest(top_n).index.tolist()
                position_scale = calc_vol_scale(data, selected, td)

        elif strategy == "v42_optimized":
            gate_open = check_gate(index_df, td)
            if not gate_open:
                selected, position_scale = [], 0.0
            else:
                panel_norm = rank_normalize(panel.copy(), ALL_FACTORS)
                X = panel_norm[ALL_FACTORS].values
                scores_arr = X @ factor_weights
                scores = pd.Series(scores_arr, index=panel["symbol"].values)
                selected = scores.nlargest(top_n).index.tolist()
                position_scale = 1.0

        elif strategy == "v42_equal":
            gate_open = check_gate(index_df, td)
            if not gate_open:
                selected, position_scale = [], 0.0
            else:
                panel_norm = rank_normalize(panel.copy(), ALL_FACTORS)
                scores = panel_norm.set_index("symbol")[ALL_FACTORS].mean(axis=1)
                selected = scores.nlargest(top_n).index.tolist()
                position_scale = 1.0

        # === 交易 ===
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]

        stock_budget = equity * position_scale
        bond_budget = equity * (1 - position_scale)
        per_stock = stock_budget / max(len(selected), 1) if selected else 0

        for sym in list(holdings.keys()):
            if sym not in selected and sym not in DEFENSIVE:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    cash += holdings[sym] * row.iloc[0]["close"] * (1 - FEE - SLIPPAGE)
                    n_trades += 1
                    del holdings[sym]

        bond_sym = "511010"
        if bond_sym in data:
            row = data[bond_sym][data[bond_sym]["trade_date"] == td]
            if not row.empty:
                price = row.iloc[0]["close"]
                cur_bond = holdings.get(bond_sym, 0)
                target_bond = int(bond_budget / price / 10) * 10
                diff = target_bond - cur_bond
                if diff > 10:
                    cost = diff * price * (1 + FEE / 2)
                    if cost <= cash:
                        cash -= cost
                        holdings[bond_sym] = cur_bond + diff
                        n_trades += 1
                elif diff < -10:
                    sell = min(-diff, cur_bond)
                    cash += sell * price * (1 - FEE / 2)
                    holdings[bond_sym] = cur_bond - sell
                    if holdings[bond_sym] <= 0:
                        holdings.pop(bond_sym, None)
                    n_trades += 1

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
                    n_trades += 1
            elif diff < -100:
                sell = int(min(-diff, cur) / 100) * 100
                if sell > 0:
                    cash += sell * price * (1 - FEE - SLIPPAGE)
                    holdings[sym] = cur - sell
                    if holdings[sym] <= 0:
                        holdings.pop(sym, None)
                    n_trades += 1

        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity})

    if not equity_history:
        return {"error": "no data"}

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
        "total_return": total_return,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "yearly": yearly,
        "n_trades": n_trades,
        "equity_curve": eq_df,
    }


# =============================================================================
# Main
# =============================================================================


def load_data():
    data = {}
    for f in DATA_DIR.glob("*.parquet"):
        if f.name in (
            "combined_long.parquet",
            "northbound.parquet",
            "pe_percentile.parquet",
            "margin_sentiment.parquet",
        ):
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


def main():
    print("=" * 75)
    print("  V7.1: 前2年训练 / 后2年测试")
    print(f"  训练集: 2022-01 ~ 2023-12 | 测试集: 2024-01 ~ 2026-04")
    print(f"  调仓: 月频({REBALANCE}天) | Top4")
    print("=" * 75)

    data, index_df = load_data()
    all_dates = sorted(index_df["trade_date"].tolist())
    print(f"\n  数据: {all_dates[0]} ~ {all_dates[-1]} ({len(all_dates)}天)")

    # 构建全周期因子面板
    print(f"  构建因子面板...")
    panels = build_factor_panel(data, all_dates[80:])
    print(f"  有效面板: {len(panels)}天")

    # 训练/测试日期
    train_dates = [d for d in sorted(panels.keys()) if d <= TRAIN_END]
    test_dates = [d for d in sorted(panels.keys()) if d >= TEST_START]
    print(f"  训练期: {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)}天)")
    print(f"  测试期: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}天)")

    # === 训练阶段 ===
    print(f"\n[训练] 在2022-2023数据上训练模型...")

    # B: NN训练
    print(f"  训练NN-8 (6因子→8隐藏→1输出)...")
    X_list, y_list = [], []
    for td in train_dates[::5]:
        if td not in panels:
            continue
        panel = panels[td]
        fwd = get_forward_returns(data, panel, td, horizon=20)
        if len(fwd) < 8:
            continue
        merged = panel[panel["symbol"].isin(fwd.index)].copy()
        if len(merged) < 8:
            continue
        merged = rank_normalize(merged, CLUSTER_REPS)
        X_list.append(merged[CLUSTER_REPS].values)
        y_list.append(merged["symbol"].map(fwd).values)

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
    X_train, y_train = X_train[mask], y_train[mask]
    print(f"    训练样本: {len(X_train)} (来自{len(X_list)}个日期)")

    nn = FactorNN(n_in=6, n_hidden=8)
    nn.train(X_train, y_train, epochs=150)
    # 验证训练集IC
    train_pred = nn.predict(X_train)
    train_ic = np.corrcoef(train_pred, y_train)[0, 1]
    print(f"    训练集IC: {train_ic:.4f}")

    # C: AdamW因子权重优化
    print(f"  优化v4.2因子权重 (AdamW, 16因子)...")
    factor_weights = optimize_factor_weights(
        panels, data, train_dates, ALL_FACTORS, lr=0.01, epochs=200
    )
    print(f"    优化后权重:")
    for fname, w in sorted(zip(ALL_FACTORS, factor_weights), key=lambda x: -x[1]):
        if w > 0.03:
            print(f"      {fname:<16} {w:.3f}")

    # === 测试阶段 ===
    print(f"\n[测试] 在2024-2026数据上回测 (纯OOS)...")

    strategies = [
        ("A: 纯动量Top4", "momentum", None, None),
        ("B: NN8+Gate+Vol", "nn_gate_vol", nn, None),
        ("C: v4.2-AdamW权重", "v42_optimized", None, factor_weights),
        ("D: v4.2-等权+Gate", "v42_equal", None, None),
    ]

    results = {}
    for name, strat, nn_model, fw in strategies:
        print(f"\n  {name}...", end=" ", flush=True)
        r = run_backtest(
            data,
            index_df,
            panels,
            strategy=strat,
            rebalance=REBALANCE,
            start_date=TEST_START,
            nn=nn_model,
            factor_weights=fw,
        )
        results[name] = r
        if "error" in r:
            print(f"ERROR")
        else:
            print(
                f"收益{r['total_return']:+.1%} | 夏普{r['sharpe']:.2f} | "
                f"回撤{r['max_drawdown']:.1%} | 交易{r['n_trades']}次"
            )

    # 也跑训练期(对比用)
    print(f"\n[对比] 训练期(2022-2023)表现...")
    train_results = {}
    for name, strat, nn_model, fw in strategies:
        r = run_backtest(
            data,
            index_df,
            panels,
            strategy=strat,
            rebalance=REBALANCE,
            end_date=TRAIN_END,
            nn=nn_model,
            factor_weights=fw,
        )
        train_results[name] = r

    # === 基准 ===
    benchmarks = {"沪深300": {}, "21ETF等权": {}}
    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {k: v for k, v in data.items() if k not in DEFENSIVE and k not in index_symbols}

    for year in range(2022, 2027):
        # 沪深300
        idx_y = index_df[index_df["trade_date"].apply(lambda x: x.year) == year]
        if len(idx_y) >= 2:
            benchmarks["沪深300"][year] = (
                idx_y["close"].iloc[-1] - idx_y["close"].iloc[0]
            ) / idx_y["close"].iloc[0]
        # 21ETF等权
        rets = []
        for sym, df in tradable.items():
            ydf = df[df["trade_date"].apply(lambda x: x.year) == year]
            if len(ydf) >= 2:
                rets.append((ydf["close"].iloc[-1] - ydf["close"].iloc[0]) / ydf["close"].iloc[0])
        if rets:
            benchmarks["21ETF等权"][year] = np.mean(rets)

    # === 输出 ===
    years = [2022, 2023, 2024, 2025, 2026]
    test_years = [2024, 2025, 2026]

    print(f"\n{'=' * 75}")
    print(f"  完整年度矩阵 (训练期 + 测试期)")
    print(f"{'=' * 75}")
    header = (
        f"  {'策略':<20}"
        + "".join(f"{y:>8}" for y in years)
        + f"{'OOS合计':>9}{'夏普':>6}{'回撤':>7}"
    )
    print(header)
    print(f"  {'-' * 75}")

    for name, _, _, _ in strategies:
        r = results[name]
        tr = train_results[name]
        if "error" in r:
            continue
        row = f"  {name:<20}"
        # 训练期年份
        for y in [2022, 2023]:
            yr_data = tr.get("yearly", {}).get(y, {})
            ret = yr_data.get("return", None)
            row += f"{ret:>+8.1%}" if ret is not None else f"{'—':>8}"
        # 测试期年份
        for y in test_years:
            yr_data = r.get("yearly", {}).get(y, {})
            ret = yr_data.get("return", None)
            row += f"{ret:>+8.1%}" if ret is not None else f"{'—':>8}"
        row += f"{r['total_return']:>+9.1%}{r['sharpe']:>6.2f}{r['max_drawdown']:>7.1%}"
        print(row)

    print(f"  {'-' * 75}")
    print(f"  {'─── 训练期 ───':<20}{'─── 测试期(纯OOS) ───':>40}")
    print(f"  {'-' * 75}")

    for bname, byears in benchmarks.items():
        row = f"  {bname:<20}"
        total_oos = 1.0
        for y in years:
            if y in byears:
                row += f"{byears[y]:>+8.1%}"
                if y >= 2024:
                    total_oos *= 1 + byears[y]
            else:
                row += f"{'—':>8}"
        row += f"{total_oos - 1:>+9.1%}"
        print(row)

    # 超额
    print(f"\n{'=' * 75}")
    print(f"  OOS超额收益 (vs 沪深300, 仅2024-2026)")
    print(f"{'=' * 75}")
    for name, _, _, _ in strategies:
        r = results[name]
        if "error" in r:
            continue
        row = f"  {name:<20}"
        for y in test_years:
            yr_data = r.get("yearly", {}).get(y, {})
            ret = yr_data.get("return", None)
            bench = benchmarks["沪深300"].get(y, None)
            if ret is not None and bench is not None:
                row += f"{ret - bench:>+8.1%}"
            else:
                row += f"{'—':>8}"
        print(row)

    # 保存
    summary = {
        "train_period": "2022-2023",
        "test_period": "2024-2026",
        "strategies": {},
        "benchmarks": {k: {str(yr): v for yr, v in vs.items()} for k, vs in benchmarks.items()},
    }
    for name, r in results.items():
        if "error" not in r:
            summary["strategies"][name] = {
                "oos_return": r["total_return"],
                "sharpe": r["sharpe"],
                "max_dd": r["max_drawdown"],
                "n_trades": r["n_trades"],
                "yearly": {str(k): v for k, v in r.get("yearly", {}).items()},
            }
    with open(OUTPUT_DIR / "v71_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  结果已保存: {OUTPUT_DIR / 'v71_results.json'}")


if __name__ == "__main__":
    main()
