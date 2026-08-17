"""V7: 4策略×不同频率 统一回测 + 年度基准对比.

策略矩阵:
  A: 日频(1天) - 纯动量Top4
  B: 季频(60天) - NN-8 + Gate + VolScale
  C: 季频(60天) - v4.2 (16因子等权 + Bond Gate)
  D: 周频(5天) - v4.2 (16因子等权 + Bond Gate)

基准:
  - 沪深300
  - 21ETF等权
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "v7_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFENSIVE = {"511010": "国债ETF", "511880": "货币ETF"}
FEE = 0.001
SLIPPAGE = 0.0005

# NN超参
NN_EPOCHS = 80
NN_LR = 0.005
NN_WD = 0.001

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

    # 动量族
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

    # 反转族
    f["reversal"] = -(close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0.0
    if len(close) >= 14:
        deltas = np.diff(close[-15:])
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        f["rsi"] = -(100 - 100 / (1 + gains / (losses + 1e-8)))
    else:
        f["rsi"] = 0.0

    # 低波族
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

    # 趋势族
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

    # 量能族
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

    # 独立
    if len(close) >= 21:
        rets = np.diff(close[-21:]) / close[-21:-1]
        m, s = np.mean(rets), np.std(rets) + 1e-8
        f["skewness"] = np.mean(((rets - m) / s) ** 3)
    else:
        f["skewness"] = 0.0

    return f


def build_factor_panel(data: dict, index_df: pd.DataFrame, dates: list) -> dict:
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


def rank_normalize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].rank(pct=True)
    return out


# =============================================================================
# Neural Network
# =============================================================================


class FactorNN:
    def __init__(self, n_in, n_hidden=8, lr=NN_LR, wd=NN_WD, seed=42):
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
        self._out = (self._h1 @ self.W2 + self.b2).flatten()
        return self._out

    def train(self, X, y, epochs=NN_EPOCHS):
        n = len(X)
        for _ in range(epochs):
            idx = np.random.permutation(n)[: min(256, n)]
            Xb, yb = X[idx], y[idx]
            pred = self.forward(Xb)
            # BP for -Pearson IC
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
            # AdamW
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
# Unified Backtest Engine
# =============================================================================


def run_backtest(
    data: dict,
    index_df: pd.DataFrame,
    panels: dict,
    strategy: str,
    rebalance: int,
    initial_capital: float = 100_000.0,
    top_n: int = 4,
) -> dict:
    """统一回测引擎.

    strategy: "momentum" | "nn_gate_vol" | "v42"
    rebalance: 调仓频率(天数)
    """
    all_dates = sorted(panels.keys())
    warmup = 100
    trading_dates = all_dates[warmup:]

    # 调仓日
    rebalance_dates = trading_dates[::rebalance]

    # WF训练用 (NN策略需要)
    use_nn = strategy == "nn_gate_vol"
    use_gate = strategy in ("nn_gate_vol", "v42")
    use_vol = strategy == "nn_gate_vol"

    cash = initial_capital
    holdings: dict[str, int] = {}
    equity_history = []
    n_trades = 0

    # NN训练窗口管理
    nn = None
    last_train_idx = -999

    for ri, td in enumerate(rebalance_dates):
        if td not in panels:
            continue
        panel = panels[td]

        # === 选股 ===
        if strategy == "momentum":
            # 纯动量
            scores = panel.set_index("symbol")["momentum"]
            selected = scores.nlargest(top_n).index.tolist()
            position_scale = 1.0

        elif strategy == "nn_gate_vol":
            # Gate检查
            gate_open = check_gate(index_df, td)
            if not gate_open:
                selected = []
                position_scale = 0.0
            else:
                # NN训练 (每60天重训)
                ri_global = ri
                if nn is None or (ri_global - last_train_idx) >= 12:
                    # 收集训练数据
                    X_list, y_list = [], []
                    train_dates = [d for d in all_dates if d < td][-60::5]
                    for ttd in train_dates:
                        if ttd not in panels:
                            continue
                        tp = panels[ttd]
                        fwd = get_forward_returns(data, tp, ttd, horizon=20)
                        if len(fwd) < 8:
                            continue
                        merged = tp[tp["symbol"].isin(fwd.index)].copy()
                        if len(merged) < 8:
                            continue
                        merged = rank_normalize(merged, CLUSTER_REPS)
                        X_list.append(merged[CLUSTER_REPS].values)
                        y_list.append(merged["symbol"].map(fwd).values)
                    if X_list:
                        X_train = np.vstack(X_list)
                        y_train = np.concatenate(y_list)
                        mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
                        if mask.sum() >= 20:
                            nn = FactorNN(n_in=6, n_hidden=8)
                            nn.train(X_train[mask], y_train[mask])
                            last_train_idx = ri_global

                if nn is not None:
                    panel_norm = rank_normalize(panel.copy(), CLUSTER_REPS)
                    X_test = panel_norm[CLUSTER_REPS].values
                    pred = nn.predict(X_test)
                    scores = pd.Series(pred, index=panel["symbol"].values)
                else:
                    scores = panel.set_index("symbol")["momentum"]

                selected = scores.nlargest(top_n).index.tolist()
                position_scale = calc_vol_scale(data, selected, td)

        elif strategy == "v42":
            # v4.2: 16因子等权 + Bond Gate
            gate_open = check_gate(index_df, td)
            if not gate_open:
                selected = []
                position_scale = 0.0
            else:
                panel_norm = rank_normalize(panel.copy(), ALL_FACTORS)
                scores = panel_norm.set_index("symbol")[ALL_FACTORS].mean(axis=1)
                selected = scores.nlargest(top_n).index.tolist()
                position_scale = 1.0

        # === 交易执行 ===
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]

        stock_budget = equity * position_scale
        bond_budget = equity * (1 - position_scale)
        per_stock = stock_budget / max(len(selected), 1) if selected else 0

        # 卖出非目标
        for sym in list(holdings.keys()):
            if sym not in selected and sym not in DEFENSIVE:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    cash += holdings[sym] * row.iloc[0]["close"] * (1 - FEE - SLIPPAGE)
                    n_trades += 1
                    del holdings[sym]

        # 国债
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

        # 买入股票
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

        # 记录equity
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
# Benchmarks
# =============================================================================


def calc_benchmarks(data: dict, index_df: pd.DataFrame, start_date, end_date) -> dict:
    """计算基准年度收益."""
    benchmarks = {}

    # 沪深300
    idx = index_df[(index_df["trade_date"] >= start_date) & (index_df["trade_date"] <= end_date)]
    if not idx.empty:
        benchmarks["沪深300"] = {}
        idx_c = idx.set_index("trade_date")["close"]
        for year in sorted(set(idx["trade_date"].apply(lambda x: x.year))):
            ydf = idx[idx["trade_date"].apply(lambda x: x.year) == year]
            if len(ydf) >= 2:
                yr = (ydf["close"].iloc[-1] - ydf["close"].iloc[0]) / ydf["close"].iloc[0]
                benchmarks["沪深300"][year] = yr

    # 21ETF等权
    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {k: v for k, v in data.items() if k not in DEFENSIVE and k not in index_symbols}
    benchmarks["21ETF等权"] = {}
    for year in range(2022, 2027):
        rets = []
        for sym, df in tradable.items():
            ydf = df[df["trade_date"].apply(lambda x: x.year) == year]
            if len(ydf) >= 2:
                yr = (ydf["close"].iloc[-1] - ydf["close"].iloc[0]) / ydf["close"].iloc[0]
                rets.append(yr)
        if rets:
            benchmarks["21ETF等权"][year] = np.mean(rets)

    return benchmarks


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
    print("  V7: 4策略×不同频率 统一回测")
    print("  A: 日频(1天) 纯动量Top4")
    print("  B: 季频(60天) NN-8+Gate+Vol")
    print("  C: 季频(60天) v4.2(16因子+Gate)")
    print("  D: 周频(5天) v4.2(16因子+Gate)")
    print("=" * 75)

    data, index_df = load_data()
    if index_df is None:
        print("ERROR: 无指数数据")
        return

    all_dates = sorted(index_df["trade_date"].tolist())
    print(f"\n  数据: {all_dates[0]} ~ {all_dates[-1]} ({len(all_dates)}天)")

    # 构建日级因子面板
    print(f"  构建因子面板...")
    warmup_dates = all_dates[80:]
    panels = build_factor_panel(data, index_df, warmup_dates)
    print(f"  有效面板: {len(panels)}天")

    # 4个策略
    strategies = [
        ("A: 日频-纯动量Top4", "momentum", 1),
        ("B: 季频-NN8+Gate+Vol", "nn_gate_vol", 60),
        ("C: 季频-v4.2", "v42", 60),
        ("D: 周频-v4.2", "v42", 5),
    ]

    results = {}
    for name, strat, rebal in strategies:
        print(f"\n  运行 {name} (rebalance={rebal}天)...", end=" ", flush=True)
        r = run_backtest(data, index_df, panels, strategy=strat, rebalance=rebal)
        results[name] = r
        if "error" in r:
            print(f"ERROR: {r['error']}")
        else:
            print(
                f"全周期{r['total_return']:+.1%} | 夏普{r['sharpe']:.2f} | "
                f"回撤{r['max_drawdown']:.1%} | 交易{r['n_trades']}次"
            )

    # 基准
    benchmarks = calc_benchmarks(data, index_df, all_dates[0], all_dates[-1])

    # === 年度矩阵 ===
    years = list(range(2022, 2027))
    print(f"\n{'=' * 75}")
    print(f"  年度收益矩阵 (含基准)")
    print(f"{'=' * 75}")

    header = (
        f"  {'策略':<22}"
        + "".join(f"{y:>8}" for y in years)
        + f"{'全周期':>8}{'夏普':>6}{'回撤':>7}{'交易':>5}"
    )
    print(header)
    print(f"  {'-' * 72}")

    for name, strat, rebal in strategies:
        r = results[name]
        if "error" in r:
            print(f"  {name:<22} ERROR")
            continue
        row = f"  {name:<22}"
        for y in years:
            yr_data = r.get("yearly", {}).get(y, {})
            ret = yr_data.get("return", None)
            if ret is not None:
                row += f"{ret:>+8.1%}"
            else:
                row += f"{'—':>8}"
        row += f"{r['total_return']:>+8.1%}{r['sharpe']:>6.2f}{r['max_drawdown']:>7.1%}{r['n_trades']:>5}"
        print(row)

    # 基准
    print(f"  {'-' * 72}")
    for bname, byears in benchmarks.items():
        row = f"  {bname:<22}"
        total = 1.0
        for y in years:
            if y in byears:
                row += f"{byears[y]:>+8.1%}"
                total *= 1 + byears[y]
            else:
                row += f"{'—':>8}"
        row += f"{total - 1:>+8.1%}"
        print(row)

    # === 超额收益 ===
    print(f"\n{'=' * 75}")
    print(f"  超额收益 (vs 沪深300)")
    print(f"{'=' * 75}")
    header2 = f"  {'策略':<22}" + "".join(f"{y:>8}" for y in years)
    print(header2)
    print(f"  {'-' * 62}")
    for name, strat, rebal in strategies:
        r = results[name]
        if "error" in r:
            continue
        row = f"  {name:<22}"
        for y in years:
            yr_data = r.get("yearly", {}).get(y, {})
            ret = yr_data.get("return", None)
            bench = benchmarks.get("沪深300", {}).get(y, None)
            if ret is not None and bench is not None:
                row += f"{ret - bench:>+8.1%}"
            else:
                row += f"{'—':>8}"
        print(row)

    # 保存
    summary = {
        "strategies": {},
        "benchmarks": {k: {str(yr): v for yr, v in vs.items()} for k, vs in benchmarks.items()},
    }
    for name, r in results.items():
        if "error" not in r:
            summary["strategies"][name] = {
                "total_return": r["total_return"],
                "sharpe": r["sharpe"],
                "max_drawdown": r["max_drawdown"],
                "n_trades": r["n_trades"],
                "yearly": {str(k): v for k, v in r.get("yearly", {}).items()},
            }
    with open(OUTPUT_DIR / "v7_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  结果已保存: {OUTPUT_DIR / 'v7_results.json'}")


if __name__ == "__main__":
    main()
