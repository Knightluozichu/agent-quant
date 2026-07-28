"""v4.2: 修复3个结构性缺陷 + 市场情绪因子.

Fixes:
1. 后视镜缺陷 → 新增 momentum_accel(动量加速度) + breakout(突破信号)
2. Z-score掩盖极端 → Rank标准化 + conviction机制(极端动量+放量=突破,不抵消)
3. 无空仓选项 → composite绝对水平 < 可学习阈值时, 切债券ETF

New factors (19 total):
- Original 16 + momentum_accel + breakout + margin_sentiment

Learnable params (20 total via AdamW):
- 19 factor weights (softmax)
- 1 bond_gate threshold (sigmoid → [0.0, 0.3])
"""

from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "long_history"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

ALL_FACTORS = [
    # Original 5
    "momentum", "reversal", "low_vol", "trend", "volume_trend",
    # C-batch 9
    "bias", "rsi", "macd", "atr_ratio", "obv",
    "skewness", "vol_change", "amplitude", "bollinger",
    # External 2
    "northbound", "pe_percentile",
    # NEW: Fix defect 1 (rear-view mirror)
    "momentum_accel", "breakout",
    # NEW: Market sentiment
    "margin_sentiment",
]
N_FACTORS = len(ALL_FACTORS)  # 19
N_PARAMS = N_FACTORS + 1  # +1 for bond_gate threshold

DEFENSIVE_ASSETS = {"511010": "国债ETF", "511880": "货币ETF"}


# =============================================================================
# Factor Calculator v4.2
# =============================================================================

class FactorCalculatorV42:
    """19 factors with rank normalization + conviction mechanism."""

    def __init__(self):
        self.northbound_data: pd.DataFrame | None = None
        self.pe_data: dict[str, pd.DataFrame] = {}
        self.margin_data: pd.DataFrame | None = None

    def load_external_data(self):
        nb_file = DATA_DIR / "northbound.parquet"
        if nb_file.exists():
            self.northbound_data = pd.read_parquet(nb_file)
        pe_file = DATA_DIR / "pe_percentile.parquet"
        if pe_file.exists():
            pe_df = pd.read_parquet(pe_file)
            for code in pe_df["index_code"].unique():
                self.pe_data[code] = pe_df[pe_df["index_code"] == code].copy()
        margin_file = DATA_DIR / "margin_sentiment.parquet"
        if margin_file.exists():
            self.margin_data = pd.read_parquet(margin_file)

    def calculate_all(self, data: dict[str, pd.DataFrame], as_of_date: date) -> pd.DataFrame:
        records = []
        for symbol, df in data.items():
            hist = df[df["trade_date"] <= as_of_date].copy()
            if len(hist) < 70:
                continue
            hist = hist.sort_values("trade_date")
            close = hist["close"].values.astype(float)
            high = hist["high"].values.astype(float)
            low = hist["low"].values.astype(float)
            volume = hist["volume"].values.astype(float) if "volume" in hist.columns else np.ones(len(close))

            factors = self._calc_single(close, high, low, volume)
            factors["symbol"] = symbol
            factors["northbound"] = self._get_northbound(as_of_date)
            factors["pe_percentile"] = self._get_pe_percentile(symbol, as_of_date)
            factors["margin_sentiment"] = self._get_margin_sentiment(as_of_date)
            records.append(factors)

        if not records:
            return pd.DataFrame()

        factor_df = pd.DataFrame(records)

        # === FIX DEFECT 2: Rank normalization instead of Z-score ===
        for col in ALL_FACTORS:
            if col in factor_df.columns:
                vals = factor_df[col].values
                mask = ~np.isnan(vals)
                if mask.sum() > 2:
                    # Rank-based: convert to percentile [-1, 1]
                    ranked = stats.rankdata(vals[mask])
                    ranked = (ranked - ranked.mean()) / (ranked.std() + 1e-8)
                    factor_df.loc[mask, col] = ranked
                factor_df[col] = factor_df[col].fillna(0.0)

        # === FIX DEFECT 2: Conviction mechanism ===
        # When momentum is extreme AND volume confirms → boost, don't cancel
        # breakout factor already captures this, but we also apply a multiplier
        # to composite later (in the backtest)

        return factor_df

    def _calc_single(self, close, high, low, volume) -> dict[str, float]:
        n = len(close)
        f = {}

        # --- Original 16 factors (same as v4) ---
        f["momentum"] = (close[-20] - close[-80]) / close[-80] if n > 80 else 0.0
        f["reversal"] = -(close[-1] - close[-5]) / close[-5] if n > 5 else 0.0
        if n > 20:
            rets = np.diff(close[-20:]) / close[-20:-1]
            f["low_vol"] = -(np.std(rets) * np.sqrt(252))
        else:
            f["low_vol"] = 0.0
        if n > 60:
            f["trend"] = (close[-20:].mean() - close[-60:].mean()) / close[-60:].mean()
        else:
            f["trend"] = 0.0
        if n > 60:
            v20, v60 = volume[-20:].mean(), volume[-60:].mean()
            f["volume_trend"] = (v20 - v60) / v60 if v60 > 0 else 0.0
        else:
            f["volume_trend"] = 0.0
        if n > 20:
            ma20 = close[-20:].mean()
            f["bias"] = (close[-1] - ma20) / ma20
        else:
            f["bias"] = 0.0
        if n > 15:
            deltas = np.diff(close[-15:])
            gains = np.where(deltas > 0, deltas, 0).mean()
            losses = np.where(deltas < 0, -deltas, 0).mean()
            f["rsi"] = (100 - 100 / (1 + gains / (losses + 1e-10)) - 50) / 50
        else:
            f["rsi"] = 0.0
        if n > 35:
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd_line = ema12 - ema26
            signal = self._ema(macd_line[-9:], 9)
            f["macd"] = (macd_line[-1] - signal[-1]) / close[-1] * 100
        else:
            f["macd"] = 0.0
        if n > 15:
            tr = np.maximum(high[-14:] - low[-14:],
                           np.maximum(np.abs(high[-14:] - close[-15:-1]),
                                      np.abs(low[-14:] - close[-15:-1])))
            f["atr_ratio"] = -(tr.mean() / close[-1])
        else:
            f["atr_ratio"] = 0.0
        if n > 21:
            obv = np.zeros(20)
            for i in range(1, 20):
                idx = n - 20 + i
                if close[idx] > close[idx-1]:
                    obv[i] = obv[i-1] + volume[idx]
                elif close[idx] < close[idx-1]:
                    obv[i] = obv[i-1] - volume[idx]
                else:
                    obv[i] = obv[i-1]
            avg_vol = volume[-20:].mean()
            f["obv"] = (obv[-1] - obv[0]) / (avg_vol * 20) if avg_vol > 0 else 0.0
        else:
            f["obv"] = 0.0
        if n > 21:
            rets = np.diff(close[-21:]) / close[-21:-1]
            f["skewness"] = float(stats.skew(rets))
        else:
            f["skewness"] = 0.0
        if n > 11:
            vol_this = np.std(np.diff(close[-5:]) / close[-5:-1])
            vol_last = np.std(np.diff(close[-10:-5]) / close[-10:-6])
            f["vol_change"] = -(vol_this - vol_last) / (vol_last + 1e-8)
        else:
            f["vol_change"] = 0.0
        if n > 5:
            f["amplitude"] = -((high[-5:] - low[-5:]) / close[-5:]).mean()
        else:
            f["amplitude"] = 0.0
        if n > 20:
            ma20, std20 = close[-20:].mean(), close[-20:].std()
            f["bollinger"] = (close[-1] - ma20) / (2 * std20) if std20 > 1e-8 else 0.0
        else:
            f["bollinger"] = 0.0

        # --- NEW: Fix Defect 1 ---
        # momentum_accel: 动量加速度 (近10日动量 - 前10日动量)
        # 正值 = 动量在加速 (真突破), 负值 = 动量在衰减 (假平静即将结束)
        if n > 40:
            mom_recent = (close[-1] - close[-10]) / close[-10]
            mom_prev = (close[-10] - close[-20]) / close[-20]
            f["momentum_accel"] = mom_recent - mom_prev
        else:
            f["momentum_accel"] = 0.0

        # breakout: 突破信号 (价格突破20日高点 + 放量确认)
        # 解决"高波动=危险"的误判: 如果是向上突破+放量, 高波动是好事
        if n > 21:
            high_20d = high[-21:-1].max()  # 前20日最高价
            price_above = close[-1] > high_20d  # 突破
            vol_ratio = volume[-5:].mean() / (volume[-20:].mean() + 1e-8)  # 放量比
            if price_above and vol_ratio > 1.2:
                # 向上突破+放量: 强正信号
                f["breakout"] = vol_ratio * 0.5
            elif close[-1] < low[-21:-1].min() and vol_ratio > 1.2:
                # 向下突破+放量: 强负信号
                f["breakout"] = -vol_ratio * 0.5
            else:
                f["breakout"] = 0.0
        else:
            f["breakout"] = 0.0

        return f

    def _ema(self, data, period):
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema

    def _get_northbound(self, as_of_date):
        if self.northbound_data is None or self.northbound_data.empty:
            return 0.0
        nb = self.northbound_data[self.northbound_data["trade_date"] <= as_of_date]
        if len(nb) < 5 or "net_flow" not in nb.columns:
            return 0.0
        flow_5d = nb["net_flow"].tail(5).sum()
        std_60 = nb["net_flow"].tail(60).std() if len(nb) > 60 else 1e8
        return flow_5d / (std_60 + 1e-8)

    def _get_pe_percentile(self, symbol, as_of_date):
        etf_to_index = {
            "510300": "000300", "510500": "000905", "159915": "399006",
            "510050": "000016", "512100": "000852", "159901": "399330",
            "510880": "000015", "512800": "399986", "512880": "399975",
        }
        index_code = etf_to_index.get(symbol)
        if index_code is None or index_code not in self.pe_data:
            return 0.0
        pe_df = self.pe_data[index_code]
        pe_hist = pe_df[pe_df["trade_date"] <= as_of_date]
        if len(pe_hist) < 60:
            return 0.0
        current_pe = pe_hist.iloc[-1]["pe"]
        percentile = (pe_hist["pe"] < current_pe).mean()
        return -(percentile - 0.5) * 2

    def _get_margin_sentiment(self, as_of_date):
        """融资余额5日变化率 (市场情绪: 加杠杆=乐观)."""
        if self.margin_data is None or self.margin_data.empty:
            return 0.0
        m = self.margin_data[self.margin_data["trade_date"] <= as_of_date]
        if len(m) < 10 or "margin_balance" not in m.columns:
            return 0.0
        bal = m["margin_balance"].values
        # 5-day change rate
        change_5d = (bal[-1] - bal[-5]) / (bal[-5] + 1e-8)
        # Normalize by 60-day std
        if len(bal) > 60:
            changes = np.diff(bal[-60:]) / bal[-60:-1]
            std = np.std(changes) + 1e-8
            return change_5d / std
        return change_5d * 100  # Rough normalization


# =============================================================================
# AdamW (19 factor weights + 1 bond gate)
# =============================================================================

class AdamW42:
    """AdamW for 19 factor weights + 1 bond gate threshold."""

    def __init__(self, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, beta1, beta2, eps, wd
        self.logits = np.zeros(N_PARAMS)
        self.m = np.zeros(N_PARAMS)
        self.v = np.zeros(N_PARAMS)
        self.t = 0

    def get_factor_weights(self) -> np.ndarray:
        fl = self.logits[:N_FACTORS]
        exp_l = np.exp(fl - fl.max())
        return exp_l / exp_l.sum()

    def get_bond_gate(self) -> float:
        """Bond gate threshold: sigmoid → [0.0, 0.3].
        When top4 avg composite < this value → switch to bonds."""
        return 0.3 / (1.0 + np.exp(-self.logits[N_FACTORS]))

    def step(self, gradient: np.ndarray):
        self.t += 1
        self.logits -= self.lr * self.wd * self.logits
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        self.logits -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self):
        return {
            "logits": self.logits.tolist(), "m": self.m.tolist(),
            "v": self.v.tolist(), "t": self.t,
            "factor_weights": self.get_factor_weights().tolist(),
            "bond_gate": self.get_bond_gate(),
        }

    def load_state_dict(self, state):
        self.logits = np.array(state["logits"])
        self.m = np.array(state["m"])
        self.v = np.array(state["v"])
        self.t = state["t"]


# =============================================================================
# Backtest v4.2
# =============================================================================

def run_backtest_v42(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    factor_weights: np.ndarray,
    bond_gate: float,
    start_date: date | None = None,
    end_date: date | None = None,
    top_n: int = 4,
    rebalance_days: int = 20,
    initial_capital: float = 100_000.0,
) -> dict:
    """Backtest with 3 defect fixes."""
    calculator = FactorCalculatorV42()
    calculator.load_external_data()

    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {k: v for k, v in data.items()
                if k not in DEFENSIVE_ASSETS and k not in index_symbols}

    all_dates = sorted(index_df["trade_date"].tolist())
    warmup = 70
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]
    trading_days = all_dates[warmup:] if len(all_dates) > warmup else all_dates

    cash = initial_capital
    holdings: dict[str, int] = {}
    equity_history = []
    days_since = rebalance_days
    n_trades = 0
    n_bond_switches = 0
    fee_rate = 0.001

    for td in trading_days:
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]

        days_since += 1
        if days_since >= rebalance_days:
            factor_df = calculator.calculate_all(tradable, td)
            if not factor_df.empty and len(factor_df) >= top_n:
                # Composite score
                factor_df["composite"] = factor_df[ALL_FACTORS].values @ factor_weights
                factor_df = factor_df.sort_values("composite", ascending=False)

                # === FIX DEFECT 3: Bond gate ===
                # If top4 average composite < threshold → switch to bonds
                top4_avg = factor_df.head(top_n)["composite"].mean()

                if top4_avg < bond_gate:
                    # Market is weak: switch to bond ETF
                    selected = []
                    bond_sym = "511010"
                    if bond_sym in data:
                        # Sell all equity
                        for sym in list(holdings.keys()):
                            if sym != bond_sym:
                                row = data[sym][data[sym]["trade_date"] == td]
                                if not row.empty:
                                    cash += holdings[sym] * row.iloc[0]["close"] * (1 - fee_rate/2)
                                    n_trades += 1
                                    del holdings[sym]
                        # Buy bond
                        bond_target = equity * 0.9
                        row = data[bond_sym][data[bond_sym]["trade_date"] == td]
                        if not row.empty:
                            price = row.iloc[0]["close"]
                            cur = holdings.get(bond_sym, 0)
                            delta = int((bond_target - cur * price) / price / 10) * 10
                            if delta > 0 and delta * price <= cash:
                                cash -= delta * price
                                holdings[bond_sym] = cur + delta
                                n_trades += 1
                                n_bond_switches += 1
                    days_since = 0
                else:
                    # Normal selection
                    selected = factor_df.head(top_n)["symbol"].tolist()

                    # === FIX DEFECT 2: Conviction boost ===
                    # If breakout factor is extreme positive, boost that stock's allocation
                    conviction_bonus = {}
                    for sym in selected:
                        row = factor_df[factor_df["symbol"] == sym].iloc[0]
                        if row["breakout"] > 1.0:  # Strong breakout signal
                            conviction_bonus[sym] = 1.3  # 30% extra weight
                        else:
                            conviction_bonus[sym] = 1.0

                    # Risk parity + conviction
                    vols = {}
                    for sym in selected:
                        hist = tradable[sym][tradable[sym]["trade_date"] <= td]
                        if len(hist) > 40:
                            rets = np.diff(hist["close"].values[-40:]) / hist["close"].values[-40:-1]
                            vols[sym] = np.std(rets) * np.sqrt(252)
                        else:
                            vols[sym] = 0.2

                    raw_weights = {}
                    for sym in selected:
                        inv_vol = 1.0 / (vols[sym] + 1e-8)
                        raw_weights[sym] = inv_vol * conviction_bonus[sym]
                    total_w = sum(raw_weights.values())
                    target_weights = {s: min(w/total_w, 0.45) for s, w in raw_weights.items()}
                    tw_sum = sum(target_weights.values())
                    target_weights = {s: w/tw_sum for s, w in target_weights.items()}

                    # Sell bond if holding
                    for bond_sym in DEFENSIVE_ASSETS:
                        if bond_sym in holdings:
                            row = data[bond_sym][data[bond_sym]["trade_date"] == td]
                            if not row.empty:
                                cash += holdings[bond_sym] * row.iloc[0]["close"] * (1 - fee_rate/2)
                                del holdings[bond_sym]
                                n_trades += 1

                    # Sell non-target equity
                    for sym in list(holdings.keys()):
                        if sym not in selected:
                            row = data[sym][data[sym]["trade_date"] == td]
                            if not row.empty:
                                cash += holdings[sym] * row.iloc[0]["close"] * (1 - fee_rate/2)
                                n_trades += 1
                                del holdings[sym]

                    # Buy targets
                    for sym, tw in target_weights.items():
                        target_val = equity * tw
                        row = data[sym][data[sym]["trade_date"] == td]
                        if row.empty:
                            continue
                        price = row.iloc[0]["close"]
                        cur = holdings.get(sym, 0)
                        diff = target_val - cur * price
                        if abs(diff) > price * 100:
                            delta = int(diff / price / 100) * 100
                            if delta > 0:
                                cost = delta * price * (1 + fee_rate/2)
                                if cost <= cash:
                                    cash -= cost
                                    holdings[sym] = cur + delta
                                    n_trades += 1
                            elif delta < 0:
                                sell = min(-delta, cur)
                                cash += sell * price * (1 - fee_rate/2)
                                holdings[sym] = cur - sell
                                if holdings[sym] <= 0:
                                    del holdings[sym]
                                n_trades += 1

                    days_since = 0

        # Record equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity})

    if not equity_history:
        return {"total_return": 0.0, "equity_curve": pd.DataFrame(), "n_bond_switches": 0}

    eq_df = pd.DataFrame(equity_history)
    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    n_days = len(eq_df)
    ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    return {
        "total_return": total_return, "ann_return": ann_return,
        "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd,
        "n_trades": n_trades, "n_days": n_days,
        "n_bond_switches": n_bond_switches, "equity_curve": eq_df,
    }


# =============================================================================
# Gradient + Flywheel
# =============================================================================

def compute_gradient(data, index_df, optimizer, sample_dates):
    """IC-based gradient for factor weights + perturbation for bond gate."""
    calculator = FactorCalculatorV42()
    calculator.load_external_data()
    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {k: v for k, v in data.items()
                if k not in DEFENSIVE_ASSETS and k not in index_symbols}

    weights = optimizer.get_factor_weights()
    factor_grad = np.zeros(N_FACTORS)
    n_samples = 0

    for td in sample_dates:
        factor_df = calculator.calculate_all(tradable, td)
        if factor_df.empty or len(factor_df) < 4:
            continue
        fwd = {}
        for sym in factor_df["symbol"].tolist():
            if sym not in data:
                continue
            future = data[sym][data[sym]["trade_date"] > td].head(20)
            current = data[sym][data[sym]["trade_date"] == td]
            if len(future) >= 10 and not current.empty:
                fwd[sym] = (future.iloc[-1]["close"] / current.iloc[0]["close"]) - 1
        if len(fwd) < 4:
            continue

        symbols = factor_df["symbol"].tolist()
        ret_vec = np.array([fwd.get(s, 0) for s in symbols])
        if np.std(ret_vec) < 1e-10:
            continue
        fm = factor_df[ALL_FACTORS].values
        comp = fm @ weights
        if np.std(comp) < 1e-10:
            continue
        base_ic = stats.spearmanr(comp, ret_vec)[0]
        if np.isnan(base_ic):
            continue

        eps = 0.01
        for i in range(N_FACTORS):
            pw = weights.copy()
            pw[i] += eps
            pw /= pw.sum()
            pc = fm @ pw
            if np.std(pc) < 1e-10:
                continue
            pic = stats.spearmanr(pc, ret_vec)[0]
            if not np.isnan(pic):
                factor_grad[i] += -(pic - base_ic) / eps
        n_samples += 1

    if n_samples > 0:
        factor_grad /= n_samples

    # Bond gate gradient (perturbation)
    bond_grad = 0.0
    if len(sample_dates) >= 2:
        bg = optimizer.get_bond_gate()
        r_start = sample_dates[-2]
        r_end = sample_dates[-1]
        base_r = run_backtest_v42(data, index_df, weights, bg,
                                   start_date=r_start, end_date=r_end)["total_return"]
        pert_r = run_backtest_v42(data, index_df, weights, bg + 0.05,
                                   start_date=r_start, end_date=r_end)["total_return"]
        bond_grad = -(pert_r - base_r) / 0.05

    return np.concatenate([factor_grad, [bond_grad]])


def run_flywheel_v42(data, index_df, n_rounds=20, lr=0.03):
    optimizer = AdamW42(lr=lr)
    all_dates = sorted(index_df["trade_date"].tolist())
    n_total = len(all_dates)
    train_size = int(n_total * 0.6)
    test_size = int(n_total * 0.1)
    step_size = (n_total - train_size - test_size) // max(n_rounds - 1, 1)

    best_return = -999.0
    best_state = None
    best_round = 0

    print(f"\n{'='*70}")
    print(f"  v4.2 飞轮: {n_rounds}轮 | AdamW lr={lr} | {N_FACTORS}因子 + bond_gate")
    print(f"  修复: 动量加速度 + Rank标准化 + Conviction + 债券切换")
    print(f"{'='*70}")

    for ri in range(n_rounds):
        train_end = min(train_size + ri * step_size, n_total - test_size)
        test_start = train_end
        test_end = min(test_start + test_size, n_total)
        train_dates = all_dates[:train_end]
        test_dates = all_dates[test_start:test_end]
        if len(test_dates) < 20:
            break

        sample_pts = train_dates[70::20][-10:]
        grad = compute_gradient(data, index_df, optimizer, sample_pts)
        optimizer.step(grad)

        weights = optimizer.get_factor_weights()
        bg = optimizer.get_bond_gate()
        result = run_backtest_v42(data, index_df, weights, bg,
                                   start_date=test_dates[0], end_date=test_dates[-1])
        test_ret = result["total_return"]

        if test_ret > best_return:
            best_return = test_ret
            best_round = ri
            best_state = optimizer.state_dict()
            ckpt = {
                "version": "v4.2", "round": ri,
                "factor_weights": weights.tolist(), "factor_names": ALL_FACTORS,
                "bond_gate": bg, "test_return": test_ret,
                "test_period": f"{test_dates[0]} ~ {test_dates[-1]}",
                "optimizer_state": optimizer.state_dict(),
            }
            with open(CHECKPOINT_DIR / "best_weights_v42.json", "w") as f:
                json.dump(ckpt, f, indent=2, default=str)

        marker = " ★" if ri == best_round else ""
        print(f"  R{ri+1:2d}/{n_rounds} | Test: {test_ret:+.2%} | "
              f"Best: {best_return:+.2%} (R{best_round+1}) | "
              f"gate={bg:.3f} bonds={result.get('n_bond_switches',0)}{marker}")

    print(f"\n  最优: R{best_round+1}, {best_return:+.2%}, bond_gate={optimizer.get_bond_gate():.3f}")
    return {"best_state": best_state, "best_return": best_return, "optimizer": optimizer}


# =============================================================================
# Main
# =============================================================================

def load_long_data():
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


def main():
    print("=" * 70)
    print("  v4.2: 19因子 + 3缺陷修复 + 市场情绪 + 20轮飞轮")
    print("=" * 70)

    print("\n[1/3] 加载数据...")
    data, index_df = load_long_data()
    if index_df is None:
        print("ERROR: 无指数数据")
        return
    print(f"  {len([k for k in data if not k.startswith('idx_')])} 标的, "
          f"{len(index_df)} 天 ({index_df['trade_date'].min()} ~ {index_df['trade_date'].max()})")

    print("\n[2/3] 飞轮进化...")
    evo = run_flywheel_v42(data, index_df, n_rounds=20, lr=0.03)

    print("\n[3/3] 全周期回测 (本金10万)...")
    if evo["best_state"]:
        evo["optimizer"].load_state_dict(evo["best_state"])
    weights = evo["optimizer"].get_factor_weights()
    bg = evo["optimizer"].get_bond_gate()

    result = run_backtest_v42(data, index_df, weights, bg, initial_capital=100_000)

    eq = result["equity_curve"]
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])
    eq["year"] = eq["trade_date"].dt.year

    print(f"\n  年度收益:")
    print(f"  {'年份':<6} {'年初':>10} {'年末':>10} {'收益':>8} {'回撤':>8}")
    print(f"  {'-'*46}")
    prev = 100_000.0
    for year in sorted(eq["year"].unique()):
        ydf = eq[eq["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev) - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        print(f"  {year:<6} {prev:>10,.0f} {end_val:>10,.0f} {yr:>+8.2%} {dd:>8.2%}")
        prev = end_val

    final = eq["equity"].iloc[-1]
    total_ret = (final / 100_000) - 1
    n_years = len(index_df) / 252
    ann_ret = (1 + total_ret) ** (1/n_years) - 1
    print(f"  {'-'*46}")
    print(f"\n  10万 → {final:,.0f} ({total_ret:+.1%}, {final/100_000:.2f}x)")
    print(f"  年化: {ann_ret:+.1%} | 夏普: {result['sharpe']:.2f} | 回撤: {result['max_drawdown']:.1%}")
    print(f"  债券切换次数: {result['n_bond_switches']}")

    # Factor weights
    print(f"\n  因子权重:")
    for i in np.argsort(weights)[::-1]:
        bar = "█" * int(weights[i] * 60)
        print(f"    {ALL_FACTORS[i]:<16}: {weights[i]*100:.1f}% {bar}")
    print(f"    bond_gate阈值: {bg:.3f}")

    idx_c = index_df["close"].values
    bench = (idx_c[-1] - idx_c[0]) / idx_c[0]
    print(f"\n  沪深300: {bench:+.1%} | 超额: {total_ret - bench:+.1%}")


if __name__ == "__main__":
    main()
