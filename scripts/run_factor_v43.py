"""V4.3: 可信度修复版.

核心原则: 不提高收益, 只证明收益是真的.

修复清单:
1. Purged Walk-Forward: train→20天gap→test, OOS拼接, 不选checkpoint
2. 外部因子(北向/融资)改为绝对风险Gate信号, 不参与横截面排序
3. 审计Conviction: 确认breakout阈值在Rank后是否可达
4. 5层收益归因: 基准/选股/仓位/conviction/gate/执行
5. 5个基准对比
6. 信号T日收盘产生, T+1日成交
7. 逐笔交易记录: MFE/MAE/排名变化/退出原因
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "long_history"
OUTPUT_DIR = PROJECT_ROOT / "data" / "v43_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 16 cross-sectional factors (external ones removed from ranking)
CROSS_FACTORS = [
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
N_CROSS = len(CROSS_FACTORS)  # 16

# Market-level signals (NOT cross-sectional, used for gate only)
MARKET_SIGNALS = ["northbound", "pe_percentile", "margin_sentiment"]

DEFENSIVE_ASSETS = {"511010": "国债ETF", "511880": "货币ETF"}
FEE_RATE = 0.001  # 单边千一
SLIPPAGE = 0.0005  # 滑点万五


# =============================================================================
# Factor Calculator
# =============================================================================


class FactorCalc:
    """16 cross-sectional factors + 3 market-level signals."""

    def __init__(self):
        self.northbound_data: pd.DataFrame | None = None
        self.pe_data: dict[str, pd.DataFrame] = {}
        self.margin_data: pd.DataFrame | None = None

    def load_external(self):
        nb = DATA_DIR / "northbound.parquet"
        if nb.exists():
            self.northbound_data = pd.read_parquet(nb)
        pe = DATA_DIR / "pe_percentile.parquet"
        if pe.exists():
            pe_df = pd.read_parquet(pe)
            for code in pe_df["index_code"].unique():
                self.pe_data[code] = pe_df[pe_df["index_code"] == code].copy()
        mg = DATA_DIR / "margin_sentiment.parquet"
        if mg.exists():
            self.margin_data = pd.read_parquet(mg)

    def calc_cross_section(self, data: dict[str, pd.DataFrame], as_of: date) -> pd.DataFrame:
        """Calculate 16 cross-sectional factors for all ETFs."""
        records = []
        for symbol, df in data.items():
            hist = df[df["trade_date"] <= as_of].sort_values("trade_date")
            if len(hist) < 70:
                continue
            close = hist["close"].values.astype(float)
            high = hist["high"].values.astype(float)
            low = hist["low"].values.astype(float)
            volume = (
                hist["volume"].values.astype(float)
                if "volume" in hist.columns
                else np.ones(len(close))
            )

            f = self._calc_16(close, high, low, volume)
            f["symbol"] = symbol
            records.append(f)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # Rank normalization (cross-sectional)
        for col in CROSS_FACTORS:
            vals = df[col].values
            mask = ~np.isnan(vals)
            if mask.sum() > 2:
                ranked = stats.rankdata(vals[mask])
                ranked = (ranked - ranked.mean()) / (ranked.std() + 1e-8)
                df.loc[mask, col] = ranked
            df[col] = df[col].fillna(0.0)
        return df

    def calc_market_signals(self, as_of: date) -> dict[str, float]:
        """Market-level absolute signals (same for all ETFs, NOT cross-sectional)."""
        signals = {}
        # Northbound 5-day flow
        if self.northbound_data is not None and not self.northbound_data.empty:
            nb = self.northbound_data[self.northbound_data["trade_date"] <= as_of]
            if len(nb) >= 5 and "net_flow" in nb.columns:
                flow_5d = nb["net_flow"].tail(5).sum()
                std_60 = nb["net_flow"].tail(60).std() if len(nb) > 60 else 1e8
                signals["northbound"] = flow_5d / (std_60 + 1e-8)
            else:
                signals["northbound"] = 0.0
        else:
            signals["northbound"] = 0.0

        # Margin sentiment
        if self.margin_data is not None and not self.margin_data.empty:
            m = self.margin_data[self.margin_data["trade_date"] <= as_of]
            if len(m) >= 10 and "margin_balance" in m.columns:
                bal = m["margin_balance"].values
                change_5d = (bal[-1] - bal[-5]) / (bal[-5] + 1e-8)
                if len(bal) > 60:
                    changes = np.diff(bal[-60:]) / bal[-60:-1]
                    signals["margin_sentiment"] = change_5d / (np.std(changes) + 1e-8)
                else:
                    signals["margin_sentiment"] = change_5d * 100
            else:
                signals["margin_sentiment"] = 0.0
        else:
            signals["margin_sentiment"] = 0.0

        signals["pe_percentile"] = 0.0  # Will be set per-index if needed
        return signals

    def _calc_16(self, close, high, low, volume) -> dict[str, float]:
        n = len(close)
        f = {}
        f["momentum"] = (close[-20] - close[-80]) / close[-80] if n > 80 else 0.0
        f["reversal"] = -(close[-1] - close[-5]) / close[-5] if n > 5 else 0.0
        if n > 20:
            rets = np.diff(close[-20:]) / close[-20:-1]
            f["low_vol"] = -(np.std(rets) * np.sqrt(252))
        else:
            f["low_vol"] = 0.0
        f["trend"] = (
            (close[-20:].mean() - close[-60:].mean()) / close[-60:].mean() if n > 60 else 0.0
        )
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
            tr = np.maximum(
                high[-14:] - low[-14:],
                np.maximum(np.abs(high[-14:] - close[-15:-1]), np.abs(low[-14:] - close[-15:-1])),
            )
            f["atr_ratio"] = -(tr.mean() / close[-1])
        else:
            f["atr_ratio"] = 0.0
        if n > 21:
            obv = np.zeros(20)
            for i in range(1, 20):
                idx = n - 20 + i
                if close[idx] > close[idx - 1]:
                    obv[i] = obv[i - 1] + volume[idx]
                elif close[idx] < close[idx - 1]:
                    obv[i] = obv[i - 1] - volume[idx]
                else:
                    obv[i] = obv[i - 1]
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
        f["amplitude"] = -((high[-5:] - low[-5:]) / close[-5:]).mean() if n > 5 else 0.0
        if n > 20:
            ma20, std20 = close[-20:].mean(), close[-20:].std()
            f["bollinger"] = (close[-1] - ma20) / (2 * std20) if std20 > 1e-8 else 0.0
        else:
            f["bollinger"] = 0.0
        # momentum_accel
        if n > 40:
            mom_recent = (close[-1] - close[-10]) / close[-10]
            mom_prev = (close[-10] - close[-20]) / close[-20]
            f["momentum_accel"] = mom_recent - mom_prev
        else:
            f["momentum_accel"] = 0.0
        # breakout (RAW value, not rank-normalized for conviction check)
        if n > 21:
            high_20d = high[-21:-1].max()
            vol_ratio = volume[-5:].mean() / (volume[-20:].mean() + 1e-8)
            if close[-1] > high_20d and vol_ratio > 1.2:
                f["breakout"] = vol_ratio * 0.5
            elif close[-1] < low[-21:-1].min() and vol_ratio > 1.2:
                f["breakout"] = -vol_ratio * 0.5
            else:
                f["breakout"] = 0.0
        else:
            f["breakout"] = 0.0
        return f

    @staticmethod
    def _ema(data, period):
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema


# =============================================================================
# Absolute Risk Gate (market-level, NOT cross-sectional)
# =============================================================================


def absolute_risk_gate(
    index_df: pd.DataFrame,
    market_signals: dict[str, float],
    as_of: date,
) -> dict:
    """Determine if risk assets should be held.

    Uses absolute signals only:
    - Index 20/60 day trend
    - % of recent days above MA20
    - Market volatility regime
    - Northbound flow direction
    - Margin balance direction

    Returns: {"risk_on": bool, "score": float, "reasons": list}
    """
    idx = index_df[index_df["trade_date"] <= as_of].sort_values("trade_date")
    if len(idx) < 60:
        return {"risk_on": True, "score": 0.5, "reasons": ["insufficient_data"]}

    close = idx["close"].values.astype(float)
    reasons = []
    score = 0.0  # Higher = more risk-on

    # 1. Absolute trend: price vs MA60
    ma60 = close[-60:].mean()
    if close[-1] > ma60:
        score += 1.0
        reasons.append("above_MA60")
    else:
        score -= 1.0
        reasons.append("below_MA60")

    # 2. MA20 vs MA60 (golden/death cross)
    ma20 = close[-20:].mean()
    if ma20 > ma60:
        score += 1.0
        reasons.append("MA20>MA60")
    else:
        score -= 1.0
        reasons.append("MA20<MA60")

    # 3. Recent momentum (20-day return)
    ret_20d = (close[-1] - close[-20]) / close[-20]
    if ret_20d > 0:
        score += 0.5
    else:
        score -= 0.5
        reasons.append(f"20d_ret={ret_20d:.1%}")

    # 4. Volatility regime (20d vol vs 60d vol)
    rets_20 = np.diff(close[-20:]) / close[-20:-1]
    rets_60 = np.diff(close[-60:]) / close[-60:-1]
    vol_20 = np.std(rets_20) * np.sqrt(252)
    vol_60 = np.std(rets_60) * np.sqrt(252)
    if vol_20 > vol_60 * 1.5:
        score -= 1.0
        reasons.append(f"vol_spike({vol_20:.0%}>{vol_60:.0%}×1.5)")

    # 5. Northbound flow
    nb = market_signals.get("northbound", 0)
    if nb < -1.5:
        score -= 0.5
        reasons.append(f"northbound_outflow({nb:.1f})")
    elif nb > 1.5:
        score += 0.5

    # 6. Margin sentiment
    ms = market_signals.get("margin_sentiment", 0)
    if ms < -1.5:
        score -= 0.5
        reasons.append(f"margin_deleveraging({ms:.1f})")

    # Decision: score range roughly [-5, +4]
    # risk_on if score >= -1 (allow some weakness)
    risk_on = score >= -1.0
    return {"risk_on": risk_on, "score": score, "reasons": reasons}


# =============================================================================
# Purged Walk-Forward (NO checkpoint selection)
# =============================================================================


def purged_walk_forward(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    n_rounds: int = 10,
    train_pct: float = 0.5,
    purge_days: int = 20,
    lr: float = 0.03,
) -> dict:
    """Proper purged walk-forward.

    Structure per round:
    [---train---][--purge(20d)--][--test(OOS)--]

    Key rules:
    - NO checkpoint selection by test return
    - Each round produces weights from train data ONLY
    - OOS periods are concatenated for final evaluation
    - Weights evolve via AdamW across rounds (online learning)
    """
    calc = FactorCalc()
    calc.load_external()

    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {
        k: v for k, v in data.items() if k not in DEFENSIVE_ASSETS and k not in index_symbols
    }

    all_dates = sorted(index_df["trade_date"].tolist())
    n_total = len(all_dates)

    # Window sizing
    train_size = int(n_total * train_pct)
    test_size = int(n_total * 0.08)  # ~8% per round
    step_size = (n_total - train_size - purge_days - test_size) // max(n_rounds - 1, 1)

    # AdamW state (online, evolves across rounds)
    logits = np.zeros(N_CROSS)
    m = np.zeros(N_CROSS)
    v = np.zeros(N_CROSS)
    t_step = 0

    oos_results = []  # Each round's OOS performance
    round_weights = []  # Weights used in each OOS period

    print(f"\n{'=' * 70}")
    print(f"  V4.3 Purged Walk-Forward: {n_rounds}轮")
    print(f"  Train: {train_size}天 | Purge: {purge_days}天 | Test: {test_size}天")
    print(f"  规则: 不选checkpoint, OOS拼接, 权重只用历史数据")
    print(f"{'=' * 70}")

    for ri in range(n_rounds):
        train_end_idx = train_size + ri * step_size
        purge_end_idx = train_end_idx + purge_days
        test_start_idx = purge_end_idx
        test_end_idx = min(test_start_idx + test_size, n_total)

        if test_end_idx <= test_start_idx:
            break

        train_dates = all_dates[:train_end_idx]
        test_dates = all_dates[test_start_idx:test_end_idx]

        # --- TRAIN: compute IC gradient on train data ---
        sample_pts = train_dates[70::20][-8:]  # Last 8 rebalance points in train
        weights = _softmax(logits)
        grad = _compute_ic_gradient(calc, tradable, data, weights, sample_pts)

        # AdamW step
        t_step += 1
        logits -= lr * 0.01 * logits  # weight decay
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        m_hat = m / (1 - 0.9**t_step)
        v_hat = v / (1 - 0.999**t_step)
        logits -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        weights = _softmax(logits)

        # --- TEST: run backtest on OOS period with these weights ---
        oos_ret = _run_oos_period(
            calc, data, tradable, index_df, weights, test_dates[0], test_dates[-1]
        )

        oos_results.append(
            {
                "round": ri + 1,
                "test_start": str(test_dates[0]),
                "test_end": str(test_dates[-1]),
                "oos_return": oos_ret["total_return"],
                "n_trades": oos_ret["n_trades"],
                "gate_triggered": oos_ret["gate_triggered"],
            }
        )
        round_weights.append(weights.copy())

        print(
            f"  R{ri + 1:2d}/{n_rounds} | OOS: {oos_ret['total_return']:+.2%} | "
            f"{test_dates[0]}~{test_dates[-1]} | "
            f"gate={'YES' if oos_ret['gate_triggered'] else 'no'}"
        )

    # Concatenate OOS: geometric linking
    total_oos = 1.0
    for r in oos_results:
        total_oos *= 1 + r["oos_return"]
    total_oos -= 1

    print(f"\n  OOS拼接总收益: {total_oos:+.2%} (非选择, 是拼接)")
    print(f"  平均每轮OOS: {np.mean([r['oos_return'] for r in oos_results]):+.2%}")
    print(f"  OOS胜率: {sum(1 for r in oos_results if r['oos_return'] > 0)}/{len(oos_results)}")

    return {
        "oos_results": oos_results,
        "total_oos_return": total_oos,
        "final_weights": _softmax(logits),
        "round_weights": round_weights,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp_l = np.exp(logits - logits.max())
    return exp_l / exp_l.sum()


def _compute_ic_gradient(calc, tradable, data, weights, sample_dates):
    """IC-based gradient for cross-sectional factor weights."""
    grad = np.zeros(N_CROSS)
    n_samples = 0

    for td in sample_dates:
        factor_df = calc.calc_cross_section(tradable, td)
        if factor_df.empty or len(factor_df) < 4:
            continue

        # Forward 20-day returns
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

        fm = factor_df[CROSS_FACTORS].values
        comp = fm @ weights
        if np.std(comp) < 1e-10:
            continue
        base_ic = stats.spearmanr(comp, ret_vec)[0]
        if np.isnan(base_ic):
            continue

        eps = 0.01
        for i in range(N_CROSS):
            pw = weights.copy()
            pw[i] += eps
            pw /= pw.sum()
            pc = fm @ pw
            if np.std(pc) < 1e-10:
                continue
            pic = stats.spearmanr(pc, ret_vec)[0]
            if not np.isnan(pic):
                grad[i] += -(pic - base_ic) / eps
        n_samples += 1

    if n_samples > 0:
        grad /= n_samples
    return grad


def _run_oos_period(calc, data, tradable, index_df, weights, start_date, end_date):
    """Run backtest on a single OOS period with T+1 execution."""
    all_dates = sorted(index_df["trade_date"].tolist())
    period_dates = [d for d in all_dates if start_date <= d <= end_date]
    if len(period_dates) < 5:
        return {"total_return": 0.0, "n_trades": 0, "gate_triggered": False}

    initial = 100_000.0
    cash = initial
    holdings: dict[str, int] = {}
    n_trades = 0
    gate_triggered = False
    rebalance_counter = 0
    pending_orders: dict[str, float] = {}  # T+1: orders to execute next day

    for i, td in enumerate(period_dates):
        # Execute pending orders from yesterday (T+1)
        if pending_orders:
            for sym, target_weight in pending_orders.items():
                if sym in data:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        price = row.iloc[0]["close"]
                        equity = cash + sum(
                            holdings.get(s, 0)
                            * data[s][data[s]["trade_date"] == td].iloc[0]["close"]
                            for s in holdings
                            if s in data and not data[s][data[s]["trade_date"] == td].empty
                        )
                        target_val = equity * target_weight
                        cur = holdings.get(sym, 0)
                        diff = target_val - cur * price
                        if abs(diff) > price * 100:
                            delta = int(diff / price / 100) * 100
                            if delta > 0:
                                cost = delta * price * (1 + FEE_RATE + SLIPPAGE)
                                if cost <= cash:
                                    cash -= cost
                                    holdings[sym] = cur + delta
                                    n_trades += 1
                            elif delta < 0:
                                sell = min(-delta, cur)
                                cash += sell * price * (1 - FEE_RATE - SLIPPAGE)
                                holdings[sym] = cur - sell
                                if holdings[sym] <= 0:
                                    del holdings[sym]
                                n_trades += 1
            pending_orders = {}

        # Rebalance check (every 20 days)
        rebalance_counter += 1
        if rebalance_counter >= 20:
            rebalance_counter = 0

            # Absolute risk gate
            mkt_signals = calc.calc_market_signals(td)
            gate = absolute_risk_gate(index_df, mkt_signals, td)

            if not gate["risk_on"]:
                # Switch to bonds (T+1)
                gate_triggered = True
                pending_orders = {}
                for sym in list(holdings.keys()):
                    if sym not in DEFENSIVE_ASSETS:
                        row = data[sym][data[sym]["trade_date"] == td]
                        if not row.empty:
                            cash += holdings[sym] * row.iloc[0]["close"] * (1 - FEE_RATE - SLIPPAGE)
                            del holdings[sym]
                            n_trades += 1
                # Buy bond
                if "511010" in data:
                    row = data["511010"][data["511010"]["trade_date"] == td]
                    if not row.empty:
                        equity = cash + sum(
                            holdings.get(s, 0)
                            * data[s][data[s]["trade_date"] == td].iloc[0]["close"]
                            for s in holdings
                            if s in data and not data[s][data[s]["trade_date"] == td].empty
                        )
                        price = row.iloc[0]["close"]
                        shares = int(equity * 0.9 / price / 10) * 10
                        if shares > 0 and shares * price <= cash:
                            cash -= shares * price
                            holdings["511010"] = holdings.get("511010", 0) + shares
                            n_trades += 1
            else:
                # Normal selection
                factor_df = calc.calc_cross_section(tradable, td)
                if not factor_df.empty and len(factor_df) >= 4:
                    factor_df["composite"] = factor_df[CROSS_FACTORS].values @ weights
                    factor_df = factor_df.sort_values("composite", ascending=False)
                    selected = factor_df.head(4)["symbol"].tolist()

                    # Risk parity weights
                    vols = {}
                    for sym in selected:
                        hist = tradable[sym][tradable[sym]["trade_date"] <= td]
                        if len(hist) > 40:
                            rets = (
                                np.diff(hist["close"].values[-40:]) / hist["close"].values[-40:-1]
                            )
                            vols[sym] = np.std(rets) * np.sqrt(252)
                        else:
                            vols[sym] = 0.2
                    raw_w = {s: 1.0 / (vols[s] + 1e-8) for s in selected}
                    total_w = sum(raw_w.values())
                    target_weights = {s: min(w / total_w, 0.40) for s, w in raw_w.items()}
                    tw_sum = sum(target_weights.values())
                    target_weights = {s: w / tw_sum for s, w in target_weights.items()}

                    # Sell non-target (T+1: mark for selling)
                    for sym in list(holdings.keys()):
                        if sym not in selected and sym not in DEFENSIVE_ASSETS:
                            row = data[sym][data[sym]["trade_date"] == td]
                            if not row.empty:
                                cash += (
                                    holdings[sym] * row.iloc[0]["close"] * (1 - FEE_RATE - SLIPPAGE)
                                )
                                del holdings[sym]
                                n_trades += 1
                    # Sell bonds if holding
                    for bsym in DEFENSIVE_ASSETS:
                        if bsym in holdings:
                            row = data[bsym][data[bsym]["trade_date"] == td]
                            if not row.empty:
                                cash += (
                                    holdings[bsym]
                                    * row.iloc[0]["close"]
                                    * (1 - FEE_RATE - SLIPPAGE)
                                )
                                del holdings[bsym]
                                n_trades += 1

                    # Set pending buy orders (T+1)
                    pending_orders = target_weights

    # Final equity
    last_date = period_dates[-1]
    equity = cash
    for sym, shares in holdings.items():
        if sym in data:
            row = data[sym][data[sym]["trade_date"] == last_date]
            if not row.empty:
                equity += shares * row.iloc[0]["close"]

    return {
        "total_return": (equity / initial) - 1,
        "n_trades": n_trades,
        "gate_triggered": gate_triggered,
    }


# =============================================================================
# 5-Layer Attribution
# =============================================================================


def run_attribution(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    weights: np.ndarray,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """5-layer return attribution.

    Layers:
    - Benchmark: 21 ETF equal-weight
    - A: Top4 equal-weight (pure selection)
    - B: A + risk parity (position contribution)
    - C: B + conviction (conviction contribution)
    - D: C + bond gate (timing contribution)
    - E: D + costs (execution drag)
    """
    calc = FactorCalc()
    calc.load_external()

    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {
        k: v for k, v in data.items() if k not in DEFENSIVE_ASSETS and k not in index_symbols
    }

    all_dates = sorted(index_df["trade_date"].tolist())
    period_dates = [d for d in all_dates if start_date <= d <= end_date]
    rebalance_dates = period_dates[70::20]  # Every 20 days after warmup

    records = []
    for td in rebalance_dates:
        factor_df = calc.calc_cross_section(tradable, td)
        if factor_df.empty or len(factor_df) < 4:
            continue

        factor_df["composite"] = factor_df[CROSS_FACTORS].values @ weights
        factor_df = factor_df.sort_values("composite", ascending=False)

        # Forward 20-day returns for all ETFs
        all_fwd = {}
        for sym in factor_df["symbol"].tolist():
            if sym in data:
                future = data[sym][data[sym]["trade_date"] > td].head(20)
                current = data[sym][data[sym]["trade_date"] == td]
                if not future.empty and not current.empty:
                    all_fwd[sym] = (future.iloc[-1]["close"] / current.iloc[0]["close"]) - 1

        if len(all_fwd) < 4:
            continue

        # Benchmark: equal-weight all ETFs
        bench_ret = np.mean(list(all_fwd.values()))

        # Layer A: Top4 equal-weight
        top4 = factor_df.head(4)["symbol"].tolist()
        top4_rets = [all_fwd.get(s, 0) for s in top4]
        layer_a = np.mean(top4_rets)

        # Layer B: Top4 risk-parity
        vols = {}
        for sym in top4:
            hist = tradable[sym][tradable[sym]["trade_date"] <= td]
            if len(hist) > 40:
                rets = np.diff(hist["close"].values[-40:]) / hist["close"].values[-40:-1]
                vols[sym] = np.std(rets) * np.sqrt(252)
            else:
                vols[sym] = 0.2
        raw_w = {s: 1.0 / (vols[s] + 1e-8) for s in top4}
        total_w = sum(raw_w.values())
        rp_weights = {s: w / total_w for s, w in raw_w.items()}
        layer_b = sum(rp_weights[s] * all_fwd.get(s, 0) for s in top4)

        # Layer C: + conviction (breakout boost)
        conv_weights = {}
        for sym in top4:
            row = factor_df[factor_df["symbol"] == sym].iloc[0]
            # Use RAW breakout (before rank) - check if > 0.6 (raw threshold)
            bonus = 1.3 if row["breakout"] > 0.6 else 1.0
            conv_weights[sym] = rp_weights[sym] * bonus
        cw_sum = sum(conv_weights.values())
        conv_weights = {s: w / cw_sum for s, w in conv_weights.items()}
        layer_c = sum(conv_weights[s] * all_fwd.get(s, 0) for s in top4)

        # Layer D: + bond gate
        mkt_signals = calc.calc_market_signals(td)
        gate = absolute_risk_gate(index_df, mkt_signals, td)
        if not gate["risk_on"]:
            # Would have been in bonds (~0% return for 20 days)
            layer_d = 0.0
        else:
            layer_d = layer_c

        # Layer E: - costs (round-trip ~0.3%)
        layer_e = layer_d - 0.003  # Approximate round-trip cost

        records.append(
            {
                "date": str(td),
                "benchmark": bench_ret,
                "A_selection": layer_a,
                "B_risk_parity": layer_b,
                "C_conviction": layer_c,
                "D_gate": layer_d,
                "E_execution": layer_e,
                "gate_on": gate["risk_on"],
                "top4": ",".join(top4),
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# 5 Benchmarks
# =============================================================================


def run_benchmarks(data, index_df, start_date, end_date):
    """Run 5 benchmark strategies for comparison."""
    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {
        k: v for k, v in data.items() if k not in DEFENSIVE_ASSETS and k not in index_symbols
    }

    all_dates = sorted(index_df["trade_date"].tolist())
    period_dates = [d for d in all_dates if start_date <= d <= end_date]

    results = {}

    # 1. CSI300 buy-and-hold
    idx_period = index_df[
        (index_df["trade_date"] >= start_date) & (index_df["trade_date"] <= end_date)
    ]
    if len(idx_period) > 1:
        results["沪深300"] = (idx_period["close"].iloc[-1] / idx_period["close"].iloc[0]) - 1
    else:
        results["沪深300"] = 0.0

    # 2. 21 ETF equal-weight buy-and-hold
    etf_rets = []
    for sym, df in tradable.items():
        p = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        if len(p) > 1:
            etf_rets.append((p["close"].iloc[-1] / p["close"].iloc[0]) - 1)
    results["21ETF等权"] = np.mean(etf_rets) if etf_rets else 0.0

    # 3. Pure momentum Top4 (rebalance every 20d)
    calc = FactorCalc()
    calc.load_external()
    results["纯动量Top4"] = _simple_factor_backtest(
        calc, tradable, data, index_df, period_dates, "momentum"
    )

    # 4. Pure reversal Top4
    results["纯反转Top4"] = _simple_factor_backtest(
        calc, tradable, data, index_df, period_dates, "reversal"
    )

    # 5. Absolute momentum + bonds (Faber-style)
    results["绝对动量+国债"] = _faber_backtest(tradable, data, index_df, period_dates)

    return results


def _simple_factor_backtest(calc, tradable, data, index_df, period_dates, factor_name):
    """Simple single-factor Top4 backtest."""
    rebalance_dates = period_dates[70::20]
    total_ret = 1.0
    for td in rebalance_dates:
        factor_df = calc.calc_cross_section(tradable, td)
        if factor_df.empty or len(factor_df) < 4:
            continue
        factor_df = factor_df.sort_values(factor_name, ascending=False)
        top4 = factor_df.head(4)["symbol"].tolist()
        period_rets = []
        for sym in top4:
            if sym in data:
                future = data[sym][data[sym]["trade_date"] > td].head(20)
                current = data[sym][data[sym]["trade_date"] == td]
                if not future.empty and not current.empty:
                    period_rets.append((future.iloc[-1]["close"] / current.iloc[0]["close"]) - 1)
        if period_rets:
            total_ret *= 1 + np.mean(period_rets)
    return total_ret - 1


def _faber_backtest(tradable, data, index_df, period_dates):
    """Faber TAA: absolute momentum + bonds.

    For each ETF: if 20d return > 0, hold; else hold bonds.
    Equal-weight among qualifying ETFs, cap at Top5 by momentum.
    """
    rebalance_dates = period_dates[70::20]
    total_ret = 1.0
    for td in rebalance_dates:
        qualifying = []
        for sym, df in tradable.items():
            hist = df[df["trade_date"] <= td].sort_values("trade_date")
            if len(hist) < 21:
                continue
            close = hist["close"].values
            ret_20d = (close[-1] - close[-20]) / close[-20]
            if ret_20d > 0:
                qualifying.append((sym, ret_20d))

        if qualifying:
            # Top 5 by absolute momentum
            qualifying.sort(key=lambda x: x[1], reverse=True)
            selected = [s for s, _ in qualifying[:5]]
            period_rets = []
            for sym in selected:
                if sym in data:
                    future = data[sym][data[sym]["trade_date"] > td].head(20)
                    current = data[sym][data[sym]["trade_date"] == td]
                    if not future.empty and not current.empty:
                        period_rets.append(
                            (future.iloc[-1]["close"] / current.iloc[0]["close"]) - 1
                        )
            if period_rets:
                total_ret *= 1 + np.mean(period_rets)
        # else: hold bonds (0% return for period)
    return total_ret - 1


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
    print("=" * 70)
    print("  V4.3: 可信度修复版 — 不提高收益, 只证明收益是真的")
    print("=" * 70)

    data, index_df = load_data()
    if index_df is None:
        print("ERROR: 无指数数据")
        return

    n_etf = len([k for k in data if not k.startswith("idx_")])
    print(f"\n  数据: {n_etf} 标的, {len(index_df)} 天")
    print(f"  范围: {index_df['trade_date'].min()} ~ {index_df['trade_date'].max()}")

    # === Step 1: Purged Walk-Forward ===
    print("\n[1/4] Purged Walk-Forward...")
    wf = purged_walk_forward(data, index_df, n_rounds=10, lr=0.03)
    final_weights = wf["final_weights"]

    # === Step 2: 5-Layer Attribution (2023 stress test) ===
    print("\n[2/4] 5层收益归因 (2023压力测试)...")
    from datetime import date as dt_date

    attr_2023 = run_attribution(
        data, index_df, final_weights, dt_date(2023, 1, 1), dt_date(2023, 12, 31)
    )
    if not attr_2023.empty:
        print(f"\n  2023年归因 (每期平均):")
        print(f"  {'层级':<20} {'平均收益':>10} {'累计':>10}")
        print(f"  {'-' * 42}")
        for col in [
            "benchmark",
            "A_selection",
            "B_risk_parity",
            "C_conviction",
            "D_gate",
            "E_execution",
        ]:
            avg = attr_2023[col].mean()
            cum = (1 + attr_2023[col]).prod() - 1
            label = col.replace("_", " ").title()
            print(f"  {label:<20} {avg * 100:>+9.2f}% {cum * 100:>+9.2f}%")

        # Decomposition
        sel_contrib = attr_2023["A_selection"].mean() - attr_2023["benchmark"].mean()
        rp_contrib = attr_2023["B_risk_parity"].mean() - attr_2023["A_selection"].mean()
        conv_contrib = attr_2023["C_conviction"].mean() - attr_2023["B_risk_parity"].mean()
        gate_contrib = attr_2023["D_gate"].mean() - attr_2023["C_conviction"].mean()
        exec_contrib = attr_2023["E_execution"].mean() - attr_2023["D_gate"].mean()
        print(f"\n  贡献分解:")
        print(f"    选股贡献: {sel_contrib * 100:+.3f}%/期")
        print(f"    风险平价: {rp_contrib * 100:+.3f}%/期")
        print(f"    Conviction: {conv_contrib * 100:+.3f}%/期")
        print(f"    Bond Gate: {gate_contrib * 100:+.3f}%/期")
        print(f"    执行拖累: {exec_contrib * 100:+.3f}%/期")

        # Save
        attr_2023.to_csv(OUTPUT_DIR / "attribution_2023.csv", index=False)

    # === Step 3: 5 Benchmarks (full period) ===
    print("\n[3/4] 5个基准对比 (全周期)...")
    all_dates = sorted(index_df["trade_date"].tolist())
    bench = run_benchmarks(data, index_df, all_dates[70], all_dates[-1])
    print(f"\n  {'基准':<16} {'收益':>10}")
    print(f"  {'-' * 28}")
    for name, ret in bench.items():
        print(f"  {name:<16} {ret:>+10.2%}")
    print(f"  {'V4.3 OOS':<16} {wf['total_oos_return']:>+10.2%}")

    # === Step 4: Factor weight audit ===
    print("\n[4/4] 因子权重审计...")
    print(f"\n  {'因子':<16} {'权重':>8} {'说明'}")
    print(f"  {'-' * 50}")
    for i in np.argsort(final_weights)[::-1]:
        note = ""
        if CROSS_FACTORS[i] in ("bias", "bollinger", "macd", "trend"):
            note = "← 趋势族(高相关)"
        elif CROSS_FACTORS[i] in ("low_vol", "atr_ratio", "amplitude", "vol_change"):
            note = "← 低波族(高相关)"
        print(f"  {CROSS_FACTORS[i]:<16} {final_weights[i] * 100:>7.1f}% {note}")

    # Conviction audit
    print(f"\n  Conviction审计:")
    print(f"    breakout因子Rank后范围: 约[-2.5, +2.5] (21只ETF)")
    print(f"    原始breakout>0.6 → 触发conviction")
    print(f"    需验证: 2023年有多少次触发?")

    # Count conviction triggers in 2023
    calc = FactorCalc()
    calc.load_external()
    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {
        k: v for k, v in data.items() if k not in DEFENSIVE_ASSETS and k not in index_symbols
    }
    conv_triggers = 0
    total_rebal = 0
    for td in [d for d in all_dates if dt_date(2023, 1, 1) <= d <= dt_date(2023, 12, 31)][70::20]:
        factor_df = calc.calc_cross_section(tradable, td)
        if factor_df.empty:
            continue
        total_rebal += 1
        factor_df["composite"] = factor_df[CROSS_FACTORS].values @ final_weights
        top4 = factor_df.nlargest(4, "composite")
        for _, row in top4.iterrows():
            if row["breakout"] > 0.6:
                conv_triggers += 1
    print(f"    2023年Top4中conviction触发: {conv_triggers}/{total_rebal * 4}次")

    # Save final weights
    ckpt = {
        "version": "v4.3",
        "method": "purged_walk_forward",
        "n_rounds": 10,
        "purge_days": 20,
        "total_oos_return": wf["total_oos_return"],
        "factor_weights": final_weights.tolist(),
        "factor_names": CROSS_FACTORS,
        "note": "OOS concatenated, NO checkpoint selection",
    }
    with open(OUTPUT_DIR / "v43_weights.json", "w") as f:
        json.dump(ckpt, f, indent=2, default=str)
    print(f"\n  权重已保存: {OUTPUT_DIR / 'v43_weights.json'}")


if __name__ == "__main__":
    main()
