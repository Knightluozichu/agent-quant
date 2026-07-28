"""Full factor engine v4: 16 factors + AdamW optimizer + 20-round flywheel.

Factor pool:
- Original 5: momentum, reversal, low_vol, trend, volume_trend
- C-batch 9 (zero-cost): bias, rsi, macd, atr_ratio, obv, skewness, vol_change, amplitude, bollinger
- External 2: northbound_flow, pe_percentile

Optimizer: AdamW (lr=0.03, weight_decay=0.01)
Evolution: 20 rounds flywheel, checkpoint best weights
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "long_history"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Factor Names
# =============================================================================

ALL_FACTORS = [
    # Original 5
    "momentum", "reversal", "low_vol", "trend", "volume_trend",
    # C-batch 9 (zero-cost, from OHLCV)
    "bias", "rsi", "macd", "atr_ratio", "obv",
    "skewness", "vol_change", "amplitude", "bollinger",
    # External 2
    "northbound", "pe_percentile",
]

N_FACTORS = len(ALL_FACTORS)  # 16


# =============================================================================
# Factor Calculator (all 16 factors)
# =============================================================================

class FactorCalculatorV4:
    """Calculate all 16 factors from OHLCV + external data."""

    def __init__(self):
        self.northbound_data: pd.DataFrame | None = None
        self.pe_data: dict[str, pd.DataFrame] = {}  # index_code -> PE df

    def load_external_data(self):
        """Load northbound and PE data if available."""
        nb_file = DATA_DIR / "northbound.parquet"
        if nb_file.exists():
            self.northbound_data = pd.read_parquet(nb_file)
            print(f"  [外部] 北向资金: {len(self.northbound_data)} 条")

        pe_file = DATA_DIR / "pe_percentile.parquet"
        if pe_file.exists():
            pe_df = pd.read_parquet(pe_file)
            for code in pe_df["index_code"].unique():
                self.pe_data[code] = pe_df[pe_df["index_code"] == code].copy()
            print(f"  [外部] PE百分位: {len(pe_df)} 条, {len(self.pe_data)} 个指数")

    def calculate_all(
        self,
        data: dict[str, pd.DataFrame],
        as_of_date: date,
    ) -> pd.DataFrame:
        """Calculate all 16 factors for all symbols."""
        records = []
        for symbol, df in data.items():
            hist = df[df["trade_date"] <= as_of_date].copy()
            if len(hist) < 70:  # Need at least 70 days
                continue
            hist = hist.sort_values("trade_date")
            close = hist["close"].values.astype(float)
            high = hist["high"].values.astype(float)
            low = hist["low"].values.astype(float)
            volume = hist["volume"].values.astype(float) if "volume" in hist.columns else np.ones(len(close))
            amount = hist["amount"].values.astype(float) if "amount" in hist.columns else volume * close

            factors = self._calc_single(close, high, low, volume, amount)
            factors["symbol"] = symbol

            # External factors
            factors["northbound"] = self._get_northbound(as_of_date)
            factors["pe_percentile"] = self._get_pe_percentile(symbol, as_of_date)

            records.append(factors)

        if not records:
            return pd.DataFrame()

        factor_df = pd.DataFrame(records)

        # Z-score normalization for each factor (cross-sectional)
        for col in ALL_FACTORS:
            if col in factor_df.columns:
                vals = factor_df[col].values
                mask = ~np.isnan(vals)
                if mask.sum() > 2:
                    mean = vals[mask].mean()
                    std = vals[mask].std()
                    if std > 1e-8:
                        factor_df.loc[mask, col] = (vals[mask] - mean) / std
                    else:
                        factor_df.loc[mask, col] = 0.0
                factor_df[col] = factor_df[col].fillna(0.0)

        return factor_df

    def _calc_single(self, close, high, low, volume, amount) -> dict[str, float]:
        """Calculate 14 price/volume factors for a single symbol."""
        n = len(close)
        f = {}

        # --- Original 5 ---
        # 1. Momentum (60d, skip 20d)
        if n > 80:
            f["momentum"] = (close[-20] - close[-80]) / close[-80]
        else:
            f["momentum"] = 0.0

        # 2. Reversal (5d)
        f["reversal"] = -(close[-1] - close[-5]) / close[-5] if n > 5 else 0.0

        # 3. Low volatility (20d realized vol, negated so low vol = high score)
        if n > 20:
            rets = np.diff(close[-20:]) / close[-20:-1]
            f["low_vol"] = -(np.std(rets) * np.sqrt(252))
        else:
            f["low_vol"] = 0.0

        # 4. Trend (MA20 vs MA60)
        if n > 60:
            ma20 = close[-20:].mean()
            ma60 = close[-60:].mean()
            f["trend"] = (ma20 - ma60) / ma60
        else:
            f["trend"] = 0.0

        # 5. Volume trend (20d vs 60d)
        if n > 60:
            vol20 = volume[-20:].mean()
            vol60 = volume[-60:].mean()
            f["volume_trend"] = (vol20 - vol60) / vol60 if vol60 > 0 else 0.0
        else:
            f["volume_trend"] = 0.0

        # --- C-batch 9 (zero-cost) ---
        # 6. BIAS (20d)
        if n > 20:
            ma20 = close[-20:].mean()
            f["bias"] = (close[-1] - ma20) / ma20
        else:
            f["bias"] = 0.0

        # 7. RSI (14d)
        if n > 15:
            deltas = np.diff(close[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            if avg_loss > 1e-10:
                rs = avg_gain / avg_loss
                f["rsi"] = 100 - 100 / (1 + rs)
            else:
                f["rsi"] = 100.0
            # Normalize: RSI 50 = neutral, >50 bullish
            f["rsi"] = (f["rsi"] - 50) / 50  # Range [-1, 1]
        else:
            f["rsi"] = 0.0

        # 8. MACD signal
        if n > 35:
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd_line = ema12 - ema26
            signal_line = self._ema(macd_line[-9:], 9) if len(macd_line) >= 9 else macd_line[-1:]
            # MACD histogram (positive = bullish)
            f["macd"] = macd_line[-1] - signal_line[-1]
            # Normalize by price
            f["macd"] = f["macd"] / close[-1] * 100
        else:
            f["macd"] = 0.0

        # 9. ATR ratio (14d ATR / close)
        if n > 15:
            tr = np.maximum(
                high[-14:] - low[-14:],
                np.maximum(
                    np.abs(high[-14:] - close[-15:-1]),
                    np.abs(low[-14:] - close[-15:-1])
                )
            )
            atr = tr.mean()
            f["atr_ratio"] = -(atr / close[-1])  # Negate: lower ATR = better
        else:
            f["atr_ratio"] = 0.0

        # 10. OBV trend (20d slope)
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
            # OBV slope (normalized)
            if volume[-20:].mean() > 0:
                f["obv"] = (obv[-1] - obv[0]) / (volume[-20:].mean() * 20)
            else:
                f["obv"] = 0.0
        else:
            f["obv"] = 0.0

        # 11. Skewness (20d returns)
        if n > 21:
            rets = np.diff(close[-21:]) / close[-21:-1]
            f["skewness"] = float(stats.skew(rets))
        else:
            f["skewness"] = 0.0

        # 12. Volatility change (this week vs last week)
        if n > 11:
            rets_this = np.diff(close[-5:]) / close[-5:-1]
            rets_last = np.diff(close[-10:-5]) / close[-10:-6]
            vol_this = np.std(rets_this)
            vol_last = np.std(rets_last)
            # Decreasing vol = good (negative change = positive signal)
            f["vol_change"] = -(vol_this - vol_last) / (vol_last + 1e-8)
        else:
            f["vol_change"] = 0.0

        # 13. Amplitude (5d average)
        if n > 5:
            amp = ((high[-5:] - low[-5:]) / close[-5:]).mean()
            f["amplitude"] = -amp  # Lower amplitude = more stable = better
        else:
            f["amplitude"] = 0.0

        # 14. Bollinger position (20d)
        if n > 20:
            ma20 = close[-20:].mean()
            std20 = close[-20:].std()
            if std20 > 1e-8:
                # Position within bands: -1 (lower) to +1 (upper)
                f["bollinger"] = (close[-1] - ma20) / (2 * std20)
            else:
                f["bollinger"] = 0.0
        else:
            f["bollinger"] = 0.0

        return f

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _get_northbound(self, as_of_date: date) -> float:
        """Get northbound flow signal (market-level, same for all symbols)."""
        if self.northbound_data is None or self.northbound_data.empty:
            return 0.0
        nb = self.northbound_data[self.northbound_data["trade_date"] <= as_of_date]
        if len(nb) < 5:
            return 0.0
        # 5-day cumulative net flow (normalized)
        recent = nb.tail(5)
        if "net_flow" in recent.columns:
            flow_5d = recent["net_flow"].sum()
            # Normalize by 60-day std
            if len(nb) > 60:
                std_60 = nb["net_flow"].tail(60).std()
                return flow_5d / (std_60 + 1e-8)
            return flow_5d / 1e8  # Rough normalization
        return 0.0

    def _get_pe_percentile(self, symbol: str, as_of_date: date) -> float:
        """Get PE percentile for the symbol's tracking index."""
        # Map ETF to its tracking index
        etf_to_index = {
            "510300": "000300", "510500": "000905", "159915": "399006",
            "510050": "000016", "512100": "000852", "159901": "399330",
            "510880": "000015", "512800": "399986", "512880": "399975",
        }
        index_code = etf_to_index.get(symbol)
        if index_code is None or index_code not in self.pe_data:
            return 0.0  # No PE data for sector ETFs, neutral

        pe_df = self.pe_data[index_code]
        pe_hist = pe_df[pe_df["trade_date"] <= as_of_date]
        if len(pe_hist) < 60:
            return 0.0

        current_pe = pe_hist.iloc[-1]["pe"]
        # Percentile over available history (lower PE = higher score = value)
        percentile = (pe_hist["pe"] < current_pe).mean()
        # Negate: low PE percentile (cheap) = positive signal
        return -(percentile - 0.5) * 2  # Range [-1, 1], cheap = positive


# =============================================================================
# AdamW Optimizer for Factor Weights
# =============================================================================

class AdamWOptimizer:
    """AdamW optimizer for factor weight optimization.

    Optimizes weights to maximize IC-weighted returns.
    Constraint: weights sum to 1 (softmax parameterization).
    """

    def __init__(
        self,
        n_factors: int = N_FACTORS,
        lr: float = 0.03,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.n = n_factors
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # Raw logits (softmax → weights)
        self.logits = np.zeros(n_factors)
        self.m = np.zeros(n_factors)  # First moment
        self.v = np.zeros(n_factors)  # Second moment
        self.t = 0  # Timestep

    def get_weights(self) -> np.ndarray:
        """Get current factor weights (softmax of logits, sum=1)."""
        # Softmax with temperature
        exp_logits = np.exp(self.logits - self.logits.max())
        weights = exp_logits / exp_logits.sum()
        return weights

    def step(self, gradient: np.ndarray):
        """One AdamW update step.

        Args:
            gradient: d(loss)/d(logits), shape (n_factors,)
                     Negative gradient = direction to improve.
        """
        self.t += 1

        # AdamW: decoupled weight decay
        self.logits = self.logits - self.lr * self.weight_decay * self.logits

        # Update biased first moment
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        # Update biased second moment
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update logits (minimize loss = follow negative gradient)
        self.logits = self.logits - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict:
        """Save optimizer state."""
        return {
            "logits": self.logits.tolist(),
            "m": self.m.tolist(),
            "v": self.v.tolist(),
            "t": self.t,
            "weights": self.get_weights().tolist(),
        }

    def load_state_dict(self, state: dict):
        """Load optimizer state."""
        self.logits = np.array(state["logits"])
        self.m = np.array(state["m"])
        self.v = np.array(state["v"])
        self.t = state["t"]


# =============================================================================
# IC-based Gradient Computation
# =============================================================================

def compute_ic_gradient(
    factor_df: pd.DataFrame,
    forward_returns: dict[str, float],
    weights: np.ndarray,
) -> np.ndarray:
    """Compute gradient of IC loss w.r.t. factor logits.

    Loss = -IC (we want to maximize IC).
    Gradient approximated by: for each factor, how much does increasing
    its weight improve the composite IC?

    Args:
        factor_df: DataFrame with factor columns and 'symbol'
        forward_returns: symbol -> forward return (next period)
        weights: current factor weights

    Returns:
        gradient array shape (n_factors,)
    """
    n = len(ALL_FACTORS)
    symbols = factor_df["symbol"].tolist()

    # Build return vector
    ret_vec = np.array([forward_returns.get(s, 0.0) for s in symbols])
    if np.std(ret_vec) < 1e-10:
        return np.zeros(n)

    # Current composite score
    factor_matrix = factor_df[ALL_FACTORS].values  # (n_symbols, n_factors)

    # IC of composite
    composite = factor_matrix @ weights
    if np.std(composite) < 1e-10:
        return np.zeros(n)

    base_ic = stats.spearmanr(composite, ret_vec)[0]
    if np.isnan(base_ic):
        return np.zeros(n)

    # Numerical gradient: perturb each factor weight
    gradient = np.zeros(n)
    eps = 0.01
    for i in range(n):
        perturbed_weights = weights.copy()
        perturbed_weights[i] += eps
        perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize

        perturbed_composite = factor_matrix @ perturbed_weights
        if np.std(perturbed_composite) < 1e-10:
            gradient[i] = 0.0
            continue
        perturbed_ic = stats.spearmanr(perturbed_composite, ret_vec)[0]
        if np.isnan(perturbed_ic):
            gradient[i] = 0.0
            continue
        # Gradient of loss (-IC) w.r.t. logit_i
        gradient[i] = -(perturbed_ic - base_ic) / eps

    return gradient


# =============================================================================
# Backtest Engine
# =============================================================================

def run_backtest_with_weights(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    weights: np.ndarray,
    start_date: date | None = None,
    end_date: date | None = None,
    top_n: int = 4,
    rebalance_days: int = 20,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """Run backtest with given factor weights.

    Returns dict with equity curve, total_return, etc.
    """
    calculator = FactorCalculatorV4()
    calculator.load_external_data()

    # Tradable universe (exclude indices and defensive)
    defensive = {"511010", "511880"}
    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {k: v for k, v in data.items()
                if k not in defensive and k not in index_symbols}

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
    days_since_rebalance = rebalance_days  # Force first rebalance
    n_trades = 0
    fee_rate = 0.001  # 万2.5 * 2 (buy+sell) + slippage

    for td in trading_days:
        # Current equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]

        days_since_rebalance += 1

        # Rebalance check
        if days_since_rebalance >= rebalance_days:
            factor_df = calculator.calculate_all(tradable, td)
            if not factor_df.empty and len(factor_df) >= top_n:
                # Composite score with optimized weights
                factor_df["composite"] = factor_df[ALL_FACTORS].values @ weights

                # Select top N
                selected = factor_df.nlargest(top_n, "composite")["symbol"].tolist()

                # Risk parity weights (inverse vol)
                target_weights = {}
                vols = {}
                for sym in selected:
                    hist = tradable[sym][tradable[sym]["trade_date"] <= td]
                    if len(hist) > 40:
                        rets = np.diff(hist["close"].values[-40:]) / hist["close"].values[-40:-1]
                        vols[sym] = np.std(rets) * np.sqrt(252)
                    else:
                        vols[sym] = 0.2

                total_inv_vol = sum(1.0 / (v + 1e-8) for v in vols.values())
                for sym in selected:
                    w = (1.0 / (vols[sym] + 1e-8)) / total_inv_vol
                    target_weights[sym] = min(w, 0.4)  # Cap at 40%

                # Normalize
                tw_sum = sum(target_weights.values())
                if tw_sum > 0:
                    target_weights = {s: w / tw_sum for s, w in target_weights.items()}

                # Execute trades
                # Sell non-target
                for sym in list(holdings.keys()):
                    if sym not in selected:
                        row = data[sym][data[sym]["trade_date"] == td]
                        if not row.empty:
                            cash += holdings[sym] * row.iloc[0]["close"] * (1 - fee_rate / 2)
                            n_trades += 1
                            del holdings[sym]

                # Buy target
                for sym, target_w in target_weights.items():
                    target_value = equity * target_w
                    row = data[sym][data[sym]["trade_date"] == td]
                    if row.empty:
                        continue
                    price = row.iloc[0]["close"]
                    current_shares = holdings.get(sym, 0)
                    current_value = current_shares * price
                    diff_value = target_value - current_value

                    if abs(diff_value) > price * 100:  # Min trade: 100 shares
                        shares_delta = int(diff_value / price / 100) * 100
                        if shares_delta > 0:
                            cost = shares_delta * price * (1 + fee_rate / 2)
                            if cost <= cash:
                                cash -= cost
                                holdings[sym] = current_shares + shares_delta
                                n_trades += 1
                        elif shares_delta < 0:
                            sell_shares = min(-shares_delta, current_shares)
                            cash += sell_shares * price * (1 - fee_rate / 2)
                            holdings[sym] = current_shares - sell_shares
                            if holdings[sym] <= 0:
                                del holdings[sym]
                            n_trades += 1

                days_since_rebalance = 0

        # Record equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity})

    if not equity_history:
        return {"total_return": 0.0, "equity_curve": pd.DataFrame()}

    eq_df = pd.DataFrame(equity_history)
    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1

    # Annualized metrics
    n_days = len(eq_df)
    ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    cummax = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - cummax) / cummax
    max_dd = drawdown.min()

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "n_days": n_days,
        "equity_curve": eq_df,
    }


# =============================================================================
# 20-Round Flywheel Evolution
# =============================================================================

def run_flywheel_evolution(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    n_rounds: int = 20,
    lr: float = 0.03,
) -> dict:
    """Run 20-round flywheel evolution with AdamW.

    Each round:
    1. Split data into train/test
    2. Compute IC gradients on train
    3. Update weights with AdamW
    4. Evaluate on test
    5. Checkpoint if best
    """
    optimizer = AdamWOptimizer(n_factors=N_FACTORS, lr=lr)
    calculator = FactorCalculatorV4()
    calculator.load_external_data()

    all_dates = sorted(index_df["trade_date"].tolist())
    n_total = len(all_dates)

    # Walk-forward splits for evolution
    train_size = int(n_total * 0.6)
    test_size = int(n_total * 0.1)
    step_size = (n_total - train_size - test_size) // max(n_rounds - 1, 1)

    best_return = -999.0
    best_weights = optimizer.get_weights().copy()
    best_round = 0
    history = []

    print(f"\n{'='*70}")
    print(f"  飞轮进化: {n_rounds} 轮 | AdamW lr={lr} | {N_FACTORS} 因子")
    print(f"  数据: {all_dates[0]} ~ {all_dates[-1]} ({n_total} 天)")
    print(f"{'='*70}")

    for round_i in range(n_rounds):
        # Walk-forward window
        train_end_idx = min(train_size + round_i * step_size, n_total - test_size)
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_size, n_total)

        train_dates = all_dates[:train_end_idx]
        test_dates = all_dates[test_start_idx:test_end_idx]

        if len(test_dates) < 20:
            break

        # --- Train phase: compute IC gradients ---
        # Sample rebalance points in training period
        train_rebalance_points = train_dates[70::20]  # Every 20 days after warmup
        # Use last 10 rebalance points for gradient
        sample_points = train_rebalance_points[-10:] if len(train_rebalance_points) > 10 else train_rebalance_points

        accumulated_gradient = np.zeros(N_FACTORS)
        n_samples = 0

        defensive = {"511010", "511880"}
        index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
        tradable = {k: v for k, v in data.items()
                    if k not in defensive and k not in index_symbols}

        for td in sample_points:
            factor_df = calculator.calculate_all(tradable, td)
            if factor_df.empty or len(factor_df) < 4:
                continue

            # Compute forward returns (20-day)
            forward_returns = {}
            for sym in factor_df["symbol"].tolist():
                if sym not in data:
                    continue
                hist = data[sym]
                future = hist[hist["trade_date"] > td].head(20)
                if len(future) >= 10:
                    current_row = hist[hist["trade_date"] == td]
                    if not current_row.empty:
                        p0 = current_row.iloc[0]["close"]
                        p1 = future.iloc[-1]["close"]
                        forward_returns[sym] = (p1 - p0) / p0

            if len(forward_returns) < 4:
                continue

            weights = optimizer.get_weights()
            grad = compute_ic_gradient(factor_df, forward_returns, weights)
            accumulated_gradient += grad
            n_samples += 1

        # Average gradient and update
        if n_samples > 0:
            avg_gradient = accumulated_gradient / n_samples
            optimizer.step(avg_gradient)

        # --- Test phase: evaluate ---
        weights = optimizer.get_weights()
        test_start = test_dates[0]
        test_end = test_dates[-1]

        result = run_backtest_with_weights(
            data, index_df, weights,
            start_date=test_start, end_date=test_end,
        )

        test_return = result["total_return"]

        # Checkpoint best
        if test_return > best_return:
            best_return = test_return
            best_weights = weights.copy()
            best_round = round_i
            # Save checkpoint
            checkpoint = {
                "round": round_i,
                "weights": weights.tolist(),
                "factor_names": ALL_FACTORS,
                "test_return": test_return,
                "test_period": f"{test_start} ~ {test_end}",
                "optimizer_state": optimizer.state_dict(),
            }
            ckpt_path = CHECKPOINT_DIR / "best_weights_v4.json"
            with open(ckpt_path, "w") as f:
                json.dump(checkpoint, f, indent=2, default=str)

        history.append({
            "round": round_i,
            "test_return": test_return,
            "test_period": f"{test_start} ~ {test_end}",
            "weights": weights.tolist(),
        })

        # Print progress
        top3_idx = np.argsort(weights)[-3:][::-1]
        top3_str = ", ".join(f"{ALL_FACTORS[i]}={weights[i]:.3f}" for i in top3_idx)
        marker = " ★" if round_i == best_round else ""
        print(f"  Round {round_i+1:2d}/{n_rounds} | "
              f"Test: {test_return:+.2%} | "
              f"Best: {best_return:+.2%} (R{best_round+1}) | "
              f"Top3: {top3_str}{marker}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  进化完成! 最优: Round {best_round+1}, 收益: {best_return:+.2%}")
    print(f"  最优权重:")
    for i, (name, w) in enumerate(zip(ALL_FACTORS, best_weights)):
        bar = "█" * int(w * 50)
        print(f"    {name:15s}: {w:.4f} ({w*100:.1f}%) {bar}")
    print(f"  Checkpoint: {CHECKPOINT_DIR / 'best_weights_v4.json'}")
    print(f"{'='*70}")

    return {
        "best_weights": best_weights,
        "best_return": best_return,
        "best_round": best_round,
        "history": history,
        "optimizer": optimizer,
    }


# =============================================================================
# Main
# =============================================================================

def load_long_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load long history data."""
    data = {}
    for f in DATA_DIR.glob("*.parquet"):
        if f.name in ("combined_long.parquet", "northbound.parquet", "pe_percentile.parquet"):
            continue
        df = pd.read_parquet(f)
        if "symbol" not in df.columns or "trade_date" not in df.columns:
            continue
        symbol = df["symbol"].iloc[0]
        if f.name.startswith("index_"):
            symbol = f"idx_{symbol}"
            df["symbol"] = symbol
        data[symbol] = df.sort_values("trade_date").reset_index(drop=True)

    # Index for regime/benchmark
    index_key = "idx_000300" if "idx_000300" in data else "000300"
    index_df = data.get(index_key)
    if index_df is None:
        # Try without prefix
        for k, v in data.items():
            if "000300" in k:
                index_df = v
                break

    return data, index_df


def main():
    print("=" * 70)
    print("  多因子策略 v4: 16因子 + AdamW + 20轮飞轮进化")
    print("=" * 70)

    # Load data
    print("\n[1/4] 加载数据...")
    data, index_df = load_long_data()
    if index_df is None:
        print("ERROR: 无法加载指数数据")
        return
    n_symbols = len([k for k in data if not k.startswith("idx_")])
    print(f"  {n_symbols} 个标的, {len(index_df)} 个交易日")
    print(f"  时间: {index_df['trade_date'].min()} ~ {index_df['trade_date'].max()}")

    # Run flywheel evolution
    print("\n[2/4] 飞轮进化 (20轮 AdamW)...")
    evo_result = run_flywheel_evolution(
        data, index_df,
        n_rounds=20,
        lr=0.03,
    )

    # Full-period backtest with best weights
    print("\n[3/4] 全周期回测 (最优权重)...")
    best_weights = evo_result["best_weights"]
    full_result = run_backtest_with_weights(data, index_df, best_weights)

    print(f"\n  全周期回测结果:")
    print(f"    总收益:   {full_result['total_return']:+.2%}")
    print(f"    年化收益: {full_result['ann_return']:+.2%}")
    print(f"    年化波动: {full_result['ann_vol']:.2%}")
    print(f"    夏普比率: {full_result['sharpe']:.2f}")
    print(f"    最大回撤: {full_result['max_drawdown']:.2%}")
    print(f"    交易次数: {full_result['n_trades']}")

    # Benchmark
    print("\n[4/4] 基准对比...")
    idx_close = index_df["close"].values
    bench_return = (idx_close[-1] - idx_close[0]) / idx_close[0]
    print(f"    沪深300:  {bench_return:+.2%}")
    print(f"    超额收益: {full_result['total_return'] - bench_return:+.2%}")

    # Save final report
    report = {
        "strategy": "v4_16factor_adamw",
        "n_factors": N_FACTORS,
        "factor_names": ALL_FACTORS,
        "best_weights": best_weights.tolist(),
        "evolution_rounds": 20,
        "lr": 0.03,
        "optimizer": "AdamW",
        "full_period": {
            "total_return": full_result["total_return"],
            "ann_return": full_result["ann_return"],
            "ann_vol": full_result["ann_vol"],
            "sharpe": full_result["sharpe"],
            "max_drawdown": full_result["max_drawdown"],
        },
        "benchmark_return": bench_return,
        "excess_return": full_result["total_return"] - bench_return,
        "best_evolution_round": evo_result["best_round"],
        "best_evolution_return": evo_result["best_return"],
    }
    report_path = CHECKPOINT_DIR / "v4_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  报告已保存: {report_path}")


if __name__ == "__main__":
    main()
