"""v4.1: 16因子 + 可学习风控参数 + 负相关资产 + regime-conditional权重.

Improvements over v4 (based on 2023 post-mortem):
1. Learnable risk params: 降仓比例/防御配比/因子健康阈值 全部由AdamW学习
2. Regime-conditional factor weighting: 震荡市自动压制趋势类因子
3. Negative-correlation assets: 国债ETF(511010)作为DOWN regime配置
4. No hardcoded stop-loss: 降仓幅度是连续可学习参数,不是固定阈值

Learnable parameters (19 total):
- 16 factor weights (softmax → sum=1)
- 3 risk params (sigmoid → bounded):
  - stress_reduce: 回撤时降仓比例 [0, 0.8]
  - defensive_ratio: DOWN时防御资产配比 [0, 0.5]
  - health_threshold: 因子健康度阈值 [0.2, 0.8]
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "long_history"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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
    "northbound",
    "pe_percentile",
]
N_FACTORS = len(ALL_FACTORS)  # 16
N_RISK_PARAMS = 3
N_TOTAL = N_FACTORS + N_RISK_PARAMS  # 19

# Factors that fail in震荡市 (from 2023 post-mortem)
TREND_FOLLOWING_FACTORS = {"momentum", "macd", "bias", "bollinger", "rsi"}

# Negative-correlation assets
DEFENSIVE_ASSETS = {"511010": "国债ETF", "511880": "货币ETF"}


# =============================================================================
# Factor Calculator (same as v4)
# =============================================================================


class FactorCalculatorV4:
    """Calculate all 16 factors."""

    def __init__(self):
        self.northbound_data: pd.DataFrame | None = None
        self.pe_data: dict[str, pd.DataFrame] = {}

    def load_external_data(self):
        nb_file = DATA_DIR / "northbound.parquet"
        if nb_file.exists():
            self.northbound_data = pd.read_parquet(nb_file)
        pe_file = DATA_DIR / "pe_percentile.parquet"
        if pe_file.exists():
            pe_df = pd.read_parquet(pe_file)
            for code in pe_df["index_code"].unique():
                self.pe_data[code] = pe_df[pe_df["index_code"] == code].copy()

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
            volume = (
                hist["volume"].values.astype(float)
                if "volume" in hist.columns
                else np.ones(len(close))
            )

            factors = self._calc_single(close, high, low, volume)
            factors["symbol"] = symbol
            factors["northbound"] = self._get_northbound(as_of_date)
            factors["pe_percentile"] = self._get_pe_percentile(symbol, as_of_date)
            records.append(factors)

        if not records:
            return pd.DataFrame()

        factor_df = pd.DataFrame(records)
        for col in ALL_FACTORS:
            if col in factor_df.columns:
                vals = factor_df[col].values
                mask = ~np.isnan(vals)
                if mask.sum() > 2:
                    mean, std = vals[mask].mean(), vals[mask].std()
                    if std > 1e-8:
                        factor_df.loc[mask, col] = (vals[mask] - mean) / std
                    else:
                        factor_df.loc[mask, col] = 0.0
                factor_df[col] = factor_df[col].fillna(0.0)
        return factor_df

    def _calc_single(self, close, high, low, volume) -> dict[str, float]:
        n = len(close)
        f = {}
        # Momentum
        f["momentum"] = (close[-20] - close[-80]) / close[-80] if n > 80 else 0.0
        # Reversal
        f["reversal"] = -(close[-1] - close[-5]) / close[-5] if n > 5 else 0.0
        # Low vol
        if n > 20:
            rets = np.diff(close[-20:]) / close[-20:-1]
            f["low_vol"] = -(np.std(rets) * np.sqrt(252))
        else:
            f["low_vol"] = 0.0
        # Trend
        if n > 60:
            f["trend"] = (close[-20:].mean() - close[-60:].mean()) / close[-60:].mean()
        else:
            f["trend"] = 0.0
        # Volume trend
        if n > 60:
            v20, v60 = volume[-20:].mean(), volume[-60:].mean()
            f["volume_trend"] = (v20 - v60) / v60 if v60 > 0 else 0.0
        else:
            f["volume_trend"] = 0.0
        # BIAS
        if n > 20:
            ma20 = close[-20:].mean()
            f["bias"] = (close[-1] - ma20) / ma20
        else:
            f["bias"] = 0.0
        # RSI
        if n > 15:
            deltas = np.diff(close[-15:])
            gains = np.where(deltas > 0, deltas, 0).mean()
            losses = np.where(deltas < 0, -deltas, 0).mean()
            rs = gains / (losses + 1e-10)
            f["rsi"] = (100 - 100 / (1 + rs) - 50) / 50
        else:
            f["rsi"] = 0.0
        # MACD
        if n > 35:
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd_line = ema12 - ema26
            signal = self._ema(macd_line[-9:], 9)
            f["macd"] = (macd_line[-1] - signal[-1]) / close[-1] * 100
        else:
            f["macd"] = 0.0
        # ATR ratio
        if n > 15:
            tr = np.maximum(
                high[-14:] - low[-14:],
                np.maximum(np.abs(high[-14:] - close[-15:-1]), np.abs(low[-14:] - close[-15:-1])),
            )
            f["atr_ratio"] = -(tr.mean() / close[-1])
        else:
            f["atr_ratio"] = 0.0
        # OBV
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
        # Skewness
        if n > 21:
            rets = np.diff(close[-21:]) / close[-21:-1]
            f["skewness"] = float(stats.skew(rets))
        else:
            f["skewness"] = 0.0
        # Vol change
        if n > 11:
            vol_this = np.std(np.diff(close[-5:]) / close[-5:-1])
            vol_last = np.std(np.diff(close[-10:-5]) / close[-10:-6])
            f["vol_change"] = -(vol_this - vol_last) / (vol_last + 1e-8)
        else:
            f["vol_change"] = 0.0
        # Amplitude
        if n > 5:
            f["amplitude"] = -((high[-5:] - low[-5:]) / close[-5:]).mean()
        else:
            f["amplitude"] = 0.0
        # Bollinger
        if n > 20:
            ma20, std20 = close[-20:].mean(), close[-20:].std()
            f["bollinger"] = (close[-1] - ma20) / (2 * std20) if std20 > 1e-8 else 0.0
        else:
            f["bollinger"] = 0.0
        return f

    def _ema(self, data, period):
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
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
            "510300": "000300",
            "510500": "000905",
            "159915": "399006",
            "510050": "000016",
            "512100": "000852",
            "159901": "399330",
            "510880": "000015",
            "512800": "399986",
            "512880": "399975",
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


# =============================================================================
# Regime Detector (lightweight)
# =============================================================================


def detect_regime(index_df: pd.DataFrame, as_of_date: date) -> str:
    """Simple regime: UP / FLAT / DOWN."""
    hist = index_df[index_df["trade_date"] <= as_of_date]
    if len(hist) < 60:
        return "FLAT"
    close = hist["close"].values
    ma20 = close[-20:].mean()
    ma60 = close[-60:].mean()
    ret_20d = (close[-1] - close[-20]) / close[-20]
    if ma20 > ma60 * 1.01 and ret_20d > 0.02:
        return "UP"
    elif ma20 < ma60 * 0.99 and ret_20d < -0.02:
        return "DOWN"
    return "FLAT"


# =============================================================================
# AdamW Optimizer (extended: 16 factor weights + 3 risk params)
# =============================================================================


class AdamWExtended:
    """AdamW for factor weights + learnable risk parameters.

    Parameterization:
    - Factor weights: softmax(logits[:16]) → sum=1
    - Risk params: sigmoid(logits[16:19]) → bounded [0,1] then scaled
    """

    def __init__(self, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.n = N_TOTAL

        self.logits = np.zeros(self.n)
        self.m = np.zeros(self.n)
        self.v = np.zeros(self.n)
        self.t = 0

    def get_factor_weights(self) -> np.ndarray:
        """Softmax over first 16 logits."""
        fl = self.logits[:N_FACTORS]
        exp_l = np.exp(fl - fl.max())
        return exp_l / exp_l.sum()

    def get_risk_params(self) -> dict[str, float]:
        """Sigmoid over last 3 logits → bounded risk parameters."""
        rl = self.logits[N_FACTORS:]
        sig = 1.0 / (1.0 + np.exp(-rl))
        return {
            "stress_reduce": sig[0] * 0.8,  # [0, 0.8] 回撤降仓比例
            "defensive_ratio": sig[1] * 0.5,  # [0, 0.5] 防御资产配比
            "health_threshold": 0.2 + sig[2] * 0.6,  # [0.2, 0.8] 因子健康阈值
        }

    def step(self, gradient: np.ndarray):
        self.t += 1
        self.logits -= self.lr * self.weight_decay * self.logits
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        self.logits -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict:
        return {
            "logits": self.logits.tolist(),
            "m": self.m.tolist(),
            "v": self.v.tolist(),
            "t": self.t,
            "factor_weights": self.get_factor_weights().tolist(),
            "risk_params": self.get_risk_params(),
        }

    def load_state_dict(self, state: dict):
        self.logits = np.array(state["logits"])
        self.m = np.array(state["m"])
        self.v = np.array(state["v"])
        self.t = state["t"]


# =============================================================================
# Regime-Conditional Factor Weighting
# =============================================================================


def apply_regime_mask(weights: np.ndarray, regime: str) -> np.ndarray:
    """Downweight trend-following factors in FLAT/DOWN regime.

    Not hardcoded zero — just a 0.5x multiplier on trend factors.
    Then renormalize to sum=1.
    """
    w = weights.copy()
    if regime in ("FLAT", "DOWN"):
        for i, name in enumerate(ALL_FACTORS):
            if name in TREND_FOLLOWING_FACTORS:
                w[i] *= 0.5  # Suppress, not kill
    return w / w.sum()


# =============================================================================
# Backtest with Learnable Risk
# =============================================================================


def run_backtest_v41(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    factor_weights: np.ndarray,
    risk_params: dict[str, float],
    start_date: date | None = None,
    end_date: date | None = None,
    top_n: int = 4,
    rebalance_days: int = 20,
    initial_capital: float = 100_000.0,
) -> dict:
    """Backtest with learnable risk management.

    Risk logic (all learnable, no hardcoded thresholds):
    - Drawdown > 10% → reduce position by stress_reduce ratio
    - Factor health < health_threshold → reduce position
    - DOWN regime → allocate defensive_ratio to bond ETF
    """
    calculator = FactorCalculatorV4()
    calculator.load_external_data()

    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {
        k: v for k, v in data.items() if k not in DEFENSIVE_ASSETS and k not in index_symbols
    }

    all_dates = sorted(index_df["trade_date"].tolist())
    warmup = 70
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]
    trading_days = all_dates[warmup:] if len(all_dates) > warmup else all_dates

    stress_reduce = risk_params["stress_reduce"]
    defensive_ratio = risk_params["defensive_ratio"]
    health_threshold = risk_params["health_threshold"]

    cash = initial_capital
    holdings: dict[str, int] = {}
    equity_history = []
    days_since = rebalance_days
    n_trades = 0
    fee_rate = 0.001
    peak_equity = initial_capital

    for td in trading_days:
        # Current equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]

        peak_equity = max(peak_equity, equity)
        current_dd = (equity - peak_equity) / peak_equity  # Negative

        # --- Learnable risk overlay ---
        position_scale = 1.0

        # 1. Drawdown-based reduction (learnable ratio)
        if current_dd < -0.10:
            # Scale reduction by drawdown severity × learned ratio
            dd_severity = min(abs(current_dd) / 0.25, 1.0)  # Normalize: 25% DD = max
            position_scale *= 1.0 - stress_reduce * dd_severity

        # 2. Factor health check (learnable threshold)
        regime = detect_regime(index_df, td)

        days_since += 1
        if days_since >= rebalance_days:
            factor_df = calculator.calculate_all(tradable, td)
            if not factor_df.empty and len(factor_df) >= top_n:
                # Compute factor health: ratio of IC-positive factors
                factor_health = _estimate_factor_health(factor_df, data, td)

                if factor_health < health_threshold:
                    # Reduce position proportionally to how unhealthy
                    unhealthy_gap = (health_threshold - factor_health) / health_threshold
                    position_scale *= 1.0 - stress_reduce * unhealthy_gap * 0.5

                # Regime-conditional weights
                adjusted_weights = apply_regime_mask(factor_weights, regime)

                # Composite score
                factor_df["composite"] = factor_df[ALL_FACTORS].values @ adjusted_weights
                selected = factor_df.nlargest(top_n, "composite")["symbol"].tolist()

                # Risk parity weights
                vols = {}
                for sym in selected:
                    hist = tradable[sym][tradable[sym]["trade_date"] <= td]
                    if len(hist) > 40:
                        rets = np.diff(hist["close"].values[-40:]) / hist["close"].values[-40:-1]
                        vols[sym] = np.std(rets) * np.sqrt(252)
                    else:
                        vols[sym] = 0.2
                total_inv = sum(1.0 / (v + 1e-8) for v in vols.values())
                target_weights = {
                    s: min((1.0 / (vols[s] + 1e-8)) / total_inv, 0.4) for s in selected
                }
                tw_sum = sum(target_weights.values())
                if tw_sum > 0:
                    target_weights = {s: w / tw_sum for s, w in target_weights.items()}

                # Apply position_scale
                target_weights = {s: w * position_scale for s, w in target_weights.items()}

                # 3. Defensive allocation in DOWN regime (learnable ratio)
                defensive_target = 0.0
                if regime == "DOWN" and defensive_ratio > 0.01:
                    defensive_target = defensive_ratio * position_scale
                    # Scale down equity targets to make room
                    equity_budget = position_scale - defensive_target
                    if equity_budget > 0:
                        scale_factor = equity_budget / position_scale
                        target_weights = {s: w * scale_factor for s, w in target_weights.items()}

                # Execute trades
                # Sell non-target (including defensive if not needed)
                for sym in list(holdings.keys()):
                    if sym not in selected and sym not in DEFENSIVE_ASSETS:
                        row = data[sym][data[sym]["trade_date"] == td]
                        if not row.empty:
                            cash += holdings[sym] * row.iloc[0]["close"] * (1 - fee_rate / 2)
                            n_trades += 1
                            del holdings[sym]

                # Buy/sell defensive asset
                if defensive_target > 0.01:
                    def_sym = "511010"  # 国债ETF
                    if def_sym in data:
                        def_target_val = equity * defensive_target
                        row = data[def_sym][data[def_sym]["trade_date"] == td]
                        if not row.empty:
                            price = row.iloc[0]["close"]
                            cur = holdings.get(def_sym, 0)
                            diff = def_target_val - cur * price
                            if abs(diff) > price * 10:
                                delta = int(diff / price / 10) * 10
                                if delta > 0 and delta * price <= cash:
                                    cash -= delta * price * (1 + fee_rate / 2)
                                    holdings[def_sym] = cur + delta
                                    n_trades += 1
                                elif delta < 0:
                                    sell = min(-delta, cur)
                                    cash += sell * price * (1 - fee_rate / 2)
                                    holdings[def_sym] = cur - sell
                                    if holdings[def_sym] <= 0:
                                        del holdings[def_sym]
                                    n_trades += 1
                else:
                    # Sell defensive if not needed
                    for def_sym in DEFENSIVE_ASSETS:
                        if def_sym in holdings:
                            row = data[def_sym][data[def_sym]["trade_date"] == td]
                            if not row.empty:
                                cash += (
                                    holdings[def_sym] * row.iloc[0]["close"] * (1 - fee_rate / 2)
                                )
                                del holdings[def_sym]
                                n_trades += 1

                # Buy equity targets
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
                            cost = delta * price * (1 + fee_rate / 2)
                            if cost <= cash:
                                cash -= cost
                                holdings[sym] = cur + delta
                                n_trades += 1
                        elif delta < 0:
                            sell = min(-delta, cur)
                            cash += sell * price * (1 - fee_rate / 2)
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
        return {"total_return": 0.0, "equity_curve": pd.DataFrame()}

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
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "n_days": n_days,
        "equity_curve": eq_df,
    }


def _estimate_factor_health(factor_df: pd.DataFrame, data: dict, as_of_date: date) -> float:
    """Estimate factor health: fraction of factors with positive recent IC.

    Uses 5-day forward returns as quick proxy.
    """
    symbols = factor_df["symbol"].tolist()
    fwd = {}
    for sym in symbols:
        if sym not in data:
            continue
        hist = data[sym]
        future = hist[hist["trade_date"] > as_of_date].head(5)
        current = hist[hist["trade_date"] == as_of_date]
        if not future.empty and not current.empty:
            fwd[sym] = (future.iloc[-1]["close"] / current.iloc[0]["close"]) - 1

    if len(fwd) < 4:
        return 0.7  # Default neutral

    ret_vec = np.array([fwd.get(s, 0) for s in symbols])
    n_positive = 0
    for f in ALL_FACTORS:
        if f in factor_df.columns:
            ic = stats.spearmanr(factor_df[f].values, ret_vec)[0]
            if not np.isnan(ic) and ic > 0:
                n_positive += 1

    return n_positive / N_FACTORS


# =============================================================================
# Gradient: factor IC + risk param sensitivity
# =============================================================================


def compute_full_gradient(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    optimizer: AdamWExtended,
    sample_dates: list[date],
) -> np.ndarray:
    """Compute gradient for all 19 parameters.

    Factor weights: IC-based gradient (same as v4)
    Risk params: perturbation-based (does reducing more/less help?)
    """
    calculator = FactorCalculatorV4()
    calculator.load_external_data()

    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {
        k: v for k, v in data.items() if k not in DEFENSIVE_ASSETS and k not in index_symbols
    }

    factor_weights = optimizer.get_factor_weights()
    risk_params = optimizer.get_risk_params()

    # --- Factor weight gradient (IC-based) ---
    factor_grad = np.zeros(N_FACTORS)
    n_samples = 0

    for td in sample_dates:
        factor_df = calculator.calculate_all(tradable, td)
        if factor_df.empty or len(factor_df) < 4:
            continue

        forward_returns = {}
        for sym in factor_df["symbol"].tolist():
            if sym not in data:
                continue
            hist = data[sym]
            future = hist[hist["trade_date"] > td].head(20)
            current = hist[hist["trade_date"] == td]
            if len(future) >= 10 and not current.empty:
                forward_returns[sym] = (future.iloc[-1]["close"] / current.iloc[0]["close"]) - 1

        if len(forward_returns) < 4:
            continue

        symbols = factor_df["symbol"].tolist()
        ret_vec = np.array([forward_returns.get(s, 0) for s in symbols])
        if np.std(ret_vec) < 1e-10:
            continue

        factor_matrix = factor_df[ALL_FACTORS].values
        composite = factor_matrix @ factor_weights
        if np.std(composite) < 1e-10:
            continue
        base_ic = stats.spearmanr(composite, ret_vec)[0]
        if np.isnan(base_ic):
            continue

        eps = 0.01
        for i in range(N_FACTORS):
            pw = factor_weights.copy()
            pw[i] += eps
            pw = pw / pw.sum()
            pc = factor_matrix @ pw
            if np.std(pc) < 1e-10:
                continue
            pic = stats.spearmanr(pc, ret_vec)[0]
            if not np.isnan(pic):
                factor_grad[i] += -(pic - base_ic) / eps
        n_samples += 1

    if n_samples > 0:
        factor_grad /= n_samples

    # --- Risk param gradient (perturbation on short backtest) ---
    risk_grad = np.zeros(N_RISK_PARAMS)

    # Use last 2 sample dates for risk gradient (expensive)
    if len(sample_dates) >= 2:
        risk_sample_start = sample_dates[-2]
        risk_sample_end = sample_dates[-1]

        # Baseline: current risk params
        base_result = run_backtest_v41(
            data,
            index_df,
            factor_weights,
            risk_params,
            start_date=risk_sample_start,
            end_date=risk_sample_end,
            initial_capital=100_000,
        )
        base_ret = base_result["total_return"]

        # Perturb each risk param
        risk_keys = ["stress_reduce", "defensive_ratio", "health_threshold"]
        eps_risk = 0.05
        for i, key in enumerate(risk_keys):
            perturbed = risk_params.copy()
            perturbed[key] = min(max(perturbed[key] + eps_risk, 0.0), 1.0)
            p_result = run_backtest_v41(
                data,
                index_df,
                factor_weights,
                perturbed,
                start_date=risk_sample_start,
                end_date=risk_sample_end,
                initial_capital=100_000,
            )
            # Gradient: how does increasing this param affect return?
            # We want to MINIMIZE loss, so gradient = -(Δreturn / Δparam)
            risk_grad[i] = -(p_result["total_return"] - base_ret) / eps_risk

    # Combine: factor_grad (16) + risk_grad (3) = 19
    full_grad = np.concatenate([factor_grad, risk_grad])
    return full_grad


# =============================================================================
# 20-Round Flywheel Evolution
# =============================================================================


def run_flywheel_v41(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    n_rounds: int = 20,
    lr: float = 0.03,
) -> dict:
    optimizer = AdamWExtended(lr=lr)

    all_dates = sorted(index_df["trade_date"].tolist())
    n_total = len(all_dates)
    train_size = int(n_total * 0.6)
    test_size = int(n_total * 0.1)
    step_size = (n_total - train_size - test_size) // max(n_rounds - 1, 1)

    best_return = -999.0
    best_state = None
    best_round = 0
    history = []

    print(f"\n{'=' * 70}")
    print(f"  v4.1 飞轮进化: {n_rounds} 轮 | AdamW lr={lr} | 19 可学习参数")
    print(f"  (16因子权重 + 3风控参数: 降仓比例/防御配比/健康阈值)")
    print(f"  数据: {all_dates[0]} ~ {all_dates[-1]} ({n_total} 天)")
    print(f"{'=' * 70}")

    for round_i in range(n_rounds):
        train_end_idx = min(train_size + round_i * step_size, n_total - test_size)
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_size, n_total)

        train_dates = all_dates[:train_end_idx]
        test_dates = all_dates[test_start_idx:test_end_idx]
        if len(test_dates) < 20:
            break

        # Train: compute gradient
        sample_points = train_dates[70::20][-10:]
        gradient = compute_full_gradient(data, index_df, optimizer, sample_points)
        optimizer.step(gradient)

        # Test: evaluate
        weights = optimizer.get_factor_weights()
        risk = optimizer.get_risk_params()
        result = run_backtest_v41(
            data,
            index_df,
            weights,
            risk,
            start_date=test_dates[0],
            end_date=test_dates[-1],
        )
        test_return = result["total_return"]

        # Checkpoint
        if test_return > best_return:
            best_return = test_return
            best_round = round_i
            best_state = optimizer.state_dict()
            ckpt = {
                "version": "v4.1",
                "round": round_i,
                "factor_weights": weights.tolist(),
                "factor_names": ALL_FACTORS,
                "risk_params": risk,
                "test_return": test_return,
                "test_period": f"{test_dates[0]} ~ {test_dates[-1]}",
                "optimizer_state": optimizer.state_dict(),
            }
            with open(CHECKPOINT_DIR / "best_weights_v41.json", "w") as f:
                json.dump(ckpt, f, indent=2, default=str)

        history.append({"round": round_i, "test_return": test_return, "risk_params": risk})

        marker = " ★" if round_i == best_round else ""
        print(
            f"  R{round_i + 1:2d}/{n_rounds} | "
            f"Test: {test_return:+.2%} | "
            f"Best: {best_return:+.2%} (R{best_round + 1}) | "
            f"降仓={risk['stress_reduce']:.2f} 防御={risk['defensive_ratio']:.2f} "
            f"阈值={risk['health_threshold']:.2f}{marker}"
        )

    print(f"\n{'=' * 70}")
    print(f"  进化完成! 最优 Round {best_round + 1}, 收益: {best_return:+.2%}")
    fw = optimizer.get_factor_weights()
    rp = optimizer.get_risk_params()
    print(f"\n  因子权重 Top5:")
    for i in np.argsort(fw)[-5:][::-1]:
        print(f"    {ALL_FACTORS[i]:15s}: {fw[i] * 100:.1f}%")
    print(f"\n  学习到的风控参数:")
    print(
        f"    降仓比例 (stress_reduce):    {rp['stress_reduce']:.3f}  (回撤时最多降{rp['stress_reduce'] * 100:.0f}%)"
    )
    print(
        f"    防御配比 (defensive_ratio):  {rp['defensive_ratio']:.3f}  (DOWN时{rp['defensive_ratio'] * 100:.0f}%配债券)"
    )
    print(
        f"    健康阈值 (health_threshold): {rp['health_threshold']:.3f}  (因子健康<{rp['health_threshold']:.0%}时降仓)"
    )
    print(f"{'=' * 70}")

    return {
        "best_state": best_state,
        "best_return": best_return,
        "best_round": best_round,
        "history": history,
        "optimizer": optimizer,
    }


# =============================================================================
# Main
# =============================================================================


def load_long_data():
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

    index_df = data.get("idx_000300")
    if index_df is None:
        for k, v in data.items():
            if "000300" in k:
                index_df = v
                break
    return data, index_df


def main():
    print("=" * 70)
    print("  v4.1: 16因子 + 可学习风控 + 负相关资产 + 20轮飞轮")
    print("=" * 70)

    print("\n[1/3] 加载数据...")
    data, index_df = load_long_data()
    if index_df is None:
        print("ERROR: 无指数数据")
        return
    n_sym = len([k for k in data if not k.startswith("idx_")])
    print(
        f"  {n_sym} 标的, {len(index_df)} 天 ({index_df['trade_date'].min()} ~ {index_df['trade_date'].max()})"
    )

    print("\n[2/3] 飞轮进化 (20轮)...")
    evo = run_flywheel_v41(data, index_df, n_rounds=20, lr=0.03)

    print("\n[3/3] 全周期回测 (最优参数, 本金10万)...")
    weights = evo["optimizer"].get_factor_weights()
    risk = evo["optimizer"].get_risk_params()

    # Use best checkpoint state
    if evo["best_state"]:
        evo["optimizer"].load_state_dict(evo["best_state"])
        weights = evo["optimizer"].get_factor_weights()
        risk = evo["optimizer"].get_risk_params()

    result = run_backtest_v41(data, index_df, weights, risk, initial_capital=100_000)

    # Yearly breakdown
    eq = result["equity_curve"]
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])
    eq["year"] = eq["trade_date"].dt.year

    print(f"\n  年度收益 (本金10万):")
    print(f"  {'年份':<6} {'年初':>10} {'年末':>10} {'收益':>8} {'回撤':>8}")
    print(f"  {'-' * 46}")
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
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1

    print(f"  {'-' * 46}")
    print(f"\n  最终: 10万 → {final:,.0f} ({total_ret:+.1%}, {final / 100_000:.2f}x)")
    print(
        f"  年化: {ann_ret:+.1%} | 夏普: {result['sharpe']:.2f} | 最大回撤: {result['max_drawdown']:.1%}"
    )

    # Benchmark
    idx_c = index_df["close"].values
    bench = (idx_c[-1] - idx_c[0]) / idx_c[0]
    print(f"  沪深300: {bench:+.1%} | 超额: {total_ret - bench:+.1%}")


if __name__ == "__main__":
    main()
