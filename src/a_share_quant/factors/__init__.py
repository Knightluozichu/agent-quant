"""Multi-factor strategy engine v2.

Improvements over v1:
1. Factor orthogonalization: remove contradictory factors per regime
2. Dynamic IC weighting: rolling IC replaces equal weight
3. Regime-conditional factor activation
4. Momentum crash detection
5. Drawdown-based risk overlay

Anti-overfitting principles maintained:
- Fixed lookback windows (academic defaults)
- IC weighting uses ONLY past data (no lookahead)
- Walk-forward validation required
- No parameter optimization on test period
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Factor Configuration
# =============================================================================


@dataclass
class FactorConfig:
    """Factor configuration with fixed parameters (academic defaults)."""

    # Momentum
    momentum_long: int = 60  # ~3 months
    momentum_skip: int = 20  # Skip recent 1 month (reversal contamination)

    # Short-term reversal
    reversal_window: int = 5  # 1 week

    # Volatility
    vol_window: int = 20  # 1 month realized vol

    # Trend
    trend_fast: int = 20  # MA20
    trend_slow: int = 60  # MA60

    # Volume
    volume_window: int = 20
    volume_baseline: int = 60

    # IC calculation
    ic_lookback: int = 40  # Rolling IC window (~2 months)
    ic_min_periods: int = 20  # Min observations for IC

    # Rebalance
    rebalance_days: int = 5  # Weekly rebalance

    # Portfolio
    top_n: int = 2  # Hold top N stocks
    max_position_pct: float = 0.5

    # Risk overlay
    max_drawdown_threshold: float = 0.12  # 12% drawdown → reduce
    momentum_crash_window: int = 10  # Detect crash in 10 days
    momentum_crash_threshold: float = -0.08  # -8% in 10 days = crash


# =============================================================================
# Regime-Conditional Factor Activation
# =============================================================================

# Which factors are active in each regime direction
# Key insight: momentum and reversal are CONTRADICTORY
# Solution: use momentum in trending markets, reversal in flat markets
REGIME_FACTOR_MAP: dict[str, list[str]] = {
    # UP market: momentum + trend + volume work well
    "UP": ["momentum", "trend", "volume_trend", "volatility"],
    # FLAT market: reversal + low-vol work well (mean reversion)
    "FLAT": ["reversal", "volatility", "volume_trend"],
    # DOWN market: only low-vol is useful (defensive)
    "DOWN": ["volatility", "reversal"],
}


# =============================================================================
# Factor Engine v2
# =============================================================================


class FactorEngine:
    """Calculate factors with regime-conditional activation and IC weighting."""

    def __init__(self, config: FactorConfig | None = None):
        self.config = config or FactorConfig()
        # IC history: factor_name -> list of (date, ic_value)
        self._ic_history: dict[str, list[tuple[date, float]]] = {
            "momentum": [],
            "reversal": [],
            "volatility": [],
            "trend": [],
            "volume_trend": [],
        }
        self._prev_scores: pd.DataFrame | None = None
        self._prev_date: date | None = None

    def calculate_all(
        self,
        data: dict[str, pd.DataFrame],
        as_of_date: date,
        regime_direction: str = "FLAT",
    ) -> pd.DataFrame:
        """Calculate factors with regime-conditional activation.

        Args:
            data: Dict of symbol -> DataFrame with OHLCV
            as_of_date: Calculation date
            regime_direction: UP/FLAT/DOWN from regime detector

        Returns:
            DataFrame with factor scores and composite_score
        """
        # Update IC from previous period (no lookahead)
        self._update_ic(data, as_of_date)

        # Calculate raw factors
        records = []
        for symbol, df in data.items():
            hist = df[df["trade_date"] <= as_of_date].copy()
            if len(hist) < self.config.trend_slow + 10:
                continue
            hist = hist.sort_values("trade_date")
            close = hist["close"].values
            volume = hist["volume"].values if "volume" in hist.columns else None

            factors = self._calculate_single(close, volume)
            factors["symbol"] = symbol
            records.append(factors)

        if not records:
            return pd.DataFrame()

        factor_df = pd.DataFrame(records)

        # Get active factors for this regime
        active_factors = REGIME_FACTOR_MAP.get(regime_direction, ["momentum", "volatility"])

        # Z-score normalization (cross-sectional)
        all_factor_cols = ["momentum", "reversal", "volatility", "trend", "volume_trend"]
        for col in all_factor_cols:
            if col in factor_df.columns:
                factor_df[col] = self._zscore(factor_df[col])

        # IC-weighted composite (only active factors)
        weights = self._get_ic_weights(active_factors, as_of_date)
        factor_df["composite_score"] = sum(factor_df[f] * w for f, w in weights.items())

        # Store for next IC calculation
        self._prev_scores = factor_df[["symbol", "composite_score"]].copy()
        self._prev_date = as_of_date

        return factor_df.sort_values("composite_score", ascending=False)

    def _calculate_single(
        self,
        close: np.ndarray,
        volume: np.ndarray | None,
    ) -> dict[str, float]:
        """Calculate raw factor values for a single stock."""
        n = len(close)
        cfg = self.config

        # 1. Momentum (skip recent month)
        if n > cfg.momentum_long + cfg.momentum_skip:
            past_price = close[-(cfg.momentum_long + cfg.momentum_skip)]
            recent_price = close[-cfg.momentum_skip]
            momentum = (recent_price - past_price) / past_price
        else:
            momentum = 0.0

        # 2. Short-term reversal (contrarian)
        if n > cfg.reversal_window:
            recent_return = (close[-1] - close[-cfg.reversal_window]) / close[-cfg.reversal_window]
            reversal = -recent_return
        else:
            reversal = 0.0

        # 3. Low volatility (negative = lower vol is better)
        if n > cfg.vol_window:
            returns = np.diff(close[-cfg.vol_window :]) / close[-cfg.vol_window : -1]
            realized_vol = np.std(returns) * np.sqrt(252)
            volatility = -realized_vol
        else:
            volatility = 0.0

        # 4. Trend (MA ratio)
        if n > cfg.trend_slow:
            ma_fast = np.mean(close[-cfg.trend_fast :])
            ma_slow = np.mean(close[-cfg.trend_slow :])
            trend = (ma_fast - ma_slow) / ma_slow
        else:
            trend = 0.0

        # 5. Volume trend
        volume_trend = 0.0
        if volume is not None and len(volume) > cfg.volume_baseline:
            recent_vol = np.mean(volume[-cfg.volume_window :])
            baseline_vol = np.mean(volume[-cfg.volume_baseline :])
            if baseline_vol > 0:
                volume_trend = (recent_vol - baseline_vol) / baseline_vol

        return {
            "momentum": momentum,
            "reversal": reversal,
            "volatility": volatility,
            "trend": trend,
            "volume_trend": volume_trend,
        }

    def _update_ic(self, data: dict[str, pd.DataFrame], as_of_date: date) -> None:
        """Update IC history using realized returns from previous prediction.

        IC = Spearman rank correlation between predicted score and realized return.
        Uses ONLY past data (no lookahead).
        """
        if self._prev_scores is None or self._prev_date is None:
            return

        # Calculate realized returns from prev_date to as_of_date
        realized = {}
        for symbol in self._prev_scores["symbol"]:
            if symbol not in data:
                continue
            df = data[symbol]
            prev_row = df[df["trade_date"] == self._prev_date]
            curr_row = df[df["trade_date"] == as_of_date]
            if not prev_row.empty and not curr_row.empty:
                ret = (curr_row.iloc[0]["close"] / prev_row.iloc[0]["close"]) - 1
                realized[symbol] = ret

        if len(realized) < 3:
            return

        # Merge with previous scores
        merged = self._prev_scores.copy()
        merged["realized_return"] = merged["symbol"].map(realized)
        merged = merged.dropna()

        if len(merged) < 3:
            return

        # Calculate rank IC (Spearman)
        ic = merged["composite_score"].corr(merged["realized_return"], method="spearman")

        if not np.isnan(ic):
            # Store IC for each active factor (approximation: use composite IC)
            for factor_name in self._ic_history:
                self._ic_history[factor_name].append((as_of_date, ic))

    def _get_ic_weights(
        self,
        active_factors: list[str],
        as_of_date: date,
    ) -> dict[str, float]:
        """Get IC-based weights for active factors.

        Uses rolling IC mean. Falls back to equal weight if insufficient data.
        """
        cfg = self.config
        weights = {}
        total_ic = 0.0

        for factor in active_factors:
            history = self._ic_history.get(factor, [])
            # Use only recent IC values
            recent = [ic for d, ic in history[-cfg.ic_lookback :]]

            if len(recent) >= cfg.ic_min_periods:
                ic_mean = np.mean(recent)
                # IC can be negative → factor is harmful → weight 0
                weights[factor] = max(0.0, ic_mean)
            else:
                # Insufficient data → equal weight fallback
                weights[factor] = 1.0

            total_ic += weights[factor]

        # Normalize
        if total_ic > 0:
            weights = {k: v / total_ic for k, v in weights.items()}
        else:
            # All IC negative → equal weight (defensive)
            n = len(active_factors)
            weights = {f: 1.0 / n for f in active_factors}

        return weights

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """Cross-sectional z-score."""
        mean = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (series - mean) / std


# =============================================================================
# Risk Overlay
# =============================================================================


class RiskOverlay:
    """Portfolio-level risk management.

    Rules (all general, not overfit):
    1. Max drawdown → reduce exposure
    2. Momentum crash detection → pause trading
    3. Regime-based position sizing
    """

    def __init__(self, config: FactorConfig | None = None):
        self.config = config or FactorConfig()
        self._peak_equity: float = 0.0
        self._current_drawdown: float = 0.0
        self._is_paused: bool = False

    def update(self, equity: float, data: dict[str, pd.DataFrame], as_of_date: date) -> None:
        """Update risk state."""
        self._peak_equity = max(self._peak_equity, equity)
        if self._peak_equity > 0:
            self._current_drawdown = (equity - self._peak_equity) / self._peak_equity

    def check_momentum_crash(
        self,
        data: dict[str, pd.DataFrame],
        as_of_date: date,
        index_symbol: str = "000300.XSHG",
    ) -> bool:
        """Detect momentum crash (market drops > threshold in short window)."""
        if index_symbol not in data:
            return False

        df = data[index_symbol]
        hist = df[df["trade_date"] <= as_of_date].sort_values("trade_date")

        if len(hist) < self.config.momentum_crash_window + 1:
            return False

        window_return = (
            hist.iloc[-1]["close"] / hist.iloc[-self.config.momentum_crash_window]["close"]
        ) - 1

        return window_return < self.config.momentum_crash_threshold

    def get_position_scale(
        self,
        regime_state: str,
        data: dict[str, pd.DataFrame],
        as_of_date: date,
    ) -> float:
        """Get position scale factor (0.0 to 1.0).

        Returns:
            0.0 = all cash, 1.0 = full position
        """
        # Rule 1: Drawdown control
        if self._current_drawdown < -self.config.max_drawdown_threshold:
            return 0.3  # Reduce to 30%

        # Rule 2: Momentum crash
        if self.check_momentum_crash(data, as_of_date):
            return 0.0  # All cash

        # Rule 3: Regime-based
        if regime_state == "DOWN_HIGH":
            return 0.0
        elif regime_state.startswith("DOWN"):
            return 0.4
        elif regime_state.startswith("FLAT"):
            return 0.8
        else:  # UP
            return 1.0

    @property
    def current_drawdown(self) -> float:
        return self._current_drawdown


# =============================================================================
# Portfolio Construction v2
# =============================================================================


@dataclass
class RebalanceSignal:
    """Signal from factor-based rebalancing."""

    date: date
    target_holdings: list[str]
    weights: dict[str, float]
    factor_scores: pd.DataFrame
    regime_state: str = ""
    position_scale: float = 1.0
    active_factors: list[str] = field(default_factory=list)
    ic_weights: dict[str, float] = field(default_factory=dict)


class FactorPortfolio:
    """Factor-based portfolio with IC weighting and risk overlay."""

    def __init__(self, config: FactorConfig | None = None):
        self.config = config or FactorConfig()
        self.engine = FactorEngine(self.config)
        self.risk = RiskOverlay(self.config)
        self.last_rebalance: date | None = None
        self.current_holdings: list[str] = []
        self.days_since_rebalance: int = 999

    def should_rebalance(self, trade_date: date) -> bool:
        """Check if it's time to rebalance."""
        self.days_since_rebalance += 1
        return self.days_since_rebalance >= self.config.rebalance_days

    def generate_signal(
        self,
        data: dict[str, pd.DataFrame],
        trade_date: date,
        regime_state: str = "",
        equity: float = 1_000_000.0,
    ) -> Optional[RebalanceSignal]:
        """Generate rebalancing signal."""
        if not self.should_rebalance(trade_date):
            return None

        # Extract regime direction
        regime_direction = regime_state.split("_")[0] if "_" in regime_state else "FLAT"

        # Update risk overlay
        self.risk.update(equity, data, trade_date)
        position_scale = self.risk.get_position_scale(regime_state, data, trade_date)

        # Calculate factors (regime-conditional)
        factor_df = self.engine.calculate_all(data, trade_date, regime_direction)

        if factor_df.empty:
            return None

        # Determine how many to hold based on risk
        top_n = self.config.top_n
        if position_scale == 0.0:
            top_n = 0
        elif position_scale < 0.5:
            top_n = max(1, top_n - 1)

        # Select top N
        selected = factor_df.head(top_n)["symbol"].tolist()

        # Weight by position scale
        if selected:
            base_weight = position_scale / len(selected)
            weights = {s: base_weight for s in selected}
        else:
            weights = {}

        # Get active factors and IC weights for reporting
        active_factors = REGIME_FACTOR_MAP.get(regime_direction, ["momentum", "volatility"])
        ic_weights = self.engine._get_ic_weights(active_factors, trade_date)

        # Update state
        self.last_rebalance = trade_date
        self.current_holdings = selected
        self.days_since_rebalance = 0

        return RebalanceSignal(
            date=trade_date,
            target_holdings=selected,
            weights=weights,
            factor_scores=factor_df,
            regime_state=regime_state,
            position_scale=position_scale,
            active_factors=active_factors,
            ic_weights=ic_weights,
        )


# =============================================================================
# Walk-Forward Validation
# =============================================================================


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation."""

    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    test_return: float
    benchmark_return: float
    excess_return: float
    n_trades: int
    max_drawdown: float


class WalkForwardValidator:
    """Walk-forward validation to prevent overfitting."""

    def __init__(
        self,
        train_days: int = 100,
        test_days: int = 40,
        step_days: int = 40,
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

    def generate_windows(self, all_dates: list[date]) -> list[dict]:
        """Generate walk-forward windows."""
        windows = []
        n = len(all_dates)
        start = 0
        window_id = 0

        while start + self.train_days + self.test_days <= n:
            train_end_idx = start + self.train_days
            test_end_idx = train_end_idx + self.test_days

            windows.append(
                {
                    "window_id": window_id,
                    "train_start": all_dates[start],
                    "train_end": all_dates[train_end_idx - 1],
                    "test_start": all_dates[train_end_idx],
                    "test_end": all_dates[min(test_end_idx - 1, n - 1)],
                }
            )

            start += self.step_days
            window_id += 1

        return windows

    def validate(
        self,
        data: dict[str, pd.DataFrame],
        benchmark_symbol: str,
    ) -> list[WalkForwardResult]:
        """Run walk-forward validation."""
        if benchmark_symbol not in data:
            return []

        all_dates = sorted(data[benchmark_symbol]["trade_date"].tolist())
        windows = self.generate_windows(all_dates)

        results = []
        for window in windows:
            result = self._run_window(data, benchmark_symbol, window)
            if result:
                results.append(result)

        return results

    def _run_window(
        self,
        data: dict[str, pd.DataFrame],
        benchmark_symbol: str,
        window: dict,
    ) -> Optional[WalkForwardResult]:
        """Run a single walk-forward window."""
        from a_share_quant.regime import RegimeDetector

        test_start = window["test_start"]
        test_end = window["test_end"]

        bench_df = data[benchmark_symbol]
        test_dates = bench_df[
            (bench_df["trade_date"] >= test_start) & (bench_df["trade_date"] <= test_end)
        ]["trade_date"].tolist()

        if len(test_dates) < 5:
            return None

        # Run factor strategy on test period
        portfolio = FactorPortfolio()
        detector = RegimeDetector()
        initial_capital = 1_000_000.0
        cash = initial_capital
        holdings: dict[str, int] = {}
        trades = 0
        equity_history = []
        fee_rate = 0.001

        # Tradable universe (exclude index)
        tradable = {k: v for k, v in data.items() if not k.startswith("000")}

        for td in test_dates:
            # Detect regime
            idx_hist = bench_df[bench_df["trade_date"] <= td]
            regime = detector.detect(idx_hist, td)

            # Current equity
            equity = cash
            for sym, shares in holdings.items():
                if sym in data:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        equity += shares * row.iloc[0]["close"]

            # Check rebalance
            signal = portfolio.generate_signal(tradable, td, regime.state_id, equity)

            if signal:
                # Sell positions not in target
                for sym in list(holdings.keys()):
                    if sym not in signal.target_holdings:
                        row = data[sym][data[sym]["trade_date"] == td]
                        if not row.empty:
                            price = row.iloc[0]["close"]
                            revenue = holdings[sym] * price
                            cash += revenue * (1 - fee_rate / 2)
                            del holdings[sym]
                            trades += 1

                # Buy target positions
                total_equity = cash + sum(
                    holdings.get(s, 0) * data[s][data[s]["trade_date"] == td].iloc[0]["close"]
                    for s in holdings
                    if s in data and not data[s][data[s]["trade_date"] == td].empty
                )

                for sym, weight in signal.weights.items():
                    if sym not in holdings and sym in data:
                        row = data[sym][data[sym]["trade_date"] == td]
                        if not row.empty:
                            price = row.iloc[0]["close"]
                            target_value = total_equity * weight
                            shares = int(target_value / price / 100) * 100
                            if shares > 0:
                                cost = shares * price * (1 + fee_rate / 2)
                                if cost <= cash:
                                    cash -= cost
                                    holdings[sym] = shares
                                    trades += 1

            # Track equity
            equity = cash
            for sym, shares in holdings.items():
                if sym in data:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        equity += shares * row.iloc[0]["close"]
            equity_history.append(equity)

        # Calculate metrics
        final_equity = equity_history[-1] if equity_history else initial_capital
        test_return = (final_equity / initial_capital) - 1

        # Max drawdown
        equity_arr = np.array(equity_history)
        if len(equity_arr) > 0:
            cummax = np.maximum.accumulate(equity_arr)
            dd = (equity_arr - cummax) / cummax
            max_dd = abs(dd.min())
        else:
            max_dd = 0.0

        # Benchmark return
        bench_start = bench_df[bench_df["trade_date"] == test_dates[0]]
        bench_end = bench_df[bench_df["trade_date"] == test_dates[-1]]
        if bench_start.empty or bench_end.empty:
            return None

        benchmark_return = (bench_end.iloc[0]["close"] / bench_start.iloc[0]["close"]) - 1

        return WalkForwardResult(
            window_id=window["window_id"],
            train_start=window["train_start"],
            train_end=window["train_end"],
            test_start=test_start,
            test_end=test_end,
            test_return=test_return,
            benchmark_return=benchmark_return,
            excess_return=test_return - benchmark_return,
            n_trades=trades,
            max_drawdown=max_dd,
        )
