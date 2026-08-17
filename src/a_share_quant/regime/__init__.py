"""Market regime detection - 9-state model.

States: Direction (UP/FLAT/DOWN) × Oscillation (LOW/MEDIUM/HIGH)

Detection based on:
- Index trend (MA20/MA60 slope)
- Volatility (ATR percentile)
- Breadth (advance/decline ratio)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import date


class Direction(StrEnum):
    """Market direction."""

    UP = "UP"
    FLAT = "FLAT"
    DOWN = "DOWN"


class Oscillation(StrEnum):
    """Market oscillation/volatility level."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class RegimeState:
    """Current market regime state."""

    direction: Direction
    oscillation: Oscillation
    confidence: float  # 0-1
    as_of_date: date

    @property
    def state_id(self) -> str:
        """Get unique state identifier."""
        return f"{self.direction.value}_{self.oscillation.value}"

    @property
    def is_bullish(self) -> bool:
        return self.direction == Direction.UP

    @property
    def is_bearish(self) -> bool:
        return self.direction == Direction.DOWN

    @property
    def is_volatile(self) -> bool:
        return self.oscillation == Oscillation.HIGH


class RegimeDetector:
    """Detect market regime from index data.

    Uses multiple indicators:
    1. Trend: MA20 vs MA60 slope
    2. Volatility: ATR percentile over 60 days
    3. Momentum: 20-day return
    """

    def __init__(
        self,
        trend_fast: int = 20,
        trend_slow: int = 60,
        vol_window: int = 20,
        vol_lookback: int = 60,
    ):
        self._trend_fast = trend_fast
        self._trend_slow = trend_slow
        self._vol_window = vol_window
        self._vol_lookback = vol_lookback

    def detect(self, df: pd.DataFrame, as_of_date: date) -> RegimeState:
        """Detect regime from daily bar data.

        Args:
            df: DataFrame with columns: trade_date, close, high, low
            as_of_date: Date to detect regime for

        Returns:
            RegimeState with direction, oscillation, and confidence
        """
        # Filter to data up to as_of_date
        df = df[df["trade_date"] <= as_of_date].copy()

        if len(df) < self._trend_slow:
            # Not enough data, return neutral
            return RegimeState(
                direction=Direction.FLAT,
                oscillation=Oscillation.MEDIUM,
                confidence=0.3,
                as_of_date=as_of_date,
            )

        df = df.sort_values("trade_date").tail(self._vol_lookback + self._trend_slow)

        # Calculate indicators
        direction, dir_confidence = self._detect_direction(df)
        oscillation, osc_confidence = self._detect_oscillation(df)

        # Combined confidence
        confidence = (dir_confidence + osc_confidence) / 2

        return RegimeState(
            direction=direction,
            oscillation=oscillation,
            confidence=confidence,
            as_of_date=as_of_date,
        )

    def _detect_direction(self, df: pd.DataFrame) -> tuple[Direction, float]:
        """Detect market direction from trend indicators."""
        close = df["close"].values

        # Moving averages
        ma_fast = pd.Series(close).rolling(self._trend_fast).mean().iloc[-1]
        ma_slow = pd.Series(close).rolling(self._trend_slow).mean().iloc[-1]

        # MA slope (5-day change)
        ma_fast_series = pd.Series(close).rolling(self._trend_fast).mean()
        ma_slope = (ma_fast_series.iloc[-1] - ma_fast_series.iloc[-5]) / ma_fast_series.iloc[-5]

        # 20-day return
        ret_20d = (close[-1] - close[-self._trend_fast]) / close[-self._trend_fast]

        # Determine direction
        if ma_fast > ma_slow and ma_slope > 0.005 and ret_20d > 0.02:
            direction = Direction.UP
            confidence = min(0.9, 0.5 + abs(ma_slope) * 10 + abs(ret_20d) * 2)
        elif ma_fast < ma_slow and ma_slope < -0.005 and ret_20d < -0.02:
            direction = Direction.DOWN
            confidence = min(0.9, 0.5 + abs(ma_slope) * 10 + abs(ret_20d) * 2)
        else:
            direction = Direction.FLAT
            confidence = 0.6 - abs(ma_slope) * 5

        return direction, max(0.3, min(0.95, confidence))

    def _detect_oscillation(self, df: pd.DataFrame) -> tuple[Oscillation, float]:
        """Detect oscillation/volatility level."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # Calculate ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr = pd.Series(tr).rolling(self._vol_window).mean().iloc[-1]
        atr_pct = atr / close[-1]  # ATR as percentage of price

        # Historical ATR percentile
        atr_series = pd.Series(tr).rolling(self._vol_window).mean().dropna()
        if len(atr_series) > 10:
            percentile = (atr_series < atr_series.iloc[-1]).sum() / len(atr_series)
        else:
            percentile = 0.5

        # Determine oscillation level
        if percentile > 0.7 or atr_pct > 0.025:
            oscillation = Oscillation.HIGH
            confidence = min(0.9, 0.5 + percentile * 0.4)
        elif percentile < 0.3 or atr_pct < 0.012:
            oscillation = Oscillation.LOW
            confidence = min(0.9, 0.5 + (1 - percentile) * 0.4)
        else:
            oscillation = Oscillation.MEDIUM
            confidence = 0.6

        return oscillation, confidence


# =============================================================================
# Strategy Selection based on Regime
# =============================================================================

# Mapping from regime to preferred strategies
REGIME_STRATEGY_MAP: dict[str, list[str]] = {
    # Bullish + Low Vol: Trend following
    "UP_LOW": ["TREND_HOLD", "PULLBACK_SWING"],
    "UP_MEDIUM": ["TREND_HOLD", "PULLBACK_SWING"],
    "UP_HIGH": ["PULLBACK_SWING", "RANGE_MEAN_REVERSION"],
    # Neutral: Mean reversion
    "FLAT_LOW": ["RANGE_MEAN_REVERSION", "CASH_DEFENSE"],
    "FLAT_MEDIUM": ["RANGE_MEAN_REVERSION", "PULLBACK_SWING"],
    "FLAT_HIGH": ["RANGE_MEAN_REVERSION", "CASH_DEFENSE"],
    # Bearish: Defense
    "DOWN_LOW": ["BEAR_REBOUND", "CASH_DEFENSE"],
    "DOWN_MEDIUM": ["BEAR_REBOUND", "CASH_DEFENSE"],
    "DOWN_HIGH": ["CASH_DEFENSE"],
}


def get_recommended_strategies(regime: RegimeState) -> list[str]:
    """Get recommended strategies for current regime."""
    return REGIME_STRATEGY_MAP.get(regime.state_id, ["CASH_DEFENSE"])
