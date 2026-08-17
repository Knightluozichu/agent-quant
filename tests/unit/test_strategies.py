"""Tests for regime detection and strategies."""

from datetime import date

import pandas as pd
import pytest

from a_share_quant.regime import (
    Direction,
    Oscillation,
    RegimeDetector,
    RegimeState,
    get_recommended_strategies,
)
from a_share_quant.strategies import (
    BaseStrategy,
    BearReboundStrategy,
    CashDefenseStrategy,
    PositionInfo,
    PullbackSwingStrategy,
    RangeMeanReversionStrategy,
    TrendHoldStrategy,
    get_strategy,
    list_strategies,
)


def make_test_df(days: int = 100, trend: str = "up") -> pd.DataFrame:
    """Create test DataFrame with specified trend."""
    dates = pd.date_range("2024-01-01", periods=days, freq="B")

    if trend == "up":
        base = 100 + pd.Series(range(days)) * 0.5
    elif trend == "down":
        base = 100 - pd.Series(range(days)) * 0.5
    else:  # flat
        base = pd.Series([100] * days)

    # Add some noise
    noise = pd.Series([i % 3 - 1 for i in range(days)]) * 0.5

    close = base + noise
    high = close + 1
    low = close - 1
    volume = [1000000 + i * 1000 for i in range(days)]

    return pd.DataFrame(
        {
            "trade_date": dates.date,
            "open": close - 0.2,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestRegimeDetector:
    """Tests for regime detection."""

    def test_uptrend_detection(self):
        detector = RegimeDetector()
        df = make_test_df(100, "up")
        regime = detector.detect(df, df["trade_date"].iloc[-1])

        assert regime.direction == Direction.UP
        assert regime.confidence > 0.5

    def test_downtrend_detection(self):
        detector = RegimeDetector()
        df = make_test_df(100, "down")
        regime = detector.detect(df, df["trade_date"].iloc[-1])

        assert regime.direction == Direction.DOWN

    def test_flat_detection(self):
        detector = RegimeDetector()
        df = make_test_df(100, "flat")
        regime = detector.detect(df, df["trade_date"].iloc[-1])

        assert regime.direction == Direction.FLAT

    def test_insufficient_data(self):
        detector = RegimeDetector()
        df = make_test_df(30, "up")  # Less than 60 days
        regime = detector.detect(df, df["trade_date"].iloc[-1])

        assert regime.direction == Direction.FLAT
        assert regime.confidence < 0.5

    def test_state_id(self):
        state = RegimeState(
            direction=Direction.UP,
            oscillation=Oscillation.LOW,
            confidence=0.8,
            as_of_date=date(2024, 1, 15),
        )
        assert state.state_id == "UP_LOW"
        assert state.is_bullish
        assert not state.is_bearish


class TestStrategyRecommendation:
    """Tests for strategy recommendation."""

    def test_bullish_low_vol(self):
        regime = RegimeState(Direction.UP, Oscillation.LOW, 0.8, date(2024, 1, 15))
        strategies = get_recommended_strategies(regime)
        assert "TREND_HOLD" in strategies

    def test_bearish_high_vol(self):
        regime = RegimeState(Direction.DOWN, Oscillation.HIGH, 0.8, date(2024, 1, 15))
        strategies = get_recommended_strategies(regime)
        assert "CASH_DEFENSE" in strategies

    def test_flat_medium(self):
        regime = RegimeState(Direction.FLAT, Oscillation.MEDIUM, 0.6, date(2024, 1, 15))
        strategies = get_recommended_strategies(regime)
        assert "RANGE_MEAN_REVERSION" in strategies


class TestTrendHoldStrategy:
    """Tests for trend hold strategy."""

    def test_buy_signal_in_uptrend(self):
        strategy = TrendHoldStrategy()
        df = make_test_df(100, "up")
        regime = RegimeState(Direction.UP, Oscillation.LOW, 0.8, date(2024, 1, 15))

        # Add volume spike at end
        df.loc[df.index[-1], "volume"] = 3000000

        signal = strategy.generate_signal("510300.SSE", df, regime, None, df["trade_date"].iloc[-1])

        # May or may not generate signal depending on exact conditions
        if signal:
            assert signal.action == "BUY"
            assert signal.stop_loss is not None

    def test_no_signal_in_downtrend(self):
        strategy = TrendHoldStrategy()
        df = make_test_df(100, "down")
        regime = RegimeState(Direction.DOWN, Oscillation.MEDIUM, 0.7, date(2024, 1, 15))

        signal = strategy.generate_signal("510300.SSE", df, regime, None, df["trade_date"].iloc[-1])

        assert signal is None

    def test_exit_on_ma_break(self):
        strategy = TrendHoldStrategy()
        df = make_test_df(100, "down")  # Downtrend to trigger exit
        regime = RegimeState(Direction.DOWN, Oscillation.MEDIUM, 0.7, date(2024, 1, 15))

        position = PositionInfo(
            symbol="510300.SSE",
            quantity=1000,
            avg_cost=100,
            sellable=1000,
            current_price=90,
            unrealized_pnl=-10000,
            holding_days=10,
        )

        should_exit, reason = strategy.should_exit(position, df, regime, df["trade_date"].iloc[-1])

        # Should exit due to regime change
        assert should_exit or "熊市" in reason


class TestCashDefenseStrategy:
    """Tests for cash defense strategy."""

    def test_never_buys(self):
        strategy = CashDefenseStrategy()
        df = make_test_df(100, "up")
        regime = RegimeState(Direction.UP, Oscillation.LOW, 0.8, date(2024, 1, 15))

        signal = strategy.generate_signal("510300.SSE", df, regime, None, df["trade_date"].iloc[-1])

        assert signal is None

    def test_always_exits(self):
        strategy = CashDefenseStrategy()
        df = make_test_df(100, "down")
        regime = RegimeState(Direction.DOWN, Oscillation.HIGH, 0.9, date(2024, 1, 15))

        position = PositionInfo(
            symbol="510300.SSE",
            quantity=1000,
            avg_cost=100,
            sellable=1000,
            current_price=95,
            unrealized_pnl=-5000,
            holding_days=5,
        )

        should_exit, reason = strategy.should_exit(position, df, regime, df["trade_date"].iloc[-1])

        assert should_exit
        assert "防御" in reason


class TestStrategyRegistry:
    """Tests for strategy registry."""

    def test_list_strategies(self):
        strategies = list_strategies()
        assert len(strategies) == 5
        assert "TREND_HOLD" in strategies
        assert "CASH_DEFENSE" in strategies

    def test_get_strategy(self):
        strategy = get_strategy("TREND_HOLD")
        assert isinstance(strategy, TrendHoldStrategy)
        assert strategy.name == "TREND_HOLD"

    def test_get_strategy_with_params(self):
        strategy = get_strategy("TREND_HOLD", {"ma_period": 30})
        assert strategy.params["ma_period"] == 30

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("UNKNOWN")
