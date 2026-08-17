"""Tests for backtest engine."""

from datetime import date

import pandas as pd
import pytest

from a_share_quant.backtest import (
    BacktestConfig,
    BacktestEngine,
    ExecutionSimulator,
    FillEvent,
    MetricsCalculator,
    OrderEvent,
    PerformanceMetrics,
)
from a_share_quant.data.providers.mock import MockProvider


class TestExecutionSimulator:
    """Tests for execution simulation."""

    def test_market_order_fill(self):
        sim = ExecutionSimulator()
        order = OrderEvent(
            trade_date=date(2024, 1, 15),
            symbol="510300.SSE",
            side="BUY",
            quantity=1000,
        )
        bar = {"open": 4.0, "high": 4.1, "low": 3.95, "close": 4.05, "volume": 1000000}
        fill = sim.simulate_fill(order, bar, prev_close=4.0, board="MAIN")
        assert fill is not None
        assert fill.fill_price == 4.0  # Market order uses open
        assert fill.quantity == 1000

    def test_limit_order_fill(self):
        sim = ExecutionSimulator()
        order = OrderEvent(
            trade_date=date(2024, 1, 15),
            symbol="510300.SSE",
            side="BUY",
            quantity=1000,
            order_type="LIMIT",
            limit_price=4.0,
        )
        bar = {"open": 4.05, "high": 4.1, "low": 3.95, "close": 4.05, "volume": 1000000}
        fill = sim.simulate_fill(order, bar, prev_close=4.0, board="MAIN")
        assert fill is not None
        assert fill.fill_price == 4.0

    def test_limit_order_no_fill(self):
        """Limit buy below low should not fill."""
        sim = ExecutionSimulator()
        order = OrderEvent(
            trade_date=date(2024, 1, 15),
            symbol="510300.SSE",
            side="BUY",
            quantity=1000,
            order_type="LIMIT",
            limit_price=3.90,  # Below low
        )
        bar = {"open": 4.0, "high": 4.1, "low": 3.95, "close": 4.05, "volume": 1000000}
        fill = sim.simulate_fill(order, bar, prev_close=4.0, board="MAIN")
        assert fill is None

    def test_fees_calculated(self):
        sim = ExecutionSimulator()
        order = OrderEvent(
            trade_date=date(2024, 1, 15),
            symbol="510300.SSE",
            side="SELL",
            quantity=1000,
        )
        bar = {"open": 4.0, "high": 4.1, "low": 3.95, "close": 4.05, "volume": 1000000}
        fill = sim.simulate_fill(order, bar, prev_close=4.0, board="MAIN")
        assert fill is not None
        assert fill.commission >= 5.0  # Min commission
        assert fill.stamp_tax > 0  # Sell has stamp tax


class TestMetricsCalculator:
    """Tests for performance metrics."""

    def test_basic_metrics(self):
        calc = MetricsCalculator()
        # Simple equity curve: 100 -> 110 -> 105 -> 115
        equity = pd.Series([100, 110, 105, 115], index=pd.date_range("2024-01-01", periods=4))
        trades = [
            {"pnl": 10, "holding_days": 5},
            {"pnl": -5, "holding_days": 3},
            {"pnl": 10, "holding_days": 4},
        ]
        metrics = calc.calculate(equity, trades)

        assert metrics.total_return == pytest.approx(0.15)  # 15%
        assert metrics.total_trades == 3
        assert metrics.win_rate == pytest.approx(2 / 3)

    def test_empty_equity(self):
        calc = MetricsCalculator()
        equity = pd.Series([100])
        metrics = calc.calculate(equity, [])
        assert metrics.total_return == 0
        assert metrics.total_trades == 0

    def test_max_drawdown(self):
        calc = MetricsCalculator()
        # Equity: 100 -> 120 -> 90 -> 110 (drawdown from 120 to 90 = 25%)
        equity = pd.Series([100, 120, 90, 110], index=pd.date_range("2024-01-01", periods=4))
        metrics = calc.calculate(equity, [])
        assert metrics.max_drawdown == pytest.approx(0.25, rel=0.01)


class TestBacktestEngine:
    """Tests for backtest engine."""

    def test_basic_backtest(self):
        provider = MockProvider()
        config = BacktestConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=1_000_000,
            symbols=["510300.SSE"],
        )
        engine = BacktestEngine(provider, config)

        # Simple buy-and-hold strategy
        def strategy(trade_date, market_data, positions):
            if "510300.SSE" in market_data and not positions.get("510300.SSE"):
                return [
                    OrderEvent(
                        trade_date=trade_date,
                        symbol="510300.SSE",
                        side="BUY",
                        quantity=10000,
                        strategy_name="buy_hold",
                    )
                ]
            return []

        result = engine.run(strategy)

        assert len(result.equity_curve) > 0
        assert result.config.initial_capital == 1_000_000

    def test_no_strategy(self):
        """Backtest without strategy should just hold cash."""
        provider = MockProvider()
        config = BacktestConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=1_000_000,
            symbols=["510300.SSE"],
        )
        engine = BacktestEngine(provider, config)
        result = engine.run()

        # Should have no trades
        assert len(result.trades) == 0
        # Equity should remain at initial capital
        assert result.equity_curve.iloc[-1] == pytest.approx(1_000_000)

    def test_metrics_calculated(self):
        provider = MockProvider()
        config = BacktestConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 29),
            initial_capital=1_000_000,
            symbols=["510300.SSE"],
        )
        engine = BacktestEngine(provider, config)

        def strategy(trade_date, market_data, positions):
            if "510300.SSE" in market_data and not positions.get("510300.SSE"):
                return [
                    OrderEvent(
                        trade_date=trade_date,
                        symbol="510300.SSE",
                        side="BUY",
                        quantity=10000,
                    )
                ]
            return []

        result = engine.run(strategy)

        # Metrics should be calculated
        assert isinstance(result.metrics, PerformanceMetrics)
