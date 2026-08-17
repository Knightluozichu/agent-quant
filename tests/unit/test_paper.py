"""Tests for paper trading."""

from datetime import date
from pathlib import Path
import tempfile

import pytest

from a_share_quant.paper import (
    DailyScheduler,
    DailySignal,
    PaperBroker,
    PaperOrder,
)


class TestPaperBroker:
    """Tests for paper broker."""

    def test_initial_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = PaperBroker(
                initial_capital=1_000_000,
                state_file=Path(tmpdir) / "state.json",
            )
            assert broker.cash == 1_000_000
            assert broker.available_cash == 1_000_000
            assert broker.get_positions() == {}

    def test_submit_buy_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = PaperBroker(
                initial_capital=1_000_000,
                state_file=Path(tmpdir) / "state.json",
            )
            order = broker.submit_order("510300.SSE", "BUY", 1000)
            assert order.status == "PENDING"
            assert order.order_id.startswith("P")

    def test_submit_sell_order_no_position(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = PaperBroker(
                initial_capital=1_000_000,
                state_file=Path(tmpdir) / "state.json",
            )
            order = broker.submit_order("510300.SSE", "SELL", 1000)
            assert order.status == "REJECTED"
            assert "T+1" in order.reject_reason

    def test_execute_buy_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = PaperBroker(
                initial_capital=1_000_000,
                state_file=Path(tmpdir) / "state.json",
            )
            broker.submit_order("510300.SSE", "BUY", 1000)

            market_data = {"510300.SSE": {"open": 4.0, "high": 4.1, "low": 3.95, "close": 4.05}}
            filled = broker.execute_pending_orders(market_data, date(2024, 1, 15))

            assert len(filled) == 1
            assert filled[0].status == "FILLED"
            assert filled[0].fill_price == 4.0

            # Check position
            pos = broker.get_position("510300.SSE")
            assert pos is not None
            assert pos["quantity"] == 1000
            assert pos["sellable"] == 0  # T+1: not sellable same day

    def test_t_plus_1_sell(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = PaperBroker(
                initial_capital=1_000_000,
                state_file=Path(tmpdir) / "state.json",
            )

            # Buy on day 1
            broker.submit_order("510300.SSE", "BUY", 1000)
            market_data = {"510300.SSE": {"open": 4.0, "high": 4.1, "low": 3.95, "close": 4.05}}
            broker.execute_pending_orders(market_data, date(2024, 1, 15))

            # Try to sell same day - should fail
            order = broker.submit_order("510300.SSE", "SELL", 1000)
            assert order.status == "REJECTED"

            # Sell next day - should work
            order = broker.submit_order("510300.SSE", "SELL", 1000)
            # Need to update settlement first
            broker._positions.update_settlement(date(2024, 1, 16))
            order2 = broker.submit_order("510300.SSE", "SELL", 1000)
            assert order2.status == "PENDING"

    def test_state_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"

            # Create broker and make a trade
            broker1 = PaperBroker(initial_capital=1_000_000, state_file=state_file)
            broker1.submit_order("510300.SSE", "BUY", 1000)
            market_data = {"510300.SSE": {"open": 4.0, "high": 4.1, "low": 3.95, "close": 4.05}}
            broker1.execute_pending_orders(market_data, date(2024, 1, 15))

            # Create new broker - should restore state
            broker2 = PaperBroker(initial_capital=1_000_000, state_file=state_file)
            assert broker2.get_position("510300.SSE") is not None
            assert broker2.cash < 1_000_000  # Cash reduced by trade

    def test_cancel_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = PaperBroker(
                initial_capital=1_000_000,
                state_file=Path(tmpdir) / "state.json",
            )
            order = broker.submit_order("510300.SSE", "BUY", 1000)
            assert broker.cancel_order(order.order_id)
            assert order.status == "CANCELLED"


class TestDailyScheduler:
    """Tests for daily scheduler."""

    def test_generate_and_execute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = PaperBroker(
                initial_capital=1_000_000,
                state_file=Path(tmpdir) / "state.json",
            )
            scheduler = DailyScheduler(broker)

            signals = [
                DailySignal(
                    trade_date=date(2024, 1, 15),
                    symbol="510300.SSE",
                    action="BUY",
                    quantity=1000,
                    strategy_name="TREND_HOLD",
                    reason="Breakout",
                ),
            ]
            scheduler.generate_signals(signals, date(2024, 1, 15))

            market_data = {"510300.SSE": {"open": 4.0, "high": 4.1, "low": 3.95, "close": 4.05}}
            filled = scheduler.execute_signals(market_data, date(2024, 1, 16))

            assert len(filled) == 1
            log = scheduler.get_execution_log()
            assert len(log) == 1
            assert log[0]["filled"] == 1
