"""Hypothetical strategy accounting, independent of personal fills."""

import pytest
from scripts.review_since_entry_20260916 import summarize_trades


def test_trade_summary_reconciles_costs_and_marks_open_position():
    trades = [
        {"action": "buy", "code": "A", "shares": 100, "price": 10.0},
        {"action": "sell", "code": "A", "price": 11.0},
        {"action": "buy", "code": "B", "shares": 100, "price": 20.0},
    ]
    result = summarize_trades(trades, {"B": 21.0}, cost=0.0015)
    assert result["estimated_cost_paid"] == pytest.approx(6.15)
    assert result["realized_pnl"] == pytest.approx(96.85)
    assert result["open_position_pnl_after_buy_cost"] == pytest.approx(97.0)
    assert result["total_pnl"] == pytest.approx(193.85)
    assert result["cash"] == pytest.approx(98093.85)
    assert result["ending_equity"] == pytest.approx(100193.85)
    assert summarize_trades([], {}, cost=0.0015)["ending_equity"] == 100000
    with pytest.raises(ValueError, match="unmatched sell"):
        summarize_trades([trades[1]], {}, cost=0.0015)
