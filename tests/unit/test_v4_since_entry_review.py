"""Pure-helper tests for the V4 since-entry review (no data or replay needed)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from scripts.review_v4_since_entry_20261102 import (
    hold_intervals,
    max_drawdown_details,
    monthly_returns,
    pair_episodes,
)


def test_max_drawdown_uses_initial_cash_as_first_high_water_mark() -> None:
    curve = pd.DataFrame(
        {
            "trade_date": [date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30)],
            "equity": [110.0, 105.0, 120.0],
        }
    )
    detail = max_drawdown_details(curve, initial=100.0)
    assert detail["max_drawdown"] == pytest.approx(105.0 / 110.0 - 1.0)
    assert detail["peak_equity"] == 110.0
    assert detail["peak_date"] == "2026-07-28"
    assert detail["trough_date"] == "2026-07-29"
    assert detail["underwater_days"] == 1
    assert detail["recovered"] is True


def test_max_drawdown_without_drawdown_has_no_trough_date() -> None:
    curve = pd.DataFrame(
        {"trade_date": [date(2026, 7, 28), date(2026, 7, 29)], "equity": [101.0, 102.0]}
    )
    detail = max_drawdown_details(curve, initial=100.0)
    assert detail["max_drawdown"] == 0.0
    assert detail["trough_date"] is None
    assert detail["underwater_days"] == 0


def test_pair_episodes_reconciles_costs_and_holding_days() -> None:
    calendar = [
        date(2026, 7, 28),
        date(2026, 7, 30),
        date(2026, 8, 3),
        date(2026, 8, 6),
    ]
    trades = [
        {
            "date": "2026-07-30",
            "signal_date": "2026-07-29",
            "action": "buy",
            "code": "A",
            "price": 10.0,
            "shares": 1000,
            "estimated_cost": 1.5,
        },
        {
            "date": "2026-08-06",
            "signal_date": "2026-08-05",
            "action": "sell",
            "code": "A",
            "price": 11.0,
            "shares": 1000,
            "estimated_cost": 1.65,
            "realized_pnl": 996.85,
        },
    ]
    episodes = pair_episodes(trades, calendar)
    assert len(episodes) == 1
    assert episodes[0]["holding_days"] == 2
    assert episodes[0]["basis"] == pytest.approx(10001.5)
    assert episodes[0]["realized_pnl"] == pytest.approx(996.85)
    assert episodes[0]["return"] == pytest.approx(996.85 / 10001.5)


def test_pair_episodes_rejects_unmatched_sell() -> None:
    with pytest.raises(ValueError, match="unmatched sell"):
        pair_episodes(
            [
                {
                    "date": "2026-08-06",
                    "signal_date": "2026-08-05",
                    "action": "sell",
                    "code": "A",
                    "price": 11.0,
                    "shares": 1000,
                    "estimated_cost": 1.65,
                    "realized_pnl": 0.0,
                }
            ],
            [date(2026, 8, 6)],
        )


def test_monthly_returns_compound_from_initial_capital() -> None:
    curve = pd.DataFrame(
        {
            "trade_date": [date(2026, 7, 28), date(2026, 7, 31), date(2026, 8, 31)],
            "equity": [101.0, 102.0, 99.0],
        }
    )
    rows = monthly_returns(curve, initial=100.0)
    assert rows[0]["month"] == "2026-07"
    assert rows[0]["return"] == pytest.approx(0.02)
    assert rows[1]["return"] == pytest.approx(99.0 / 102.0 - 1.0)


def test_hold_intervals_marks_cash_and_contiguous_runs() -> None:
    curve = pd.DataFrame(
        {
            "trade_date": [date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30)],
            "equity": [100.0, 100.0, 101.0],
            "holding": [None, "A", "A"],
        }
    )
    intervals = hold_intervals(curve)
    assert [item["holding"] for item in intervals] == ["CASH", "A"]
    assert intervals[0]["days"] == 1
    assert intervals[1]["days"] == 2
    assert intervals[1]["start_date"] == "2026-07-29"
