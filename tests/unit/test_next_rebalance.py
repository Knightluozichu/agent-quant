"""绝对网格调仓日计算: 锚点落后时必须沿网格向前推进, 不能返回过去日期."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import next_rebalance as nr  # noqa: E402


def _weekdays(count: int) -> list[date]:
    start = date(2026, 1, 5)
    out: list[date] = []
    cursor = start
    while len(out) < count:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


@pytest.mark.unit
def test_no_last_rebalance_returns_latest_as_today():
    dates = _weekdays(10)
    assert nr.find_next_rebalance(dates, None) == (dates[-1], 0)


@pytest.mark.unit
def test_on_grid_anchor_advances_one_period():
    dates = _weekdays(16)  # 最新交易日 = dates[15], 恰好是网格日
    next_rb, days_left = nr.find_next_rebalance(dates, str(dates[10]))
    assert next_rb == dates[15]
    assert days_left == 0


@pytest.mark.unit
def test_stale_anchor_never_returns_negative_days():
    """待确认票抑制多轮后, 锚点可能落后多个网格周期."""
    latest_idx = 20
    dates = _weekdays(latest_idx + 1)
    next_rb, days_left = nr.find_next_rebalance(dates, str(dates[0]))

    assert next_rb >= dates[latest_idx]
    assert days_left is not None and days_left >= 0


@pytest.mark.unit
def test_anchor_missing_from_dates_falls_back_to_calendar():
    dates = _weekdays(10)
    next_rb, days_left = nr.find_next_rebalance(dates, "2025-12-31")
    assert next_rb == date(2026, 1, 7)
    assert days_left is None
