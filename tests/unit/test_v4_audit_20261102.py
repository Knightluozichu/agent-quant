"""Pure-helper and risk-adapter tests for the V4 strategy audit."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from scripts import audit_v4_current_20261102 as audit
from scripts import live_signal as live


def curve(values: list[float], start: date = date(2020, 6, 19)) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(values)).date
    return pd.DataFrame({"trade_date": dates, "equity": values})


def test_metrics_uses_initial_capital_for_drawdown() -> None:
    frame = curve([110.0, 105.0, 120.0])
    result = audit.metrics(frame, initial=100.0)
    assert result["max_drawdown"] == pytest.approx(105.0 / 110.0 - 1.0)
    assert result["final_equity"] == 120.0
    assert result["total_return"] == pytest.approx(0.20)
    assert result["peak_date"] == "2020-06-19"
    assert result["trough_date"] == "2020-06-22"


def test_yearly_continuous_compounds_from_previous_year_end() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": [date(2020, 12, 31), date(2021, 12, 31), date(2022, 12, 30)],
            "equity": [110.0, 121.0, 108.9],
        }
    )
    rows = audit.yearly_continuous(frame, initial=100.0)
    assert [row["year"] for row in rows] == [2020, 2021, 2022]
    assert rows[0]["return"] == pytest.approx(0.10)
    assert rows[1]["return"] == pytest.approx(0.10)
    assert rows[2]["return"] == pytest.approx(-0.10)


def test_rolling_stats_counts_negative_windows() -> None:
    frame = curve([100.0, 110.0, 99.0, 120.0, 90.0, 130.0])
    rows, summary = audit.rolling_stats(frame, window=2, initial=100.0)
    assert len(rows) == 5
    assert summary["windows"] == 5
    assert summary["negative_windows"] == 2
    assert summary["return_min"] == pytest.approx(90.0 / 99.0 - 1.0)


def _risk_kwargs() -> dict[str, object]:
    data = {"A": pd.DataFrame({"trade_date": [date(2020, 6, 19)], "open": [10.0], "close": [10.0]})}
    return {
        "target": "A",
        "holding": "A",
        "state": {"cash": 0.0, "shares": 100.0, "holding": "A", "peak_equity": 2000.0},
        "data": data,
        "td": date(2020, 6, 19),
        "idx_map": {"A": 0},
        "is_rebalance": False,
        "common_dates": [date(2020, 6, 19)],
        "spot_map": {},
    }


def test_peak_reset_rebases_after_emergency(monkeypatch: pytest.MonkeyPatch) -> None:
    emergency = live.RiskDecision(
        final_target=live.DEFENSE,
        exposure=1.0,
        action=live.ACTION_EMERGENCY,
        reason="test",
        events=[],
        cooldown_until=date(2020, 6, 19),
    )
    monkeypatch.setattr(audit, "ORIGINAL_RISK", lambda **kwargs: emergency)
    kwargs = _risk_kwargs()
    audit.peak_reset(**kwargs)
    assert kwargs["state"]["peak_equity"] == pytest.approx(1000.0)


def test_cb_off_keeps_target_when_emergency(monkeypatch: pytest.MonkeyPatch) -> None:
    emergency = live.RiskDecision(
        final_target=live.DEFENSE,
        exposure=1.0,
        action=live.ACTION_EMERGENCY,
        reason="test",
        events=[],
        cooldown_until=date(2020, 6, 19),
    )
    monkeypatch.setattr(audit, "ORIGINAL_RISK", lambda **kwargs: emergency)
    kwargs = _risk_kwargs()
    decision = audit.cb_off(**kwargs)
    assert decision.final_target == "A"
    assert decision.action == "hold"
    assert decision.cooldown_until is None


def test_fmt_year_counts_is_readable() -> None:
    assert audit.fmt_year_counts({"2022": 41, "2023": 55}) == "2022:41、2023:55"
    assert audit.fmt_year_counts(None) == "—"
