"""H6 fixed-rule research: causal execution and fail-closed boundaries."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scripts import live_signal as live
from scripts import run_qixing_v3 as rq
from scripts.review_rotation_20260916 import DefensiveOverride, replay_next_open
from scripts.review_v4_reversal_20261020 import (
    RULES,
    ReversalFilter,
    assert_evaluation_window,
    audit_ledger,
    is_weak,
    load_frozen_data,
)
from scripts.risk_overrides import RiskDecision


def fixture_bars():
    dates = pd.bdate_range("2025-01-01", periods=170).date.tolist()
    prices = np.r_[np.linspace(100, 130, 131), np.linspace(129, 110, 39)]
    data = {
        code: pd.DataFrame(
            {
                "trade_date": dates,
                "open": prices + 1,
                "close": prices,
                "high": prices + 2,
                "low": prices - 1,
                "volume": 10000,
            }
        )
        for code in [*live.ETF_POOL, live.DEFENSE]
    }
    data[live.DEFENSE].loc[:, ["open", "close", "high", "low"]] = 100.0
    return dates, data


def force_gold(monkeypatch):
    monkeypatch.setattr(live, "select_target", lambda *a: ("518880", [("518880", 1)], [], []))
    monkeypatch.setattr(
        live, "risk_assess", lambda **kw: RiskDecision("518880", 1.0, "hold", "fixture")
    )


def test_fixed_weakness_requires_both_conditions_and_enough_history():
    fast = RULES[0]
    assert not is_weak(np.arange(100.0, 121), fast)
    assert is_weak(np.r_[np.arange(100.0, 120), 110.0, 109.0, 108.0], fast)
    # r3 negative but close still above MA10 -> not weak.
    assert not is_weak(np.r_[np.ones(7) * 50, 110, 109, 108, 107], fast)
    # Below MA10 but r3 positive -> not weak.
    assert not is_weak(np.r_[np.ones(7) * 120, 90, 91, 92, 93], fast)
    with pytest.raises(ValueError, match="history"):
        is_weak(np.ones(3), fast)
    for bad in [np.nan, 0, -1, np.inf]:
        with pytest.raises(ValueError, match="price"):
            is_weak(np.r_[np.ones(20), bad], fast)


def test_exit_preserves_existing_rotation_and_defense():
    dates, data = fixture_bars()
    filt = ReversalFilter(data, RULES[0])
    td = dates[-1]
    assert filt(td, "518880", "518880", False).target == live.DEFENSE
    assert filt(td, "518880", "159985", True).target == "159985"
    assert not filt(td, live.DEFENSE, live.DEFENSE, False).force_rebalance
    with pytest.raises(ValueError, match="observation"):
        filt(date(2030, 1, 1), "518880", "518880", False)


def test_gate_only_applies_to_an_executable_entry():
    dates, data = fixture_bars()
    filt = ReversalFilter(data, RULES[1])
    assert filt(dates[-1], None, "518880", False).target == "518880"
    assert filt(dates[-1], None, "518880", True).target == live.DEFENSE


def test_noop_filter_is_baseline_identical():
    dates, data = fixture_bars()
    kw = {"start": dates[130], "end": dates[-1], "grid": set(dates[130::5])}
    baseline = replay_next_open(data, **kw)
    noop = replay_next_open(
        data, **kw, target_filter=lambda td, holding, target, allowed: DefensiveOverride(target)
    )
    pd.testing.assert_frame_equal(baseline["curve"], noop["curve"])
    for key in ["trades", "metrics", "decisions", "final_state", "unfilled_next_open"]:
        assert baseline[key] == noop[key]


def test_reversal_uses_next_open_and_cost_ledger(monkeypatch):
    force_gold(monkeypatch)
    dates, data = fixture_bars()
    result = replay_next_open(
        data,
        start=dates[130],
        end=dates[140],
        grid={dates[130]},
        target_filter=ReversalFilter(data, RULES[0]),
        cost_multiplier=2,
    )
    events = result["research_events"]
    assert events and events[0]["reason"] == "reversal_exit"
    sell = next(t for t in result["trades"] if t["action"] == "sell")
    signal = date.fromisoformat(sell["signal_date"])
    fill = date.fromisoformat(sell["date"])
    assert fill == dates[dates.index(signal) + 1]
    assert sell["price"] == data["518880"].set_index("trade_date").loc[fill, "open"]
    assert sell["price"] != data["518880"].set_index("trade_date").loc[signal, "close"]
    stats = audit_ledger(result, data, cost_multiplier=2)
    assert stats["total_cost"] > 0
    assert stats["max_accounting_error"] < 1e-7
    assert all(t.get("shares", 100) % 100 == 0 for t in result["trades"])
    assert all(t["signal_date"] < t["date"] for t in result["trades"])
    # Even an exit is not allowed to fill at the final signal day's close.
    last_day = signal
    pending = replay_next_open(
        data,
        start=dates[130],
        end=last_day,
        grid={dates[130]},
        target_filter=ReversalFilter(data, RULES[0]),
    )
    assert pending["unfilled_next_open"][0] == live.DEFENSE
    assert not any(t["action"] == "sell" for t in pending["trades"])


def test_guard_replay_prefix_and_future_perturbation(monkeypatch):
    force_gold(monkeypatch)
    dates, data = fixture_bars()
    cut = dates[140]
    kw = {"start": dates[130], "grid": set(dates[130::5])}
    prefix = replay_next_open(data, end=cut, **kw, target_filter=ReversalFilter(data, RULES[1]))
    changed = {c: f.copy() for c, f in data.items()}
    for frame in changed.values():
        frame.loc[frame.trade_date > cut, ["open", "close", "high", "low"]] *= 4
    full = replay_next_open(
        changed, end=dates[-1], **kw, target_filter=ReversalFilter(changed, RULES[1])
    )
    pd.testing.assert_frame_equal(
        full["curve"].query("trade_date <= @cut").reset_index(drop=True), prefix["curve"]
    )
    assert [t for t in full["trades"] if t["date"] <= str(cut)] == prefix["trades"]
    assert [e for e in full["research_events"] if e["date"] <= str(cut)] == prefix[
        "research_events"
    ]


def test_filter_cannot_add_risk_or_force_risk_entry():
    dates, data = fixture_bars()
    kw = {"start": dates[130], "end": dates[131], "grid": set()}
    with pytest.raises(ValueError, match="defensive"):
        replay_next_open(
            data,
            **kw,
            target_filter=lambda *a: DefensiveOverride("159985", True, "bad"),
        )


@pytest.mark.parametrize(
    "start,end",
    [("2023-01-01", "2023-02-01"), ("2022-12-30", "2025-01-03"), ("2024-12-31", "2025-02-01")],
)
def test_locked_windows_are_rejected(start, end):
    with pytest.raises(ValueError, match="locked"):
        assert_evaluation_window(date.fromisoformat(start), date.fromisoformat(end))


def test_valid_windows_and_reversed_window():
    assert_evaluation_window(date(2025, 1, 1), date(2026, 10, 19))
    with pytest.raises(ValueError, match="window"):
        assert_evaluation_window(date(2026, 1, 1), date(2025, 1, 1))


def test_manifest_hash_fails_closed(tmp_path):
    import json

    (tmp_path / "bars").mkdir()
    (tmp_path / "bars" / "518880.parquet").write_bytes(b"not parquet")
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"requested_cutoff": "2026-10-19", "assets": {"518880": {"sha256": "bad"}}})
    )
    with pytest.raises(ValueError, match="hash"):
        load_frozen_data(tmp_path)


def test_independent_ledger_rejects_corrupt_fills(monkeypatch):
    force_gold(monkeypatch)
    dates, data = fixture_bars()
    result = replay_next_open(data, start=dates[130], end=dates[132], grid={dates[130]})
    result["trades"][0]["shares"] += 1
    with pytest.raises(ValueError, match=r"lot|accounting"):
        audit_ledger(result, data, cost_multiplier=1)
    assert pytest.approx(0.0015) == rq.FEE + rq.SLIPPAGE


def test_gate_blocks_weak_buy_without_moving_execution_day(monkeypatch):
    force_gold(monkeypatch)
    dates, data = fixture_bars()
    result = replay_next_open(
        data,
        start=dates[140],
        end=dates[143],
        grid={dates[140]},
        target_filter=ReversalFilter(data, RULES[1]),
    )
    assert result["trades"][0]["code"] == live.DEFENSE
    assert result["trades"][0]["signal_date"] == str(dates[140])
    assert result["trades"][0]["date"] == str(dates[141])
    assert result["research_events"][0]["reason"] == "weak_entry_blocked"
    assert result["final_state"]["v4_state"]["last_early_rotation_date"] is None
    audit_ledger(result, data, cost_multiplier=1)


def test_incomplete_calendar_cannot_pass_even_with_good_diagnostic_metrics():
    from scripts.review_v4_reversal_20261020 import verdict

    rows = []
    for pool in ("with_oil", "without_oil"):
        for window in ("2025", "2026"):
            for cost in (1, 2):
                for name in ("V4", *[r.name for r in RULES]):
                    rows.append(
                        {
                            "pool": pool,
                            "window": window,
                            "cost": cost,
                            "strategy": name,
                            "calendar_status": "complete"
                            if window == "2025"
                            else "incomplete_diagnostic",
                            "max_drawdown": -0.3 if name == "V4" else -0.1,
                            "total_return": 0.1,
                        }
                    )
    outputs = verdict(pd.DataFrame(rows))
    assert all(r["status"] == "blocked_incomplete_calendar" for r in outputs.values())
    assert all(r["passing_cells"] == 4 for r in outputs.values())
