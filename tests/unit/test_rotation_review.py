"""Causality checks for the next-open, fixed-grid review."""

from datetime import date

import pandas as pd
from scripts import live_signal as live
from scripts.review_rotation_20260916 import replay_next_open


def test_next_open_replay_is_causal_and_prefix_invariant():
    dates = pd.bdate_range("2025-01-01", periods=180).date.tolist()
    data = {}
    for n, code in enumerate([*live.ETF_POOL, live.DEFENSE]):
        prices = [10 + i * (n + 1) / 1000 for i in range(len(dates))]
        data[code] = pd.DataFrame(
            {
                "trade_date": dates,
                "open": prices,
                "close": prices,
                "high": prices,
                "low": prices,
                "volume": 10000,
            }
        )
    grid = set(dates[130::5])
    result = replay_next_open(data, start=dates[130], end=dates[-1], grid=grid)
    assert result["trades"]
    assert all(
        date.fromisoformat(t["signal_date"]) < date.fromisoformat(t["date"])
        for t in result["trades"]
    )
    assert (result["curve"]["cash"] >= 0).all()
    cut = dates[160]
    prefix = replay_next_open(data, start=dates[130], end=cut, grid=grid)
    pd.testing.assert_frame_equal(
        result["curve"].query("trade_date <= @cut").reset_index(drop=True), prefix["curve"]
    )
    changed = {k: v.copy() for k, v in data.items()}
    for frame in changed.values():
        frame.loc[frame.trade_date > cut, ["open", "close", "high", "low"]] *= 3
    perturbed = replay_next_open(changed, start=dates[130], end=dates[-1], grid=grid)
    pd.testing.assert_frame_equal(
        perturbed["curve"].query("trade_date <= @cut").reset_index(drop=True), prefix["curve"]
    )


def test_replay_persists_production_confirmation_contract(monkeypatch):
    dates = pd.bdate_range("2025-01-01", periods=140).date.tolist()
    data = {
        code: pd.DataFrame(
            {
                "trade_date": dates,
                "open": 10.0,
                "close": 10.0,
                "high": 10.0,
                "low": 10.0,
                "volume": 10000,
            }
        )
        for code in [*live.ETF_POOL, live.DEFENSE]
    }
    monkeypatch.setattr(
        live.v4, "raw_candidate", lambda *args: live.v4.FullPoolDecision(True, "518880")
    )
    original = live.evaluate_v4_overlay
    hits = []
    last_inputs = {}

    def observe(**kwargs):
        last_inputs.update(kwargs)
        result = original(**kwargs)
        hits.append(result["signal_hits"])
        return result

    monkeypatch.setattr(live, "evaluate_v4_overlay", observe)
    replay_next_open(data, start=dates[130], end=dates[132], grid=set())
    assert hits == [1, 2, 2]
    repeated = original(**last_inputs)
    assert repeated["signal_hits"] == 2
    assert len(repeated["history"]) == 2  # Same day is not another confirmation.
    monkeypatch.setattr(live, "POOL_VERSION", "different-research-pool")
    changed_pool = original(**last_inputs)
    assert changed_pool["signal_hits"] == 1
    assert len(changed_pool["history"]) == 1


def test_research_early_lock_starts_on_fill_and_v3_disables_handoff(monkeypatch):
    dates = pd.bdate_range("2025-01-01", periods=140).date.tolist()
    data = {
        code: pd.DataFrame(
            {
                "trade_date": dates,
                "open": 10.0,
                "close": 10.0,
                "high": 10.0,
                "low": 10.0,
                "volume": 10000,
            }
        )
        for code in [*live.ETF_POOL, live.DEFENSE]
    }
    monkeypatch.setattr(
        live.v4, "raw_candidate", lambda *args: live.v4.FullPoolDecision(True, "518880")
    )
    monkeypatch.setattr(
        live.v4,
        "decide_full_pool_handoff",
        lambda **kwargs: live.v4.FullPoolDecision(kwargs["signal_hits"] >= 2, "518880"),
    )
    result = replay_next_open(data, start=dates[130], end=dates[132], grid=set())
    assert result["trades"][0]["signal_date"] == str(dates[131])
    assert result["trades"][0]["date"] == str(dates[132])
    assert result["final_state"]["v4_state"]["last_early_rotation_date"] == str(dates[132])
    assert result["decisions"][-1]["days_since_early_rotation"] == 0
    baseline = replay_next_open(data, start=dates[130], end=dates[132], grid=set(), mode="V3-G")
    assert baseline["trades"] == []
    assert baseline["final_state"]["v4_state"]["last_early_rotation_date"] is None
