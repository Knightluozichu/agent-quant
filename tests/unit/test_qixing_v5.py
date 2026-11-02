"""V5 risk adapter tests: only the post-circuit-breaker peak may change."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from scripts import qixing_v5 as v5
from scripts import risk_overrides as ro


def _kwargs(*, holding: str = "A", action: str = ro.ACTION_EMERGENCY) -> dict[str, object]:
    decision = ro.RiskDecision(
        final_target=ro.DEFENSE_SEQ[-1] if action == ro.ACTION_EMERGENCY else "A",
        exposure=1.0,
        action=action,
        reason="test",
        events=[],
        cooldown_until=date(2020, 6, 19),
    )

    def fake_assess(**_: object) -> ro.RiskDecision:
        return decision

    return {
        "target": "A",
        "holding": holding,
        "state": {
            "cash": 0.0,
            "shares": 100.0,
            "holding": holding,
            "peak_equity": 2000.0,
        },
        "data": {
            "A": pd.DataFrame({"trade_date": [date(2020, 6, 19)], "open": [10.0], "close": [10.0]})
        },
        "td": date(2020, 6, 19),
        "idx_map": {"A": 0},
        "is_rebalance": False,
        "common_dates": [date(2020, 6, 19)],
        "spot_map": {},
        "_assess": fake_assess,
    }


def test_v5_rebases_peak_after_circuit_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _kwargs()
    monkeypatch.setattr(v5.ro, "assess", kwargs.pop("_assess"))
    decision = v5.assess_v5(**kwargs)
    assert decision.action == ro.ACTION_EMERGENCY
    # 100 shares * 10.0 = 1000; stale peak 2000 must be rebased.
    assert kwargs["state"]["peak_equity"] == pytest.approx(1000.0)


def test_v5_does_not_touch_peak_on_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _kwargs(action=ro.ACTION_HOLD)
    monkeypatch.setattr(v5.ro, "assess", kwargs.pop("_assess"))
    v5.assess_v5(**kwargs)
    assert kwargs["state"]["peak_equity"] == 2000.0


def test_v5_does_not_change_decision_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _kwargs()
    fake = kwargs.pop("_assess")
    monkeypatch.setattr(v5.ro, "assess", fake)
    decision = v5.assess_v5(**kwargs)
    assert decision.final_target == ro.DEFENSE_SEQ[-1]
    assert decision.exposure == 1.0
    assert decision.action == ro.ACTION_EMERGENCY
    assert decision.cooldown_until == date(2020, 6, 19)


def test_v5_config_hash_is_stable() -> None:
    assert v5.config_hash() == v5.CONFIG_HASH
    assert len(v5.CONFIG_HASH) == 64


def test_strategy_mode_v5_identity(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import live_signal

    mode_file = tmp_path / "strategy_mode.json"
    monkeypatch.setattr(live_signal, "STRATEGY_MODE_FILE", mode_file)
    monkeypatch.delenv("QIXING_STRATEGY_MODE", raising=False)
    live_signal.set_strategy_mode("V5")
    assert live_signal.get_strategy_mode() == "V5"
    assert live_signal.active_strategy_id() == v5.STRATEGY_ID
    assert live_signal.active_config_hash() == v5.CONFIG_HASH
    assert live_signal.strategy_identity("V4")[1] == live_signal.v4.STRATEGY_ID
