"""实时信号风控通知测试."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import live_signal  # noqa: E402
import qixing_v4  # noqa: E402


@pytest.mark.unit
def test_full_exposure_risk_warning_does_not_send_downsize_bark(monkeypatch):
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        live_signal,
        "push_bark",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    risk = SimpleNamespace(
        action="hold",
        exposure=1.0,
        events=[{"type": "改进-H1/H2降仓", "reason": "day=-5.7%"}],
    )

    live_signal.notify_risk_reduction(risk)

    assert calls == []


@pytest.mark.unit
def test_actual_reduction_sends_one_downsize_bark(monkeypatch):
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        live_signal,
        "push_bark",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    risk = SimpleNamespace(
        action="hold",
        exposure=0.7,
        events=[{"type": "改进-H1/H2降仓", "reason": "day=-5.7%"}],
    )

    live_signal.notify_risk_reduction(risk)

    assert len(calls) == 1
    assert calls[0][0][0] == "⚠️ 自动降仓至70%仓位"


@pytest.mark.unit
def test_v4_confirmation_history_requires_two_contiguous_trading_days():
    dates = ["2026-08-10", "2026-08-11", "2026-08-12"]
    history, hits = qixing_v4.update_candidate_history(
        [], trade_date=dates[0], raw_target="518880", trading_dates=dates
    )
    assert hits == 1

    history, hits = qixing_v4.update_candidate_history(
        history, trade_date=dates[2], raw_target="518880", trading_dates=dates
    )
    assert hits == 1
    assert history == [{"date": "2026-08-12", "target": "518880"}]

    history, hits = qixing_v4.update_candidate_history(
        history, trade_date=dates[1], raw_target="518880", trading_dates=dates
    )
    history, hits = qixing_v4.update_candidate_history(
        history, trade_date=dates[2], raw_target="518880", trading_dates=dates
    )
    assert hits == 2


@pytest.mark.unit
def test_load_state_migrates_v4_runtime(monkeypatch, tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        '{"initial_capital": 100000, "cash": 100000}', encoding="utf-8"
    )
    monkeypatch.setattr(live_signal, "STATE_FILE", state_file)

    state = live_signal.load_state()

    assert state["v4_state"]["strategy_id"] == qixing_v4.STRATEGY_ID
    assert state["v4_state"]["candidate_history"] == []
    assert state["last_decision"] is None


@pytest.mark.unit
def test_confirm_v4_order_validates_nested_target_and_records_rotation(
    monkeypatch, tmp_path
):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(live_signal, "STATE_FILE", state_file)
    monkeypatch.setattr(live_signal, "STATE_TMP_FILE", tmp_path / "state.json.tmp")
    monkeypatch.setattr(live_signal, "LOCK_FILE", tmp_path / "state.lock")
    state = live_signal.default_state(100_000)
    state.update({"holding": "161226", "shares": 1000, "entry_price": 5.0})
    state["pending_order"] = {
        "order_id": "order-1",
        "expected_state_version": 0,
        "date": "2026-08-12",
        "status": "pending",
        "decision_kind": "v4_early_rotation",
        "sell": {"code": "161226", "shares": 1000},
        "buy": {"code": "518880", "shares": 500},
    }
    state_file.write_text(__import__("json").dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="不匹配"):
        live_signal.confirm_order(
            {"shares": 1000, "price": 5.0},
            {"code": "513100", "shares": 500, "price": 10.0},
            order_id="order-1",
            expected_state_version=0,
        )

    confirmed = live_signal.confirm_order(
        {"shares": 1000, "price": 5.0},
        {"code": "518880", "shares": 500, "price": 10.0},
        order_id="order-1",
        expected_state_version=0,
        idempotency_key="request-1",
    )
    assert confirmed["holding"] == "518880"
    assert confirmed["pending_order"]["status"] == "confirmed"
    assert confirmed["v4_state"]["last_early_rotation_date"] == "2026-08-12"
    assert confirmed["confirm_receipts"]["request-1"]["order_id"] == "order-1"
