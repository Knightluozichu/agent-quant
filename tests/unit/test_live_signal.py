"""实时信号风控通知测试."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
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
    state_file.write_text(json.dumps(state), encoding="utf-8")

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


# --------------------------------------------------------------------------- #
# 3a20ac2 审核修复的回归测试: 卖出零股 / 停牌回退无未来函数 / 时区 / 空集守卫
# --------------------------------------------------------------------------- #
def _patch_state_files(monkeypatch, tmp_path, state: dict) -> None:
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(live_signal, "STATE_FILE", state_file)
    monkeypatch.setattr(live_signal, "STATE_TMP_FILE", tmp_path / "state.json.tmp")
    monkeypatch.setattr(live_signal, "LOCK_FILE", tmp_path / "state.lock")
    state_file.write_text(json.dumps(state), encoding="utf-8")


@pytest.mark.unit
def test_confirm_order_allows_odd_lot_full_sell(monkeypatch, tmp_path):
    """A股卖出允许零股: 持仓150股一次性全卖150股应成功 (不做整百校验)."""
    state = live_signal.default_state(100_000)
    state.update({"holding": "161226", "shares": 150, "entry_price": 5.0, "cash": 0.0})
    state["pending_order"] = {
        "order_id": "order-odd",
        "date": "2026-08-12",
        "status": "pending",
        "sell": {"code": "161226", "shares": 150},
        "buy": None,
    }
    _patch_state_files(monkeypatch, tmp_path, state)

    confirmed = live_signal.confirm_order({"shares": 150, "price": 5.0}, None)

    expected_cash = 150 * 5.0 * (1 - live_signal.FEE - live_signal.SLIPPAGE)
    assert confirmed["cash"] == pytest.approx(expected_cash)
    assert confirmed["holding"] is None
    assert confirmed["shares"] == 0
    assert confirmed["pending_order"]["status"] == "confirmed"


@pytest.mark.unit
def test_confirm_order_rejects_non_round_lot_buy(monkeypatch, tmp_path):
    """买入仍必须100股整数倍: 150股买入被拒绝."""
    state = live_signal.default_state(100_000)
    state["pending_order"] = {
        "order_id": "order-buy",
        "date": "2026-08-12",
        "status": "pending",
        "sell": None,
        "buy": {"code": "518880", "shares": 100},
    }
    _patch_state_files(monkeypatch, tmp_path, state)

    with pytest.raises(ValueError, match="整数倍"):
        live_signal.confirm_order(
            None, {"code": "518880", "shares": 150, "price": 10.0}
        )


@pytest.mark.unit
def test_account_value_suspension_fallback_uses_no_future_data():
    """停牌回退只允许用 td 当天或之前的最后收盘, 不得偷看未来行."""
    df = pd.DataFrame(
        {
            "trade_date": [
                date(2026, 8, 7),
                date(2026, 8, 10),   # td 前最后已知
                date(2026, 8, 12),   # td 之后的未来行 (停牌期间不应可见)
            ],
            "close": [9.0, 10.0, 99.0],
        }
    )
    state = {"cash": 0.0, "holding": "518880", "shares": 100}
    td = date(2026, 8, 11)  # 停牌日, 当日无价

    value = live_signal.account_value(state, {"518880": df}, td)

    assert value == pytest.approx(100 * 10.0)  # 回退到 8-10 的收盘, 而非 99.0


@pytest.mark.unit
def test_inject_realtime_empty_data_does_not_silently_skip(capsys):
    """空 data / 无交易池标的: all() 恒真陷阱, 必须告警并原样返回 (fail-closed)."""
    result = live_signal.inject_realtime({})
    assert result == {}
    assert "无交易池行情数据" in capsys.readouterr().out

    df = pd.DataFrame({"trade_date": [date(2026, 8, 10)], "close": [1.0]})
    result = live_signal.inject_realtime({"999999": df})
    assert list(result.keys()) == ["999999"]
    assert "无交易池行情数据" in capsys.readouterr().out


@pytest.mark.unit
def test_today_sh_uses_asia_shanghai():
    """_today_sh 必须按 Asia/Shanghai (+08:00) 口径取日期, 与本地时区无关."""
    assert live_signal._SH_TZ.utcoffset(datetime(2026, 8, 17)) == timedelta(hours=8)
    assert live_signal._today_sh() == datetime.now(live_signal._SH_TZ).date()
