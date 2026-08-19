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
    state_file.write_text('{"initial_capital": 100000, "cash": 100000}', encoding="utf-8")
    monkeypatch.setattr(live_signal, "STATE_FILE", state_file)

    state = live_signal.load_state()

    assert state["v4_state"]["strategy_id"] == qixing_v4.STRATEGY_ID
    assert state["v4_state"]["candidate_history"] == []
    assert state["last_decision"] is None


@pytest.mark.unit
def test_confirm_v4_order_validates_nested_target_and_records_rotation(monkeypatch, tmp_path):
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
        live_signal.confirm_order(None, {"code": "518880", "shares": 150, "price": 10.0})


@pytest.mark.unit
def test_account_value_suspension_fallback_uses_no_future_data():
    """停牌回退只允许用 td 当天或之前的最后收盘, 不得偷看未来行."""
    df = pd.DataFrame(
        {
            "trade_date": [
                date(2026, 8, 7),
                date(2026, 8, 10),  # td 前最后已知
                date(2026, 8, 12),  # td 之后的未来行 (停牌期间不应可见)
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


# --------------------------------------------------------------------------- #
# I-V4A-01: 待确认订单只抑制新交易信号, 不得阻塞风控监控与熔断告警
# --------------------------------------------------------------------------- #
def _make_run_env(monkeypatch, tmp_path, *, risk, overlay, bark_calls, notify_calls):
    """为 run() 搭建全打桩环境, 返回写入磁盘的 state 路径."""
    today = live_signal._today_sh()
    trading_dates = [today - timedelta(days=i) for i in range(200, -1, -1)]

    state = {
        "_version": 0,
        "initial_capital": 100000.0,
        "cash": 5000.0,
        "holding": "518880",
        "shares": 1000,
        "entry_price": 5.0,
        "entry_date": "2026-08-01",
        "peak_equity": 100000.0,
        "risk_exposure": 1.0,
        "cooldown_until": None,
        "trade_log": [],
        "risk_log": [],
        "pending_order": {
            "order_id": "old-order-1",
            "date": str(today - timedelta(days=1)),
            "status": "pending",
            "sell": {"code": "518880", "shares": 1000},
            "buy": {"code": "513100", "shares": 500},
        },
        "v4_state": live_signal.default_v4_state(),
        "last_run_date": None,
    }
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(live_signal, "STATE_FILE", state_file)
    monkeypatch.setattr(live_signal, "STATE_TMP_FILE", tmp_path / "state.json.tmp")
    monkeypatch.setattr(live_signal, "LOCK_FILE", tmp_path / "state.lock")
    monkeypatch.setattr(live_signal, "SNAPSHOT_DIR", tmp_path / "snapshots")

    fake_data = {"518880": pd.DataFrame({"trade_date": trading_dates, "close": [5.0] * 201})}
    monkeypatch.setattr(live_signal, "load_data", lambda: fake_data)
    monkeypatch.setattr(live_signal, "update_data", lambda d: d)
    monkeypatch.setattr(live_signal, "get_strategy_mode", lambda: "V4")
    monkeypatch.setattr(live_signal, "is_trading_day", lambda d: True)
    spot = {"518880": {"price": 5.0, "prev_close": 5.0}}
    monkeypatch.setattr(live_signal, "_fetch_tencent_spot", lambda: spot)
    monkeypatch.setattr(live_signal, "validate_realtime_data", lambda s: (True, ""))
    monkeypatch.setattr(live_signal, "inject_realtime", lambda d, s: d)
    monkeypatch.setattr(live_signal, "get_trading_dates", lambda d: trading_dates)
    monkeypatch.setattr(live_signal, "build_etf_data_at_date", lambda d, td: {})
    monkeypatch.setattr(
        live_signal,
        "select_target",
        lambda d, idx, holding: ("513100", [("513100", 0.10)], 0.10, False),
    )
    monkeypatch.setattr(live_signal, "evaluate_v4_overlay", lambda **kw: overlay)
    monkeypatch.setattr(live_signal, "account_value", lambda s, d, td: 100000.0)
    monkeypatch.setattr(live_signal, "check_data_availability", lambda d, td: (True, []))
    monkeypatch.setattr(live_signal, "price_on", lambda d, code, td: 5.0)
    monkeypatch.setattr(live_signal, "risk_assess", lambda **kw: risk)
    for fn in ("print_header", "print_account", "print_momentum_board", "print_orders"):
        monkeypatch.setattr(live_signal, fn, lambda *a, **kw: None)
    monkeypatch.setattr(live_signal, "notify_risk_reduction", lambda r: None)
    monkeypatch.setattr(live_signal, "is_paper_mode", lambda: False)
    monkeypatch.setattr(
        live_signal, "push_bark", lambda *a, **kw: bark_calls.append((a, kw)) or True
    )
    monkeypatch.setattr(live_signal, "notify_trade", lambda *a, **kw: notify_calls.append((a, kw)))
    monkeypatch.setattr(live_signal, "notify_hold", lambda *a, **kw: None)
    return state_file, trading_dates


def _fake_overlay(*, is_rebalance: bool, target: str = "513100"):
    decision = SimpleNamespace(triggered=False, target=None, blocked_by=None, reasons=[])
    return {
        "target": target,
        "is_rebalance": is_rebalance,
        "raw_target": None,
        "signal_hits": 0,
        "scheduled_lock": False,
        "decision": decision,
        "factors": {},
        "mode": "V4",
        "base_target": target,
        "days_since_early_rotation": None,
        "proposed_target": None,
        "history": [],
    }


@pytest.mark.unit
def test_pending_order_suppresses_signal_but_risk_still_runs(monkeypatch, tmp_path):
    """pending 存在: 风控照常评估+落盘, 新订单被抑制且不推送交易指令."""
    risk = SimpleNamespace(
        events=[{"date": "2026-08-19", "type": "H1硬触发", "reason": "test"}],
        final_target="513100",
        action="none",
        exposure=1.0,
        cooldown_until=None,
    )
    bark_calls: list = []
    notify_calls: list = []
    state_file, trading_dates = _make_run_env(
        monkeypatch,
        tmp_path,
        risk=risk,
        overlay=_fake_overlay(is_rebalance=True),
        bark_calls=bark_calls,
        notify_calls=notify_calls,
    )

    rc = live_signal.run()

    assert rc == 0
    # 新订单未覆盖旧 pending
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["pending_order"]["order_id"] == "old-order-1"
    # 风控结论已落盘 (risk_log 记录事件)
    assert any(e["type"] == "H1硬触发" for e in saved["risk_log"])
    # 抑制日不写 last_run_date: 处理 pending 后当日可重跑补信号
    assert saved.get("last_run_date") != str(trading_dates[-1])
    # 推送了抑制告警, 未推送交易指令
    assert any("抑制" in str(c[0][0]) for c in bark_calls)
    assert notify_calls == []


@pytest.mark.unit
def test_pending_order_does_not_block_emergency_alert(monkeypatch, tmp_path):
    """pending + 组合熔断: 必须发 timeSensitive 熔断告警且返回非零."""
    risk = SimpleNamespace(
        events=[{"date": "2026-08-19", "type": "组合熔断", "reason": "dd=-31%"}],
        final_target="511880",
        action=live_signal.ACTION_EMERGENCY,
        exposure=1.0,
        cooldown_until=None,
    )
    bark_calls: list = []
    notify_calls: list = []
    state_file, _ = _make_run_env(
        monkeypatch,
        tmp_path,
        risk=risk,
        overlay=_fake_overlay(is_rebalance=False, target="513100"),
        bark_calls=bark_calls,
        notify_calls=notify_calls,
    )

    rc = live_signal.run()

    assert rc == 1
    emergency = [c for c in bark_calls if c[1].get("level") == "timeSensitive"]
    assert emergency, "熔断被 pending 阻塞时必须升级告警"
    assert "熔断" in str(emergency[0][0][0])
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["pending_order"]["order_id"] == "old-order-1"
    assert notify_calls == []


@pytest.mark.unit
def test_pending_order_non_rebalance_day_records_risk(monkeypatch, tmp_path):
    """pending + 非调仓日: 风控评估照常, 状态正常落盘, 返回 0."""
    risk = SimpleNamespace(
        events=[],
        final_target=None,
        action="none",
        exposure=1.0,
        cooldown_until=None,
    )
    bark_calls: list = []
    notify_calls: list = []
    state_file, trading_dates = _make_run_env(
        monkeypatch,
        tmp_path,
        risk=risk,
        overlay=_fake_overlay(is_rebalance=False),
        bark_calls=bark_calls,
        notify_calls=notify_calls,
    )

    rc = live_signal.run()

    assert rc == 0
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved.get("last_run_date") == str(trading_dates[-1])
    assert notify_calls == []
