"""16:30 每日简报口径回归: 不得与 14:50 官方决策输出互相矛盾的调仓指令."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import daily_calibrate as dc  # noqa: E402
import live_signal as ls  # noqa: E402


def _state(**overrides):
    state = {
        "initial_capital": 100_000.0,
        "cash": 7.45,
        "holding": None,
        "shares": 0,
        "last_rebalance_date": "2026-11-05",
        "last_decision": None,
        "pending_order": None,
    }
    state.update(overrides)
    return state


def _official(
    target: str = "518880",
    trade_date: str = "2026-11-05",
    *,
    mode: str | None = None,
    config_hash: str | None = None,
) -> dict:
    return {
        "trade_date": trade_date,
        "mode": ls.get_strategy_mode() if mode is None else mode,
        "config_hash": ls.active_config_hash() if config_hash is None else config_hash,
        "final_target": target,
        "factors": {target: {"slow_momentum": 0.12}},
    }


def _pending(*, status: str = "pending") -> dict:
    return {
        "order_id": "order-gold",
        "date": "2026-11-05",
        "status": status,
        "schema_version": 2,
        "ticket_id": "tkt_20261105_abc123",
        "legs": [
            {
                "leg_id": "buy_518880",
                "side": "buy",
                "code": "518880",
                "name": "黄金ETF",
                "shares": 11400,
                "reference_price": 8.417,
                "limit_low": 8.375,
                "limit_high": 8.459,
                "estimated_amount": 95953.8,
                "status": "PENDING",
            }
        ],
        "execution_windows": {
            "same_day": {"date": "2026-11-05", "start": "14:50", "end": "15:00"},
            "next_open": {"date": "2026-11-06", "start": "09:25", "end": "09:30"},
        },
    }


# --------------------------------------------------------------------------- #
# official_decision: 只认当天(交易日)/最后交易日(非交易日)且模式配置一致
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_official_decision_accepts_today_snapshot_on_trading_day():
    official = _official(trade_date="2026-11-05")
    state = _state(last_decision=official)

    assert (
        dc.official_decision(
            state, today=date(2026, 11, 5), latest=date(2026, 11, 4), is_td=True
        )
        is official
    )


@pytest.mark.unit
def test_official_decision_rejects_yesterday_snapshot_on_trading_day():
    """16:30 数据 latest=T-1 时, 昨天的官方快照不能冒充今天."""
    state = _state(last_decision=_official(trade_date="2026-11-04"))

    assert (
        dc.official_decision(
            state, today=date(2026, 11, 5), latest=date(2026, 11, 4), is_td=True
        )
        is None
    )


@pytest.mark.unit
def test_official_decision_accepts_latest_snapshot_on_holiday():
    official = _official(trade_date="2026-10-07")
    state = _state(last_decision=official)

    assert (
        dc.official_decision(
            state, today=date(2026, 10, 8), latest=date(2026, 10, 7), is_td=False
        )
        is official
    )


@pytest.mark.unit
def test_official_decision_rejects_mode_or_config_mismatch(monkeypatch):
    monkeypatch.setattr(ls, "get_strategy_mode", lambda: "V5")
    monkeypatch.setattr(ls, "active_config_hash", lambda: "cfg-v5")

    state = _state(last_decision=_official(mode="V4", config_hash="cfg-v5"))
    assert (
        dc.official_decision(
            state, today=date(2026, 11, 5), latest=date(2026, 11, 4), is_td=True
        )
        is None
    )

    state = _state(last_decision=_official(mode="V5", config_hash="stale-hash"))
    assert (
        dc.official_decision(
            state, today=date(2026, 11, 5), latest=date(2026, 11, 4), is_td=True
        )
        is None
    )


@pytest.mark.unit
def test_official_decision_rejects_target_outside_pool():
    state = _state(last_decision=_official(target="000001"))
    assert (
        dc.official_decision(
            state, today=date(2026, 11, 5), latest=date(2026, 11, 4), is_td=True
        )
        is None
    )


# --------------------------------------------------------------------------- #
# build_brief: 有官方快照只复述; 有待执行票不生成新指令; 无快照只给观察预览
# --------------------------------------------------------------------------- #
def _brief(state: dict, **overrides) -> str:
    kwargs = {
        "state": state,
        "data": {},
        "latest": date(2026, 11, 4),
        "next_rb": date(2026, 11, 12),
        "days_left": 3,
        "today": date(2026, 11, 5),
        "is_td": True,
        "total": 97_848.05,
    }
    kwargs.update(overrides)
    return "\n".join(dc.build_brief(**kwargs))


@pytest.mark.unit
def test_brief_with_official_restates_only():
    state = _state(last_decision=_official())

    text = _brief(state)

    assert "14:50官方决策" in text
    assert "本简报不重复下单" in text
    assert "易淘金" not in text
    assert "买入" not in text


@pytest.mark.unit
def test_brief_defense_target_with_no_holding_says_cash_not_hold_cash():
    state = _state(last_decision=_official(target=ls.DEFENSE))

    text = _brief(state)

    assert "保持空仓（货币基金）" in text
    assert "继续持有 空仓" not in text


@pytest.mark.unit
def test_brief_warns_manual_holding_deviation_without_new_order():
    state = _state(
        holding="161226",
        shares=62_300,
        last_decision=_official(target="518880"),
    )

    text = _brief(state)

    assert "手动成交" in text
    assert "下次调仓按实际持仓计算" in text
    assert "易淘金" not in text


@pytest.mark.unit
def test_brief_with_pending_and_official_suppresses_new_signal():
    state = _state(last_decision=_official(), pending_order=_pending())

    text = _brief(state)

    assert "待执行票" in text
    assert "旧票未处理前，14:50 新信号会被抑制" in text
    assert "易淘金" not in text


@pytest.mark.unit
def test_brief_with_pending_and_no_official_does_not_preview():
    state = _state(pending_order=_pending())

    text = _brief(state)

    assert "今日无有效官方快照" in text
    assert "请先处理" in text
    assert "易淘金" not in text
    assert "非交易预览" not in text


@pytest.mark.unit
def test_brief_without_official_is_labeled_observation_only(monkeypatch):
    monkeypatch.setattr(dc, "build_idx_map", lambda data, latest: {})
    monkeypatch.setattr(
        dc.rq, "select_target", lambda data, idx_map, holding: ("501018", [], 0.057, False)
    )

    text = _brief(_state())

    assert "非交易预览" in text
    assert "501018" in text
    assert "不构成买入指令" in text
    assert "易淘金搜索代码" not in text


@pytest.mark.unit
def test_brief_skips_preview_when_close_data_not_injected(monkeypatch):
    monkeypatch.setattr(dc, "build_idx_map", lambda data, latest: {})
    monkeypatch.setattr(
        dc.rq, "select_target", lambda data, idx_map, holding: ("518880", [], 0.12, False)
    )

    text = _brief(_state())

    assert "当日收盘行情未注入" in text
    assert "数据基于 2026-11-04" in text


# --------------------------------------------------------------------------- #
# 调仓日展示: 过期网格不能再显示成"就是今天"
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_next_rebalance_line_marks_overdue_grid():
    text = dc._next_rebalance_line(date(2026, 10, 29), -5)
    assert "已过期 5 个交易日" in text
    assert "今天" not in text


@pytest.mark.unit
def test_next_rebalance_line_today():
    assert "（今天）" in dc._next_rebalance_line(date(2026, 11, 5), 0)


# --------------------------------------------------------------------------- #
# pending_active: 终态票不再提醒
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_pending_active_only_for_open_status():
    assert dc.pending_active(_state(pending_order=_pending())) is not None
    assert dc.pending_active(_state(pending_order=_pending(status="skipped"))) is None
    assert dc.pending_active(_state(pending_order=_pending(status="confirmed"))) is None
    assert dc.pending_active(_state()) is None
