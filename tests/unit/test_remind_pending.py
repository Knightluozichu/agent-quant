"""Reminder escalation writes ticket state and is idempotent."""

from __future__ import annotations

import json
from datetime import date, datetime

from scripts import remind_pending, ticket


def _state(tmp_path, monkeypatch):
    state = remind_pending.live_signal.default_state(100000.0)
    pending = {
        "order_id": "abc123",
        "decision_id": "dec1",
        "date": "2026-11-02",
        "status": "pending",
        "sell": None,
        "buy": {
            "code": "501018",
            "name": "南方原油",
            "shares": 43300,
            "price": 2.27,
            "amount": 98291.0,
        },
        "reason": "test",
    }
    ticket.attach_ticket(pending, date(2026, 11, 2), next_date=date(2026, 11, 3))
    state["pending_order"] = pending
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(remind_pending.live_signal, "STATE_FILE", path)
    monkeypatch.setattr(remind_pending.live_signal, "STATE_TMP_FILE", tmp_path / "state.json.tmp")
    monkeypatch.setattr(remind_pending.live_signal, "LOCK_FILE", tmp_path / "state.lock")
    monkeypatch.setattr(remind_pending.ticket, "is_trading_day", lambda _: True)
    calls: list[tuple] = []
    monkeypatch.setattr(
        remind_pending, "push_bark", lambda *args, **kwargs: calls.append((args, kwargs)) or True
    )
    return path, calls


def test_same_day_reminder_marks_sent_and_is_idempotent(tmp_path, monkeypatch) -> None:
    path, calls = _state(tmp_path, monkeypatch)
    monkeypatch.setattr(
        remind_pending.ticket, "now_sh", lambda: datetime(2026, 11, 2, 14, 56, tzinfo=ticket.SH_TZ)
    )
    assert remind_pending.run("same_day_warn") == 0
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["pending_order"]["reminders"]["sent"]["same_day_warn"]
    assert len(calls) == 1
    assert remind_pending.run("same_day_warn") == 0
    assert len(calls) == 1


def test_expired_marks_unresolved_without_skipping(tmp_path, monkeypatch) -> None:
    path, calls = _state(tmp_path, monkeypatch)
    monkeypatch.setattr(
        remind_pending.ticket, "now_sh", lambda: datetime(2026, 11, 3, 9, 36, tzinfo=ticket.SH_TZ)
    )
    assert remind_pending.run("expired") == 0
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["pending_order"]["status"] == "expired_unresolved"
    assert saved["pending_order"]["status_history"][-1]["status"] == "EXPIRED_UNRESOLVED"
    assert len(calls) == 1
