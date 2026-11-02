"""Semi-automatic execution ticket helpers."""

from __future__ import annotations

from datetime import date, datetime

from scripts import ticket


def _pending() -> dict:
    return {
        "order_id": "abc123",
        "decision_id": "dec1",
        "date": "2026-11-02",
        "status": "pending",
        "sell": {
            "code": "518880",
            "name": "黄金ETF",
            "shares": 10900,
            "price": 7.934,
            "amount": 86480.0,
        },
        "buy": {
            "code": "501018",
            "name": "南方原油",
            "shares": 43300,
            "price": 2.27,
            "amount": 98291.0,
        },
        "reason": "test",
    }


def test_attach_ticket_builds_dual_windows_and_legs() -> None:
    pending = ticket.attach_ticket(_pending(), date(2026, 11, 2), next_date=date(2026, 11, 3))
    assert pending["schema_version"] == 2
    assert pending["ticket_id"].startswith("tkt_20261102_")
    assert pending["execution_windows"]["same_day"]["end"] == "15:00"
    assert pending["execution_windows"]["next_open"]["date"] == "2026-11-03"
    assert [leg["side"] for leg in pending["legs"]] == ["sell", "buy"]
    assert pending["legs"][1]["limit_low"] < pending["legs"][1]["reference_price"]
    assert pending["status_history"][0]["status"] == "PENDING"


def test_stage_due_honours_window_and_dedupe() -> None:
    pending = ticket.attach_ticket(_pending(), date(2026, 11, 2), next_date=date(2026, 11, 3))
    now = datetime(2026, 11, 2, 14, 56, tzinfo=ticket.SH_TZ)
    assert ticket.stage_due(pending, "same_day_warn", now)
    ticket.mark_reminder(pending, "same_day_warn", now)
    assert not ticket.stage_due(pending, "same_day_warn", now)
    assert not ticket.stage_due(
        pending, "same_day_warn", datetime(2026, 11, 2, 14, 54, tzinfo=ticket.SH_TZ)
    )


def test_snooze_blocks_reminders_and_caps_at_two() -> None:
    pending = ticket.attach_ticket(_pending(), date(2026, 11, 2), next_date=date(2026, 11, 3))
    now = datetime(2026, 11, 2, 14, 56, tzinfo=ticket.SH_TZ)
    assert ticket.snooze(pending, 10, now)
    assert not ticket.stage_due(pending, "same_day_warn", now)
    assert ticket.snooze(pending, 10, now)
    assert not ticket.snooze(pending, 10, now)
    assert pending["snooze_count"] == 2


def test_ticket_view_exposes_deadline_and_status() -> None:
    pending = ticket.attach_ticket(_pending(), date(2026, 11, 2), next_date=date(2026, 11, 3))
    view = ticket.ticket_view(pending, datetime(2026, 11, 2, 14, 51, tzinfo=ticket.SH_TZ))
    assert view is not None
    assert view["status"] == "PENDING"
    assert view["deadline"] == "2026-11-02 15:00"
    assert view["seconds_remaining"] > 0


def test_mark_status_appends_history() -> None:
    pending = ticket.attach_ticket(_pending(), date(2026, 11, 2), next_date=date(2026, 11, 3))
    ticket.mark_status(pending, "skipped", actor="user", reason="forgot")
    assert pending["status"] == "skipped"
    assert pending["status_history"][-1]["status"] == "SKIPPED"
