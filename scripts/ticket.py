"""Semi-automatic execution tickets: dual windows, deadlines, reminders.

No broker access, no order placement.  A ticket is a human-executable plan with
an auditable lifecycle.  Existing ``pending_order`` fields remain the source of
truth; the new ticket fields are additive and backward compatible.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

SH_TZ = ZoneInfo("Asia/Shanghai")
SAME_DAY_START = time(14, 50)
SAME_DAY_END = time(15, 0)
NEXT_OPEN_START = time(9, 25)
NEXT_OPEN_END = time(9, 30)
GRID_ANCHOR = date(2020, 6, 19)
REBALANCE_DAYS = 5

# 只保留四个关键节点: 盘前预告 / 同日收尾 / 次日开盘最后机会 / 过期。
# 15:02、15:10、09:15 三个中间阶段已移除, 避免同一张票反复推送。
STAGE_TIMES: dict[str, time] = {
    "pre": time(14, 45),
    "same_day_warn": time(14, 55),
    "next_open_final": time(9, 25),
    "expired": time(9, 35),
}
SAME_DAY_STAGES = {"same_day_warn"}
NEXT_OPEN_STAGES = {"next_open_final", "expired"}

_CALENDAR_CACHE: dict[str, Any] = {"at": 0.0, "dates": []}


def now_sh() -> datetime:
    return datetime.now(SH_TZ)


def trading_dates() -> list[date]:
    """A-share trading calendar, cached for one hour; weekday fallback on error."""
    import time as _time

    if _CALENDAR_CACHE["dates"] and _time.time() - _CALENDAR_CACHE["at"] < 3600:
        return list(_CALENDAR_CACHE["dates"])
    try:
        import akshare as ak

        frame = ak.tool_trade_date_hist_sina()
        dates = sorted(pd_to_date(value) for value in frame["trade_date"])
        if dates:
            _CALENDAR_CACHE["dates"] = dates
            _CALENDAR_CACHE["at"] = _time.time()
            return dates
    except Exception as exc:
        print(f"  ⚠️ 交易日历获取失败, 使用工作日回退: {exc}")
    # Fail-open only for scheduling; callers still verify data freshness.
    fallback = [date.today() + timedelta(days=offset) for offset in range(-30, 370)]
    return [item for item in fallback if item.weekday() < 5]


def pd_to_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def is_trading_day(td: date) -> bool:
    return td in set(trading_dates())


def next_trading_day(td: date) -> date:
    for item in trading_dates():
        if item > td:
            return item
    raise ValueError(f"No next trading day after {td}")


def is_rebalance_day(td: date) -> bool:
    dates = trading_dates()
    try:
        anchor_index = dates.index(GRID_ANCHOR)
    except ValueError:
        return td.weekday() == 4
    if td not in dates:
        return False
    return (dates.index(td) - anchor_index) % REBALANCE_DAYS == 0


def _limit_band(price: float) -> tuple[float, float]:
    tick = 0.01 if price >= 100 else 0.001
    band = max(tick, round(price * 0.005, 3))
    low = round(max(tick, price - band), 3)
    high = round(price + band, 3)
    return low, high


def _legs_from_pending(pending: dict[str, Any]) -> list[dict[str, Any]]:
    legs: list[dict[str, Any]] = []
    for side in ("sell", "buy"):
        item = pending.get(side)
        if not item:
            continue
        price = float(item.get("price", 0.0))
        low, high = _limit_band(price) if price > 0 else (0.0, 0.0)
        legs.append(
            {
                "leg_id": f"{side}_{item.get('code')}",
                "side": side,
                "code": item.get("code"),
                "name": item.get("name"),
                "shares": int(item.get("shares", 0)),
                "reference_price": price,
                "limit_low": low,
                "limit_high": high,
                "estimated_amount": float(item.get("amount", 0.0)),
                "status": "PENDING",
            }
        )
    return legs


def ticket_id_for(pending: dict[str, Any], signal_date: date) -> str:
    order_id = str(pending.get("order_id", ""))
    digest = hashlib.sha256(f"{signal_date}:{order_id}".encode()).hexdigest()[:6]
    return f"tkt_{signal_date.strftime('%Y%m%d')}_{digest}"


def attach_ticket(
    pending: dict[str, Any],
    signal_date: date,
    *,
    next_date: date | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Add/refresh ticket fields in place; safe to call on an old pending."""
    now = now or now_sh()
    if next_date is None:
        existing = (pending.get("execution_windows") or {}).get("next_open") or {}
        raw_next = existing.get("date")
        next_date = pd_to_date(raw_next) if raw_next else next_trading_day(signal_date)
    pending["schema_version"] = 2
    pending.setdefault("ticket_id", ticket_id_for(pending, signal_date))
    pending.setdefault("signal_date", str(signal_date))
    pending.setdefault(
        "execution_windows",
        {
            "same_day": {
                "date": str(signal_date),
                "start": SAME_DAY_START.strftime("%H:%M"),
                "end": SAME_DAY_END.strftime("%H:%M"),
            },
            "next_open": {
                "date": str(next_date),
                "start": NEXT_OPEN_START.strftime("%H:%M"),
                "end": NEXT_OPEN_END.strftime("%H:%M"),
            },
        },
    )
    pending.setdefault("legs", _legs_from_pending(pending))
    reminders = pending.setdefault("reminders", {})
    reminders.setdefault("sent", {})
    reminders.setdefault("count", 0)
    reminders.setdefault("last_at", None)
    pending.setdefault("snooze_until", None)
    pending.setdefault("snooze_count", 0)
    history = pending.setdefault("status_history", [])
    if not history:
        history.append(
            {
                "status": str(pending.get("status", "pending")).upper(),
                "at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "actor": "system",
                "reason": "ticket_created",
            }
        )
    return pending


def _window_date(pending: dict[str, Any], stage: str) -> date | None:
    windows = pending.get("execution_windows") or {}
    if stage in SAME_DAY_STAGES:
        raw = (windows.get("same_day") or {}).get("date")
    elif stage in NEXT_OPEN_STAGES:
        raw = (windows.get("next_open") or {}).get("date")
    else:
        return None
    return pd_to_date(raw) if raw else None


def stage_due(pending: dict[str, Any], stage: str, now: datetime | None = None) -> bool:
    if stage not in STAGE_TIMES:
        raise ValueError(f"unknown stage: {stage}")
    if stage == "pre":
        return False
    now = now or now_sh()
    if str(pending.get("status", "")).lower() not in {"pending", "partial"}:
        return False
    snooze_until = pending.get("snooze_until")
    if snooze_until:
        try:
            if now < datetime.fromisoformat(str(snooze_until)).replace(tzinfo=SH_TZ):
                return False
        except ValueError:
            pass
    target = _window_date(pending, stage)
    if target is None or now.date() != target:
        return False
    if now.time() < STAGE_TIMES[stage]:
        return False
    sent = (pending.get("reminders") or {}).get("sent") or {}
    return stage not in sent


def mark_reminder(pending: dict[str, Any], stage: str, now: datetime | None = None) -> None:
    now = now or now_sh()
    reminders = pending.setdefault("reminders", {})
    sent = reminders.setdefault("sent", {})
    sent[stage] = now.strftime("%Y-%m-%d %H:%M:%S")
    reminders["count"] = int(reminders.get("count", 0)) + 1
    reminders["last_at"] = now.strftime("%Y-%m-%d %H:%M:%S")


def mark_status(
    pending: dict[str, Any],
    status: str,
    *,
    actor: str,
    reason: str = "",
    now: datetime | None = None,
) -> None:
    now = now or now_sh()
    pending["status"] = status
    pending.setdefault("status_history", []).append(
        {
            "status": status.upper(),
            "at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "actor": actor,
            "reason": reason,
        }
    )


def snooze(pending: dict[str, Any], minutes: int = 10, now: datetime | None = None) -> bool:
    now = now or now_sh()
    if int(pending.get("snooze_count", 0)) >= 2:
        return False
    until = now + timedelta(minutes=minutes)
    pending["snooze_until"] = until.strftime("%Y-%m-%d %H:%M:%S")
    pending["snooze_count"] = int(pending.get("snooze_count", 0)) + 1
    return True


def _deadline(pending: dict[str, Any], now: datetime) -> datetime | None:
    windows = pending.get("execution_windows") or {}
    candidates: list[datetime] = []
    for key in ("same_day", "next_open"):
        item = windows.get(key) or {}
        raw_date, raw_end = item.get("date"), item.get("end")
        if not raw_date or not raw_end:
            continue
        candidates.append(
            datetime.combine(pd_to_date(raw_date), time.fromisoformat(raw_end), tzinfo=SH_TZ)
        )
    future = [item for item in candidates if item >= now]
    return min(future) if future else None


def ticket_view(
    pending: dict[str, Any] | None, now: datetime | None = None
) -> dict[str, Any] | None:
    if not pending:
        return None
    now = now or now_sh()
    if pending.get("schema_version") != 2:
        if str(pending.get("status", "")).lower() not in {"pending", "partial"}:
            return None
        # Legacy pending: derive windows without mutating state here.
        try:
            signal_date = pd_to_date(pending.get("date"))
            attach_ticket(pending, signal_date, now=now)
        except Exception:
            return None
    deadline = _deadline(pending, now)
    return {
        "ticket_id": pending.get("ticket_id"),
        "status": str(pending.get("status", "")).upper(),
        "signal_date": pending.get("signal_date"),
        "execution_windows": pending.get("execution_windows"),
        "deadline": deadline.strftime("%Y-%m-%d %H:%M") if deadline else None,
        "seconds_remaining": int((deadline - now).total_seconds()) if deadline else 0,
        "legs": pending.get("legs", []),
        "reason": pending.get("reason", ""),
        "reminders": pending.get("reminders", {}),
        "status_history": pending.get("status_history", []),
        "snooze_until": pending.get("snooze_until"),
        "snooze_count": pending.get("snooze_count", 0),
    }
