"""Send one semi-automatic execution reminder stage. No broker access.

Run by cron at:
  14:45 pre, 14:55 same_day_warn, 15:02 same_day_close, 15:10 same_day_closed,
  09:15 next_open_warn, 09:25 next_open_final, 09:35 expired.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import live_signal  # noqa: E402
import ticket  # noqa: E402
from notify import push_bark  # noqa: E402

WEB_URL = os.environ.get("QIXING_WEB_URL", "https://dj.luozichu.ink")
STAGES = tuple(ticket.STAGE_TIMES)


def _name(code: str) -> str:
    return str(live_signal.name_of(code))


def _ticket_body(view: dict[str, Any], stage: str) -> tuple[str, str, str]:
    legs = view.get("legs", [])
    lines = []
    for leg in legs:
        side = "卖出" if leg["side"] == "sell" else "买入"
        lines.append(
            f"{side} {leg['code']} {_name(leg['code'])} "
            f"{leg['shares']}份 参考{leg['reference_price']:.3f} "
            f"限价{leg['limit_low']:.3f}-{leg['limit_high']:.3f}"
        )
    deadline = view.get("deadline") or "无"
    titles = {
        "same_day_warn": "V5 还有 5 分钟到同日窗口",
        "same_day_close": "V5 同日窗口即将关闭",
        "same_day_closed": "V5 同日窗口已关闭，明日开盘仍可执行",
        "next_open_warn": "V5 昨日信号：今日开盘窗口",
        "next_open_final": "V5 最后 5 分钟：开盘窗口",
        "expired": "V5 已错过两个执行窗口",
    }
    title = titles.get(stage, "V5 执行提醒")
    body = "\n".join(lines)
    body += f"\n截止：{deadline}\n状态：{view.get('status')}"
    if stage == "expired":
        body += "\n请到网页补登实际成交或标记未交易；不会自动跳过。"
    body += f"\n{WEB_URL}"
    return title, body, deadline


def _send(title: str, body: str, *, urgent: bool) -> bool:
    return bool(
        push_bark(
            title,
            body,
            level="timeSensitive" if urgent else "active",
            group="七星V5执行",
            url=WEB_URL,
        )
    )


def run(stage: str, *, force: bool = False, dry_run: bool = False) -> int:
    now = ticket.now_sh()
    today = now.date()
    if not force and not ticket.is_trading_day(today):
        print(f"非交易日，跳过 {stage}")
        return 0

    state = live_signal.load_state()
    if state is None:
        print("账户未初始化")
        return 1
    pending = state.get("pending_order")

    if stage == "pre":
        if pending and str(pending.get("status", "")).lower() in {"pending", "partial"}:
            print("已有未决 ticket，跳过 pre")
            return 0
        grid = ticket.is_rebalance_day(today)
        history = (state.get("v4_state") or {}).get("candidate_history") or []
        possible_early = any(isinstance(item, dict) and item.get("target") for item in history[-2:])
        if not force and not (grid or possible_early):
            print("今日非网格日且无确认候选，跳过 pre")
            return 0
        sent = state.setdefault("pre_reminder_sent", {})
        key = str(today)
        if not force and key in sent:
            print("pre 已发送，跳过")
            return 0
        title = "V5 今日可能有信号"
        body = (
            f"{today} 是{'网格日' if grid else '候选确认日'}，"
            "14:50 可能生成 V5 执行票。\n准备易淘金，收到正式票后再操作。\n"
            f"{WEB_URL}"
        )
        if dry_run:
            print(f"[dry-run] {title}\n{body}")
            return 0
        if not _send(title, body, urgent=False):
            print("Bark 发送失败")
            return 1
        if not force:
            with live_signal.state_transaction() as fresh:
                fresh.setdefault("pre_reminder_sent", {})[key] = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"pre 已发送 {today}")
        return 0

    if not pending:
        print("无 pending，跳过")
        return 0
    if str(pending.get("status", "")).lower() not in {"pending", "partial"}:
        print(f"pending 状态 {pending.get('status')}，跳过")
        return 0
    try:
        ticket.attach_ticket(pending, ticket.pd_to_date(pending.get("date")))
    except Exception as exc:
        print(f"ticket 迁移失败: {exc}")
        return 1
    if not force and not ticket.stage_due(pending, stage, now):
        print(f"{stage} 未到期/已发送/不在窗口，跳过")
        return 0

    view = ticket.ticket_view(pending, now)
    assert view is not None
    title, body, _deadline = _ticket_body(view, stage)
    if dry_run:
        print(f"[dry-run] {title}\n{body}")
        return 0
    if not _send(title, body, urgent=stage in {"same_day_close", "next_open_final", "expired"}):
        print("Bark 发送失败")
        return 1

    with live_signal.state_transaction() as fresh:
        current = fresh.get("pending_order")
        if not current or str(current.get("status", "")).lower() not in {"pending", "partial"}:
            print("发送后 pending 已被处理，不写提醒记录")
            return 0
        ticket.attach_ticket(current, ticket.pd_to_date(current.get("date")))
        ticket.mark_reminder(current, stage, now)
        if stage == "expired":
            ticket.mark_status(
                current, "expired_unresolved", actor="system", reason="both_windows_missed", now=now
            )
    print(f"{stage} 已发送")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--force", action="store_true", help="测试用：绕过日期/到期/去重")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不推送、不写状态")
    args = parser.parse_args()
    return run(args.stage, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
