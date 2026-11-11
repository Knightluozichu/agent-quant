"""每日校准: 收盘后更新数据 + 校准下个调仓日 + 推送每日状态.

每天16:30由cron运行: 更新到最终收盘数据, 校准下个调仓日, 推送每日校准状态到手机。
与14:50的信号推送区分: 14:50=调仓日实际信号(timeSensitive); 本脚本=每日简报(active)。
16:30 只复述当天 14:50 的官方决策与待执行票, 不重新计算、不输出新的买入指令。
用法: uv run python scripts/daily_calibrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import date

sys.path.insert(0, str(Path(__file__).parent))

import live_signal as ls
import run_qixing_v3 as rq
from next_rebalance import find_next_rebalance
from notify import push_bark
from strategy_lab.engine import build_idx_map, get_common_dates


def official_decision(
    state: dict[str, Any], *, today: date, latest: date, is_td: bool
) -> dict[str, Any] | None:
    """只认当天(交易日)或最后交易日(非交易日)的官方快照, 且模式/配置必须一致."""
    official = state.get("last_decision") or {}
    expected_date = str(today) if is_td else str(latest)
    if official.get("trade_date") != expected_date:
        return None
    if official.get("mode") != ls.get_strategy_mode():
        return None
    if official.get("config_hash") != ls.active_config_hash():
        return None
    target = official.get("final_target")
    if target not in {*ls.ETF_POOL, ls.DEFENSE}:
        return None
    return official


def pending_active(state: dict[str, Any]) -> dict[str, Any] | None:
    pending = state.get("pending_order")
    if not pending:
        return None
    if str(pending.get("status", "")).lower() not in {"pending", "partial"}:
        return None
    return pending


def _pending_lines(pending: dict[str, Any]) -> list[str]:
    view = ls.ticket_engine.ticket_view(pending) or {}
    legs = view.get("legs") or []
    lines = []
    for leg in legs:
        side = "卖出" if leg.get("side") == "sell" else "买入"
        lines.append(
            f"⚠️ 待执行票: {side} {leg.get('code')} {leg.get('name')} "
            f"{leg.get('shares')}份"
        )
    if not lines and pending.get("buy"):
        buy = pending["buy"]
        lines.append(f"⚠️ 待执行票: 买入 {buy.get('code')} {buy.get('name')} {buy.get('shares')}份")
    deadline = view.get("deadline")
    if deadline:
        lines.append(f"   截止 {deadline}，请先在网页确认/跳过")
    else:
        lines.append("   请先在网页确认/跳过")
    return lines


def _next_rebalance_line(next_rb: date | None, days_left: int | None) -> str:
    if next_rb is None:
        return "🎯 调仓日: 未知"
    if days_left is None:
        return f"🎯 调仓日: 约{next_rb}"
    if days_left < 0:
        return f"🎯 调仓日: {next_rb} 已过期 {abs(days_left)} 个交易日，等待补单/跳过"
    if days_left == 0:
        return f"🎯 调仓日: {next_rb}（今天）"
    return f"🎯 调仓日: {next_rb}（还有{days_left}个交易日）"


def build_brief(
    *,
    state: dict[str, Any],
    data: dict[str, Any],
    latest: date,
    next_rb: date | None,
    days_left: int | None,
    today: date,
    is_td: bool,
    total: float,
) -> list[str]:
    """16:30 简报正文. 有官方快照时只复述, 没有时只给不可执行的观察预览."""
    holding = state.get("holding")
    hold_disp = f"【{holding}】{ls.name_of(holding)}" if holding else "空仓"
    ret = (total / state["initial_capital"] - 1) * 100
    lines = [
        f"策略: {ls.get_strategy_mode()} ({ls.active_strategy_id()})",
        f"持仓: {hold_disp}",
        f"💰 盈亏: {total / 10000:.2f}万 ({ret:+.1f}%)",
        _next_rebalance_line(next_rb, days_left),
    ]

    official = official_decision(state, today=today, latest=latest, is_td=is_td)
    pending = pending_active(state)
    if pending:
        lines.extend(_pending_lines(pending))

    if official:
        target = official["final_target"]
        factor = (official.get("factors") or {}).get(target, {})
        score = factor.get("slow_momentum")
        if target == holding and holding:
            lines.append(f"👉 14:50官方决策: 继续持有 {hold_disp}")
        elif target == ls.DEFENSE and holding is None:
            lines.append("👉 14:50官方决策: 保持空仓（货币基金）")
        elif score is not None:
            lines.append(
                f"👉 14:50官方决策: → 【{target}】{ls.name_of(target)} "
                f"(动量{score * 100:+.1f}%)"
            )
        else:
            lines.append(f"👉 14:50官方决策: → 【{target}】{ls.name_of(target)}")
        lines.append(f"   以 {official['trade_date']} 14:50 官方快照为准；本简报不重复下单")
        if holding and target != holding and not pending and target != ls.DEFENSE:
            lines.append(
                f"⚠️ 实际持仓 {holding} 与模型目标 {target} 不一致（手动成交），"
                "下次调仓按实际持仓计算"
            )
        if pending:
            lines.append("   旧票未处理前，14:50 新信号会被抑制")
        return lines

    if pending:
        lines.append("👉 今日无有效官方快照；请先处理上面的待执行票")
        return lines

    idx_map = build_idx_map(data, latest)
    target, _candidates, best_score, a_share_weak = rq.select_target(data, idx_map, holding)
    if target == holding and holding:
        lines.append(f"👉 非交易预览（基于 {latest} 收盘，未计V5风控）: 继续持有 {hold_disp}")
    elif target == ls.DEFENSE and holding is None:
        lines.append(
            f"👉 非交易预览（基于 {latest} 收盘，未计V5风控）: 保持空仓（货币基金）"
        )
    else:
        lines.append(
            f"👉 非交易预览（基于 {latest} 收盘，未计V5风控）: → "
            f"【{target}】{ls.name_of(target)} (动量{best_score * 100:+.1f}%)"
        )
    lines.append("   仅供观察，不构成买入指令；下一调仓日以14:50正式信号为准")
    if a_share_weak:
        lines.append("⚠️ A股走弱, 已排除创业板")
    if is_td and str(latest) != str(today):
        lines.append(f"⚠️ 当日收盘行情未注入, 数据基于 {latest}")
    return lines


def main() -> None:
    data = ls.load_data()
    data = ls.update_data(data)  # 收盘后更新到最终收盘数据

    state = ls.load_state()
    if state is None:
        push_bark("⚠️ 校准异常", "账户未初始化, 请检查", level="timeSensitive")
        return

    # 收盘后腾讯行情时间戳固定在 15:00, 不能用盘中 5 分钟门禁; 按当日收盘快照校验
    today = ls._today_sh()
    is_td = ls.is_trading_day(today)
    if is_td:
        data = ls.inject_realtime(data, after_close=True)

    trading_dates = get_common_dates(data)
    latest = trading_dates[-1]
    next_rb, days_left = find_next_rebalance(trading_dates, state.get("last_rebalance_date"))
    holding = state.get("holding")

    # 账户总值 (优先用实时行情当日真实价, 失败回退历史价)
    total = state["cash"]
    if holding:
        p = ls.get_realtime_price(holding)
        if p is None and holding in data:
            p = ls.price_on(data, holding, latest)
        if p:
            total += state.get("shares", 0) * p

    lines = build_brief(
        state=state,
        data=data,
        latest=latest,
        next_rb=next_rb,
        days_left=days_left,
        today=today,
        is_td=is_td,
        total=total,
    )
    push_bark("📅 每日简报", "\n".join(lines), level="active")
    print("  ✓ 每日校准已推送")
    print("\n".join("  " + ln for ln in lines))


if __name__ == "__main__":
    main()
