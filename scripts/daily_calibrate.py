"""每日校准: 收盘后更新数据 + 校准下个调仓日 + 预览调仓数据 + 推送每日状态.

每天16:30由cron运行: 更新到最终收盘数据, 校准下个调仓日及调仓预览, 推送每日校准状态到手机。
与14:50的信号推送区分: 14:50=调仓日实际信号(timeSensitive); 本脚本=每日校准预告(active)。
用法: uv run python scripts/daily_calibrate.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import live_signal as ls
import run_qixing_v3 as rq
from next_rebalance import find_next_rebalance
from notify import push_bark
from strategy_lab.engine import build_idx_map, get_common_dates


def main() -> None:
    data = ls.load_data()
    data = ls.update_data(data)  # 收盘后更新到最终收盘数据

    state = ls.load_state()
    if state is None:
        push_bark("⚠️ 校准异常", "账户未初始化, 请检查", level="timeSensitive")
        return

    # 方案A: 注入当日实时收盘价 (腾讯源), 解决新浪当日K线发布延迟导致校准滞后一天
    # 交易日才注入; 非交易日(节假日)或实时源不可用时, 回退昨日数据并在消息中标注
    today = date.today()
    is_td = ls.is_trading_day(today)
    if is_td:
        data = ls.inject_realtime(data)

    trading_dates = get_common_dates(data)
    latest = trading_dates[-1]
    last_rb = state.get("last_rebalance_date")
    holding = state.get("holding")

    next_rb, days_left = find_next_rebalance(trading_dates, last_rb)

    # 14:50官方决策是唯一交易口径；16:30不重复推进V4确认历史。
    idx_map = build_idx_map(data, latest)
    target, candidates, best_score, a_share_weak = rq.select_target(data, idx_map, holding)
    official = state.get("last_decision")
    has_official = bool(official and official.get("trade_date") == str(latest))
    if has_official:
        target = official["final_target"]
        best_score = dict(candidates).get(target, 0.0)

    # 账户总值 (优先用实时行情当日真实价, 失败回退历史价)
    total = state["cash"]
    if holding:
        p = ls.get_realtime_price(holding)
        if p is None and holding in data:
            p = ls.price_on(data, holding, latest)
        if p:
            total += state.get("shares", 0) * p
    ret = (total / state["initial_capital"] - 1) * 100

    # 组装每日校准消息
    hold_disp = f"【{holding}】{ls.name_of(holding)}" if holding else "空仓"
    lines = [f"策略: {ls.get_strategy_mode()} ({ls.v4.STRATEGY_ID})", f"持仓: {hold_disp}"]
    lines.append(f"💰 盈亏: {total/10000:.2f}万 ({ret:+.1f}%)")
    if days_left is not None and days_left <= 0:
        lines.append(f"🎯 调仓日: {next_rb} (就是今天/即将)")
    elif days_left is not None:
        lines.append(f"🎯 调仓日: {next_rb} (还有{days_left}个交易日)")
    else:
        lines.append(f"🎯 调仓日: 约{next_rb}")
    if target == holding:
        label = "14:50官方决策" if has_official else "V3-G基础预览"
        lines.append(f"👉 {label}: 继续持有 {hold_disp}")
    else:
        label = "14:50官方调仓" if has_official else "V3-G基础预览"
        lines.append(
            f"👉 {label}: → 【{target}】{ls.name_of(target)} "
            f"(动量{best_score * 100:+.1f}%)"
        )
        lines.append(f"   易淘金搜索代码 {target} 买入")
    if a_share_weak:
        lines.append("⚠️ A股走弱, 已排除创业板")
    if is_td and str(latest) != str(today):
        lines.append("⚠️ 实时行情不可用, 数据基于昨日收盘")

    push_bark("📅 每日简报", "\n".join(lines), level="active")
    print("  ✓ 每日校准已推送")
    print("\n".join("  " + ln for ln in lines))


if __name__ == "__main__":
    main()
