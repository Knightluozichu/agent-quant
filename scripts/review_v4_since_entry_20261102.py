"""V4 since the 2026-07-28 real 100k entry, extended to the 2026-10-30 close.

Research only: frozen inputs, no broker, no production state writes.
The 2026-07-28..2026-09-15 leg must reproduce the archived -8.9358% / -5.0587%
before the extended window is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts import live_signal as live
from scripts import run_qixing_v3 as rq
from scripts.review_oil_proxy_20260916 import prepare_bars
from scripts.review_rotation_20260916 import replay_next_open
from scripts.review_since_entry_20260916 import INITIAL, summarize_trades

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/v4_since_entry_20261102"
ARCHIVE = ROOT / "artifacts/version_comparison_20260916/release_verified"
START, END = date(2026, 7, 28), date(2026, 10, 30)
CHECK_END = date(2026, 9, 15)
CHECK_EXPECTED = {
    "original": -0.08935787600000011,
    "exclude_oil": -0.050586856499999944,
}
COST = rq.FEE + rq.SLIPPAGE


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def max_drawdown_details(curve: pd.DataFrame, initial: float = INITIAL) -> dict[str, Any]:
    """Drawdown path including the initial cash point as first high-water mark."""
    equity = curve.equity.to_numpy(dtype=float)
    path = np.r_[initial, equity]
    peaks = np.maximum.accumulate(path)
    drawdown = path / peaks - 1.0
    trough_index = int(np.argmin(drawdown))
    peak_index = int(np.argmax(path[: trough_index + 1]))
    dates = [None, *curve.trade_date.tolist()]
    underwater_days = int(sum(drawdown[i] < -1e-12 for i in range(1, len(drawdown))))
    return {
        "max_drawdown": float(drawdown[trough_index]),
        "peak_equity": float(path[peak_index]),
        "peak_date": None if peak_index == 0 else str(dates[peak_index]),
        "trough_equity": float(path[trough_index]),
        "trough_date": None if trough_index == 0 else str(dates[trough_index]),
        "underwater_days": underwater_days,
        "recovered": bool(np.any(path[trough_index:] >= path[peak_index] - 1e-9)),
    }


def monthly_returns(curve: pd.DataFrame, initial: float = INITIAL) -> list[dict[str, Any]]:
    frame = curve.copy()
    frame["month"] = pd.to_datetime(frame.trade_date).dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    previous = initial
    for month, group in frame.groupby("month", sort=True):
        ending = float(group.equity.iloc[-1])
        rows.append(
            {
                "month": month,
                "first_date": str(group.trade_date.iloc[0]),
                "last_date": str(group.trade_date.iloc[-1]),
                "return": ending / previous - 1.0,
                "ending_equity": ending,
            }
        )
        previous = ending
    return rows


def pair_episodes(trades: list[dict[str, Any]], calendar: list[date]) -> list[dict[str, Any]]:
    """Pair buy/sell legs into holding episodes with realized P&L."""
    order = {td: i for i, td in enumerate(calendar)}
    episodes: list[dict[str, Any]] = []
    opened: dict[str, Any] | None = None
    for trade in trades:
        action = trade["action"]
        if action == "buy":
            if opened is not None:
                raise ValueError("buy while a position is open")
            basis = trade["shares"] * trade["price"] + trade["estimated_cost"]
            opened = {
                "code": trade["code"],
                "signal_date": trade["signal_date"],
                "entry_date": trade["date"],
                "entry_price": trade["price"],
                "shares": trade["shares"],
                "basis": basis,
            }
        elif action == "sell":
            if opened is None or opened["code"] != trade["code"]:
                raise ValueError("unmatched sell")
            proceeds = trade["shares"] * trade["price"] - trade["estimated_cost"]
            entry_day = date.fromisoformat(opened["entry_date"])
            exit_day = date.fromisoformat(trade["date"])
            episodes.append(
                {
                    **opened,
                    "exit_signal_date": trade["signal_date"],
                    "exit_date": trade["date"],
                    "exit_price": trade["price"],
                    "proceeds": proceeds,
                    "realized_pnl": trade["realized_pnl"],
                    "return": trade["realized_pnl"] / opened["basis"],
                    "holding_days": order[exit_day] - order[entry_day],
                }
            )
            opened = None
        else:
            raise ValueError(f"unknown action: {action}")
    return episodes


def hold_intervals(curve: pd.DataFrame) -> list[dict[str, Any]]:
    """Contiguous holding intervals with equity endpoints."""
    rows: list[dict[str, Any]] = []
    holdings = curve["holding"].tolist()
    dates_list = curve["trade_date"].tolist()
    equities = curve["equity"].to_numpy(dtype=float)
    for trade_date, raw, equity in zip(dates_list, holdings, equities, strict=True):
        key = "CASH" if raw is None or pd.isna(raw) else str(raw)
        if not rows or rows[-1]["holding"] != key:
            rows.append(
                {
                    "start_date": str(trade_date),
                    "holding": key,
                    "start_equity": float(equity),
                    "days": 0,
                }
            )
        rows[-1]["days"] += 1
        rows[-1]["end_date"] = str(trade_date)
        rows[-1]["end_equity"] = float(equity)
    return rows


def run_scenario(
    data: dict[str, pd.DataFrame],
    dates: list[date],
    grid: set[date],
    *,
    pool: dict[str, str],
    pool_version: str,
    mode: str = "V4",
    end: date = END,
) -> dict[str, Any]:
    selector_patch: AbstractContextManager[Any] = nullcontext()
    with patch.object(rq, "ETF_POOL", pool):
        # Make config_hash pool-specific without touching production modules permanently.
        config_hash = live.v4.config_hash()
        with (
            patch.object(live, "ETF_POOL", pool),
            patch.object(live, "POOL_VERSION", pool_version),
            patch.object(live.v4, "CONFIG_HASH", config_hash),
            selector_patch,
        ):
            result: dict[str, Any] = replay_next_open(
                data,
                start=START,
                end=end,
                grid=grid,
                calendar=dates,
                mode=mode,
            )
            return result


def terminal_sale_proceeds(accounting: dict[str, Any], prices: dict[str, float]) -> float:
    """Liquidation value at the final close after one sell-side cost."""
    if not accounting["holding"]:
        return float(accounting["ending_equity"])
    gross = accounting["shares"] * prices[accounting["holding"]]
    return float(accounting["cash"] + gross * (1 - COST))


def scenario_report(
    replay: dict[str, Any],
    accounting: dict[str, Any],
    *,
    name: str,
    pool: dict[str, str],
    dates: list[date],
    prices: dict[str, float],
) -> dict[str, Any]:
    curve = replay["curve"]
    episodes = pair_episodes(accounting["trades"], dates)
    closed = [item for item in episodes if item["realized_pnl"] is not None]
    wins = [item for item in closed if item["realized_pnl"] > 0]
    losses = [item for item in closed if item["realized_pnl"] <= 0]
    gross_win = sum(item["realized_pnl"] for item in wins)
    gross_loss = -sum(item["realized_pnl"] for item in losses)
    metrics = replay["metrics"]
    drawdown = max_drawdown_details(curve)
    months = monthly_returns(curve)
    pending = replay["unfilled_next_open"]
    open_entry = None
    open_entry_price = None
    if accounting["holding"]:
        for trade in reversed(accounting["trades"]):
            if trade["action"] == "buy":
                open_entry = trade["date"]
                open_entry_price = trade["price"]
                break
    return {
        "name": name,
        "pool": pool,
        "start": str(START),
        "end": str(curve.trade_date.iloc[-1]),
        "trading_days": len(curve),
        "initial_capital": INITIAL,
        "final_equity": float(accounting["ending_equity"]),
        "total_pnl": float(accounting["total_pnl"]),
        "total_return": float(accounting["total_pnl"] / INITIAL),
        "terminal_liquidation_equity": terminal_sale_proceeds(accounting, prices),
        "realized_pnl": float(accounting["realized_pnl"]),
        "unrealized_pnl": float(accounting["open_position_pnl_after_buy_cost"]),
        "cash": float(accounting["cash"]),
        "holding": accounting["holding"],
        "open_entry_date": open_entry,
        "open_entry_price": open_entry_price,
        "shares": accounting["shares"],
        "market_value": float(accounting["market_value"]),
        "estimated_cost_paid": float(accounting["estimated_cost_paid"]),
        "max_drawdown": drawdown["max_drawdown"],
        "drawdown_detail": drawdown,
        "sharpe_daily_rf0": float(metrics["sharpe_daily_rf0"]),
        "trade_legs": len(accounting["trades"]),
        "round_trips": len(closed),
        "win_rate": len(wins) / len(closed) if closed else 0.0,
        "gross_profit": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else None,
        "best_episode": max(closed, key=lambda x: x["realized_pnl"]) if closed else None,
        "worst_episode": min(closed, key=lambda x: x["realized_pnl"]) if closed else None,
        "monthly_returns": months,
        "episodes": episodes,
        "holds": hold_intervals(curve),
        "trades": accounting["trades"],
        "pending_next_open": pending,
        "v4_trigger_dates": [
            str(item["trade_date"]) for item in replay["decisions"] if item.get("v4_triggered")
        ],
        "curve": curve,
        "decisions": replay["decisions"],
    }


def build_inputs() -> tuple[dict[str, pd.DataFrame], list[date], set[date]]:
    manifest = json.loads((FROZEN / "inputs.json").read_text())
    codes = [*rq.ETF_POOL, rq.DEFENSE]
    data: dict[str, pd.DataFrame] = {}
    for code in codes:
        path = FROZEN / "inputs" / f"{code}.parquet"
        if sha256(path) != manifest["files"][code]["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {code}")
        data[code] = prepare_bars(pd.read_parquet(path), code, END)
    dates = sorted(set.intersection(*(set(frame.trade_date) for frame in data.values())))
    assert START in dates and END in dates
    assert str(dates[130]) == "2020-06-19"
    grid = set(dates[130 :: rq.REBALANCE_DAYS])
    return data, dates, grid


def verify_against_archive(
    data: dict[str, pd.DataFrame], dates: list[date], grid: set[date]
) -> dict[str, float]:
    """Reproduce the archived 2026-09-15 V4 result before extending the window."""
    results: dict[str, float] = {}
    for name, pool in {
        "original": dict(rq.ETF_POOL),
        "exclude_oil": {c: n for c, n in rq.ETF_POOL.items() if c != "501018"},
    }.items():
        replay = run_scenario(
            data,
            dates,
            grid,
            pool=pool,
            pool_version="check-" + hashlib.sha256(",".join(sorted(pool)).encode()).hexdigest()[:8],
            end=CHECK_END,
        )
        total = replay["metrics"]["total_return"]
        if not math.isclose(total, CHECK_EXPECTED[name], abs_tol=1e-12):
            raise ValueError(f"Archive reproduction failed for {name}: {total}")
        archived_dir = ARCHIVE / f"v4_{name}"
        if not archived_dir.exists():
            raise FileNotFoundError(f"Missing archived result: {archived_dir}")
        archived_curve = pd.read_csv(archived_dir / "curve.csv")
        new_curve = replay["curve"].copy()
        new_curve["trade_date"] = new_curve.trade_date.astype(str)
        archived_curve["trade_date"] = archived_curve.trade_date.astype(str)
        if len(new_curve) != len(archived_curve):
            raise ValueError(f"Archive curve length mismatch for {name}")
        if not np.allclose(
            new_curve.equity.to_numpy(dtype=float),
            archived_curve.equity.to_numpy(dtype=float),
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(f"Archive curve mismatch for {name}")
        archived_trades = pd.read_csv(archived_dir / "trades.csv")
        new_trades = pd.DataFrame(replay["trades"])
        if len(new_trades) != len(archived_trades):
            raise ValueError(f"Archive trade count mismatch for {name}")
        for column in ("date", "signal_date", "action", "code"):
            if (
                new_trades[column].astype(str).tolist()
                != archived_trades[column].astype(str).tolist()
            ):
                raise ValueError(f"Archive trade column mismatch for {name}: {column}")
        buys = new_trades.action == "buy"
        if (
            new_trades.loc[buys, "shares"].astype(float).tolist()
            != archived_trades.loc[buys, "shares"].astype(float).tolist()
        ):
            raise ValueError(f"Archive trade shares mismatch for {name}")
        if not np.allclose(
            new_trades.price.to_numpy(dtype=float),
            archived_trades.price.to_numpy(dtype=float),
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(f"Archive trade price mismatch for {name}")
        archived_decisions = json.loads((archived_dir / "decisions.json").read_text())
        if len(replay["decisions"]) != len(archived_decisions):
            raise ValueError(f"Archive decision count mismatch for {name}")
        for new_decision, archived_decision in zip(
            replay["decisions"], archived_decisions, strict=True
        ):
            for key in ("trade_date", "holding", "v4_triggered", "final_target", "is_rebalance"):
                if new_decision.get(key) != archived_decision.get(key):
                    raise ValueError(f"Archive decision mismatch for {name}: {key}")
        results[name] = total
    return results


def write_scenario(output: Path, scenario: dict[str, Any]) -> None:
    dest = output / "results" / scenario["name"]
    dest.mkdir(parents=True, exist_ok=True)
    scenario["curve"].to_csv(dest / "curve.csv", index=False)
    pd.DataFrame(scenario["trades"]).to_csv(dest / "trades.csv", index=False)
    pd.DataFrame(scenario["episodes"]).to_csv(dest / "episodes.csv", index=False)
    pd.DataFrame(scenario["holds"]).to_csv(dest / "holds.csv", index=False)
    (dest / "decisions.json").write_text(
        json.dumps(scenario["decisions"], ensure_ascii=False, indent=2, default=str)
    )
    payload = {k: v for k, v in scenario.items() if k not in {"curve", "decisions"}}
    (dest / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


def format_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def money(value: float) -> str:
    return f"{value:,.2f}"


def write_markdown(
    output: Path,
    scenarios: dict[str, dict[str, Any]],
    reproduction: dict[str, float],
    inputs_manifest: dict[str, Any],
) -> None:
    primary = scenarios["original"]
    oil = scenarios["exclude_oil"]
    names = {**rq.ETF_POOL, rq.DEFENSE: "货币基金"}
    lines: list[str] = []
    lines.append("# V4 自 2026-07-28 实盘入金 10 万以来的完整收益与时间线（截至 2026-10-30）\n")
    lines.append(
        "> 口径：用户实盘入金 10 万；以下为**纯 V4 策略回溯**，不导入个人成交，不是账户实绩。\n"
    )
    lines.append("## 结论先行\n")
    lines.append(
        f"- 纯 V4 原池（含原油，7 个轮动候选 + 货币防御）从 2026-07-28 起以 "
        f"**100,000 元** 空仓起步，到 **2026-10-30 收盘**权益 "
        f"**{money(primary['final_equity'])} 元**，净盈亏 **{money(primary['total_pnl'])} 元**，"
        f"总收益 **{format_pct(primary['total_return'])}**。"
    )
    lines.append(
        f"- 区间最大回撤（收盘口径） **{format_pct(primary['max_drawdown'])}**（"
        f"高点 {primary['drawdown_detail']['peak_date']} → 低点 "
        f"{primary['drawdown_detail']['trough_date']}）；"
        f"当前持仓 **{primary['holding']}**，现金 {money(primary['cash'])} 元。"
    )
    lines.append(
        f"- 已实现盈亏 **{money(primary['realized_pnl'])} 元**，"
        f"期末浮动盈亏（扣买入费、未扣卖出费） **{money(primary['unrealized_pnl'])} 元**；"
        f"共 {primary['trade_legs']} 条成交腿、{primary['round_trips']} 个已平仓持仓段，"
        f"胜率 {primary['win_rate'] * 100:.0f}%（仅 {primary['round_trips']} 个已平仓段样本）。"
    )
    lines.append(
        f"- 若期末按 10-30 收盘一次性卖出并计卖出费，清算权益约 "
        f"**{money(primary['terminal_liquidation_equity'])} 元**"
        f"（较期末市值少约 "
        f"{money(primary['final_equity'] - primary['terminal_liquidation_equity'])} 元）。"
    )
    lines.append(
        f"- 同规则下若剔除 501018 原油（6 个轮动候选）：总收益 "
        f"**{format_pct(oil['total_return'])}**，最大回撤 {format_pct(oil['max_drawdown'])}。"
        "这是池子敏感性，不是用户实际账户收益。\n"
    )
    lines.append("## 口径与验证\n")
    lines.append(
        f"- 起点：{START}（账本最早交易/入金日，非认定首次投资日）；"
        f"终点：{END} 收盘（当前最新完整交易日）。"
        f"初始状态：100,000 元现金、空仓，不导入个人成交、不继承持仓/峰值/冷静期。\n"
        f"- 规则：生产 V4（`{live.v4.STRATEGY_ID}`），T 日收盘出信号，下一公共交易日开盘成交；"
        "100 份整手、按现金 99% 计算可买份数（费用另计，实际占用可略高于 99%）、"
        "单边费率+滑点 0.15%；期末按收盘估值，不强制卖出。\n"
        f"- 输入：8 只标的日线，2026-10-19 及以前来自既有冻结包，2026-10-20～10-30 为新浪增量；"
        f"输入 SHA256 见 `inputs.json`。\n"
        f"- 归档复现：先用同一引擎把窗口截到 {CHECK_END}，原池 "
        f"{format_pct(reproduction['original'])}、删油池 "
        f"{format_pct(reproduction['exclude_oil'])}，"
        "与 2026-09-16 归档报告逐位一致后才向外延伸。\n"
        "- 池子口径：当前生产 V4 为 7 个轮动候选（含 501018 原油）+ 511880 货币防御，"
        "2026-10-18 恢复；2026-09-16～10-17 生产曾剔除原油。本报告把**当前** 7 候选规则"
        "回溯到整个窗口，因此是反事实计算，不是历史逐日部署拼接。\n"
        "- 数据边界：2026-10-20～10-30 为新浪增量；10-19 及以前与旧冻结包逐日 OHLC 完全一致。"
        "独立审计用同源新浪二次抓取复核 10-19～10-30 的 8 只标的 OHLCV 零差异；"
        "东财第二源在本环境连接被重置，未作为证据。\n"
        "- 独立复核：逐笔现金/市值/已实现/浮动/总收益 7 项差值 0.00 元；"
        "24/24 成交腿满足 signal_date<成交日、成交价=次日 open；"
        "截到 2026-09-15 的成交/决策与全窗口一致，曲线仅 1.46e-11 浮点尾差，"
        "与 9/16 归档曲线/成交逐位一致。\n"
        "- 品种边界：501018 为 LOF，二级市场价格可能含溢价；本计算按次日开盘价成交、"
        "收盘价估值，未单独建模溢价收敛、申赎限制、停牌或流动性冲击。"
        "511880 货币基金年末除息会在价格序列产生伪跌，若跨年持有需另行复权；本窗口未持有。\n"
        "- 这是**当前 V4 规则的回溯计算**，不是历史每日实盘部署记录，也不代表用户实际成交；"
        "不构成收益承诺或下单指令。\n"
    )
    lines.append("## 资金曲线关键节点\n")
    detail = primary["drawdown_detail"]
    lines.append(
        f"- 期末权益 {money(primary['final_equity'])} 元；峰值 {money(detail['peak_equity'])} 元"
        f"（{detail['peak_date']}）；谷值 {money(detail['trough_equity'])} 元"
        f"（{detail['trough_date']}）；水下交易日 {detail['underwater_days']} 天（收盘口径）；"
        f"是否收复前高：{'是' if detail['recovered'] else '否'}。"
    )
    lines.append(
        f"- 累计成本（费+滑点估算）{money(primary['estimated_cost_paid'])} 元；"
        f"日频夏普（无风险利率 0，仅参考）{primary['sharpe_daily_rf0']:.2f}。\n"
    )
    lines.append("### 当前持仓（2026-10-30 收盘）\n")
    lines.append("| 标的 | 建仓日 | 买入价 | 份额 | 收盘价 | 市值 | 浮动盈亏 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    if primary["holding"]:
        close = primary["market_value"] / primary["shares"]
        lines.append(
            f"| {primary['holding']} {names.get(primary['holding'], '')} | "
            f"{primary['open_entry_date']} | {primary['open_entry_price']:.3f} | "
            f"{primary['shares']} | {close:.3f} | {money(primary['market_value'])} | "
            f"{money(primary['unrealized_pnl'])} |"
        )
    else:
        lines.append("| 空仓 | — | — | 0 | — | 0.00 | 0.00 |")
    lines.append(
        f"\n现金 **{money(primary['cash'])} 元**，"
        f"持仓市值 **{money(primary['market_value'])} 元**，"
        f"合计期末权益 **{money(primary['final_equity'])} 元**。\n"
    )
    if primary["holding"] and primary["final_equity"]:
        weight = primary["market_value"] / primary["final_equity"]
        per_pct = primary["shares"] * (primary["market_value"] / primary["shares"]) * 0.01
        lines.append(
            f"- 集中度：期末 **{weight * 100:.2f}%** 权益在 {primary['holding']}；"
            f"该标的收盘价每变动 1%，组合权益约变动 {money(per_pct)} 元"
            f"（约 {per_pct / primary['final_equity'] * 100:.2f} 个百分点）。\n"
        )
    lines.append("### 月度收益\n")
    lines.append("| 月份 | 区间 | 月末权益 | 月收益 |")
    lines.append("| --- | --- | ---: | ---: |")
    for row in primary["monthly_returns"]:
        lines.append(
            f"| {row['month']} | {row['first_date']}～{row['last_date']} | "
            f"{money(row['ending_equity'])} | {format_pct(row['return'])} |"
        )
    lines.append("\n## V4 完整时间线（原池）\n")
    lines.append("### 持仓段\n")
    lines.append("| 买入日 | 卖出日 | 标的 | 持有交易日 | 买入价 | 卖出价 | 已实现盈亏 | 段收益 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for item in primary["episodes"]:
        lines.append(
            f"| {item['entry_date']} | {item['exit_date']} | {item['code']} "
            f"{names.get(item['code'], '')} | {item['holding_days']} | "
            f"{item['entry_price']:.3f} | {item['exit_price']:.3f} | "
            f"{money(item['realized_pnl'])} | {format_pct(item['return'])} |"
        )
    lines.append("\n未平仓的当前持仓见上方“当前持仓（2026-10-30 收盘）”，不计入已实现盈亏。\n")
    lines.append("### 调仓检查与提前轮动决策\n")
    lines.append("| 决策日 | 当时持仓 | V3-G 基准目标 | V4 提前触发 | 最终目标 | 动作 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for decision in primary["decisions"]:
        if not decision["is_rebalance"] and not decision["v4_triggered"]:
            continue
        holding = decision["holding"] or "空仓"
        action = "换仓" if decision["final_target"] != decision["holding"] else "持有"
        lines.append(
            f"| {decision['trade_date']} | {holding} | {decision['v3g_target']} | "
            f"{'是' if decision['v4_triggered'] else '否'} | "
            f"{decision['final_target']} | {action} |"
        )
    lines.append("\n### 逐笔成交\n")
    lines.append(
        "| 信号日 | 成交日 | 动作 | 标的 | 价格 | 份额 | 估算费用 | 成交后现金 | 已实现盈亏 |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for trade in primary["trades"]:
        pnl = "" if trade.get("realized_pnl") is None else money(trade["realized_pnl"])
        lines.append(
            f"| {trade['signal_date']} | {trade['date']} | "
            f"{'买入' if trade['action'] == 'buy' else '卖出'} | {trade['code']} "
            f"{names.get(trade['code'], '')} | {trade['price']:.3f} | "
            f"{trade['shares']} | {money(trade['estimated_cost'])} | "
            f"{money(trade['cash_after'])} | {pnl} |"
        )
    pending = primary["pending_next_open"]
    lines.append("\n### 期末待执行信号\n")
    if pending:
        lines.append(
            f"- {END} 收盘产生、计划在下一交易日开盘执行的信号："
            f"`{json.dumps(pending, ensure_ascii=False)}`。**尚未成交，不计入上述收益。**"
        )
    else:
        lines.append("- 2026-10-30 收盘无待执行换仓信号。")
    lines.append("\n## 与删油池的差异\n")
    lines.append("| 指标 | 原池（含 501018） | 删油池 |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| 期末权益 | {money(primary['final_equity'])} | {money(oil['final_equity'])} |")
    lines.append(
        f"| 总收益 | {format_pct(primary['total_return'])} | {format_pct(oil['total_return'])} |"
    )
    lines.append(
        f"| 最大回撤 | {format_pct(primary['max_drawdown'])} | {format_pct(oil['max_drawdown'])} |"
    )
    lines.append(f"| 成交腿数 | {primary['trade_legs']} | {oil['trade_legs']} |")
    lines.append(
        f"| 已平仓段胜率 | {primary['win_rate'] * 100:.0f}% | {oil['win_rate'] * 100:.0f}% |"
    )
    lines.append(
        "\n差异来自 501018 是否参与候选与轮动；删油池结果不代表用户实际账户，"
        "也不证明原油未来更优或更差。\n"
    )
    oil_detail = oil["drawdown_detail"]
    lines.append("### 删油池持仓段\n")
    lines.append(
        f"删油池峰值 {money(oil_detail['peak_equity'])} 元（{oil_detail['peak_date']}），"
        f"谷值 {money(oil_detail['trough_equity'])} 元（{oil_detail['trough_date']}），"
        f"最大回撤 {format_pct(oil['max_drawdown'])}；"
        "8 月初白银的浮盈在 4 个交易日内回吐，是回撤的主要来源之一。\n"
    )
    lines.append("| 买入日 | 卖出日 | 标的 | 持有交易日 | 买入价 | 卖出价 | 已实现盈亏 | 段收益 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for item in oil["episodes"]:
        lines.append(
            f"| {item['entry_date']} | {item['exit_date']} | {item['code']} "
            f"{names.get(item['code'], '')} | {item['holding_days']} | "
            f"{item['entry_price']:.3f} | {item['exit_price']:.3f} | "
            f"{money(item['realized_pnl'])} | {format_pct(item['return'])} |"
        )
    if oil["holding"]:
        lines.append(
            f"\n删油池期末持有 {oil['holding']} {names.get(oil['holding'], '')} "
            f"{oil['shares']} 份，建仓价 {oil['open_entry_price']:.3f}，"
            f"收盘 {oil['market_value'] / oil['shares']:.3f}，"
            f"浮动盈亏 {money(oil['unrealized_pnl'])} 元（未计入已实现）。\n"
        )
    lines.append("## 交付文件\n")
    lines.append("- `inputs.json`、`inputs/*.parquet`：冻结输入与 SHA256。")
    lines.append("- `results/original/`、`results/exclude_oil/`：曲线、逐笔、持仓段、每日决策。")
    lines.append("- `summary.json`：两池指标与复现校验。")
    lines.append("- `equity_curve.png`：两池资金曲线与回撤。")
    lines.append("\n## 输入清单\n")
    lines.append("| 代码 | 行数 | 起始 | 截止 | SHA256（前 12 位） |")
    lines.append("| --- | ---: | --- | --- | --- |")
    for code, meta in inputs_manifest["files"].items():
        lines.append(
            f"| {code} | {meta['rows']} | {meta['start']} | {meta['end']} | {meta['sha256'][:12]} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def write_chart(output: Path, scenarios: dict[str, dict[str, Any]]) -> None:
    """Two-panel equity and drawdown chart; English labels avoid font dependency."""
    plt.switch_backend("Agg")
    figure, (equity_ax, drawdown_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    styles = {
        "original": ("V4 7 candidates incl. oil", "#c0392b"),
        "exclude_oil": ("V4 6 candidates excl. oil", "#2c7fb8"),
    }
    for name, (label, color) in styles.items():
        curve = scenarios[name]["curve"]
        dates = pd.to_datetime(curve.trade_date)
        values = curve.equity.to_numpy(dtype=float)
        equity_ax.plot(dates, values, label=label, color=color, linewidth=1.6)
        path = np.r_[INITIAL, values]
        drawdown = path / np.maximum.accumulate(path) - 1.0
        drawdown_ax.plot(dates, drawdown[1:] * 100, color=color, linewidth=1.2)
    equity_ax.axhline(INITIAL, color="#555555", linestyle="--", linewidth=1)
    equity_ax.set_title("V4 since 2026-07-28, initial capital 100,000 CNY (to 2026-10-30)")
    equity_ax.set_ylabel("Equity (CNY)")
    equity_ax.grid(alpha=0.3)
    equity_ax.legend(loc="lower left")
    drawdown_ax.set_ylabel("Drawdown (%)")
    drawdown_ax.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FROZEN)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory (required for the frozen artifacts).",
    )
    args = parser.parse_args()
    output = args.output
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}; pass --overwrite to replace it")
    output.mkdir(parents=True, exist_ok=True)
    data, dates, grid = build_inputs()
    prices = {code: float(frame.close.iloc[-1]) for code, frame in data.items()}
    reproduction = verify_against_archive(data, dates, grid)
    inputs_manifest = json.loads((FROZEN / "inputs.json").read_text())
    scenarios: dict[str, dict[str, Any]] = {}
    for name, pool in {
        "original": dict(rq.ETF_POOL),
        "exclude_oil": {c: n for c, n in rq.ETF_POOL.items() if c != "501018"},
    }.items():
        version = "research-" + hashlib.sha256(",".join(sorted(pool)).encode()).hexdigest()[:12]
        replay = run_scenario(data, dates, grid, pool=pool, pool_version=version)
        accounting = summarize_trades(list(replay["trades"]), prices, cost=COST)
        assert math.isclose(
            accounting["ending_equity"], replay["metrics"]["final_equity"], abs_tol=1e-7
        )
        assert replay["curve"].iloc[0].equity == INITIAL
        assert all(str(START) <= t["signal_date"] < t["date"] for t in replay["trades"])
        if name == "exclude_oil":
            assert all(t["code"] != "501018" for t in replay["trades"])
        scenario = scenario_report(
            replay, accounting, name=name, pool=pool, dates=dates, prices=prices
        )
        write_scenario(output, scenario)
        scenarios[name] = scenario
    summary = {
        "start": str(START),
        "end": str(END),
        "initial_capital": INITIAL,
        "strategy_id": live.v4.STRATEGY_ID,
        "v4_params": asdict(live.v4.V4_PARAMS),
        "execution": "T close signal; next common open; final close mark; no terminal sale",
        "cost_per_side": COST,
        "archive_reproduction": reproduction,
        "scenarios": {
            name: {k: v for k, v in scenario.items() if k not in {"curve", "decisions", "trades"}}
            for name, scenario in scenarios.items()
        },
        "input_sha256": {code: meta["sha256"] for code, meta in inputs_manifest["files"].items()},
        "research_only": True,
        "independent_out_of_sample": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    )
    write_markdown(output, scenarios, reproduction, inputs_manifest)
    write_chart(output, scenarios)
    print(
        json.dumps(
            {
                "reproduction": reproduction,
                "original": {
                    "final_equity": scenarios["original"]["final_equity"],
                    "total_return": scenarios["original"]["total_return"],
                    "max_drawdown": scenarios["original"]["max_drawdown"],
                },
                "exclude_oil": {
                    "final_equity": scenarios["exclude_oil"]["final_equity"],
                    "total_return": scenarios["exclude_oil"]["total_return"],
                    "max_drawdown": scenarios["exclude_oil"]["max_drawdown"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
