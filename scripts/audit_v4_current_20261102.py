"""Professional audit of the current V4 original-pool strategy (research only).

Full backtest, yearly segments, zero-fit forward restarts, rolling windows,
cost/pool/grid-phase stress, and a circuit-breaker state-machine ablation.

Run:
    uv run --no-sync python -m scripts.audit_v4_current_20261102
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from contextlib import nullcontext
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts import live_signal as live
from scripts import qixing_v4 as v4
from scripts import run_qixing_v3 as rq
from scripts.exp_v3g_full_pool_fast_slow import run_full_pool_strategy
from scripts.review_oil_proxy_20260916 import prepare_bars
from scripts.review_rotation_20260916 import replay_next_open
from scripts.review_since_entry_20260916 import summarize_trades

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "artifacts/v4_since_entry_20261102/inputs"
INPUT_MANIFEST = ROOT / "artifacts/v4_since_entry_20261102/inputs.json"
OUT = ROOT / "artifacts/v4_audit_20261102"
START, END = date(2020, 6, 19), date(2026, 10, 30)
INITIAL = 100_000.0
COST = rq.FEE + rq.SLIPPAGE
DEFENSE_CODES = {"511260", "511220", "511880"}
ORIGINAL_RISK = live.risk_assess


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Risk-layer ablations.  All are research adapters around the production
# function; the production module is never edited.
# --------------------------------------------------------------------------- #
def neutral_risk(**kwargs: Any) -> Any:
    """Disable the whole risk layer for a raw-signal reference only."""
    return live.RiskDecision(
        final_target=kwargs["target"],
        exposure=1.0,
        action="trade" if kwargs["is_rebalance"] else "hold",
        reason="no_risk_reference",
        events=[],
        cooldown_until=None,
    )


def cb_off(**kwargs: Any) -> Any:
    """Keep all risk logic except the -30% portfolio circuit breaker."""
    decision = ORIGINAL_RISK(**kwargs)
    if decision.action == live.ACTION_EMERGENCY:
        return live.RiskDecision(
            final_target=kwargs["target"],
            exposure=1.0,
            action="trade" if kwargs["is_rebalance"] else "hold",
            reason="circuit_breaker_off",
            events=decision.events,
            cooldown_until=None,
        )
    return decision


def peak_reset(**kwargs: Any) -> Any:
    """After a CB exit, rebase the high-water mark to de-risked equity.

    The current production logic keeps the old peak.  Once drawdown is below
    -30%, every re-entry is immediately below the same stale peak and triggers
    again.  Rebasing makes the breaker a one-shot de-risking event instead of a
    ratchet.
    """
    decision = ORIGINAL_RISK(**kwargs)
    if decision.action == live.ACTION_EMERGENCY:
        state = kwargs["state"]
        holding = state.get("holding")
        equity = float(state["cash"])
        if holding:
            equity += float(state["shares"]) * float(
                kwargs["data"][holding].iloc[kwargs["idx_map"][holding]]["close"]
            )
        state["peak_equity"] = equity
    return decision


RISK_MODES: dict[str, Any] = {
    "baseline": None,
    "cb_off": cb_off,
    "peak_reset": peak_reset,
    "no_risk": neutral_risk,
}


def load_data() -> tuple[dict[str, pd.DataFrame], list[date], dict[str, Any]]:
    manifest = json.loads(INPUT_MANIFEST.read_text())
    data: dict[str, pd.DataFrame] = {}
    for code in [*rq.ETF_POOL, rq.DEFENSE]:
        path = INPUTS / f"{code}.parquet"
        if sha256(path) != manifest["files"][code]["sha256"]:
            raise ValueError(f"Input hash mismatch: {code}")
        data[code] = prepare_bars(pd.read_parquet(path), code, END)
    dates = sorted(set.intersection(*(set(frame.trade_date) for frame in data.values())))
    return data, dates, manifest


def grid_for(dates: list[date], phase: int = 0) -> set[date]:
    if not 0 <= phase < rq.REBALANCE_DAYS:
        raise ValueError("phase must be 0..4")
    return set(dates[130 + phase :: rq.REBALANCE_DAYS])


def run_v4(
    data: dict[str, pd.DataFrame],
    dates: list[date],
    *,
    pool: dict[str, str],
    risk_mode: str = "baseline",
    cost_multiplier: float = 1.0,
    phase: int = 0,
    start: date = START,
    end: date = END,
) -> dict[str, Any]:
    pool_hash = hashlib.sha256(",".join(sorted(pool)).encode()).hexdigest()[:12]
    risk_patch = (
        nullcontext()
        if RISK_MODES[risk_mode] is None
        else patch.object(live, "risk_assess", RISK_MODES[risk_mode])
    )
    with (
        patch.object(rq, "ETF_POOL", pool),
        patch.object(live, "ETF_POOL", pool),
        patch.object(live, "POOL_VERSION", f"audit-{risk_mode}-{pool_hash}"),
        risk_patch,
    ):
        result: dict[str, Any] = replay_next_open(
            data,
            start=start,
            end=end,
            grid=grid_for(dates, phase),
            calendar=dates,
            mode="V4",
            cost_multiplier=cost_multiplier,
        )
        return result


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def drawdown_path(curve: pd.DataFrame, initial: float = INITIAL) -> tuple[np.ndarray, np.ndarray]:
    equity = curve.equity.to_numpy(dtype=float)
    path = np.r_[initial, equity]
    peaks = np.maximum.accumulate(path)
    return path, path / peaks - 1.0


def metrics(curve: pd.DataFrame, initial: float = INITIAL) -> dict[str, Any]:
    path, drawdown = drawdown_path(curve, initial)
    equity = path[1:]
    returns = np.diff(path) / path[:-1]
    years = len(equity) / 252.0
    total = equity[-1] / initial - 1.0
    cagr = (equity[-1] / initial) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    vol = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else 0.0
    )
    downside = returns[returns < 0]
    sortino = (
        float(returns.mean() / downside.std(ddof=1) * np.sqrt(252))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else 0.0
    )
    mdd = float(drawdown.min())
    trough = int(np.argmin(drawdown))
    peak = int(np.argmax(path[: trough + 1]))
    underwater = int((drawdown[1:] < -1e-12).sum())
    five_day = pd.Series(equity).pct_change(5).min()
    twenty_day = pd.Series(equity).pct_change(20).min()
    return {
        "start": str(curve.trade_date.iloc[0]),
        "end": str(curve.trade_date.iloc[-1]),
        "days": len(curve),
        "final_equity": float(equity[-1]),
        "total_return": float(total),
        "cagr": float(cagr),
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "peak_equity": float(path[peak]),
        "peak_date": None if peak == 0 else str(curve.trade_date.iloc[peak - 1]),
        "trough_equity": float(path[trough]),
        "trough_date": None if trough == 0 else str(curve.trade_date.iloc[trough - 1]),
        "underwater_days": underwater,
        "worst_day": float(returns.min()) if len(returns) else 0.0,
        "worst_5d": float(five_day) if pd.notna(five_day) else 0.0,
        "worst_20d": float(twenty_day) if pd.notna(twenty_day) else 0.0,
        "ulcer": float(np.sqrt(np.mean(np.square(drawdown[1:])))),
    }


def yearly_continuous(curve: pd.DataFrame, initial: float = INITIAL) -> list[dict[str, Any]]:
    frame = curve.copy()
    frame["year"] = pd.to_datetime(frame.trade_date).dt.year
    rows: list[dict[str, Any]] = []
    previous = initial
    for year, group in frame.groupby("year", sort=True):
        path = np.r_[previous, group.equity.to_numpy(dtype=float)]
        drawdown = path / np.maximum.accumulate(path) - 1.0
        rows.append(
            {
                "year": int(str(year)),
                "start_date": str(group.trade_date.iloc[0]),
                "end_date": str(group.trade_date.iloc[-1]),
                "start_equity": previous,
                "end_equity": float(group.equity.iloc[-1]),
                "return": float(group.equity.iloc[-1] / previous - 1.0),
                "max_drawdown": float(drawdown.min()),
                "days": len(group),
            }
        )
        previous = float(group.equity.iloc[-1])
    return rows


def rolling_stats(
    curve: pd.DataFrame, window: int, initial: float = INITIAL
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    equity = curve.equity.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for i in range(0, len(equity) - window + 1):
        base = initial if i == 0 else equity[i - 1]
        segment = equity[i : i + window]
        returns = np.diff(np.r_[base, segment]) / np.r_[base, segment[:-1]]
        path = np.r_[base, segment]
        drawdown = path / np.maximum.accumulate(path) - 1.0
        vol = returns.std(ddof=1)
        rows.append(
            {
                "start": str(curve.trade_date.iloc[i]),
                "end": str(curve.trade_date.iloc[i + window - 1]),
                "return": float(segment[-1] / base - 1.0),
                "sharpe": float(returns.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0,
                "max_drawdown": float(drawdown.min()),
            }
        )
    values = np.array([[row["return"], row["sharpe"], row["max_drawdown"]] for row in rows])
    worst = min(rows, key=lambda row: row["return"])
    summary = {
        "window": window,
        "windows": len(rows),
        "negative_windows": int((values[:, 0] < 0).sum()),
        "negative_fraction": float((values[:, 0] < 0).mean()) if rows else 0.0,
        "return_min": float(values[:, 0].min()),
        "return_median": float(np.median(values[:, 0])),
        "return_max": float(values[:, 0].max()),
        "sharpe_median": float(np.median(values[:, 1])),
        "max_drawdown_median": float(np.median(values[:, 2])),
        "max_drawdown_worst": float(values[:, 2].min()),
        "worst_window": worst,
    }
    return rows, summary


def trade_diagnostics(
    result: dict[str, Any], data: dict[str, pd.DataFrame], *, cost_multiplier: float = 1.0
) -> dict[str, Any]:
    curve = result["curve"]
    prices = {code: float(frame.close.iloc[-1]) for code, frame in data.items()}
    accounting = summarize_trades(list(result["trades"]), prices, cost=COST * cost_multiplier)
    trades = pd.DataFrame(accounting["trades"])
    traded_notional = float((trades.shares * trades.price).sum())
    average_equity = float(curve.equity.mean())
    holding = curve.holding.where(curve.holding.notna(), "")
    bond_days = int((holding == "511220").sum())
    money_days = int((holding == "511880").sum())
    treasury_days = int((holding == "511260").sum())
    emergency = [
        item for item in result["decisions"] if item["risk_action"] == live.ACTION_EMERGENCY
    ]
    emergency_by_year = dict(
        sorted(Counter(str(item["trade_date"])[:4] for item in emergency).items())
    )
    return {
        "trade_legs": len(accounting["trades"]),
        "buy_legs": int((trades.action == "buy").sum()),
        "sell_legs": int((trades.action == "sell").sum()),
        "round_trips": int((trades.action == "sell").sum()),
        "emergency_count": len(emergency),
        "first_emergency": emergency[0]["trade_date"] if emergency else None,
        "last_emergency": emergency[-1]["trade_date"] if emergency else None,
        "emergency_by_year": emergency_by_year,
        "estimated_cost_paid": float(accounting["estimated_cost_paid"]),
        "cost_drag": float(accounting["estimated_cost_paid"]) / average_equity
        if average_equity
        else 0.0,
        "realized_pnl": float(accounting["realized_pnl"]),
        "unrealized_pnl": float(accounting["open_position_pnl_after_buy_cost"]),
        "ending_equity": float(accounting["ending_equity"]),
        "traded_notional": traded_notional,
        "turnover_x": traded_notional / average_equity if average_equity else 0.0,
        "bond_day_fraction": bond_days / len(curve) if len(curve) else 0.0,
        "money_day_fraction": money_days / len(curve) if len(curve) else 0.0,
        "treasury_day_fraction": treasury_days / len(curve) if len(curve) else 0.0,
    }


def concentration(curve: pd.DataFrame, initial: float = INITIAL) -> dict[str, Any]:
    frame = curve.copy()
    frame["year"] = pd.to_datetime(frame.trade_date).dt.year
    year_end = frame.groupby("year").equity.last()
    previous = initial
    rows = []
    total_log = math.log(year_end.iloc[-1] / initial)
    for year, equity in year_end.items():
        contribution = math.log(float(equity) / previous)
        rows.append(
            {
                "year": int(str(year)),
                "log_contribution": contribution,
                "share_of_total_log": contribution / total_log if total_log else 0.0,
            }
        )
        previous = float(equity)
    return {"years": rows, "total_log": total_log}


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def equal_weight(
    data: dict[str, pd.DataFrame],
    dates: list[date],
    grid: set[date],
    pool: dict[str, str],
    *,
    start: date = START,
    end: date = END,
    cost: float = COST,
) -> pd.DataFrame:
    index = {code: {td: i for i, td in enumerate(data[code].trade_date)} for code in pool}
    cash = INITIAL
    shares = dict.fromkeys(pool, 0.0)
    rows = []
    for td in dates:
        if td < start:
            continue
        if td > end:
            break
        if td in grid:
            equity = cash + sum(
                shares[code] * float(data[code].iloc[index[code][td]].open) for code in pool
            )
            target = equity / len(pool)
            for code in pool:
                price = float(data[code].iloc[index[code][td]].open)
                current = shares[code] * price
                delta = target - current
                cash -= delta + abs(delta) * cost
                shares[code] += delta / price
        equity = cash + sum(
            shares[code] * float(data[code].iloc[index[code][td]].close) for code in pool
        )
        rows.append({"trade_date": td, "equity": equity})
    return pd.DataFrame(rows)


def buy_hold(
    data: dict[str, pd.DataFrame],
    code: str,
    dates: list[date],
    *,
    start: date = START,
    end: date = END,
) -> pd.DataFrame:
    index = {td: i for i, td in enumerate(data[code].trade_date)}
    cash = INITIAL
    shares = 0.0
    rows = []
    for td in dates:
        if td < start:
            continue
        if td > end:
            break
        if shares == 0.0:
            price = float(data[code].iloc[index[td]].open)
            shares = cash * (1 - COST) / price
            cash = 0.0
        rows.append({"trade_date": td, "equity": shares * float(data[code].iloc[index[td]].close)})
    return pd.DataFrame(rows)


def same_close_reference(
    data: dict[str, pd.DataFrame], pool: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with patch.object(rq, "ETF_POOL", pool):
        result = run_full_pool_strategy(data, v4.V4_PARAMS, cost_multiplier=1.0)
    curve = result["equity_curve"].copy()
    curve = curve.rename(columns={"equity": "equity"})
    return curve, result["metrics"]


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def write_charts(results: dict[str, Any], out: Path) -> None:
    plt.switch_backend("Agg")
    styles = {
        "baseline_1x": ("Baseline (current CB)", "#c0392b"),
        "peak_reset_1x": ("CB peak reset", "#1f7a4d"),
        "same_close_1x": ("Same-close reference", "#8e44ad"),
        "ew7": ("Equal-weight 7 ETF", "#2c7fb8"),
        "bh159915": ("ChiNext buy & hold", "#7f8c8d"),
    }
    figure, (equity_ax, drawdown_ax) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for name, (label, color) in styles.items():
        curve = results[name]["curve"]
        dates = pd.to_datetime(curve.trade_date)
        equity_ax.plot(dates, curve.equity, label=label, color=color, linewidth=1.4)
        if name in {"baseline_1x", "peak_reset_1x", "same_close_1x"}:
            _, drawdown = drawdown_path(curve)
            drawdown_ax.plot(dates, drawdown[1:] * 100, color=color, linewidth=1.2)
    equity_ax.set_yscale("log")
    equity_ax.axhline(INITIAL, color="#555555", linestyle="--", linewidth=1)
    equity_ax.set_title("V4 original pool: full history 2020-06-19 to 2026-10-30 (next-open, 1x)")
    equity_ax.set_ylabel("Equity (CNY, log scale)")
    equity_ax.grid(alpha=0.3)
    equity_ax.legend(loc="upper left", fontsize=9)
    drawdown_ax.set_ylabel("Drawdown (%)")
    drawdown_ax.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / "equity_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    years = [row["year"] for row in results["peak_reset_1x"]["yearly"]]
    baseline_values = [row["return"] * 100 for row in results["baseline_1x"]["yearly"]]
    reset_values = [row["return"] * 100 for row in results["peak_reset_1x"]["yearly"]]
    figure, ax = plt.subplots(figsize=(12, 5))
    width = 0.38
    positions = np.arange(len(years))
    ax.bar(positions - width / 2, baseline_values, width, label="Baseline", color="#c0392b")
    ax.bar(positions + width / 2, reset_values, width, label="CB peak reset", color="#1f7a4d")
    ax.set_xticks(positions, [str(year) for year in years])
    ax.set_ylabel("Calendar-year return (%)")
    ax.set_title("Yearly returns: baseline vs CB peak-reset")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    figure.tight_layout()
    figure.savefig(out / "yearly_returns.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def fmt_money(value: float) -> str:
    return f"{value:,.2f}"


def fmt_year_counts(counts: dict[str, int] | None) -> str:
    if not counts:
        return "—"
    return "、".join(f"{year}:{count}" for year, count in counts.items())


def write_report(results: dict[str, Any], data_audit: dict[str, Any], out: Path) -> None:
    baseline = results["baseline_1x"]["metrics"]
    reset = results["peak_reset_1x"]["metrics"]
    same_close = results["same_close_1x"]["metrics"]
    ew7 = results["ew7"]["metrics"]
    lines: list[str] = []
    lines.append("# V4 原池策略专业审计（全历史、分段、滚动、零拟合前推与压力测试）\n")
    lines.append(
        "> 研究审计，不修改生产参数、不写账户、不下单。所有收益为规则回溯，不是账户实绩。\n"
    )
    lines.append(
        "> 池子口径：当前 7 候选池是 2026-10-18 恢复后的口径，回溯到 2020；"
        "历史生产池曾变动，本报告不是逐日部署拼接。\n"
    )
    lines.append("## 结论先行\n")
    lines.append(
        f"1. **当前最大问题不是选品，而是 −30% 组合熔断的状态机。** 在次日开盘口径下，"
        f"当前风控触发 {results['baseline_1x']['diagnostics']['emergency_count']} 次，"
        f"全周期 {fmt_pct(baseline['total_return'])}、"
        f"最大回撤 {fmt_pct(baseline['max_drawdown'])}、"
        f"{results['baseline_1x']['diagnostics']['trade_legs']} 条成交腿；"
        f"仅改熔断状态机（同一信号/同一输入，peak reset），结果变为 "
        f"{fmt_pct(reset['total_return'])}、{fmt_pct(reset['max_drawdown'])}、"
        f"{results['peak_reset_1x']['diagnostics']['trade_legs']} 条腿。\n"
    )
    lines.append(
        f"2. **当前基线在成本压力下失效。** 1x/2x/3x 成本下总收益分别为 "
        f"{fmt_pct(results['baseline_1x']['metrics']['total_return'])} / "
        f"{fmt_pct(results['baseline_2x']['metrics']['total_return'])} / "
        f"{fmt_pct(results['baseline_3x']['metrics']['total_return'])}；"
        f"peak-reset 为 {fmt_pct(results['peak_reset_1x']['metrics']['total_return'])} / "
        f"{fmt_pct(results['peak_reset_2x']['metrics']['total_return'])} / "
        f"{fmt_pct(results['peak_reset_3x']['metrics']['total_return'])}。"
        "高换手不是主因，熔断反复触发才是。\n"
    )
    lines.append(
        f"3. **执行口径仍有不确定性。** 同期 same-close 参考为 "
        f"CAGR {fmt_pct(same_close['cagr'])}、回撤 {fmt_pct(same_close['max_drawdown'])}；"
        f"次日开盘 peak-reset 为 CAGR {fmt_pct(reset['cagr'])}、"
        f"回撤 {fmt_pct(reset['max_drawdown'])}。"
        "14:50 实际成交价介于两者之间，但缺盘中快照，不能把 90%+ 的 same-close CAGR "
        "当成可实现收益。\n"
    )
    lines.append(
        f"4. **单资产集中度是结构性风险。** V4 任何时点只持有一只 ETF；peak-reset 全周期最大回撤 "
        f"{fmt_pct(reset['max_drawdown'])}，最差单日 {fmt_pct(reset['worst_day'])}，"
        f"最差 20 日 {fmt_pct(reset['worst_20d'])}。等权 7 只 ETF 基准为 "
        f"CAGR {fmt_pct(ew7['cagr'])}、回撤 {fmt_pct(ew7['max_drawdown'])}、"
        f"Sharpe {ew7['sharpe']:.2f}；V4 的收益更高，但风险预算完全不同的产品。\n"
    )
    lines.append(
        f"5. **收益高度集中在 2025–2026。** peak-reset 的 2025 年 "
        f"{fmt_pct(results['peak_reset_1x']['yearly'][5]['return'])}、2026 年至 10/30 "
        f"{fmt_pct(results['peak_reset_1x']['yearly'][6]['return'])}；"
        "2020–2024 虽然每年为正，但量级小得多。不能把 2025 的尾部当成稳定期望。\n"
    )
    lines.append(
        "6. **严格零样本不存在，而且封存样本已经被全历史回放覆盖。** 本报告的全周期回放"
        "客观经过 2023–2024，不能再声称该段“未回放”；本次不把 2023–2024 作为验收依据。"
        "能提供的是“零拟合年度前推”（每年 10 万重启、参数冻结），真正的零样本只能从 "
        "2026-11-02 之后重新积累 252 个交易日。\n"
    )
    lines.append("## 全周期结果（2020-06-19 → 2026-10-30，次日开盘，1x）\n")
    lines.append(
        "| 方案 | 期末权益 | 总收益 | CAGR | 年化波动 | Sharpe | 最大回撤 | Calmar | 成交腿 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, label in [
        ("baseline_1x", "当前 V4 风控（次日开盘）"),
        ("cb_off_1x", "关闭熔断"),
        ("peak_reset_1x", "熔断后重置高水位"),
        ("same_close_1x", "same-close 参考"),
        ("ew7", "等权 7 只 ETF"),
        ("bh159915", "创业板买入持有"),
    ]:
        item = results[name]
        item_metrics = item["metrics"]
        legs = item.get("diagnostics", {}).get("trade_legs", item.get("trade_legs", "—"))
        lines.append(
            f"| {label} | {fmt_money(item_metrics['final_equity'])} | "
            f"{fmt_pct(item_metrics['total_return'])} | {fmt_pct(item_metrics['cagr'])} | "
            f"{fmt_pct(item_metrics['ann_vol'])} | {item_metrics['sharpe']:.2f} | "
            f"{fmt_pct(item_metrics['max_drawdown'])} | {item_metrics['calmar']:.2f} | {legs} |"
        )
    lines.append(
        f"\n当前 V4 峰值 {fmt_money(baseline['peak_equity'])}（{baseline['peak_date']}）→ 谷值 "
        f"{fmt_money(baseline['trough_equity'])}（{baseline['trough_date']}），水下 "
        f"{baseline['underwater_days']} 个交易日；peak-reset 峰值 {fmt_money(reset['peak_equity'])}"
        f"（{reset['peak_date']}）→ 谷值 {fmt_money(reset['trough_equity'])}"
        f"（{reset['trough_date']}），水下 {reset['underwater_days']} 个交易日。"
        f"peak-reset 最差单日 {fmt_pct(reset['worst_day'])}、"
        f"最差 5 日 {fmt_pct(reset['worst_5d'])}、"
        f"最差 20 日 {fmt_pct(reset['worst_20d'])}。\n"
    )
    lines.append("\n### 执行口径 × 风控消融\n")
    lines.append("| 口径 / 风控 | CAGR | 最大回撤 | 成交腿 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, label in [
        ("same_close_1x", "same-close，熔断未触发"),
        ("cb_off_1x", "次日开盘，关闭熔断"),
        ("peak_reset_1x", "次日开盘，peak-reset"),
        ("baseline_1x", "次日开盘，当前熔断"),
    ]:
        item = results[name]["metrics"]
        legs = results[name].get("diagnostics", {}).get("trade_legs", "—")
        lines.append(
            f"| {label} | {fmt_pct(item['cagr'])} | {fmt_pct(item['max_drawdown'])} | {legs} |"
        )
    lines.append(
        "\n*CAGR 统一按 252 交易日/年。项目旧报告使用日历年口径，same-close 约 88–91%，"
        "不是同一分母。peak-reset 按日历年口径约 66.6%。same-close 行来自另一个回放引擎"
        "（同日收盘成交），差异包含执行时点"
        "与引擎实现两部分，不等同于纯执行效应。*\n"
    )
    lines.append("\n### 熔断行为诊断\n")
    base_diag = results["baseline_1x"]["diagnostics"]
    reset_diag = results["peak_reset_1x"]["diagnostics"]
    lines.append("| 指标 | 当前 V4 | peak-reset |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| 触发次数 | {base_diag['emergency_count']} | {reset_diag['emergency_count']} |")
    lines.append(f"| 首次触发 | {base_diag['first_emergency']} | {reset_diag['first_emergency']} |")
    lines.append(f"| 成交腿 | {base_diag['trade_legs']} | {reset_diag['trade_legs']} |")
    lines.append(
        f"| 511220 天数占比 | {base_diag['bond_day_fraction'] * 100:.1f}% | "
        f"{reset_diag['bond_day_fraction'] * 100:.1f}% |"
    )
    lines.append(
        f"| 511880 天数占比 | {base_diag['money_day_fraction'] * 100:.1f}% | "
        f"{reset_diag['money_day_fraction'] * 100:.1f}% |"
    )
    lines.append(
        f"| 估算费用 | {fmt_money(base_diag['estimated_cost_paid'])} | "
        f"{fmt_money(reset_diag['estimated_cost_paid'])} |"
    )
    lines.append(
        f"| 费用/平均权益 | {base_diag['cost_drag'] * 100:.1f}% | "
        f"{reset_diag['cost_drag'] * 100:.1f}% |"
    )
    lines.append(f"| 换手倍数 | {base_diag['turnover_x']:.0f}x | {reset_diag['turnover_x']:.0f}x |")
    lines.append(
        f"| 触发年份 | {fmt_year_counts(base_diag['emergency_by_year'])} | "
        f"{fmt_year_counts(reset_diag['emergency_by_year'])} |"
    )
    lines.append(
        "\n*511220 也是正常轮动候选，不能把它的持仓天数全部解释为“防御”；"
        "本表拆分 511220/511880，避免把正常轮动误读成风控。触发次数也不等于成交次数；"
        "791/233 腿是实际成交腿。*\n"
    )
    lines.append("\n## 逐年收益与年内回撤（连续资金曲线）\n")
    lines.append(
        "| 年份 | 当前 V4 收益 | 当前 V4 年内回撤 | peak-reset 收益 | peak-reset 年内回撤 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for base_row, reset_row in zip(
        results["baseline_1x"]["yearly"], results["peak_reset_1x"]["yearly"], strict=True
    ):
        lines.append(
            f"| {base_row['year']} | {fmt_pct(base_row['return'])} | "
            f"{fmt_pct(base_row['max_drawdown'])} | {fmt_pct(reset_row['return'])} | "
            f"{fmt_pct(reset_row['max_drawdown'])} |"
        )
    lines.append(
        "\n注意：连续曲线里 2023/2024 仍为负，是 2022 触发熔断后高水位未重置的余波；"
        "年度重启每年重置高水位，所以 2023/2024 转正。这个差异本身就是 P0 问题的证据，"
        "不是两张表互相矛盾。\n"
    )
    lines.append("\n## 零拟合年度前推（每年初 10 万重启，参数冻结）\n")
    lines.append("| 年份 | 当前 V4 收益 / 回撤 / 腿数 | peak-reset 收益 / 回撤 / 腿数 |")
    lines.append("| --- | --- | --- |")
    for year in [row["year"] for row in results["peak_reset_1x"]["restart_yearly"]]:
        base_row = next(
            row for row in results["baseline_1x"]["restart_yearly"] if row["year"] == year
        )
        reset_row = next(
            row for row in results["peak_reset_1x"]["restart_yearly"] if row["year"] == year
        )
        lines.append(
            f"| {year} | {fmt_pct(base_row['return'])} / {fmt_pct(base_row['max_drawdown'])} / "
            f"{base_row['trade_legs']} | {fmt_pct(reset_row['return'])} / "
            f"{fmt_pct(reset_row['max_drawdown'])} / {reset_row['trade_legs']} |"
        )
    lines.append(
        "\n这是“零拟合”而非“零样本”：每年起点重启、参数不重新拟合，但策略设计历史已被看过。"
        "不能替代真正的未来样本外。\n"
    )
    lines.append("\n## 滚动窗口（peak-reset，全周期）\n")
    lines.append(
        "| 窗口 | 窗口数 | 负收益占比 | 收益 min / median / max | Sharpe 中位 | 回撤中位 / 最差 |"
    )
    lines.append("| --- | ---: | ---: | --- | ---: | --- |")
    for summary in results["peak_reset_1x"]["rolling"]:
        lines.append(
            f"| {summary['window']} 日 | {summary['windows']} | "
            f"{summary['negative_fraction'] * 100:.1f}% | "
            f"{fmt_pct(summary['return_min'])} / {fmt_pct(summary['return_median'])} / "
            f"{fmt_pct(summary['return_max'])} | {summary['sharpe_median']:.2f} | "
            f"{fmt_pct(summary['max_drawdown_median'])} / "
            f"{fmt_pct(summary['max_drawdown_worst'])} |"
        )
    lines.append("\n滚动窗口为重叠窗口，不是独立样本；只用于看路径稳定性，不能据此计算显著性。\n")
    lines.append("\n## 成本压力\n")
    lines.append("| 方案 | 1x | 2x | 3x |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(
        f"| 当前 V4 总收益 | {fmt_pct(results['baseline_1x']['metrics']['total_return'])} | "
        f"{fmt_pct(results['baseline_2x']['metrics']['total_return'])} | "
        f"{fmt_pct(results['baseline_3x']['metrics']['total_return'])} |"
    )
    lines.append(
        f"| 当前 V4 最大回撤 | {fmt_pct(results['baseline_1x']['metrics']['max_drawdown'])} | "
        f"{fmt_pct(results['baseline_2x']['metrics']['max_drawdown'])} | "
        f"{fmt_pct(results['baseline_3x']['metrics']['max_drawdown'])} |"
    )
    lines.append(
        f"| peak-reset 总收益 | {fmt_pct(results['peak_reset_1x']['metrics']['total_return'])} | "
        f"{fmt_pct(results['peak_reset_2x']['metrics']['total_return'])} | "
        f"{fmt_pct(results['peak_reset_3x']['metrics']['total_return'])} |"
    )
    lines.append(
        f"| peak-reset 最大回撤 | {fmt_pct(results['peak_reset_1x']['metrics']['max_drawdown'])} | "
        f"{fmt_pct(results['peak_reset_2x']['metrics']['max_drawdown'])} | "
        f"{fmt_pct(results['peak_reset_3x']['metrics']['max_drawdown'])} |"
    )
    lines.append("\n## 网格相位与池子敏感性（peak-reset，1x）\n")
    lines.append("| 相位 | 总收益 | CAGR | 最大回撤 | 成交腿 | 熔断触发 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for phase in range(5):
        item = results[f"phase_{phase}"]
        lines.append(
            f"| {phase} | {fmt_pct(item['metrics']['total_return'])} | "
            f"{fmt_pct(item['metrics']['cagr'])} | {fmt_pct(item['metrics']['max_drawdown'])} | "
            f"{item['diagnostics']['trade_legs']} | {item['diagnostics']['emergency_count']} |"
        )
    phase_totals = [results[f"phase_{phase}"]["metrics"]["total_return"] for phase in range(5)]
    phase_cagrs = [results[f"phase_{phase}"]["metrics"]["cagr"] for phase in range(5)]
    phase_mdds = [results[f"phase_{phase}"]["metrics"]["max_drawdown"] for phase in range(5)]
    lines.append(
        f"\n5 个相位的平均总收益 {fmt_pct(float(np.mean(phase_totals)))}、"
        f"平均 CAGR {fmt_pct(float(np.mean(phase_cagrs)))}、最差回撤 "
        f"{fmt_pct(float(min(phase_mdds)))}。相位 0 是 5 个相位中最好的；"
        "把相位平均与次日开盘放在一起，V4 的合理中心预期更接近 CAGR 60% 左右，"
        "而不是 same-close 的 90%+。\n"
    )
    lines.append("\n| 池子 | 1x 总收益 / 回撤 | 2x 总收益 / 回撤 |")
    lines.append("| --- | --- | --- |")
    for label, key1, key2 in [
        ("含油 7 候选", "peak_reset_1x", "peak_reset_2x"),
        ("删油 6 候选", "peak_reset_nooil_1x", "peak_reset_nooil_2x"),
    ]:
        lines.append(
            f"| {label} | {fmt_pct(results[key1]['metrics']['total_return'])} / "
            f"{fmt_pct(results[key1]['metrics']['max_drawdown'])} | "
            f"{fmt_pct(results[key2]['metrics']['total_return'])} / "
            f"{fmt_pct(results[key2]['metrics']['max_drawdown'])} |"
        )
    lines.append("\n## 收益集中度（peak-reset，对数收益贡献）\n")
    conc = concentration(results["peak_reset_1x"]["curve"])
    lines.append("| 年份 | 对数收益贡献 | 占总对数收益 |")
    lines.append("| --- | ---: | ---: |")
    for row in conc["years"]:
        lines.append(
            f"| {row['year']} | {row['log_contribution']:.4f} | "
            f"{row['share_of_total_log'] * 100:.1f}% |"
        )
    lines.append("\n## 问题、证据与方案\n")
    lines.append("| 优先级 | 问题 | 证据 | 方案 | 验收 |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        "| P0 | −30% 熔断高水位不重置，触发后反复砍仓/追回 | 当前 269 次触发、791 腿、+25%/−64%；"
        "peak-reset 1 次触发（phase 0；phase 2 为 2 次）、233 腿、+2,472%/−32% | "
        "熔断退出后重置风险高水位；"
        "或要求收复前高才允许再风险；加重复触发单测 | 2022 段不再出现 137 腿；"
        "2x 正收益且回撤不劣于 −35%；3x MDD −36.4% 为压力边界，不宣称通过 |"
    )
    lines.append(
        "| P0 | 当前基线对成本高度敏感 | 2x −59%、3x −86%，而 peak-reset 2x/3x 仍为正 | "
        "先修熔断；再把成本压测列为上线闸门 | 2x 正收益、3x 不崩 |"
    )
    lines.append(
        "| P1 | 执行口径未验证：same-close 93.6% vs 次日开盘 70.3% | 两种口径差约 23pp CAGR；"
        "14:50 真实成交价缺快照 | 采集 14:50/14:55 快照与真实成交，量化滑点；"
        "以次日开盘为保守基线 | 20–60 个交易日内完成口径校准 |"
    )
    lines.append(
        "| P1 | 单资产 100% 集中 | 全周期回撤 −32%，最差单日/20 日见上表 | "
        "预注册单标的上限或 2–3 资产分散/波动率目标；不直接调到历史最优 | "
        "MDD 改善 ≥5pp 且 CAGR 保留 ≥70% |"
    )
    lines.append(
        "| P1 | 5 日网格相位敏感 | 5 个相位总收益 14x–26x，回撤 −26% 到 −39% | "
        "固定相位由预注册决定；报告相位平均而非只报相位 0 | 相位平均仍正；不选最好相位 |"
    )
    lines.append(
        "| P1 | 收益集中在 2025–2026 | 连续曲线 2025 +211%、2026 +105%；"
        "年度重启 2025 +229%、2026 +99%；2020–2024 量级小 | "
        "按 regime 分层监控；不把尾部年份当稳定期望 | 2027 前瞻期不出现负收益自然年 |"
    )
    lines.append(
        "| P2 | 数据缺口与 LOF 溢价 | 2021-02-08、2022-01-13、2026-01-22/30、03-24 缺行；"
        "501018 为 LOF | 补数据/交叉源；LOF 溢价监控；缺行日 fail-closed | "
        "缺口清零后才宣称完整回测 |"
    )
    lines.append(
        "| P1 | 封存样本已被全历史回放覆盖，真正零样本为零 | 2023–2024 在本报告全周期路径内；"
        "此前项目全历史研究也已覆盖 | 停止用全历史回放冒充 OOS；从 2026-11-02 起预注册"
        "252 交易日前瞻；期间不改参数 | 63/126/252 日三次检查，不提前调参；"
        "2023–2024 不再作为验证卖点 |"
    )
    lines.append(
        "\n> **当前账户尚未触发 −30%，所以这是潜在缺陷，不是已经发生的账户亏损；"
        "但一旦触发，历史路径显示它会锁死。**\n"
        "\n> **peak-reset 的边界：** 它不是免费午餐。重置高水位后，第二次熔断的门槛变成"
        "“重置点再跌 30%”，相当于原高点的约 −51%；本次路径没有触发第二次，但不能保证未来。"
        "必须与“收复前高才允许再风险”的 recovery gate 并行影子比较，不能直接上线。\n"
    )
    lines.append("\n## 执行方案（不改生产的顺序）\n")
    lines.append(
        "1. **本周：** 冻结本报告；为熔断增加“重复触发/高水位重置/冷却再入场”单测；"
        "peak-reset 只做影子，不上生产。\n"
        "2. **2–4 周：** 采集 14:50/14:55 快照与真实成交，量化执行差异；补齐 5 个缺行；"
        "把 2x 成本列为上线闸门。\n"
        "3. **1–3 个月：** 预注册一个风险预算候选（单标的上限或 2–3 资产分散/波动率目标），"
        "只允许一个候选进入影子，不做参数网格。\n"
        "4. **252 个交易日：** 真零样本复核；期间不改参数；63/126/252 日三次检查。\n"
    )
    lines.append(
        "\n**当前生产判断：** 不因为本报告直接替换 V4，也不把 peak-reset 的历史收益当成"
        "已验证收益。生产保持 V4 原规则不变；先做熔断状态机单测与影子，再谈是否上线。"
        "如果只能做一件事：本周只做“熔断重复触发”单测 + peak-reset 影子，不做参数调优。\n"
    )
    lines.append("\n## 什么情况下推翻本方案\n")
    lines.append(
        "- peak-reset 前瞻 63 个交易日跑输当前基线超过 10pp，或回撤没有改善 → 否决。\n"
        "- 14:50 真实成交相对收盘的滑点超过单边 30bps → 重新评估全部收益。\n"
        "- 5 个缺行未补齐、或 2026 数据继续缺行 → 不宣称完整回测。\n"
        "- 任何新参数被 2023–2024 的回放结果影响过 → 该候选作废，不得进入前瞻。\n"
    )
    lines.append("\n## 零样本与封存边界\n")
    lines.append(
        "- **2023-01-01～2024-12-31 不能再称为未触碰封存样本。** 本次全历史回放（以及项目"
        "既有全历史研究）客观经过该段；本报告不把该段作为验收依据，也不据其调参。"
        "这是治理缺口，不是可以靠措辞绕过的证据。\n"
        "- **真零样本**从 2026-11-02 起算，预注册 252 个交易日：V4 原池、次日开盘、"
        "1x 成本；baseline 与 peak-reset 并行影子；期间禁止改参数。\n"
        "- **零拟合年度前推**见上表：每年 10 万重启、参数冻结；它证明路径不依赖单一年份，"
        "但不等于独立样本外。\n"
    )
    lines.append("## 数据审计\n")
    lines.append(
        f"- 公共交易日 {data_audit['common_days']} 天；缺行 {len(data_audit['missing'])} 天。"
        f"缺行：{'、'.join(data_audit['missing']) if data_audit['missing'] else '无'}。\n"
        "- 输入 SHA256 已按 `artifacts/v4_since_entry_20261102/inputs.json` 校验。"
        "缺行会让后续 5 日网格相位偏移；相位压力测试覆盖 0–4 日平移，"
        "但不能完全替代完整日历。\n"
    )
    lines.append("## 限制与未覆盖\n")
    lines.append(
        "- 14:50 盘中价、真实成交滑点、涨跌停/停牌申报拒单未审；日线收盘/次日开盘只是代理。\n"
        "- 501018 为 LOF，溢价收敛、申赎限制、流动性冲击未建模；期末集中度仍是 100% 单资产。\n"
        "- 5 个缺行日未补造；公共交集日历会让 5 日网格相位在缺行后偏移。\n"
        "- 单一路径 + 5 个相位，不能把相位 0 的 +2,472% 当期望；相位平均 CAGR 约 61.9%。\n"
        "- 6.4 年、233 条腿，未做显著性检验；不声称 Sharpe 1.65 已证明可持续。\n"
        "- same-close 参考来自不同回放引擎，不能当作纯执行时点对照。\n"
    )
    lines.append("## 交付文件\n")
    lines.append("- `summary.json`：全部配置指标、诊断、滚动与分段。")
    lines.append("- `manifest.json`：脚本/引擎/风控/输入 SHA256 与生成时间。")
    lines.append("- `metrics.csv`、`yearly.csv`、`rolling_*.csv`、`trades/*.csv`。")
    lines.append("- `equity_drawdown.png`、`yearly_returns.png`。")
    lines.append("- `REPORT.md`：本报告。\n")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.output
    if out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out}; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "trades").mkdir(exist_ok=True)

    data, dates, _manifest = load_data()
    pool = dict(rq.ETF_POOL)
    no_oil = {code: name for code, name in pool.items() if code != "501018"}
    data_audit = {
        "common_days": len([td for td in dates if START <= td <= END]),
        "missing": [
            str(td)
            for td in sorted(set.union(*(set(frame.trade_date) for frame in data.values())))
            if td not in set(dates) and td >= START
        ],
    }
    results: dict[str, Any] = {}

    def record(name: str, result: dict[str, Any], *, cost_multiplier: float = 1.0) -> None:
        prices = {code: float(frame.close.iloc[-1]) for code, frame in data.items()}
        enriched = summarize_trades(list(result["trades"]), prices, cost=COST * cost_multiplier)[
            "trades"
        ]
        item: dict[str, Any] = {
            "curve": result["curve"],
            "trades": enriched,
            "metrics": metrics(result["curve"]),
            "diagnostics": trade_diagnostics(result, data, cost_multiplier=cost_multiplier),
            "yearly": yearly_continuous(result["curve"]),
        }
        results[name] = item

    configs: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        ("baseline_1x", {"risk_mode": "baseline", "cost_multiplier": 1.0}, {}),
        ("baseline_2x", {"risk_mode": "baseline", "cost_multiplier": 2.0}, {}),
        ("baseline_3x", {"risk_mode": "baseline", "cost_multiplier": 3.0}, {}),
        ("cb_off_1x", {"risk_mode": "cb_off", "cost_multiplier": 1.0}, {}),
        ("peak_reset_1x", {"risk_mode": "peak_reset", "cost_multiplier": 1.0}, {}),
        ("peak_reset_2x", {"risk_mode": "peak_reset", "cost_multiplier": 2.0}, {}),
        ("peak_reset_3x", {"risk_mode": "peak_reset", "cost_multiplier": 3.0}, {}),
        (
            "peak_reset_nooil_1x",
            {"risk_mode": "peak_reset", "cost_multiplier": 1.0, "pool": no_oil},
            {},
        ),
        (
            "peak_reset_nooil_2x",
            {"risk_mode": "peak_reset", "cost_multiplier": 2.0, "pool": no_oil},
            {},
        ),
    ]
    for phase in range(5):
        configs.append(
            (
                f"phase_{phase}",
                {"risk_mode": "peak_reset", "cost_multiplier": 1.0, "phase": phase},
                {},
            )
        )

    for name, kwargs, _extra in configs:
        kwargs = dict(kwargs)
        pool_used = kwargs.pop("pool", pool)
        result = run_v4(data, dates, pool=pool_used, **kwargs)
        record(name, result, cost_multiplier=kwargs.get("cost_multiplier", 1.0))
        for key in ("baseline_1x", "peak_reset_1x"):
            if name == key:
                result["restart_rows"] = []
                for year in range(2020, 2027):
                    year_start = max(date(year, 1, 1), START)
                    year_end = END if year == 2026 else date(year, 12, 31)
                    year_result = run_v4(
                        data,
                        dates,
                        pool=pool,
                        risk_mode=kwargs.get("risk_mode", "baseline"),
                        cost_multiplier=1.0,
                        start=year_start,
                        end=year_end,
                    )
                    row = metrics(year_result["curve"])
                    row["year"] = year
                    row["return"] = row["total_return"]
                    row["trade_legs"] = len(year_result["trades"])
                    result["restart_rows"].append(row)
                results[name]["restart_yearly"] = result["restart_rows"]

    # Same-close reference and benchmarks.
    same_curve, same_metrics = same_close_reference(data, pool)
    results["same_close_1x"] = {
        "curve": same_curve,
        "metrics": metrics(same_curve),
        "diagnostics": {"trade_legs": same_metrics.get("trade_legs", 0)},
        "yearly": yearly_continuous(same_curve),
    }
    for name, curve in [
        ("ew7", equal_weight(data, dates, grid_for(dates), pool)),
        ("ew6", equal_weight(data, dates, grid_for(dates), no_oil)),
        ("bh159915", buy_hold(data, "159915", dates)),
    ]:
        results[name] = {
            "curve": curve,
            "metrics": metrics(curve),
            "diagnostics": {"trade_legs": "—"},
            "yearly": yearly_continuous(curve),
        }

    for name in ("baseline_1x", "peak_reset_1x"):
        rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for window in (126, 252):
            window_rows, summary = rolling_stats(results[name]["curve"], window)
            rows.extend({"config": name, **row} for row in window_rows)
            summaries.append(summary)
        results[name]["rolling"] = summaries
        pd.DataFrame(rows).to_csv(out / f"rolling_{name}.csv", index=False)

    # Save curves, metrics and diagnostics.
    curves = []
    metric_rows = []
    for name, item in results.items():
        curve = item["curve"].copy()
        curve["config"] = name
        curves.append(curve)
        metric_rows.append({"config": name, **item["metrics"], **item.get("diagnostics", {})})
        if name in {"baseline_1x", "peak_reset_1x"}:
            pd.DataFrame(item["yearly"]).to_csv(out / f"yearly_{name}.csv", index=False)
            pd.DataFrame(item["rolling"]).to_csv(out / f"rolling_summary_{name}.csv", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(out / "curves.parquet", index=False)
    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {"config": name, **row}
            for name, item in results.items()
            for row in item.get("yearly", [])
        ]
    ).to_csv(out / "yearly.csv", index=False)
    for name in ("baseline_1x", "peak_reset_1x"):
        pd.DataFrame(results[name]["restart_yearly"]).to_csv(
            out / f"restart_{name}.csv", index=False
        )
        pd.DataFrame(results[name]["trades"]).to_csv(out / "trades" / f"{name}.csv", index=False)

    summary = {
        "start": str(START),
        "end": str(END),
        "initial_capital": INITIAL,
        "cost_per_side": COST,
        "execution": "T close signal; next common open; marked at close; no terminal sale",
        "data_audit": data_audit,
        "results": {
            name: {
                "metrics": item["metrics"],
                "diagnostics": item.get("diagnostics", {}),
                "yearly": item.get("yearly", []),
                "restart_yearly": item.get("restart_yearly", []),
                "rolling": item.get("rolling", []),
            }
            for name, item in results.items()
        },
        "concentration": concentration(results["peak_reset_1x"]["curve"]),
        "research_only": True,
        "locked_test_touched": True,
        "locked_test_replayed": True,
        "locked_test_note": (
            "Full-history replay includes 2023-2024; do not treat that period as untouched OOS."
        ),
        "true_oos": False,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    )
    version_manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script_sha256": sha256(Path(__file__).resolve()),
        "engine_sha256": sha256(ROOT / "scripts/review_rotation_20260916.py"),
        "engine_note": (
            "Actual executed file; repository may contain uncommitted edits. "
            "Hash pins the bytes used for this audit."
        ),
        "risk_layer_sha256": sha256(ROOT / "scripts/risk_overrides.py"),
        "live_signal_sha256": sha256(ROOT / "scripts/live_signal.py"),
        "v4_core_sha256": sha256(ROOT / "scripts/qixing_v4.py"),
        "input_manifest_sha256": sha256(INPUT_MANIFEST),
        "input_sha256": {
            code: sha256(INPUTS / f"{code}.parquet") for code in [*rq.ETF_POOL, rq.DEFENSE]
        },
        "execution": "T close signal; next common open; final close mark",
        "locked_test_replayed": True,
        "true_oos": False,
    }
    (out / "manifest.json").write_text(
        json.dumps(version_manifest, ensure_ascii=False, indent=2, default=str)
    )
    write_charts(results, out)
    write_report(results, data_audit, out)
    print(
        json.dumps(
            {
                "baseline": results["baseline_1x"]["metrics"],
                "cb_off": results["cb_off_1x"]["metrics"],
                "peak_reset": results["peak_reset_1x"]["metrics"],
                "same_close": results["same_close_1x"]["metrics"],
                "ew7": results["ew7"]["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
