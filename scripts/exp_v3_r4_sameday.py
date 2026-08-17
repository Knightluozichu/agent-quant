"""V3 vs V3+R4提前换手 — 14:50 同日成交口径全量回测对比.

基座 = run_qixing_v3_same_day (生产对齐口径: 同日收盘成交 + 涨跌停检查 +
卖出失败卡仓 + 每日净值), 插入 R4 提前换手层:

  非调仓日: 检测昨日单日涨幅>thr 且 10日动量>0 的资产 (选动量评分最高者),
  若空仓/防御 → 直接入场; 若持有风险资产且事件资产评分 > 持仓评分+缓冲 → 提前换手.
  换手同样受涨跌停可交易检查约束 (与调仓日一致).

配置: V3基线 / V3+thr2.0%_换_b2% / V3+thr1.5%_换_b0%
输出: 期末金额/总收益/年化/夏普/回撤/年度/交易/触发次数
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from exp_short_window_patterns import close_matrix
from run_qixing_v3 import (
    DEFENSE,
    ETF_POOL,
    FEE,
    REBALANCE_DAYS,
    SLIPPAGE,
    calc_momentum_score,
    load_data,
    select_target,
)

WARMUP = 130
INITIAL_CAPITAL = 100_000.0


def run_v3_r4_sameday(
    data: dict,
    mat: dict,
    thr: float = 1.0,
    buffer: float = 0.0,
    mom_period: int = 10,
    start_idx: int = 0,
    end_idx: int | None = None,
    cost_multiplier: float = 1.0,
) -> dict:
    """14:50 同日口径 + R4 提前换手层 (thr=1.0 时等价纯 V3).

    Args:
        start_idx/end_idx: trading_dates (warmup后) 的相对索引区间, 用于公平分段对比.
    """
    common_dates: set = set()
    for code in ETF_POOL:
        if code not in data:
            continue
        ds = set(data[code]["trade_date"].tolist())
        common_dates = ds if not common_dates else common_dates & ds
    if DEFENSE in data:
        common_dates &= set(data[DEFENSE]["trade_date"].tolist())
    all_dates = sorted(common_dates)
    trading_dates = all_dates[WARMUP:]
    rebalance_set = set(trading_dates[::REBALANCE_DAYS])

    cash = float(INITIAL_CAPITAL)
    holding: str | None = None
    holding_shares = 0
    equity_history: list[dict] = []
    trade_log: list[dict] = []
    r4_events: list[dict] = []
    signal_counter = 0

    def _check_close_tradable(code: str, td) -> tuple[bool, str]:
        """收盘口径可交易检查: 有数据 + 未涨跌停 (与生产 same_day 一致)."""
        df = data[code]
        row = df[df["trade_date"] == td]
        if row.empty:
            return (False, f"{code} 在 {td} 无数据")
        price = float(row.iloc[0]["close"])
        if price <= 0:
            return (False, f"{code} 在 {td} 收盘价无效")
        hist = df[df["trade_date"] < td]
        if not hist.empty:
            prev_close = float(hist.iloc[-1]["close"])
            if prev_close > 0 and abs(price / prev_close - 1) >= 0.099:
                return (False, f"{code} 在 {td} 收盘涨跌停")
        return (True, "")

    def _sell(code: str, td, tag: str) -> bool:
        """卖出持仓 (可交易检查, 失败卡仓)."""
        nonlocal cash, holding, holding_shares
        can, reason = _check_close_tradable(code, td)
        if not can:
            trade_log.append(
                {
                    "date": str(td),
                    "action": "sell",
                    "code": code,
                    "status": "cancelled",
                    "reason": f"卖出失败: {reason}",
                    "tag": tag,
                }
            )
            return False
        price = float(data[code][data[code]["trade_date"] == td].iloc[0]["close"])
        amount = holding_shares * price * (1 - (FEE + SLIPPAGE) * cost_multiplier)
        cash += amount
        holding, holding_shares = None, 0
        trade_log.append(
            {
                "date": str(td),
                "action": "sell",
                "code": code,
                "shares": 0,
                "price": price,
                "amount": round(amount, 2),
                "status": "executed",
                "reason": "",
                "tag": tag,
            }
        )
        return True

    def _buy(code: str, td, tag: str) -> bool:
        """买入目标 (可交易检查)."""
        nonlocal cash, holding, holding_shares
        can, reason = _check_close_tradable(code, td)
        if not can:
            trade_log.append(
                {
                    "date": str(td),
                    "action": "buy",
                    "code": code,
                    "status": "cancelled",
                    "reason": f"买入失败: {reason}",
                    "tag": tag,
                }
            )
            return False
        price = float(data[code][data[code]["trade_date"] == td].iloc[0]["close"])
        shares = int(cash * 0.99 / price / 100) * 100
        if shares <= 0:
            return False
        cash -= shares * price * (1 + (FEE + SLIPPAGE) * cost_multiplier)
        holding, holding_shares = code, shares
        trade_log.append(
            {
                "date": str(td),
                "action": "buy",
                "code": code,
                "shares": shares,
                "price": price,
                "amount": round(shares * price * (1 + FEE + SLIPPAGE), 2),
                "status": "executed",
                "reason": "",
                "tag": tag,
            }
        )
        return True

    def _mom_score(code: str, td) -> float:
        close = data[code][data[code]["trade_date"] <= td]["close"].values.astype(float)
        if len(close) < 121:
            return -np.inf
        return float(calc_momentum_score(close))

    idx_of = {d: i for i, d in enumerate(all_dates)}  # 日期→矩阵索引 (事件fwd用)
    if end_idx is None:
        end_idx = len(trading_dates)

    for td in trading_dates[start_idx:end_idx]:
        if td in rebalance_set:
            # ===== 调仓日: V3 决策 (与生产 same_day 一致) =====
            etf_data_at_date = {}
            for code in [*list(ETF_POOL.keys()), DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() < WARMUP:
                    continue
                etf_data_at_date[code] = mask.sum() - 1
            target, _c, _s, _a = select_target(data, etf_data_at_date, holding)

            signal_counter += 1
            if target != holding:
                sell_ok = True
                if holding and holding in data:
                    sell_ok = _sell(holding, td, "rebalance")
                if sell_ok and target and target in data:
                    _buy(target, td, "rebalance")
        elif thr < 1.0:
            # ===== 非调仓日: R4 事件检测 (昨日涨幅>thr, 10日动量>0) =====
            i = idx_of[td]
            if i >= 2:
                best_code, best_s = None, -np.inf
                ev_ret = 0.0
                for code in ETF_POOL:
                    p1, p2 = mat[code][i - 1], mat[code][i - 2]
                    if not (p1 > 0 and p2 > 0 and np.isfinite(p1) and np.isfinite(p2)):
                        continue
                    r1 = p1 / p2 - 1.0
                    if r1 <= thr:
                        continue
                    if mom_period > 0:
                        if i < mom_period + 1:
                            continue
                        mom = mat[code][i - 1] / mat[code][i - 1 - mom_period] - 1.0
                        if mom <= 0:
                            continue
                    s = _mom_score(code, td)
                    if s > best_s:
                        best_s, best_code, ev_ret = s, code, r1

                if best_code:
                    if holding is None or holding == DEFENSE:
                        # 关键: 从货币基金入场须先卖出旧持仓 (否则持仓市值丢失)
                        if holding == DEFENSE:
                            _sell(DEFENSE, td, "r4_enter")
                        _buy(best_code, td, "r4_enter")
                        r4_events.append(
                            {
                                "date": str(td),
                                "type": "r4_enter",
                                "asset": best_code,
                                "prev_ret": round(float(ev_ret), 4),
                                "score": round(float(best_s), 4),
                                "idx": i,
                            }
                        )
                    elif holding in ETF_POOL:
                        cur_s = _mom_score(holding, td)
                        if best_s > cur_s + buffer:
                            sold = _sell(holding, td, "r4_switch")
                            if sold:
                                _buy(best_code, td, "r4_switch")
                                r4_events.append(
                                    {
                                        "date": str(td),
                                        "type": "r4_switch",
                                        "asset": best_code,
                                        "from": holding,
                                        "prev_ret": round(float(ev_ret), 4),
                                        "score": round(float(best_s), 4),
                                        "from_score": round(float(cur_s), 4),
                                        "idx": i,
                                    }
                                )

        # ===== 每日净值 (与生产一致) =====
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * float(row.iloc[0]["close"])
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

    if not equity_history:
        return {"error": "no data"}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year
    total_return = eq_df["equity"].iloc[-1] / INITIAL_CAPITAL - 1
    rets = eq_df["equity"].pct_change().dropna()
    ann_vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = float(((eq_df["equity"] - cummax) / cummax).min())
    yearly = {}
    prev_val = INITIAL_CAPITAL
    for year in sorted(eq_df["year"].unique()):
        ydf = eq_df[eq_df["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev_val) - 1
        cm = ydf["equity"].cummax()
        dd = float(((ydf["equity"] - cm) / cm).min())
        yearly[int(year)] = {"return": round(yr, 4), "max_dd": round(dd, 4)}
        prev_val = end_val

    # 事件后续收益
    for ev in r4_events:
        code, i = ev["asset"], ev["idx"]
        for h in (5, 20):
            j = i + h
            if j < len(mat[code]) and mat[code][i] > 0:
                ev[f"fwd{h}"] = round(float(mat[code][j] / mat[code][i] - 1.0), 4)

    n_exec = sum(1 for t in trade_log if t.get("status") == "executed")
    return {
        "total_return": round(float(total_return), 4),
        "final_value": round(float(eq_df["equity"].iloc[-1]), 0),
        "ann_return": round(float(ann_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(max_dd, 4),
        "yearly": yearly,
        "n_trades": n_exec,
        "n_cancelled": sum(1 for t in trade_log if t.get("status") == "cancelled"),
        "n_events": len(r4_events),
        "events": r4_events,
        "equity_curve": eq_df,
    }


def main() -> None:
    print("=" * 74)
    print("  V3 vs V3+R4提前换手 — 14:50 同日成交口径 全量回测")
    print("  基座: run_qixing_v3_same_day (涨跌停检查/卖出失败卡仓/每日净值)")
    print("=" * 74)

    data = load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    mat = close_matrix(data, dates)
    print(f"\n  数据: {len(dates)} 交易日 ({dates[0]} ~ {dates[-1]})")

    configs = [
        ("V3基线", {"thr": 1.0}),
        ("V3+thr2.0%_换_b2%", {"thr": 0.02, "buffer": 0.02}),
        ("V3+thr1.5%_换_b0%", {"thr": 0.015, "buffer": 0.0}),
    ]
    results = {}
    print(
        f"\n  {'配置':<20} {'期末金额':>12} {'总收益':>9} {'年化':>8} {'夏普':>6} "
        f"{'回撤':>8} {'交易':>4} {'取消':>4} {'触发':>4}"
    )
    for name, kw in configs:
        r = run_v3_r4_sameday(data, mat, **kw)
        results[name] = {k: v for k, v in r.items() if k not in ("events", "equity_curve")}
        results[name]["_events"] = r["events"]
        print(
            f"  {name:<20} {r['final_value']:>12,.0f} {r['total_return']:>+9.1%} "
            f"{r['ann_return']:>+8.1%} {r['sharpe']:>6.2f} {r['max_drawdown']:>8.1%} "
            f"{r['n_trades']:>4} {r['n_cancelled']:>4} {r['n_events']:>4}"
        )

    # 年度对比
    print("\n  年度收益对比:")
    base_y = results["V3基线"]["yearly"]
    print(f"  {'年份':<6} {'V3基线':>9} {'thr2.0%_b2%':>11} {'thr1.5%_b0%':>11}")
    for y in sorted(base_y.keys()):
        v = results["V3基线"]["yearly"].get(y, {}).get("return", 0)
        a = results["V3+thr2.0%_换_b2%"]["yearly"].get(y, {}).get("return", 0)
        b = results["V3+thr1.5%_换_b0%"]["yearly"].get(y, {}).get("return", 0)
        print(f"  {y:<6} {v:>+9.1%} {a:>+11.1%} {b:>+11.1%}")

    # 事件日志 (thr2.0%_b2% 前10次)
    print("\n  事件日志 (thr2.0%_换_b2%, 前10次):")
    evs = results["V3+thr2.0%_换_b2%"]["_events"]
    for e in evs[:10]:
        f5 = e.get("fwd5")
        print(
            f"    {e['date']} {e['type']} {ETF_POOL.get(e['asset'], '?')} "
            f"事件{e['prev_ret']:+.1%} → 5日 {f5 if f5 is None else f'{f5:+.1%}'}"
        )

    out_path = OUTPUT_DIR / "v3_r4_sameday_full.json"
    with open(out_path, "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != "_events"} for k, v in results.items()},
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
