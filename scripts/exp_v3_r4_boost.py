"""V3基线 vs V3+R4事件加速器 A/B (全量回测 + 滚动OOS对称 + 敏感性 + 事件日志).

R4 规律 (阶段1幸存, d=1.49): 单日暴涨>3% 后5日继续涨.
事件加速器 (正向用法, 突破固定调仓日的动态入场):
  - 非调仓日: 若空仓/防御, 且某资产昨日单日涨幅>阈值 且 10日动量>0 → 今日收盘入场
  - 多个触发时选 10日动量最高者 (与V3动量逻辑一致)
  - 入场后持有到下一调仓日, 由 V3 决策接管 (换仓/继续/切防御)
  - 无未来函数: 事件检测用昨日数据, 今日收盘成交

验证 (项目黄金标准):
  1. 全量回测: 10万→最终金额优先, 年化/夏普/回撤/Calmar/换手率
  2. 滚动 OOS 4段对称: 至少 3/4 段跑赢基线
  3. 参数敏感性: 事件阈值×动量确认开关 网格
  4. 事件日志: 每次加速入场 日期/资产/事件涨幅/动量/后续5-20日

输出: data/v9_results/v3_r4_boost_ab.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from exp_short_window_patterns import close_matrix, common_dates  # noqa: E402
from run_qixing_v3 import (  # noqa: E402
    DEFENSE,
    ETF_POOL,
    FEE,
    REBALANCE_DAYS,
    SLIPPAGE,
    load_data,
    select_target,
)

WARMUP = 130
INITIAL_CAPITAL = 100_000.0

# === 滚动 OOS 段 ===
TRAIN_DAYS = 260
TEST_DAYS = 260
BUFFER_DAYS = 20
STEP = (TRAIN_DAYS + TEST_DAYS) // 2

# === R4 加速器参数网格 ===
R4_THRESHOLDS = (0.02, 0.03, 0.04)
R4_MOM_CONFIRM = (True, False)


# --------------------------------------------------------------------------- #
# 回测引擎: V3 周频调仓 + 可选 R4 事件加速器 (日频动态入场)
# --------------------------------------------------------------------------- #
def run_v3_r4(
    data: dict,
    dates: list,
    mat: dict,
    start_idx: int,
    end_idx: int,
    use_r4: bool = False,
    thr: float = 0.03,
    mom_confirm: bool = True,
) -> dict:
    """日频净值回测: 调仓日 V3 引擎; 非调仓日 R4 事件加速入场."""
    trading_dates = dates[WARMUP:]
    global_rebalance = set(trading_dates[::REBALANCE_DAYS])

    cash = float(INITIAL_CAPITAL)
    holding: str | None = None
    holding_shares: int = 0
    equity_history: list[float] = []
    events: list[dict] = []
    n_trades = 0

    for i in range(start_idx, end_idx):
        td = dates[i]
        td_s = str(td)
        is_reb = td in global_rebalance

        if is_reb:
            # ---------- 调仓日: V3 引擎决策 (与生产 select_target 一致) ----------
            etf_idx = {}
            for code in [*list(ETF_POOL.keys()), DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() >= WARMUP:
                    etf_idx[code] = mask.sum() - 1
            target, _c, _s, _a = select_target(data, etf_idx, holding)

            if target != holding:
                if holding and holding in data:
                    row = data[holding][data[holding]["trade_date"] == td]
                    if not row.empty:
                        price = row.iloc[0]["close"]
                        cash += holding_shares * price * (1 - FEE - SLIPPAGE)
                        n_trades += 1
                        holding, holding_shares = None, 0
                if target in data:
                    row = data[target][data[target]["trade_date"] == td]
                    if not row.empty:
                        price = row.iloc[0]["close"]
                        shares = int(cash * 0.99 / price / 100) * 100
                        if shares > 0:
                            cash -= shares * price * (1 + FEE + SLIPPAGE)
                            holding, holding_shares = target, shares
                            n_trades += 1
        elif use_r4 and (holding is None or holding == DEFENSE):
            # ---------- 非调仓日: R4 事件加速入场 ----------
            # 事件: 昨日(i-1)单日涨幅>thr; 确认: 10日动量>0; 选动量最高者
            best_code, best_mom = None, -np.inf
            ev_ret = 0.0
            if i >= 2:
                for code in ETF_POOL:
                    p1, p2 = mat[code][i - 1], mat[code][i - 2]
                    if not (p1 > 0 and p2 > 0 and np.isfinite(p1) and np.isfinite(p2)):
                        continue
                    r1 = p1 / p2 - 1.0
                    if r1 <= thr:
                        continue
                    mom10 = (mat[code][i - 1] / mat[code][i - 11] - 1.0) if i >= 11 else 0.0
                    if mom_confirm and mom10 <= 0:
                        continue
                    if mom10 > best_mom:
                        best_mom, best_code, ev_ret = mom10, code, r1
            if best_code:
                # 若原持仓为货币基金, 先卖出释放市值 (避免持仓市值丢失)
                if holding == DEFENSE:
                    price_mf = mat[DEFENSE][i]
                    if price_mf > 0 and np.isfinite(price_mf):
                        cash += holding_shares * price_mf * (1 - FEE - SLIPPAGE)
                        n_trades += 1
                        holding, holding_shares = None, 0
                price = mat[best_code][i]
                if price > 0 and np.isfinite(price):
                    shares = int(cash * 0.99 / price / 100) * 100
                    if shares > 0:
                        cash -= shares * price * (1 + FEE + SLIPPAGE)
                        holding, holding_shares = best_code, shares
                        n_trades += 1
                        events.append(
                            {
                                "date": td_s,
                                "type": "r4_boost",
                                "asset": best_code,
                                "prev_ret": round(float(ev_ret), 4),
                                "mom10": round(float(best_mom), 4),
                                "idx": i,
                                "on_reb": False,
                            }
                        )

        # 每日净值
        value = cash
        if holding and holding in mat:
            p = mat[holding][i]
            if p > 0 and np.isfinite(p):
                value += holding_shares * p
        equity_history.append(value)

    eq = np.array(equity_history)
    return calc_metrics(eq, INITIAL_CAPITAL, n_trades, events, mat)


def calc_metrics(eq: np.ndarray, init: float, n_trades: int, events: list[dict], mat: dict) -> dict:
    """日频净值指标 (全周期复权优先) + 事件后续收益."""
    if len(eq) < 2:
        return {"error": "insufficient"}
    total = eq[-1] / init - 1.0
    rets = np.diff(eq) / eq[:-1]
    ann_ret = (1 + total) ** (252 / max(len(eq), 1)) - 1.0
    ann_vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    cummax = np.maximum.accumulate(eq)
    max_dd = float(((eq - cummax) / cummax).min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    # 事件后续收益
    for ev in events:
        code, i = ev["asset"], ev["idx"]
        for h in (5, 20):
            j = i + h
            if j < len(mat[code]) and mat[code][i] > 0:
                ev[f"fwd{h}"] = round(float(mat[code][j] / mat[code][i] - 1.0), 4)
    return {
        "total_return": round(float(total), 4),
        "final_value": round(float(eq[-1]), 0),
        "ann_return": round(float(ann_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(float(calmar), 3),
        "n_trades": n_trades,
        "turnover_ann": round(n_trades / max(len(eq) / 252, 1e-9), 1),
        "n_events": len(events),
        "events": events,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("  V3基线 vs V3+R4事件加速器 A/B (全量 + 滚动OOS + 敏感性)")
    print(f"  加速器: 非调仓日 昨日涨幅>{R4_THRESHOLDS} 且10日动量>0 → 入场 (选动量最高)")
    print("=" * 74)

    data = load_data()
    dates = common_dates(data)
    mat = close_matrix(data, dates)
    n = len(dates)
    print(f"\n  数据: {n} 交易日 ({dates[0]} ~ {dates[-1]})")

    # ---------- 1. 全量回测 A/B ----------
    print("\n" + "=" * 74)
    print("  全量回测 (10万本金, 全周期复权)")
    print("=" * 74)
    full = {}
    for name, use_r4, thr, mc in [
        ("V3基线", False, 0.03, True),
        ("V3+R4(3%,动量确认)", True, 0.03, True),
    ]:
        r = run_v3_r4(data, dates, mat, WARMUP, n, use_r4, thr, mc)
        full[name] = r
        print(
            f"  {name:<20} 期末 {r['final_value']:>12,.0f}  ({r['total_return']:+.1%})  "
            f"年化{r['ann_return']:+.1%} 夏普{r['sharpe']:.2f} 回撤{r['max_drawdown']:.1%} "
            f"Calmar{r['calmar']:.2f} 交易{r['n_trades']} 触发{r['n_events']}"
        )
    if "error" not in full["V3+R4(3%,动量确认)"]:
        diff = full["V3+R4(3%,动量确认)"]["final_value"] - full["V3基线"]["final_value"]
        print(f"  → R4加速器 期末差: {diff:+,.0f} 元 ({diff / full['V3基线']['final_value']:+.1%})")

    # ---------- 2. 参数敏感性 (全量) ----------
    print("\n" + "=" * 74)
    print("  参数敏感性 (全量, 事件阈值 × 动量确认)")
    print("=" * 74)
    sens = {}
    for thr in R4_THRESHOLDS:
        for mc in R4_MOM_CONFIRM:
            name = f"thr{thr:.0%}_mom{'on' if mc else 'off'}"
            r = run_v3_r4(data, dates, mat, WARMUP, n, True, thr, mc)
            sens[name] = r
            print(
                f"  {name:<14} 期末 {r['final_value']:>12,.0f} ({r['total_return']:+.1%})  "
                f"夏普{r['sharpe']:.2f} 回撤{r['max_drawdown']:.1%} 触发{r['n_events']}"
            )
    best_sens = max(sens.items(), key=lambda kv: kv[1]["final_value"])
    print(f"  → 全量最优: {best_sens[0]} (期末 {best_sens[1]['final_value']:,.0f})")

    # ---------- 3. 滚动 OOS 对称 A/B ----------
    print("\n" + "=" * 74)
    print("  滚动 OOS 4段对称 (每段测试260日, V3 vs V3+R4)")
    print("=" * 74)
    starts = list(range(0, n - TRAIN_DAYS - BUFFER_DAYS - TEST_DAYS, STEP))[:4]
    oos = []
    wins = 0
    for wi, s0 in enumerate(starts, 1):
        test_start = s0 + TRAIN_DAYS + BUFFER_DAYS
        test_end = test_start + TEST_DAYS
        rb = run_v3_r4(data, dates, mat, test_start, test_end, False, 0.03, True)
        rr = run_v3_r4(data, dates, mat, test_start, test_end, True, 0.03, True)
        beat = rr["final_value"] > rb["final_value"]
        wins += int(beat)
        print(
            f"  W{wi} {dates[test_start]}~{dates[test_end - 1]}: "
            f"基线 {rb['final_value']:>8,.0f} vs R4 {rr['final_value']:>8,.0f} "
            f"({'跑赢' if beat else '跑输'}, 触发{rr['n_events']})"
        )
        oos.append(
            {
                "window": f"W{wi}",
                "base_final": rb["final_value"],
                "r4_final": rr["final_value"],
                "beat": beat,
                "r4_events": rr["n_events"],
            }
        )
    n_win = wins
    n_valid = len(starts)
    print(f"\n  OOS 判定: R4 跑赢 {n_win}/{n_valid} 段 (标准: ≥3/4)")
    passed = n_win >= 3 if n_valid == 4 else n_win / n_valid >= 0.75

    # ---------- 4. 事件日志 (全量 R4 加速入场明细) ----------
    print("\n" + "=" * 74)
    print("  事件日志: 全量 R4 加速入场明细 (前12次)")
    print("=" * 74)
    evs = full["V3+R4(3%,动量确认)"]["events"]
    win_ev = sum(1 for e in evs if (e.get("fwd5") or 0) > 0)
    print(
        f"  共 {len(evs)} 次加速入场 | 后续5日胜率 {win_ev / len(evs):.1%}" if evs else "  无事件"
    )
    for e in evs[:12]:
        f5 = e.get("fwd5")
        f20 = e.get("fwd20")
        print(
            f"  {e['date']} {ETF_POOL.get(e['asset'], '?')} 事件涨幅{e['prev_ret']:+.1%} "
            f"动量{e['mom10']:+.1%} → 5日 {f5 if f5 is None else f'{f5:+.1%}'} "
            f"20日 {f20 if f20 is None else f'{f20:+.1%}'}"
        )

    # ---------- 保存 ----------
    out = {
        "meta": {
            "note": "R4事件加速器: 非调仓日 昨日涨幅>thr且10日动量>0 → 收盘入场, "
            "V3调仓日接管; 无未来函数 (事件检测用昨日数据)",
            "golden_standard": "全量10万→最终金额优先; 滚动OOS ≥3/4段跑赢; Calmar/换手率",
        },
        "full": full,
        "sensitivity": sens,
        "oos": oos,
        "oos_wins": n_win,
        "oos_valid": n_valid,
        "oos_passed": passed,
        "events_full": evs,
    }
    out_path = OUTPUT_DIR / "v3_r4_boost_ab.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
