"""门控版暴跌过滤 + H3放行止损 — 参数扫描 A/B (全周期 + IS/OOS).

背景: 门控版(thr=0.00)在IS段(2020-2023)比基线 -12.1%, 归因定位为
      "放行品种缓跌无兜底": 2022-10-31 放行纳指后每天阴跌1.5~3%,
      V32的当日-5%急跌兜底(H2)全部失效, 扛到动量转负才在底部割肉.
H3 层: 对"放行类型"持仓 (ret60<0 且 近5日有单日跌>3%):
        自最近暴跌日收盘价起 回撤 > δ → 降仓0.5 或 退出切防御+冷却.
        (无状态可重算: 不依赖放行标记, 回测/实盘同构)

扫描: H3_DELTA ∈ {2%, 3%, 5%} × H3_ACTION ∈ {reduce(0.5), exit}
验证: 全周期 + IS/OOS, 目标: 门控+H3 全周期不劣化基线(≥-1%), IS段亏损收窄.
用法: uv run python scripts/exp_drop_gate_h3.py
输出: data/v9_results/drop_gate_h3.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
from exp_v32_tail_risk import (  # noqa: E402
    DEFENSE_SEQ,
    EXPO_REDUCE,
    H1_DD,
    H2_DAY,
    INITIAL_CAPITAL,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    OUTPUT_DIR,
    REBALANCE_DAYS,
    USE_IMPROVED,
    WARMUP,
    select_target,
)

ORIG_CHECK = rq.check_single_day_drop

# === H3 参数 (由命令行/变体注入) ===
H3_ENABLED = False
H3_EXPO = 1.0  # V3-G 关闭降仓层 (1.0=不降)
EXPO_REDUCE = 1.0  # V3-G 关闭降仓层 (覆盖 import, 1.0=不降)
H3_DELTA = 0.03
H3_ACTION = "reduce"  # reduce(降仓0.5) | exit(退出切防御)
GATE_STATS: dict = {"passed": 0, "excluded": 0}


def make_gated(ret60_thr: float = 0.0, mom_guard: bool = True):
    def gated(close: np.ndarray) -> bool:
        if len(close) < rq.DROP_LOOKBACK + 1:
            return True
        triggered = False
        for i in range(-rq.DROP_LOOKBACK, 0):
            dr = (close[i] - close[i - 1]) / close[i - 1]
            if dr < rq.DROP_THRESHOLD:
                triggered = True
                break
        if not triggered:
            return True
        if len(close) <= 61:
            return True
        ret60 = (close[-1] - close[-61]) / close[-61]
        if ret60 >= ret60_thr:
            GATE_STATS["excluded"] += 1
            return False
        if mom_guard:
            ret10 = (close[-1] - close[-11]) / close[-11] if len(close) > 11 else 0.0
            ret20 = (close[-1] - close[-21]) / close[-21] if len(close) > 21 else 0.0
            if 0.5 * ret10 + 0.5 * ret20 <= 0:
                GATE_STATS["excluded"] += 1
                return False
        GATE_STATS["passed"] += 1
        return True

    return gated


def run_v3_risk_h3(
    data: dict, start_idx: int = 0, end_idx: int | None = None, cost_multiplier: float = 1.0
) -> dict:
    """V3 + V32风控 + H3放行止损 (复制自 exp_v32_tail_risk, 零改生产)."""
    common_dates: set = set()
    for code in rq.ETF_POOL:
        if code not in data:
            continue
        ds = set(data[code]["trade_date"].tolist())
        common_dates = ds if not common_dates else common_dates & ds
    if rq.DEFENSE in data:
        common_dates &= set(data[rq.DEFENSE]["trade_date"].tolist())
    all_dates = sorted(common_dates)
    trading_dates = all_dates[WARMUP:]
    rebalance_set = set(trading_dates[::REBALANCE_DAYS])

    cash = float(INITIAL_CAPITAL)
    holding: str | None = None
    holding_shares = 0.0
    entry_price = 0.0
    peak_equity = float(INITIAL_CAPITAL)
    cooldown_until = None
    exposure = 1.0
    equity_history: list[dict] = []
    risk_events: list[dict] = []
    n_trades = 0
    # H3 trailing-peak 状态 (自最近暴跌日以来高点, 换仓/类型变化时重置)
    h3_peak = 0.0
    h3_holding: str | None = None

    def _price(code: str, td) -> float:
        row = data[code][data[code]["trade_date"] == td]
        return float(row.iloc[0]["close"]) if not row.empty else 0.0

    def _tradable(code: str, td) -> bool:
        row = data[code][data[code]["trade_date"] == td]
        if row.empty:
            return False
        price = float(row.iloc[0]["close"])
        if price <= 0:
            return False
        hist = data[code][data[code]["trade_date"] < td]
        if not hist.empty:
            prev = float(hist.iloc[-1]["close"])
            if prev > 0 and abs(price / prev - 1) >= 0.099:
                return False
        return True

    def _close_series(code: str, td) -> np.ndarray:
        return data[code][data[code]["trade_date"] <= td]["close"].values.astype(float)

    def _vol20(code: str, td) -> float:
        close = _close_series(code, td)
        if len(close) < 21:
            return 0.35
        dr = np.diff(close[-21:]) / close[-21:-1]
        return float(np.std(dr) * np.sqrt(252))

    def _mom(code: str, td, period: int) -> float:
        close = _close_series(code, td)
        if len(close) <= period or close[-period - 1] <= 0:
            return 0.0
        return float(close[-1] / close[-period - 1] - 1.0)

    def _mom_score(code: str, td) -> float:
        return 0.5 * _mom(code, td, 10) + 0.5 * _mom(code, td, 20)

    def _pick_defense(td) -> str:
        for code in DEFENSE_SEQ:
            if code not in data:
                continue
            if code == "511880":
                return code
            if _mom(code, td, 10) > 0:
                return code
        return rq.DEFENSE

    def _trade_to(code: str, td, expo: float = 1.0) -> None:
        nonlocal cash, holding, holding_shares, entry_price, n_trades
        if holding and holding in data:
            px = _price(holding, td)
            if px > 0:
                cash += holding_shares * px * (1 - (rq.FEE + rq.SLIPPAGE) * cost_multiplier)
                n_trades += 1
        if code in data:
            px = _price(code, td)
            if px > 0:
                shares = int(cash * expo * 0.99 / px / 100) * 100
                if shares > 0:
                    cash -= shares * px * (1 + (rq.FEE + rq.SLIPPAGE) * cost_multiplier)
                    holding, holding_shares = code, float(shares)
                    entry_price = px
                    n_trades += 1

    idx_of = {d: i for i, d in enumerate(all_dates)}
    if end_idx is None:
        end_idx = len(trading_dates)
    for td in trading_dates[start_idx:end_idx]:
        equity = cash
        if holding and holding in data:
            px = _price(holding, td)
            if px > 0:
                equity += holding_shares * px
        if equity > peak_equity:
            peak_equity = equity
        dd = equity / peak_equity - 1.0

        # === 层3 日频硬触发 H1/H2 ===
        if holding and holding != rq.DEFENSE and cooldown_until is None:
            cur = _price(holding, td)
            entry_dd = (cur / entry_price - 1.0) if entry_price > 0 else 0.0
            prev = _price(holding, all_dates[idx_of[td] - 1]) if idx_of[td] > 0 else cur
            day_ret = (cur / prev - 1.0) if prev > 0 else 0.0
            if entry_dd < H1_DD or day_ret < H2_DAY:
                if USE_IMPROVED:
                    exposure = min(exposure, EXPO_REDUCE)
                    risk_events.append(
                        {
                            "date": str(td),
                            "type": "改进-H1/H2降仓",
                            "from": holding,
                            "reason": f"entry_dd={entry_dd:.1%} day={day_ret:.1%}",
                        }
                    )
                else:
                    target_d = _pick_defense(td)
                    _trade_to(target_d, td)
                    cooldown_until = td
                    risk_events.append(
                        {
                            "date": str(td),
                            "type": "H1/H2硬触发",
                            "from": holding,
                            "reason": f"entry_dd={entry_dd:.1%} day={day_ret:.1%}",
                        }
                    )

        # === H3 放行止损: 放行类型持仓的缓跌兜底 (门控版新层) ===
        # 基准: 自最近暴跌日收盘以来的持仓高点 (trailing peak) — 避免被暴跌后
        # 的反弹日甩开, 使后续阴跌能正确触发止损
        if H3_ENABLED and holding and holding != rq.DEFENSE and cooldown_until is None:
            hclose = _close_series(holding, td)
            drop_idx: list[int] = []
            if len(hclose) > rq.DROP_LOOKBACK:
                for k in range(1, rq.DROP_LOOKBACK + 1):
                    if len(hclose) > k:
                        dr = (hclose[-k] - hclose[-k - 1]) / hclose[-k - 1]
                        if dr < rq.DROP_THRESHOLD:
                            drop_idx.append(len(hclose) - k)
            if drop_idx and len(hclose) > 61:
                ret60 = (hclose[-1] - hclose[-61]) / hclose[-61]
                if ret60 < 0.0:  # 门控放行类型
                    if h3_holding != holding:
                        h3_holding = holding
                        h3_peak = float(hclose[min(drop_idx)])
                    cur = _price(holding, td)
                    h3_peak = max(h3_peak, cur)
                    dd_peak = cur / h3_peak - 1.0 if h3_peak > 0 else 0.0
                    if dd_peak < -H3_DELTA:
                        if H3_ACTION == "exit":
                            target_d = _pick_defense(td)
                            _trade_to(target_d, td)
                            cooldown_until = td
                            h3_holding, h3_peak = None, 0.0
                            risk_events.append(
                                {
                                    "date": str(td),
                                    "type": "H3放行止损-退出",
                                    "from": holding,
                                    "reason": f"ret60={ret60:.3f} dd_from_peak={dd_peak:.1%}",
                                }
                            )
                        else:
                            exposure = min(exposure, H3_EXPO)
                            risk_events.append(
                                {
                                    "date": str(td),
                                    "type": "H3放行止损-降仓",
                                    "from": holding,
                                    "reason": f"ret60={ret60:.3f} dd_from_peak={dd_peak:.1%}",
                                }
                            )
                else:
                    h3_holding, h3_peak = None, 0.0
            else:
                h3_holding, h3_peak = None, 0.0

        # === 层4 组合熔断 ===
        if cooldown_until is None:
            if dd < -0.30:
                target_d = _pick_defense(td)
                _trade_to(target_d, td)
                cooldown_until = td
                exposure = 1.0
                risk_events.append(
                    {"date": str(td), "type": "熔断-30%清仓", "dd": round(float(dd), 4)}
                )
            elif dd < -0.25:
                risk_events.append(
                    {"date": str(td), "type": "熔断-25%告警", "dd": round(float(dd), 4)}
                )
            elif dd < -0.12:
                exposure = 1.0  # V3-G 关闭降仓层: 仅告警不降仓

        if cooldown_until is not None and td in rebalance_set:
            cooldown_until = None

        # === 调仓日: V3 决策 + 风控层 ===
        if td in rebalance_set and cooldown_until is None:
            etf_idx = {}
            for code in [*list(rq.ETF_POOL.keys()), rq.DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() >= WARMUP:
                    etf_idx[code] = mask.sum() - 1
            target, _c, _s, _a = select_target(data, etf_idx, holding)
            s_score = _mom_score(target, td) if target in rq.ETF_POOL else 0.0

            if target in rq.ETF_POOL:
                close = _close_series(target, td)
                ma10 = float(np.mean(close[-10:])) if len(close) >= 10 else 0.0
                mom5_prev = _mom(target, all_dates[idx_of[td] - 5], 10) if idx_of[td] >= 5 else 0.0
                delta_s = _mom_score(target, td) - mom5_prev
                vol_t = _vol20(target, td)
                decay_triple = delta_s < -0.02 and close[-1] < ma10 and s_score < 0.08
                if USE_IMPROVED:
                    if vol_t > 0.45 and decay_triple:
                        exposure = min(exposure, EXPO_REDUCE)
                        risk_events.append(
                            {
                                "date": str(td),
                                "type": "改进-高波动衰减降仓",
                                "vol": round(float(vol_t), 3),
                                "delta_s": round(float(delta_s), 4),
                            }
                        )
                elif decay_triple:
                    exposure = min(exposure, 0.5)
                    entry_dd = (_price(target, td) / entry_price - 1.0) if entry_price > 0 else 0.0
                    if entry_dd < -0.06:
                        target = _pick_defense(td)
                        risk_events.append(
                            {
                                "date": str(td),
                                "type": "层2衰减退出",
                                "target": target,
                                "delta_s": round(float(delta_s), 4),
                            }
                        )

            if target in rq.ETF_POOL:
                if _price(target, td) <= 0 or not _tradable(target, td):
                    target = _pick_defense(td)

            if target != holding:
                _trade_to(target, td, expo=exposure)
            exposure = 1.0

        equity_history.append(
            {"trade_date": td, "equity": equity, "holding": holding or rq.DEFENSE}
        )

    eq_df = __import__("pandas").DataFrame(equity_history)
    eq_df["trade_date"] = __import__("pandas").to_datetime(eq_df["trade_date"])
    return build_report(eq_df, n_trades, risk_events)


def build_report(eq_df, n_trades: int, risk_events: list) -> dict:
    init = INITIAL_CAPITAL
    total = eq_df["equity"].iloc[-1] / init - 1
    rets = eq_df["equity"].pct_change().dropna()
    ann_vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
    span_days = max((eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days, 1)
    ann_ret = (1 + total) ** (365.25 / span_days) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    dd = eq_df["equity"] / cummax - 1.0
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    deep_time = float((dd < -0.20).mean())
    return {
        "final_value": round(float(eq_df["equity"].iloc[-1]), 0),
        "total_return": round(float(total), 4),
        "ann_return": round(float(ann_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(float(calmar), 3),
        "deep_dd_time": round(deep_time, 4),
        "n_trades": n_trades,
        "n_risk_events": len(risk_events),
        "risk_events": risk_events,
        "equity_curve": eq_df,
    }


KEYS = ("final_value", "total_return", "ann_return", "sharpe", "max_drawdown", "calmar", "n_trades")


def main() -> None:
    global H3_ENABLED, H3_DELTA, H3_ACTION, H3_EXPO
    print("=" * 78)
    print("  门控 + H3放行止损 参数扫描 | 全周期 + IS/OOS | V32风控叠加")
    print("=" * 78)

    data = rq.load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    n = len(dates)

    def seg(s0: str, s1: str) -> tuple[int, int]:
        a = next(i for i, d in enumerate(dates) if str(d) >= s0)
        b = next(i for i, d in enumerate(dates) if str(d) >= s1)
        return max(a - WARMUP, 0), min(b, n)

    segs = [("全周期", 0, n), ("IS", *seg(IS_START, IS_END)), ("OOS", *seg(OOS_START, OOS_END))]

    variants = [("基线(原版)", None, 0.0, "none", 1.0), ("门控(无H3)", 0.00, 0.0, "none", 1.0)]
    for delta, act, expo in (
        (0.02, "reduce", 0.5),
        (0.03, "reduce", 0.5),
        (0.05, "reduce", 0.5),
        (0.02, "exit", 1.0),
        (0.03, "exit", 1.0),
        (0.05, "exit", 1.0),
        (0.01, "reduce", 0.5),  # 更紧止损
        (0.015, "reduce", 0.5),
        (0.02, "reduce", 0.3),
    ):  # 更深降仓
        variants.append(
            (
                f"门控+H3 δ={delta:.0%} {act}" + (f" expo={expo:g}" if expo != 0.5 else ""),
                0.00,
                delta,
                act,
                expo,
            )
        )

    results: dict = {"variants": {}}
    for vname, thr, delta, act, expo in variants:
        rq.check_single_day_drop = make_gated(thr, True) if thr is not None else ORIG_CHECK
        GATE_STATS["passed"] = GATE_STATS["excluded"] = 0
        H3_ENABLED = act != "none"
        H3_DELTA = delta
        H3_ACTION = act if act != "none" else "reduce"
        H3_EXPO = expo
        seg_results = {}
        for name, s0, s1 in segs:
            seg_results[name] = run_v3_risk_h3(data, start_idx=s0, end_idx=max(s1 - WARMUP, 0))
        results["variants"][vname] = {"segs": seg_results, "gate_stats": dict(GATE_STATS)}
        row = seg_results["全周期"]
        is_row = seg_results["IS"]
        oos_row = seg_results["OOS"]
        print(f"\n  [{vname}]")
        print(
            f"    全周期 {row['final_value']:>12,.0f} 年化{row['ann_return'] * 100:>+6.1f}% "
            f"夏普{row['sharpe']:>5.2f} 回撤{row['max_drawdown'] * 100:>6.1f}%"
        )
        print(
            f"    IS     {is_row['final_value']:>12,.0f} 年化{is_row['ann_return'] * 100:>+6.1f}% "
            f"回撤{is_row['max_drawdown'] * 100:>6.1f}% | "
            f"OOS    {oos_row['final_value']:>12,.0f} 年化{oos_row['ann_return'] * 100:>+6.1f}% "
            f"回撤{oos_row['max_drawdown'] * 100:>6.1f}%"
        )
        h3_evts = [e for e in row.get("risk_events", []) if str(e.get("type", "")).startswith("H3")]
        print(
            f"    H3事件: {len(h3_evts)}次 | 放行{results['variants'][vname]['gate_stats'].get('passed', 0)} "
            f"排除{results['variants'][vname]['gate_stats'].get('excluded', 0)}"
        )
    rq.check_single_day_drop = ORIG_CHECK

    # === 判定: 每个变体 vs 基线 ===
    base = results["variants"]["基线(原版)"]["segs"]
    print("\n" + "=" * 78)
    print("  判定 (变体 vs 基线):")
    verdict = {}
    for vname in results["variants"]:
        if vname == "基线(原版)":
            continue
        v = results["variants"][vname]["segs"]
        checks = {
            "全周期收益>=基线-1%": v["全周期"]["final_value"]
            >= base["全周期"]["final_value"] * 0.99,
            "全周期回撤不劣化": v["全周期"]["max_drawdown"] >= base["全周期"]["max_drawdown"],
            "IS收益>=基线-1%": v["IS"]["final_value"] >= base["IS"]["final_value"] * 0.99,
            "OOS收益>=基线-1%": v["OOS"]["final_value"] >= base["OOS"]["final_value"] * 0.99,
        }
        passed_all = all(checks.values())
        verdict[vname] = {k: bool(v) for k, v in checks.items()}
        mark = "✅ 全部通过" if passed_all else "❌"
        print(f"  {vname:<26} {mark} ({'/'.join('✓' if c else '✗' for c in checks.values())})")
    results["verdict"] = verdict

    out = OUTPUT_DIR / "drop_gate_h3.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  结果已保存: {out}")


if __name__ == "__main__":
    main()
