"""V3 + ML 动态层完整研究 (Guard减仓 / Boost加仓 / 动态阈值 / 参数网格).

回应三个问题:
  1. 为什么只有减仓层没有加仓层 → 补 Boost 加仓层 (RF涨概率>阈值 → 动态入场)
  2. 为什么不是动态的 → 层为日频检查 (不锁死调仓日); 阈值支持动态分位数
     (用训练集概率分布 q85/q90/q95 作为测试期阈值, 随数据自适应)
  3. 参数网格全筛 → 层阈值全网格 (guard×boost) + RF超参邻域 (depth×leaf),
     全部在 4 段 walk-forward OOS 下评估, 报告"最佳"及邻域稳健性

结构:
  - 调仓日: V3 引擎决策 (含调仓日 guard)
  - 非调仓日: guard 日频出场 (持仓跌概率>阈值 → 切防御)
              boost 日频入场 (防御且某资产涨概率>阈值 → 买入概率最高者)
  - 评价: 全周期复权优先 (10万→最终金额), 同时报告年化/夏普/回撤/交易次数

输出: data/v9_results/v3_ml_layers.json
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

from exp_ml_up_down import build_features  # noqa: E402
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

# === OOS 段 (与之前一致) ===
TRAIN_DAYS = 260
TEST_DAYS = 260
BUFFER_DAYS = 20
STEP = (TRAIN_DAYS + TEST_DAYS) // 2

# === 网格 ===
GUARD_THRS = (0.50, 0.55, 0.60)
BOOST_THRS = (0.55, 0.60, 0.65)
QUANTILES = (0.85, 0.90, 0.95)
RF_DEPTHS = (4, 6, 8)
RF_LEAVES = (25, 50, 100)


# --------------------------------------------------------------------------- #
# 日频动态层回测引擎
# --------------------------------------------------------------------------- #
def run_v3_daily(
    data: dict, dates: list, mat: dict,
    start_idx: int, end_idx: int,
    dn_map: dict, up_map: dict,
    guard_thr: float | None, boost_thr: float | None,
) -> dict:
    """日频 guard/boost 动态层 + 周频 V3 引擎.

    Args:
        dn_map: {(date_str, code): 跌概率} (guard 用)
        up_map: {(date_str, code): 涨概率} (boost 用)
        guard_thr: 跌概率出场阈值 (None=关闭)
        boost_thr: 涨概率入场阈值 (None=关闭)
    """
    trading_dates = dates[WARMUP:]
    global_rebalance = set(trading_dates[::REBALANCE_DAYS])

    cash = float(INITIAL_CAPITAL)
    holding: str | None = None
    holding_shares: int = 0
    equity_history: list[float] = []
    events: list[dict] = []
    n_trades = 0
    blocked_until = -1  # guard/boost 动作后的冷却 (避免同日卖出又买入)

    for i in range(start_idx, end_idx):
        td = dates[i]
        td_s = str(td)
        is_reb = td in global_rebalance

        # ---------- 调仓日: V3 引擎决策 ----------
        if is_reb:
            etf_idx = {}
            for code in [*list(ETF_POOL.keys()), DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() >= WARMUP:
                    etf_idx[code] = mask.sum() - 1
            target, _c, _s, _a = select_target(data, etf_idx, holding)

            # 调仓日 guard
            if guard_thr is not None and holding and holding != DEFENSE:
                p = dn_map.get((td_s, holding))
                if p is not None and p > guard_thr:
                    target = DEFENSE
                    events.append({"date": td_s, "type": "guard", "asset": holding,
                                   "prob": round(float(p), 3), "idx": i, "on_reb": True})

            # 执行交易
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
        else:
            # ---------- 非调仓日: guard 日频出场 ----------
            if (guard_thr is not None and holding and holding != DEFENSE
                    and i >= blocked_until):
                p = dn_map.get((td_s, holding))
                if p is not None and p > guard_thr:
                    asset = holding
                    price = mat[asset][i]
                    if price > 0 and np.isfinite(price):
                        cash += holding_shares * price * (1 - FEE - SLIPPAGE)
                        n_trades += 1
                        holding, holding_shares = None, 0
                        events.append({"date": td_s, "type": "guard", "asset": asset,
                                       "prob": round(float(p), 3), "idx": i,
                                       "on_reb": False})
                        blocked_until = i + 1  # 次日才允许重新入场

            # ---------- 非调仓日: boost 日频入场 ----------
            if (boost_thr is not None and (holding is None or holding == DEFENSE)
                    and i >= blocked_until):
                best_code, best_p = None, 0.0
                for code in ETF_POOL:
                    p = up_map.get((td_s, code))
                    if p is not None and p > boost_thr and p > best_p:
                        best_p, best_code = p, code
                if best_code:
                    price = mat[best_code][i]
                    if price > 0 and np.isfinite(price):
                        shares = int(cash * 0.99 / price / 100) * 100
                        if shares > 0:
                            cash -= shares * price * (1 + FEE + SLIPPAGE)
                            holding, holding_shares = best_code, shares
                            n_trades += 1
                            events.append({"date": td_s, "type": "boost", "asset": best_code,
                                           "prob": round(float(best_p), 3), "idx": i,
                                           "on_reb": False})

        # 每日净值
        value = cash
        if holding and holding in mat:
            p = mat[holding][i]
            if p > 0 and np.isfinite(p):
                value += holding_shares * p
        equity_history.append(value)

    eq = np.array(equity_history)
    return calc_metrics(eq, INITIAL_CAPITAL, n_trades, events, mat)


def calc_metrics(eq: np.ndarray, init: float, n_trades: int,
                 events: list[dict], mat: dict) -> dict:
    """日频净值指标 + 事件后续收益 (fwd5/fwd20)."""
    if len(eq) < 2:
        return {"error": "insufficient"}
    total = eq[-1] / init - 1.0
    rets = np.diff(eq) / eq[:-1]
    ann_ret = (1 + total) ** (252 / max(len(eq), 1)) - 1.0
    ann_vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    cummax = np.maximum.accumulate(eq)
    max_dd = float(((eq - cummax) / cummax).min())
    # 事件后续收益
    for ev in events:
        code, i = ev["asset"], ev["idx"]
        fwds = {}
        for h in (5, 20):
            j = i + h
            if j < len(mat[code]) and mat[code][i] > 0:
                fwds[f"fwd{h}"] = round(float(mat[code][j] / mat[code][i] - 1.0), 4)
        ev.update(fwds)
    return {
        "total_return": round(float(total), 4),
        "final_value": round(float(eq[-1]), 0),
        "ann_return": round(float(ann_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(max_dd, 4),
        "n_trades": n_trades,
        "n_events": len(events),
        "events": events,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("  V3 + ML 动态层完整研究 (Guard出场 / Boost加仓 / 动态阈值 / 全网格)")
    print("=" * 74)

    data = load_data()
    dates = common_dates(data)
    mat = close_matrix(data, dates)
    vol_mat = {c: data[c].set_index("trade_date")["volume"].reindex(dates).values
               for c in ETF_POOL}
    print("\n  构建特征面板...")
    x, meta, _feat_names = build_features(mat, vol_mat, dates)
    order = sorted(range(len(meta)), key=lambda i: meta[i]["date"])
    x = x[order]
    meta = [meta[i] for i in order]
    y_up = np.array([m["y_up"] for m in meta])
    y_dn = np.array([m["y_dn"] for m in meta])
    dates_s = [m["date"] for m in meta]

    starts = list(range(0, len(dates) - TRAIN_DAYS - BUFFER_DAYS - TEST_DAYS, STEP))[:4]
    print(f"  OOS段: {len(starts)} | 样本: {len(meta)}")

    from sklearn.ensemble import RandomForestClassifier

    # 预训练各段 RF (涨/跌模型), 记录测试段概率映射
    seg_models = []
    for s0 in starts:
        test_start = s0 + TRAIN_DAYS + BUFFER_DAYS
        test_end = test_start + TEST_DAYS
        test_start_date = str(dates[test_start])
        tr_mask = np.array([d < test_start_date for d in dates_s])
        te_dates_set = {str(dates[i]) for i in range(test_start, test_end)}
        te_mask = np.array([d in te_dates_set for d in dates_s])
        rf_up = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=50,
                                       random_state=42, n_jobs=-1)
        rf_dn = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=50,
                                       random_state=42, n_jobs=-1)
        rf_up.fit(x[tr_mask], y_up[tr_mask])
        rf_dn.fit(x[tr_mask], y_dn[tr_mask])
        pu = rf_up.predict_proba(x)[:, 1]
        pd_ = rf_dn.predict_proba(x)[:, 1]
        up_map = {(meta[i]["date"], meta[i]["code"]): float(pu[i]) for i in np.where(te_mask)[0]}
        dn_map = {(meta[i]["date"], meta[i]["code"]): float(pd_[i]) for i in np.where(te_mask)[0]}
        # 动态分位数阈值 (基于训练集概率分布)
        q = {"up": {q_: float(np.quantile(pu[tr_mask], q_)) for q_ in QUANTILES},
             "dn": {q_: float(np.quantile(pd_[tr_mask], q_)) for q_ in QUANTILES}}
        seg_models.append({"start": s0, "up_map": up_map, "dn_map": dn_map,
                           "quantiles": q,
                           "spans": [str(dates[test_start]), str(dates[test_end - 1])]})

    # 定义一个评估助手: 对给定 guard/boost 配置跑全部段
    def evaluate_config(name: str, guard: float | None, boost: float | None,
                        segs: list, use_quantile: bool = False) -> dict:
        agg = {"total": [], "final": [], "sharpe": [], "dd": [], "trades": []}
        events_all = []
        for seg in segs:
            s0 = seg["start"]
            test_start = s0 + TRAIN_DAYS + BUFFER_DAYS
            test_end = test_start + TEST_DAYS
            if use_quantile:
                g_thr = seg["quantiles"]["dn"][guard] if guard is not None else None
                b_thr = seg["quantiles"]["up"][boost] if boost is not None else None
            else:
                g_thr, b_thr = guard, boost
            res = run_v3_daily(data, dates, mat, test_start, test_end,
                               seg["dn_map"], seg["up_map"], g_thr, b_thr)
            if "error" in res:
                continue
            agg["total"].append(res["total_return"])
            agg["final"].append(res["final_value"])
            agg["sharpe"].append(res["sharpe"])
            agg["dd"].append(res["max_drawdown"])
            agg["trades"].append(res["n_trades"])
            events_all.extend(res["events"])
        if not agg["total"]:
            return {"name": name, "error": "no valid segments"}
        anns = [(1 + t) ** (252 / TEST_DAYS) - 1 for t in agg["total"]]
        return {
            "name": name,
            "avg_ann": round(float(np.mean(anns)), 4),
            "final_avg": round(float(np.mean(agg["final"])), 0),
            "final_sum": round(float(np.sum(agg["final"])), 0),
            "sharpe_avg": round(float(np.mean(agg["sharpe"])), 3),
            "dd_avg": round(float(np.mean(agg["dd"])), 4),
            "trades_avg": round(float(np.mean(agg["trades"])), 1),
            "events_total": len(events_all),
            "events": events_all,
        }

    # 1) 主配置对比
    print("\n" + "=" * 74)
    print("  主配置对比 (4段OOS平均, 10万→期末金额)")
    print("=" * 74)
    main_configs = [
        ("V3基线", None, None, False),
        ("V3+Guard(0.55)", 0.55, None, False),
        ("V3+Boost(0.60)", None, 0.60, False),
        ("V3+GB(0.55,0.60)", 0.55, 0.60, False),
        ("V3+GB(动态q90)", 0.90, 0.90, True),
        ("V3+GB(动态q85)", 0.85, 0.85, True),
    ]
    main_results = {}
    for name, g, b, q in main_configs:
        r = evaluate_config(name, g, b, seg_models, q)
        main_results[name] = r
        if "error" not in r:
            print(f"  {name:<18} 期末均值 {r['final_avg']:>9,.0f} "
                  f"夏普{r['sharpe_avg']:.2f} 回撤{r['dd_avg']:.1%} "
                  f"交易{r['trades_avg']:.0f} 触发{r['events_total']}次")

    # 2) 阈值全网格 (GB 组合)
    print("\n" + "=" * 74)
    print("  阈值全网格 (Guard×Boost 固定阈值, 4段合计期末金额)")
    print("=" * 74)
    grid = {}
    for g in GUARD_THRS:
        for b in BOOST_THRS:
            r = evaluate_config(f"GB({g:.2f},{b:.2f})", g, b, seg_models, False)
            grid[f"{g:.2f}x{b:.2f}"] = r
            print(f"  Guard{g:.2f}×Boost{b:.2f}  期末合计 {r['final_sum']:>10,.0f} "
                  f"夏普{r['sharpe_avg']:.2f} 回撤{r['dd_avg']:.1%}")
    best_grid = max(grid.items(), key=lambda kv: kv[1]["final_sum"])
    print(f"  → 网格最优: {best_grid[0]} (期末合计 {best_grid[1]['final_sum']:,.0f})")

    # 3) RF 超参敏感性 (对网格最优阈值组合)
    g_best, b_best = (float(v) for v in best_grid[0].split("x"))
    print("\n" + "=" * 74)
    print(f"  RF超参敏感性 (Guard{g_best:.2f}×Boost{b_best:.2f}, depth×leaf)")
    print("=" * 74)
    hp_results = {}
    for depth in RF_DEPTHS:
        for leaf in RF_LEAVES:
            # 重新训练4段模型 (该超参)
            segs2 = []
            for s0 in starts:
                test_start = s0 + TRAIN_DAYS + BUFFER_DAYS
                test_start_date = str(dates[test_start])
                tr_mask = np.array([d < test_start_date for d in dates_s])
                te_dates_set = {str(dates[i]) for i in range(test_start, test_start + TEST_DAYS)}
                te_mask = np.array([d in te_dates_set for d in dates_s])
                rf_u = RandomForestClassifier(n_estimators=200, max_depth=depth,
                                              min_samples_leaf=leaf, random_state=42, n_jobs=-1)
                rf_d = RandomForestClassifier(n_estimators=200, max_depth=depth,
                                              min_samples_leaf=leaf, random_state=42, n_jobs=-1)
                rf_u.fit(x[tr_mask], y_up[tr_mask])
                rf_d.fit(x[tr_mask], y_dn[tr_mask])
                pu = rf_u.predict_proba(x)[:, 1]
                pd_ = rf_d.predict_proba(x)[:, 1]
                segs2.append({
                    "start": s0,
                    "up_map": {(meta[i]["date"], meta[i]["code"]): float(pu[i])
                               for i in np.where(te_mask)[0]},
                    "dn_map": {(meta[i]["date"], meta[i]["code"]): float(pd_[i])
                               for i in np.where(te_mask)[0]},
                })
            # 用 segs2 跑 (超参组合)
            r = evaluate_config(f"d{depth}l{leaf}", g_best, b_best, segs2, False)
            hp_results[f"d{depth}l{leaf}"] = r
            print(f"  depth={depth} leaf={leaf:<3} 期末合计 {r['final_sum']:>10,.0f} "
                  f"夏普{r['sharpe_avg']:.2f}")
    best_hp = max(hp_results.items(), key=lambda kv: kv[1]["final_sum"])
    print(f"  → 超参最优: {best_hp[0]} (期末合计 {best_hp[1]['final_sum']:,.0f})")

    # 4) 段4 (白银大牛市) 误杀归因
    print("\n" + "=" * 74)
    print("  段4 (2024-12~2026-01 白银大牛市) Guard 触发事件归因")
    print("=" * 74)
    best_cfg = best_grid[1]
    w4_events = [e for e in best_cfg["events"] if e["type"] == "guard"
                 and e["date"] >= seg_models[3]["spans"][0]]
    if not w4_events:
        print("  (该配置在段4无 guard 触发)")
    for e in w4_events[:10]:
        f5 = e.get("fwd5")
        f20 = e.get("fwd20")
        print(f"  {e['date']} guard {ETF_POOL.get(e['asset'], '?')} 概率{e['prob']:.2f} "
              f"后续5日 {f5 if f5 is None else f'{f5:+.2%}'} 后续20日 "
              f"{f20 if f20 is None else f'{f20:+.2%}'}")

    # 保存
    out = {"meta": {
        "note": "日频guard/boost动态层 + 周频V3引擎; 4段OOS段前训练RF(无未来函数)",
        "grid": {"guard": list(GUARD_THRS), "boost": list(BOOST_THRS),
                 "quantiles": list(QUANTILES), "rf_depth": list(RF_DEPTHS),
                 "rf_leaf": list(RF_LEAVES)},
        "evaluation": "全周期复权优先: 4段期末金额合计, 参考夏普/回撤",
    }, "main_configs": {k: {kk: vv for kk, vv in v.items() if kk != "events"}
                        for k, v in main_results.items()},
      "threshold_grid": {k: {kk: vv for kk, vv in v.items() if kk != "events"}
                         for k, v in grid.items()},
      "rf_hyperparam": {k: {kk: vv for kk, vv in v.items() if kk != "events"}
                        for k, v in hp_results.items()},
      "best_grid": best_grid[0], "best_hyperparam": best_hp[0],
      "w4_guard_events": w4_events,
      "all_events": {k: v["events"] for k, v in {**main_results, **grid}.items()},
    }
    out_path = OUTPUT_DIR / "v3_ml_layers.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
