"""R4 事件加速器 v2 — 提前换手版 + 网格寻优.

v1 (exp_v3_r4_boost.py) 仅允许空仓/防御时入场, OOS 4段触发 0 次.
v2 放宽触发:
  - mode='switch' (提前换手): 非调仓日任何持仓状态下, 若事件资产动量评分
    超过当前持仓评分+缓冲 → 提前换仓 (不等调仓日)
  - mode='conservative': 仅空仓/防御时入场 (v1 逻辑)
  - 事件阈值/动量确认周期 可调低以增加触发

网格 (主): 阈值 {1.5%,2%,2.5%,3%} × 模式 {保守,换手} × 换手缓冲 {0,0.02,0.05}
  动量确认固定 10日; 对主网格最优再做动量周期敏感性 {10,5,0}

判定 (项目黄金标准):
  - 全量回测: 10万→期末金额优先 (用户高收益导向)
  - 滚动 OOS 4段: 至少 3/4 段跑赢基线
  - 最优 = 全量期末最高 且 OOS≥3/4; 事件日志验证换手质量

输出: data/v9_results/v3_r4_grid.json
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
    calc_momentum_score,
    load_data,
    select_target,
)

WARMUP = 130
INITIAL_CAPITAL = 100_000.0
TRAIN_DAYS = 260
TEST_DAYS = 260
BUFFER_DAYS = 20
STEP = (TRAIN_DAYS + TEST_DAYS) // 2

# === 网格 ===
GRID_THR = (0.015, 0.02, 0.025, 0.03)
GRID_MODES = ("conservative", "switch")
GRID_BUF = (0.0, 0.02, 0.05)
MOM_PERIODS = (10, 5, 0)


# --------------------------------------------------------------------------- #
# 引擎: V3 周频 + R4 加速器 v2 (提前换手)
# --------------------------------------------------------------------------- #
def run_v3_r4_v2(
    data: dict, dates: list, mat: dict,
    start_idx: int, end_idx: int,
    thr: float = 0.03, mom_period: int = 10,
    mode: str = "switch", buffer: float = 0.0,
) -> dict:
    """非调仓日: 事件资产动量评分超持仓评分+缓冲 → 提前换仓."""
    trading_dates = dates[WARMUP:]
    global_rebalance = set(trading_dates[::REBALANCE_DAYS])

    cash = float(INITIAL_CAPITAL)
    holding: str | None = None
    holding_shares: int = 0
    equity_history: list[float] = []
    events: list[dict] = []
    n_trades = 0

    def mom_score(code: str, i: int) -> float:
        """V3 动量评分 (截至 i 日, 与生产 calc_momentum_score 同源)."""
        close = mat[code][: i + 1].astype(float)
        if len(close) < 121:
            return -np.inf
        return float(calc_momentum_score(close))

    def trade_to(code: str, i: int, action: str, extra: dict | None = None) -> bool:
        """卖出持仓并买入 code (统一成本). Returns: 是否成交."""
        nonlocal cash, holding, holding_shares, n_trades
        if holding and holding in mat:
            p = mat[holding][i]
            if p > 0 and np.isfinite(p):
                cash += holding_shares * p * (1 - FEE - SLIPPAGE)
                n_trades += 1
                holding, holding_shares = None, 0
        price = mat[code][i]
        if not (price > 0 and np.isfinite(price)):
            return False
        shares = int(cash * 0.99 / price / 100) * 100
        if shares <= 0:
            return False
        cash -= shares * price * (1 + FEE + SLIPPAGE)
        holding, holding_shares = code, shares
        n_trades += 1
        ev = {"date": str(dates[i]), "type": action, "asset": code, "idx": i}
        if extra:
            ev.update(extra)
        events.append(ev)
        return True

    for i in range(start_idx, end_idx):
        td = dates[i]
        is_reb = td in global_rebalance

        if is_reb:
            # ---------- 调仓日: V3 引擎决策 ----------
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
                if holding and holding in mat:
                    p = mat[holding][i]
                    if p > 0 and np.isfinite(p):
                        cash += holding_shares * p * (1 - FEE - SLIPPAGE)
                        n_trades += 1
                        holding, holding_shares = None, 0
                if target in mat:
                    price = mat[target][i]
                    if price > 0 and np.isfinite(price):
                        shares = int(cash * 0.99 / price / 100) * 100
                        if shares > 0:
                            cash -= shares * price * (1 + FEE + SLIPPAGE)
                            holding, holding_shares = target, shares
                            n_trades += 1
        elif i >= 2:
            # ---------- 非调仓日: R4 事件检测 ----------
            # 事件: 昨日涨幅>thr 且 动量确认 (mom_period=0 关闭)
            best_code, best_score = None, -np.inf
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
                s = mom_score(code, i)
                if s > best_score:
                    best_score, best_code, ev_ret = s, code, r1

            if best_code is None:
                pass
            elif mode == "conservative":
                # 仅空仓/防御时入场
                if holding is None or holding == DEFENSE:
                    trade_to(best_code, i, "r4_enter",
                             {"prev_ret": round(float(ev_ret), 4),
                              "score": round(float(best_score), 4)})
            else:  # switch 提前换手
                if holding is None or holding == DEFENSE:
                    trade_to(best_code, i, "r4_enter",
                             {"prev_ret": round(float(ev_ret), 4),
                              "score": round(float(best_score), 4)})
                elif holding in ETF_POOL:
                    cur_s = mom_score(holding, i)
                    if best_score > cur_s + buffer:
                        trade_to(best_code, i, "r4_switch",
                                 {"prev_ret": round(float(ev_ret), 4),
                                  "score": round(float(best_score), 4),
                                  "from": holding,
                                  "from_score": round(float(cur_s), 4)})

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
        "n_events": len(events),
        "events": events,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("  R4 事件加速器 v2 — 提前换手 + 网格寻优")
    print("  主网格: 阈值×模式×缓冲 | 判定: 全量期末 + OOS≥3/4")
    print("=" * 74)

    data = load_data()
    dates = common_dates(data)
    mat = close_matrix(data, dates)
    n = len(dates)
    starts = list(range(0, n - TRAIN_DAYS - BUFFER_DAYS - TEST_DAYS, STEP))[:4]
    print(f"\n  数据: {n} 交易日 | OOS段: {len(starts)}")

    # 基线 (全量 + OOS), thr=1.0 永不触发 = 纯 V3
    base_full = run_v3_r4_v2(data, dates, mat, WARMUP, n, thr=1.0, mom_period=10,
                             mode="conservative", buffer=0.0)
    print(f"\n  V3基线 (全量): 期末 {base_full['final_value']:,.0f} "
          f"({base_full['total_return']:+.1%}) 夏普{base_full['sharpe']:.2f} "
          f"回撤{base_full['max_drawdown']:.1%}")
    base_oos = []
    for s0 in starts:
        ts = s0 + TRAIN_DAYS + BUFFER_DAYS
        te = ts + TEST_DAYS
        base_oos.append(run_v3_r4_v2(data, dates, mat, ts, te, thr=1.0,
                                     mom_period=10, mode="conservative", buffer=0.0))

    # ---------- 主网格 ----------
    print("\n" + "=" * 74)
    print("  主网格 (全量期末金额; OOS跑赢段数; 触发次数)")
    print("=" * 74)
    print(f"  {'配置':<22} {'全量期末':>10} {'夏普':>5} {'回撤':>7} {'触发':>4} {'OOS胜':>5}")
    grid_out = {}
    for thr in GRID_THR:
        for mode in GRID_MODES:
            for buf in GRID_BUF:
                key = f"thr{thr:.1%}_{'换' if mode=='switch' else '保'}_b{buf:.0%}"
                r_full = run_v3_r4_v2(data, dates, mat, WARMUP, n, thr=thr,
                                      mom_period=10, mode=mode, buffer=buf)
                wins = 0
                for s0, rb in zip(starts, base_oos, strict=False):
                    ts = s0 + TRAIN_DAYS + BUFFER_DAYS
                    te = ts + TEST_DAYS
                    rr = run_v3_r4_v2(data, dates, mat, ts, te, thr=thr,
                                      mom_period=10, mode=mode, buffer=buf)
                    wins += int(rr["final_value"] > rb["final_value"])
                grid_out[key] = {
                    "thr": thr, "mode": mode, "buffer": buf,
                    "full": r_full, "oos_wins": wins,
                }
                print(f"  {key:<22} {r_full['final_value']:>10,.0f} "
                      f"{r_full['sharpe']:>5.2f} {r_full['max_drawdown']:>7.1%} "
                      f"{r_full['n_events']:>4} {wins:>3}/4")

    # 最优: 全量期末最高 且 OOS≥3/4
    ranked = sorted(grid_out.items(), key=lambda kv: kv[1]["full"]["final_value"], reverse=True)
    best = None
    for key, v in ranked:
        if v["oos_wins"] >= 3:
            best = (key, v)
            break
    best_hint = (best[0] if best else "无候选(没有配置满足OOS≥3/4)")
    print("\n  → 主网格最优 (全量最高且OOS≥3/4):", best_hint)

    # ---------- mom 敏感性 (对最优配置) ----------
    if best:
        key_b, vb = best
        print("\n" + "=" * 74)
        print(f"  动量确认敏感性 ({key_b}, mom周期)")
        print("=" * 74)
        mom_out = {}
        for mp in MOM_PERIODS:
            r = run_v3_r4_v2(data, dates, mat, WARMUP, n, thr=vb["thr"],
                             mom_period=mp, mode=vb["mode"], buffer=vb["buffer"])
            mom_out[str(mp)] = r
            print(f"  mom{mp:>3}  期末 {r['final_value']:>10,.0f} "
                  f"({r['total_return']:+.1%}) 触发{r['n_events']}")
        best_mom = max(mom_out.items(), key=lambda kv: kv[1]["final_value"])
        print(f"  → mom 最优: {best_mom[0]} (期末 {best_mom[1]['final_value']:,.0f})")

    # ---------- 事件日志 ----------
    print("\n" + "=" * 74)
    if best:
        print(f"  事件日志: 最优配置 {best[0]} (前12次)")
        print("=" * 74)
        evs = best[1]["full"]["events"]
        enters = [e for e in evs if e["type"] == "r4_enter"]
        switches = [e for e in evs if e["type"] == "r4_switch"]
        win5 = sum(1 for e in evs if (e.get("fwd5") or 0) > 0)
        print(f"  入场 {len(enters)} 次 | 提前换手 {len(switches)} 次 | "
              f"总触发 {len(evs)} 次 | 后续5日胜率 {win5/len(evs):.1%}" if evs else "  无事件")
        for e in evs[:12]:
            f5 = e.get("fwd5")
            tag = "换手" if e["type"] == "r4_switch" else "入场"
            print(f"  {e['date']} {tag} {ETF_POOL.get(e['asset'],'?')} 事件{e['prev_ret']:+.1%} "
                  f"评分{e['score']:+.1%} → 5日 {f5 if f5 is None else f'{f5:+.1%}'}")
    else:
        print("  主网格无配置满足 OOS≥3/4, 不输出事件日志")

    # ---------- 保存 ----------
    out = {
        "meta": {
            "note": "R4加速器v2: 非调仓日事件资产动量评分超持仓+缓冲→提前换手; "
                    "网格: thr×mode×buffer, mom固定10; 判定: 全量期末+OOS≥3/4",
            "golden_standard": "全量10万→期末优先; 滚动OOS≥3/4段跑赢",
        },
        "baseline_full": {k: v for k, v in base_full.items() if k != "events"},
        "grid": {k: {kk: (vv if kk != "full" else {x: y for x, y in vv.items() if x != "events"})
                     for kk, vv in v.items()} for k, v in grid_out.items()},
        "best": best[0] if best else None,
        "mom_sensitivity": {k: {kk: vv for kk, vv in v.items() if kk != "events"}
                            for k, v in mom_out.items()} if best else {},
        "best_events": best[1]["full"]["events"] if best else [],
    }
    out_path = OUTPUT_DIR / "v3_r4_grid.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
