"""验证套件: 5项防过拟合检验 + 体检报告生成 + 通过/警惕/否决判定.

5项检验:
  1. 全周期回测   (基线: 年化/夏普/回撤)
  2. 样本外验证   (前70%训练 / 后30%验证, 按时间切, 算衰减)
  3. 参数稳健性   (param_grid 扫描, 盈利比例/夏普分布/孤立尖峰检测)
  4. 滚动一致性   (walk-forward 2年训练+6月验证, 逐窗滚动)
  5. 基准对比     (vs 等权持有 / vs 持有黄金, 需有超额)
"""

from __future__ import annotations

import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_qixing_v3 as rq

from .engine import WARMUP, backtest, get_common_dates
from .strategies import Strategy

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "qixing_results"
TRAIN_RATIO = 0.70  # 样本外切分: 前70%训练
MAX_PARAM_COMBOS = 200  # 参数扫描上限
GOLD = "518880"  # 黄金基准

_DATA = None


def get_data() -> dict:
    global _DATA
    if _DATA is None:
        _DATA = rq.load_data()
    return _DATA


# --------------------------------------------------------------------------- #
# 检验 1: 全周期回测
# --------------------------------------------------------------------------- #
def check_full(strategy: Strategy, data: dict) -> dict:
    return backtest(data, strategy.select, strategy.params, strategy.rebalance_days)


# --------------------------------------------------------------------------- #
# 检验 2: 样本外验证 (前70%训练 / 后30%验证)
# --------------------------------------------------------------------------- #
def check_oos(strategy: Strategy, data: dict) -> dict:
    dates = get_common_dates(data)[WARMUP:]
    split = dates[int(len(dates) * TRAIN_RATIO)]
    is_end = dates[int(len(dates) * TRAIN_RATIO) - 1]
    is_res = backtest(
        data, strategy.select, strategy.params, strategy.rebalance_days, end_date=is_end
    )
    oos_res = backtest(
        data, strategy.select, strategy.params, strategy.rebalance_days, start_date=split
    )
    is_ann = is_res["ann_return"] * 100
    oos_ann = oos_res["ann_return"] * 100
    decay = oos_ann - is_ann  # 衰减(百分点)
    return {
        "is_ann_return": is_ann,
        "is_sharpe": is_res["sharpe"],
        "oos_ann_return": oos_ann,
        "oos_sharpe": oos_res["sharpe"],
        "oos_total_return": oos_res["total_return"] * 100,
        "oos_max_dd": oos_res["max_drawdown"] * 100,
        "decay_pp": decay,
        "split_date": str(split),
    }


# --------------------------------------------------------------------------- #
# 检验 3: 参数稳健性 (param_grid 扫描)
# --------------------------------------------------------------------------- #
def check_param_robustness(strategy: Strategy, data: dict) -> dict:
    grid = strategy.param_grid
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if len(combos) > MAX_PARAM_COMBOS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(combos), MAX_PARAM_COMBOS, replace=False)
        combos = [combos[i] for i in idx]

    sharpes, returns = [], []
    for combo in combos:
        params = dict(strategy.params)
        params.update(dict(zip(keys, combo)))
        res = backtest(data, strategy.select, params, strategy.rebalance_days)
        sharpes.append(res["sharpe"])
        returns.append(res["total_return"] * 100)

    sharpes = np.array(sharpes)
    returns = np.array(returns)
    pct_profitable = float((returns > 0).mean() * 100)
    best_i = int(np.argmax(sharpes))
    best_sharpe = float(sharpes[best_i])
    median_sharpe = float(np.median(sharpes))
    std_sharpe = float(np.std(sharpes))
    # 孤立尖峰: 最优看着好(夏普>1.5)但多数参数不盈利(<50%) → 过拟合
    is_isolated = bool(pct_profitable < 50 and best_sharpe > 1.5)
    return {
        "n_combos": len(combos),
        "pct_profitable": pct_profitable,
        "sharpe_median": median_sharpe,
        "sharpe_std": std_sharpe,
        "best_sharpe": best_sharpe,
        "best_params": dict(zip(keys, combos[best_i])),
        "is_isolated_spike": is_isolated,
    }


# --------------------------------------------------------------------------- #
# 检验 4: 滚动一致性 (walk-forward, 固定参数测各时段)
# --------------------------------------------------------------------------- #
def check_walk_forward(
    strategy: Strategy, data: dict, train_days: int = 504, test_days: int = 126
) -> dict:
    dates = get_common_dates(data)[WARMUP:]
    n = len(dates)
    windows = []
    start = 0
    while start + train_days + test_days <= n:
        test_start = dates[start + train_days]
        test_end = dates[min(start + train_days + test_days - 1, n - 1)]
        res = backtest(
            data,
            strategy.select,
            strategy.params,
            strategy.rebalance_days,
            start_date=test_start,
            end_date=test_end,
        )
        windows.append(
            {
                "period": f"{str(test_start)[:7]}~{str(test_end)[:7]}",
                "return": res["total_return"] * 100,
            }
        )
        start += test_days  # 滚动一个测试窗
    if not windows:
        return {"n_windows": 0, "pct_positive": 0.0, "windows": []}
    rets = [w["return"] for w in windows]
    return {
        "n_windows": len(windows),
        "pct_positive": float(sum(1 for r in rets if r > 0) / len(rets) * 100),
        "windows": windows,
    }


# --------------------------------------------------------------------------- #
# 检验 5: 基准对比 (等权持有 / 持有黄金)
# --------------------------------------------------------------------------- #
def _buy_hold_return(data: dict, code: str, start, end) -> float:
    if code not in data:
        return 0.0
    df = data[code]
    sub = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
    if len(sub) < 2:
        return 0.0
    return float(sub["close"].iloc[-1] / sub["close"].iloc[0] - 1)


def check_benchmark(strategy: Strategy, data: dict, full: dict) -> dict:
    dates = get_common_dates(data)[WARMUP:]
    start, end = dates[0], dates[-1]
    pool_rets = [_buy_hold_return(data, c, start, end) for c in rq.ETF_POOL if c in data]
    eq_weight = float(np.mean(pool_rets)) * 100 if pool_rets else 0.0
    gold = _buy_hold_return(data, GOLD, start, end) * 100
    strat = full["total_return"] * 100
    return {
        "strategy_total": strat,
        "eqweight_total": eq_weight,
        "gold_total": gold,
        "excess_eqweight": strat - eq_weight,
        "excess_gold": strat - gold,
    }


# --------------------------------------------------------------------------- #
# 检验 6: 牛熊震荡分段 (风控视角: 不同市场环境下的表现)
# --------------------------------------------------------------------------- #
def _yearly_returns(data: dict, code: str) -> dict:
    """某只ETF的逐年收益."""
    df = data[code].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["year"] = df["trade_date"].dt.year
    out = {}
    for y, g in df.groupby("year"):
        if len(g) >= 2:
            out[int(y)] = float(g["close"].iloc[-1] / g["close"].iloc[0] - 1)
    return out


def check_regime(data: dict, full: dict) -> dict:
    """牛熊震荡分段: 以等权池年收益为市场代理, 分类各年并汇总策略表现."""
    years = sorted(full["yearly"].keys())
    pool_yearly: dict[int, list] = {}
    for c in rq.ETF_POOL:
        if c not in data:
            continue
        for y, ret in _yearly_returns(data, c).items():
            pool_yearly.setdefault(y, []).append(ret)
    market = {y: float(np.mean(rs)) for y, rs in pool_yearly.items()}

    rows = []
    buckets: dict[str, list] = {"牛市": [], "熊市": [], "震荡": []}
    for y in years:
        m = market.get(y, 0.0)
        regime = "牛市" if m > 0.15 else ("熊市" if m < -0.05 else "震荡")
        strat = full["yearly"][y]["return"] * 100
        rows.append({"year": y, "market": m * 100, "regime": regime, "strategy": strat})
        buckets[regime].append(strat)
    summary = {k: {"avg": float(np.mean(v)) if v else 0.0, "n": len(v)} for k, v in buckets.items()}
    return {"rows": rows, "summary": summary}


# --------------------------------------------------------------------------- #
# 判定 + 风险评级
# --------------------------------------------------------------------------- #
def compute_verdict(r: dict) -> tuple[str, list[str], str]:
    """返回 (结论, 原因列表, 风险评级)."""
    # 否决条件 (任一触发)
    reject = []
    if not r["hypothesis"].strip():
        reject.append("无经济逻辑(hypothesis为空)")
    if r["oos"]["oos_ann_return"] < 0:
        reject.append(f"样本外年化为负 ({r['oos']['oos_ann_return']:+.1f}%)")
    if r["param"]["pct_profitable"] < 30:
        reject.append(f"参数稳健性过差 (仅{r['param']['pct_profitable']:.0f}%参数盈利)")
    if r["param"]["is_isolated_spike"]:
        reject.append("最优参数为孤立尖峰 (过拟合特征)")
    if reject:
        return "否决", reject, "高"

    # 通过条件 (全部满足)
    checks = {
        "样本外为正": r["oos"]["oos_ann_return"] > 0,
        "参数稳健(>50%盈利)": r["param"]["pct_profitable"] > 50,
        "跑赢等权基准": r["benchmark"]["excess_eqweight"] > 0,
        "滚动一致(多数窗口为正)": r["walkforward"]["pct_positive"] >= 50,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if not failed:
        return "通过", ["各项检验达标"], "低"
    return "警惕", ["部分检验未过: " + ", ".join(failed)], "中"


# --------------------------------------------------------------------------- #
# 主流程 + 报告
# --------------------------------------------------------------------------- #
def run_validation(strategy: Strategy) -> dict:
    data = get_data()
    print(f"\n  ▶ 检验策略: {strategy.name}")
    print("  ▶ [1/5] 全周期回测...")
    full = check_full(strategy, data)
    print("  ▶ [2/5] 样本外验证...")
    oos = check_oos(strategy, data)
    print("  ▶ [3/5] 参数稳健性扫描...")
    param = check_param_robustness(strategy, data)
    print("  ▶ [4/5] 滚动一致性 (walk-forward)...")
    wf = check_walk_forward(strategy, data)
    print("  ▶ [5/5] 基准对比...")
    bench = check_benchmark(strategy, data, full)
    regime = check_regime(data, full)

    report = {
        "name": strategy.name,
        "hypothesis": strategy.hypothesis,
        "full": full,
        "oos": oos,
        "param": param,
        "walkforward": wf,
        "benchmark": bench,
        "regime": regime,
    }
    verdict, reasons, risk = compute_verdict(report)
    report["verdict"], report["reasons"], report["risk"] = verdict, reasons, risk

    md = render_markdown(report)
    print(md)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"lab_{strategy.name}_{datetime.now():%Y%m%d}.md"
    out.write_text(md, encoding="utf-8")
    print(f"\n  ✓ 报告已保存: {out}")
    return report


def render_markdown(r: dict) -> str:
    f, o, p, w, b = r["full"], r["oos"], r["param"], r["walkforward"], r["benchmark"]
    rg = r["regime"]
    verdict_icon = {"通过": "✅ 通过", "警惕": "⚠️ 警惕", "否决": "❌ 否决"}[r["verdict"]]
    lines = []
    lines.append("=" * 60)
    lines.append(f"  策略体检报告: {r['name']}")
    lines.append("=" * 60)
    lines.append(f"\n【经济逻辑】 {r['hypothesis']}")
    lines.append("\n【1. 全周期回测】")
    lines.append(
        f"  年化 {f['ann_return'] * 100:+.1f}% | 夏普 {f['sharpe']:.2f} | "
        f"回撤 {f['max_drawdown'] * 100:.1f}% | 交易 {f['n_trades']} 次"
    )
    yr = "  ".join(f"{y}:{v['return'] * 100:+.0f}%" for y, v in sorted(f["yearly"].items()))
    lines.append(f"  逐年: {yr}")
    worst_y, worst_v = min(f["yearly"].items(), key=lambda kv: kv[1]["return"])
    lines.append(f"  最差年份: {worst_y} ({worst_v['return'] * 100:+.1f}%)")
    lines.append("\n【2. 样本外验证】 (前70%训练 / 后30%验证)")
    lines.append(f"  训练集: 年化 {o['is_ann_return']:+.1f}% (夏普 {o['is_sharpe']:.2f})")
    lines.append(
        f"  验证集: 年化 {o['oos_ann_return']:+.1f}% (夏普 {o['oos_sharpe']:.2f}, "
        f"回撤 {o['oos_max_dd']:.1f}%)"
    )
    lines.append(f"  衰减: {o['decay_pp']:+.1f} 百分点 (切分点 {o['split_date'][:7]})")
    lines.append("\n【3. 参数稳健性】 (扫描 %d 组参数)" % p["n_combos"])
    lines.append(f"  盈利比例: {p['pct_profitable']:.0f}%")
    lines.append(
        f"  夏普分布: 中位数 {p['sharpe_median']:.2f} ± {p['sharpe_std']:.2f} "
        f"(最优 {p['best_sharpe']:.2f})"
    )
    lines.append(f"  孤立尖峰: {'是 ⚠️' if p['is_isolated_spike'] else '否'}")
    lines.append("\n【4. 滚动一致性】 (walk-forward, %d 个窗口)" % w["n_windows"])
    lines.append(f"  正收益窗口: {w['pct_positive']:.0f}%")
    if w["windows"]:
        wr = "  ".join(f"{x['period']}:{x['return']:+.0f}%" for x in w["windows"])
        lines.append(f"  各窗口: {wr}")
    lines.append("\n【5. 基准对比】 (全周期累计)")
    lines.append(
        f"  策略 {b['strategy_total']:+.0f}% vs 等权 {b['eqweight_total']:+.0f}% "
        f"(超额 {b['excess_eqweight']:+.0f}%) vs 黄金 {b['gold_total']:+.0f}% "
        f"(超额 {b['excess_gold']:+.0f}%)"
    )
    s = rg["summary"]
    lines.append("\n【6. 牛熊震荡分段】 (以等权池年收益划分市场环境)")
    lines.append(
        f"  牛市({s['牛市']['n']}年)平均 {s['牛市']['avg']:+.1f}% | "
        f"震荡({s['震荡']['n']}年) {s['震荡']['avg']:+.1f}% | "
        f"熊市({s['熊市']['n']}年) {s['熊市']['avg']:+.1f}%"
    )
    rg_rows = "  ".join(f"{x['year']}({x['regime']}):{x['strategy']:+.0f}%" for x in rg["rows"])
    lines.append(f"  逐年: {rg_rows}")
    lines.append("\n" + "=" * 60)
    lines.append(f"  过拟合风险: {r['risk']}    结论: {verdict_icon}")
    for reason in r["reasons"]:
        lines.append(f"    · {reason}")
    lines.append("=" * 60)
    return "\n".join(lines)
