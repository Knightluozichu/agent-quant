"""R4 暴涨后持续规律 — Walk-forward 验证 (C队阶段2).

候选规律 R4 (阶段1筛选通过): 单日暴涨>3% 后5日继续涨 (d=1.49, 事件聚类后 d=1.26).

策略化 (无未来函数):
  - 触发: 资产昨日单日涨幅 > 阈值 (T+1可知) → 今日收盘入场
  - 持有: H 个交易日后收盘平仓
  - 多资产同日触发 → 现金等权分配; 无触发 → 空仓
  - 成本: 万五单边 + 千一滑点 (与项目一致)

Walk-forward 协议 (C队):
  - 4 个非重叠窗口: 训练260日 → 缓冲20日 → 测试260日
    (1450个交易日限制下的最大4窗口配置, 约1年训练+1年测试)
  - 训练期: 网格搜索 (阈值∈{2%,3%,4%,5%} × 持有期∈{3,5,8}), 选训练夏普最优
  - 测试期: 冻结参数独立运行 (不回头看)
  - 基准: 等权买入持有 + M0动量轮动 (10日×0.5+20日×0.5, 近似V3)

通过标准: 至少 3/4 窗口测试期夏普>0, 且平均年化跑赢等权基准.

输出: data/v9_results/walk_forward_r4.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from exp_momentum_models import DEFENSE, run_backtest, score_m0
from exp_short_window_patterns import ETF_POOL, close_matrix, common_dates, load_data

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 交易参数 (与V3一致) ===
FEE = 0.0005
SLIPPAGE = 0.001
INITIAL_CAPITAL = 100_000.0

# === Walk-forward 窗口 ===
TRAIN_DAYS = 260
TEST_DAYS = 260
BUFFER_DAYS = 20
STEP = (TRAIN_DAYS + TEST_DAYS) // 2  # 窗口步进 (非重叠测试段)
N_WINDOWS = 4

# === R4 参数网格 (训练期学习) ===
GRID_THRESHOLD = (0.02, 0.03, 0.04, 0.05)
GRID_HOLD = (3, 5, 8)


# --------------------------------------------------------------------------- #
# R4 事件驱动回测
# --------------------------------------------------------------------------- #
def run_event_strategy(
    mat: dict,
    codes: list,
    threshold: float,
    hold: int,
    start_idx: int,
    end_idx: int,
) -> dict:
    """在日期索引区间 [start_idx, end_idx) 内运行暴涨事件策略.

    决策基于昨日涨幅 (无未来函数), 今日收盘成交, 持有 hold 日后收盘平仓.
    """
    cash = float(INITIAL_CAPITAL)
    positions: dict[str, dict] = {}  # code -> {shares, exit_idx}
    equity = []
    n_trades = 0

    for i in range(start_idx, end_idx):
        # 1. 平仓到期持仓
        for code in list(positions):
            if i >= positions[code]["exit_idx"]:
                p = mat[code][i]
                if p > 0 and np.isfinite(p):
                    cash += positions[code]["shares"] * p * (1 - FEE - SLIPPAGE)
                    n_trades += 1
                    del positions[code]

        # 2. 触发检查: 昨日 (i-1) 单日涨幅 > 阈值
        triggers = []
        if i >= start_idx + 2:
            for code in codes:
                p1, p2 = mat[code][i - 1], mat[code][i - 2]
                if (
                    p1 > 0
                    and p2 > 0
                    and np.isfinite(p1)
                    and np.isfinite(p2)
                    and (p1 / p2 - 1.0) > threshold
                ):
                    triggers.append(code)

        # 3. 入场: 今日收盘等权买入新触发资产
        if triggers:
            available = [c for c in triggers if c not in positions]
            if available:
                per = cash / len(available)
                for code in available:
                    p = mat[code][i]
                    if p <= 0 or not np.isfinite(p):
                        continue
                    shares = int(per * 0.99 / p / 100) * 100
                    if shares > 0:
                        cost = shares * p * (1 + FEE + SLIPPAGE)
                        if cost <= cash + 1e-6:
                            cash -= cost
                            positions[code] = {"shares": shares, "exit_idx": i + hold}
                            n_trades += 1

        # 4. 净值
        value = cash
        for code, pos in positions.items():
            p = mat[code][i]
            if p > 0 and np.isfinite(p):
                value += pos["shares"] * p
        equity.append(value)

    eq = np.array(equity)
    return calc_metrics(eq, INITIAL_CAPITAL, n_trades), eq


def calc_metrics(eq: np.ndarray, init: float, n_trades: int) -> dict:
    """从净值序列计算绩效 (年化/夏普/回撤)."""
    n = len(eq)
    if n < 2:
        return {"error": "insufficient"}
    total_ret = eq[-1] / init - 1.0
    rets = np.diff(eq) / eq[:-1]
    ann_vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
    ann_ret = (1 + total_ret) ** (252 / max(n, 1)) - 1.0
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    cummax = np.maximum.accumulate(eq)
    max_dd = float(((eq - cummax) / cummax).min())
    return {
        "total_return": round(float(total_ret), 4),
        "ann_return": round(float(ann_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(max_dd, 4),
        "n_trades": n_trades,
        "win_days": round(float((rets > 0).mean()), 4) if len(rets) else None,
    }


def run_equal_weight(mat: dict, codes: list, start_idx: int, end_idx: int) -> dict:
    """等权买入持有基准: 区间首日建仓, 持有至末."""
    if end_idx - start_idx < 2:
        return {"error": "insufficient"}
    init_prices = {c: mat[c][start_idx] for c in codes if mat[c][start_idx] > 0}
    if not init_prices:
        return {"error": "no prices"}
    per = INITIAL_CAPITAL / len(init_prices)
    shares = {c: int(per / p / 100) * 100 for c, p in init_prices.items()}
    cash = INITIAL_CAPITAL - sum(s * init_prices[c] for c, s in shares.items())
    eq = []
    for i in range(start_idx, end_idx):
        v = cash
        for c, s in shares.items():
            p = mat[c][i]
            if p > 0 and np.isfinite(p):
                v += s * p
        eq.append(v)
    return calc_metrics(np.array(eq), INITIAL_CAPITAL, 0)


def run_mom_baseline(mat: dict, dates: list, codes: list, start_idx: int, end_idx: int) -> dict:
    """M0 动量轮动基准 (10日×0.5+20日×0.5, 近似V3主框架)."""
    frames = {}
    for c in codes:
        frames[c] = pd.DataFrame({"trade_date": dates, "close": mat[c]})
    # 防御资产用货币基金 (mat 中无 511880? 有, close_matrix 含全部资产)
    frames[DEFENSE] = pd.DataFrame({"trade_date": dates, "close": mat[DEFENSE]})
    result = run_backtest(
        frames,
        {c: c for c in codes},
        score_m0,
        start_date=dates[start_idx].isoformat(),
        end_date=dates[end_idx - 1].isoformat(),
        use_a_share_filter=True,
        use_drop_filter=True,
    )
    if "error" in result:
        return {"error": result["error"]}
    return {k: v for k, v in result.items() if k != "equity_curve"}


# --------------------------------------------------------------------------- #
# Walk-forward 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("  R4 暴涨后持续规律 — Walk-forward 验证")
    print(f"  窗口: {TRAIN_DAYS}日训练 → {BUFFER_DAYS}日缓冲 → {TEST_DAYS}日测试 × {N_WINDOWS}窗口")
    print(f"  参数网格: 阈值{GRID_THRESHOLD} × 持有{GRID_HOLD}日 (训练期按夏普学习)")
    print("=" * 74)

    data = load_data()
    dates = common_dates(data)
    mat = close_matrix(data, dates)
    codes = list(ETF_POOL.keys())
    print(f"\n  数据: {len(dates)} 个交易日 ({dates[0]} ~ {dates[-1]})")

    # 窗口起点 (确保测试段不越界)
    max_start = len(dates) - TRAIN_DAYS - BUFFER_DAYS - TEST_DAYS
    starts = list(range(0, max_start, STEP))[:N_WINDOWS]
    if len(starts) < N_WINDOWS:
        print(f"  ⚠️ 数据不足, 实际窗口数: {len(starts)}")

    results = {
        "meta": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "buffer_days": BUFFER_DAYS,
            "grid": {"threshold": list(GRID_THRESHOLD), "hold": list(GRID_HOLD)},
            "cost": {"fee": FEE, "slippage": SLIPPAGE},
            "pass_criteria": "≥3/4窗口测试夏普>0 且平均年化>等权基准",
        },
        "windows": [],
    }

    test_sharpes = []
    test_anns = []
    test_best = []
    baseline_anns = []

    for wi, s0 in enumerate(starts, 1):
        train_end = s0 + TRAIN_DAYS
        test_start = train_end + BUFFER_DAYS
        test_end = test_start + TEST_DAYS
        print(
            f"\n  ── W{wi}: 训练 {dates[s0]}~{dates[train_end - 1]} "
            f"| 测试 {dates[test_start]}~{dates[test_end - 1]}"
        )

        # 1. 训练期网格搜索
        best_param, best_sharpe = None, -999.0
        train_results = {}
        for thr in GRID_THRESHOLD:
            for hold in GRID_HOLD:
                res, _eq = run_event_strategy(mat, codes, thr, hold, s0, train_end)
                if "error" in res:
                    continue
                train_results[f"thr{thr:.0%}_h{hold}"] = res
                if res["sharpe"] > best_sharpe:
                    best_sharpe, best_param = res["sharpe"], (thr, hold)
        if best_param is None:
            print("    ✗ 训练期全部参数无效")
            continue
        thr_b, hold_b = best_param
        print(f"    [训练] 最优参数: 阈值{thr_b:.0%} 持有{hold_b}日 (夏普{best_sharpe:.2f})")

        # 2. 测试期: 冻结参数 + 全组合稳健性 (冻结参数须处于测试期12组合的中位以上)
        test_all = {}
        for thr in GRID_THRESHOLD:
            for hold in GRID_HOLD:
                res, _eq = run_event_strategy(mat, codes, thr, hold, test_start, test_end)
                test_all[f"thr{thr:.0%}_h{hold}"] = res
        test_res = test_all[f"thr{thr_b:.0%}_h{hold_b}"]
        if "error" in test_res:
            print("    ✗ 测试期运行失败")
            continue
        sharpes = sorted(r["sharpe"] for r in test_all.values())
        rank = sum(1 for s in sharpes if s < test_res["sharpe"]) + 1
        n_combos = len(sharpes)
        robust = rank >= n_combos * 0.5  # 冻结参数夏普排在中位及以上
        test_sharpes.append(test_res["sharpe"])
        test_anns.append(test_res["ann_return"])
        test_best.append({"threshold": thr_b, "hold": hold_b})
        passed = test_res["sharpe"] > 0
        print(
            f"    [测试] 年化{test_res['ann_return']:+.1%} 夏普{test_res['sharpe']:.2f} "
            f"回撤{test_res['max_drawdown']:.1%} 交易{test_res['n_trades']}笔 "
            f"{'PASS' if passed else 'FAIL'}"
        )
        print(
            f"    [稳健] 冻结参数夏普在测试期{n_combos}组合中排第{rank}名 {'✓' if robust else '✗'}"
        )

        # 3. 基准
        ew = run_equal_weight(mat, codes, test_start, test_end)
        ew_ann = ew.get("ann_return", 0.0)
        mom = run_mom_baseline(mat, dates, codes, test_start, test_end)
        mom_ann = mom.get("ann_return", 0.0)
        mom_shp = mom.get("sharpe", 0.0)
        baseline_anns.append(ew_ann)
        beat_ew = test_res["ann_return"] > ew_ann
        print(
            f"    [基准] 等权年化{ew_ann:+.1%} | M0动量年化{mom_ann:+.1%} 夏普{mom_shp:.2f} "
            f"| R4 vs 等权: {'跑赢' if beat_ew else '跑输'}"
        )

        results["windows"].append(
            {
                "window": f"W{wi}",
                "train_span": [str(dates[s0]), str(dates[train_end - 1])],
                "test_span": [str(dates[test_start]), str(dates[test_end - 1])],
                "train_best": {"threshold": thr_b, "hold": hold_b, "sharpe": round(best_sharpe, 3)},
                "train_grid": train_results,
                "test": test_res,
                "test_grid_all": test_all,
                "test_sharpe_rank": rank,
                "test_n_combos": n_combos,
                "robust": robust,
                "baselines": {
                    "equal_weight": ew,
                    "mom_m0": mom,
                    "beat_equal_weight": beat_ew,
                },
            }
        )

    # === 综合判定 ===
    print("\n" + "=" * 74)
    n_win = sum(1 for s in test_sharpes if s > 0)
    n_valid = len(test_sharpes)
    n_robust = sum(1 for r in results["windows"] if r.get("robust"))
    avg_ann = float(np.mean(test_anns)) if test_anns else 0.0
    avg_base = float(np.mean(baseline_anns)) if baseline_anns else 0.0
    print(
        f"  测试期: {n_win}/{n_valid} 窗口夏普>0 | 参数稳健 {n_robust}/{n_valid} 窗口 | "
        f"平均年化 {avg_ann:+.1%} vs 等权基准 {avg_base:+.1%}"
    )
    passed_overall = (
        n_win >= max(3, (n_valid * 3 + 3) // 4)
        and avg_ann > avg_base
        and n_robust >= max(2, (n_valid + 1) // 2)
    )
    if not passed_overall:
        detail = []
        if n_win < max(3, (n_valid * 3 + 3) // 4):
            detail.append("夏普>0窗口数不足")
        if avg_ann <= avg_base:
            detail.append("平均年化未跑赢等权")
        if n_robust < max(2, (n_valid + 1) // 2):
            detail.append("参数稳健性不足")
        print(f"  未达标项: {', '.join(detail)}")
    verdict = (
        "✅ R4 通过 walk-forward 验证 (含稳健性)"
        if passed_overall
        else "❌ R4 未通过 walk-forward 验证 (形式达标但稳健性/收益不足)"
    )
    print(f"  {verdict}")
    print("=" * 74)

    results["summary"] = {
        "windows_valid": n_valid,
        "windows_sharpe_pos": n_win,
        "windows_robust": n_robust,
        "avg_test_ann": round(avg_ann, 4),
        "avg_equal_weight_ann": round(avg_base, 4),
        "passed": passed_overall,
        "verdict": verdict,
        "best_params_by_window": test_best,
    }
    out_path = OUTPUT_DIR / "walk_forward_r4.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
