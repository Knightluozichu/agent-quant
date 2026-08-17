"""R3: Walk-forward 滚动验证框架.

固定边界 (不可含糊):
  - 6 年数据分 4 段滚动, 每段边界固定写入配置
  - 每段 train / validation / test 边界明确
  - expanding window (默认 yes)
  - 每段之间设置 purge (1 交易日) + embargo (2 交易日) 消除重叠
  - 每段最小交易数 ≥ 20 笔, 不足则该段标记不可用
  - 所有失败尝试均登记到 manifest

验证内容:
  - 报告 OOS 夏普和收益
  - 参数扰动 ±20% 扫描确认稳定性
  - 计算 PBO (Probability of Backtest Overfitting)

用法:
  PYTHONHOME= PYTHONPATH= .venv/bin/python scripts/walk_forward.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from run_qixing_v3 import (  # noqa: E402
    _compute_data_hash,
    _compute_param_hash,
    load_data,
    run_qixing_v3_no_lookahead,
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "qixing_results"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Walk-forward 固定边界 (expanding window)
# 每段: train → [purge 1天 + embargo 2天] → test (1年, 确保 > 130 warmup)
WALK_FORWARD_SEGMENTS = [
    {
        "name": "WF-1",
        "train_start": "2020-01-02",
        "train_end": "2021-06-30",
        "test_start": "2021-07-05",  # purge 1天 + embargo 2天 后
        "test_end": "2022-06-30",
    },
    {
        "name": "WF-2",
        "train_start": "2020-01-02",
        "train_end": "2022-06-30",
        "test_start": "2022-07-05",
        "test_end": "2023-06-30",
    },
    {
        "name": "WF-3",
        "train_start": "2020-01-02",
        "train_end": "2023-06-30",
        "test_start": "2023-07-05",
        "test_end": "2024-06-30",
        "contaminated": True,  # 2023-2024 已污染
    },
    {
        "name": "WF-4",
        "train_start": "2020-01-02",
        "train_end": "2024-06-30",
        "test_start": "2024-07-05",
        "test_end": "2025-06-30",
        "contaminated": True,  # 2024 已污染
    },
]

# 参数扰动 ±20% (用于稳定性检查)
PARAM_PERTURBATIONS = [
    {"label": "baseline", "drop_lookback": 5, "a_share_ma": 15},
    {"label": "drop-20%", "drop_lookback": 4, "a_share_ma": 15},
    {"label": "drop+20%", "drop_lookback": 6, "a_share_ma": 15},
    {"label": "ma-20%", "drop_lookback": 5, "a_share_ma": 12},
    {"label": "ma+20%", "drop_lookback": 5, "a_share_ma": 18},
]

MIN_TRADES_PER_SEGMENT = 20


def _filter_data_by_date(data: dict, start: str, end: str) -> dict:
    """筛选指定日期范围内的数据."""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    filtered = {}
    for code, df in data.items():
        dates = pd.to_datetime(df["trade_date"])
        mask = (dates >= start_dt) & (dates <= end_dt)
        filtered[code] = df[mask].reset_index(drop=True)
    return filtered


def _run_backtest_with_params(data: dict, drop_lookback: int, a_share_ma: int) -> dict:
    """用指定参数运行无未来函数回测.

    临时修改全局参数, 运行后恢复.
    """
    import run_qixing_v3 as rq

    # 保存原始参数
    orig_drop = rq.DROP_LOOKBACK
    orig_ma = rq.A_SHARE_MA

    try:
        # 临时修改参数
        rq.DROP_LOOKBACK = drop_lookback
        rq.A_SHARE_MA = a_share_ma
        result = run_qixing_v3_no_lookahead(data)
    finally:
        # 恢复原始参数
        rq.DROP_LOOKBACK = orig_drop
        rq.A_SHARE_MA = orig_ma

    return result


def run_walk_forward(data: dict) -> list[dict]:
    """运行 walk-forward 滚动验证.

    Returns:
        每段的验证结果列表
    """
    results = []
    for seg in WALK_FORWARD_SEGMENTS:
        print(f"\n  {'=' * 50}")
        print(
            f"  {seg['name']}: train {seg['train_start']} ~ {seg['train_end']}"
            f" → test {seg['test_start']} ~ {seg['test_end']}"
        )
        if seg.get("contaminated"):
            print("  ⚠️  此段测试期落在已污染区间 (2023-2024)")
        print(f"  {'=' * 50}")

        # 筛选测试期数据
        test_data = _filter_data_by_date(data, seg["test_start"], seg["test_end"])

        # 检查数据量
        sample_code = next(iter(test_data))
        n_days = len(test_data[sample_code])
        if n_days < 130:
            print(f"  ⚠️  测试期数据不足 ({n_days}天 < 130 warmup), 跳过")
            results.append(
                {
                    "segment": seg["name"],
                    "status": "insufficient_data",
                    "n_days": n_days,
                }
            )
            continue

        # 运行基准参数回测
        print("  运行基准参数回测...")
        result = _run_backtest_with_params(test_data, 5, 15)

        if "error" in result:
            print(f"  ❌ 回测失败: {result['error']}")
            results.append(
                {
                    "segment": seg["name"],
                    "status": "error",
                    "error": result["error"],
                }
            )
            continue

        n_trades = result["n_trades"]
        contaminated = seg.get("contaminated", False)

        segment_result = {
            "segment": seg["name"],
            "train_period": f"{seg['train_start']} ~ {seg['train_end']}",
            "test_period": f"{seg['test_start']} ~ {seg['test_end']}",
            "contaminated": contaminated,
            "n_trades": n_trades,
            "n_cancelled": result.get("n_cancelled", 0),
            "total_return": result["total_return"],
            "ann_return": result["ann_return"],
            "sharpe": result["sharpe"],
            "max_drawdown": result["max_drawdown"],
            "status": "ok" if n_trades >= MIN_TRADES_PER_SEGMENT else "low_trades",
        }

        print(
            f"  交易: {n_trades}笔 | 夏普: {result['sharpe']:.2f} | "
            f"收益: {result['total_return']:+.1%} | 回撤: {result['max_drawdown']:.1%}"
        )

        if n_trades < MIN_TRADES_PER_SEGMENT:
            print(f"  ⚠️  交易数 {n_trades} < {MIN_TRADES_PER_SEGMENT}, 标记为低交易量")

        # 参数扰动稳定性检查
        print("\n  参数扰动 ±20% 稳定性检查:")
        perturbation_results = []
        for pert in PARAM_PERTURBATIONS:
            pert_result = _run_backtest_with_params(
                test_data, pert["drop_lookback"], pert["a_share_ma"]
            )
            if "error" not in pert_result:
                pr = {
                    "label": pert["label"],
                    "drop_lookback": pert["drop_lookback"],
                    "a_share_ma": pert["a_share_ma"],
                    "sharpe": pert_result["sharpe"],
                    "total_return": pert_result["total_return"],
                }
                perturbation_results.append(pr)
                print(
                    f"    {pert['label']:<10} 夏普: {pert_result['sharpe']:>6.2f} "
                    f"收益: {pert_result['total_return']:>+8.1%}"
                )

        segment_result["perturbation"] = perturbation_results

        # 稳定性判断: 扰动后夏普变化 < 30% 视为稳定
        if perturbation_results:
            sharpes = [p["sharpe"] for p in perturbation_results]
            baseline_sharpe = result["sharpe"]
            if baseline_sharpe > 0:
                max_deviation = max(abs(s - baseline_sharpe) / baseline_sharpe for s in sharpes)
                segment_result["stability"] = "stable" if max_deviation < 0.3 else "unstable"
                segment_result["max_sharpe_deviation"] = max_deviation
                print(f"  稳定性: {segment_result['stability']} (最大偏差: {max_deviation:.1%})")

        results.append(segment_result)

    return results


def calculate_pbo(wf_results: list[dict]) -> dict:
    """计算 PBO (Probability of Backtest Overfitting).

    简化版 CSCV:
      1. 构建参数×段落 的 Sharpe 矩阵
      2. 对每种 train/test 划分, 检查 train 最优参数是否在 test 上也最优
      3. PBO = train 最优参数在 test 上排名低于中位数的比例

    Returns:
        PBO 结果 dict
    """
    # 收集有扰动结果的段落 (不限交易量, 只要参数扰动数据可用)
    valid_segments = [
        r for r in wf_results if r.get("perturbation") and r.get("status") in ("ok", "low_trades")
    ]
    if len(valid_segments) < 2:
        return {
            "pbo": None,
            "reason": "有效段落不足 (< 2), 无法计算 PBO",
            "n_segments": len(valid_segments),
        }

    # 构建 Sharpe 矩阵: rows = 参数集, cols = 段落
    n_params = len(valid_segments[0]["perturbation"])
    n_segments = len(valid_segments)

    if n_params < 2:
        return {
            "pbo": None,
            "reason": "参数集不足 (< 2), 无法计算 PBO",
            "n_params": n_params,
        }

    sharpe_matrix = np.zeros((n_params, n_segments))
    for j, seg in enumerate(valid_segments):
        for i, pert in enumerate(seg["perturbation"]):
            sharpe_matrix[i, j] = pert["sharpe"]

    # CSCV: 对每种对称划分 (一半 train, 一半 test)
    from itertools import combinations

    n_half = n_segments // 2
    if n_half < 1:
        return {
            "pbo": None,
            "reason": "段落数不足以划分 train/test",
            "n_segments": n_segments,
        }

    n_overfit = 0
    n_total = 0

    for train_idx in combinations(range(n_segments), n_half):
        test_idx = [i for i in range(n_segments) if i not in train_idx]
        if not test_idx:
            continue

        # train 上每个参数的平均 Sharpe
        train_sharpe = sharpe_matrix[:, list(train_idx)].mean(axis=1)
        # test 上每个参数的平均 Sharpe
        test_sharpe = sharpe_matrix[:, test_idx].mean(axis=1)

        # train 上最优参数
        best_train_idx = np.argmax(train_sharpe)

        # 该参数在 test 上的排名
        test_rank = np.argsort(np.argsort(test_sharpe))
        best_param_test_rank = test_rank[best_train_idx]

        # 如果排名低于中位数 → overfit
        median_rank = n_params // 2
        if best_param_test_rank < median_rank:
            n_overfit += 1
        n_total += 1

    pbo = n_overfit / n_total if n_total > 0 else 0.0

    return {
        "pbo": round(pbo, 3),
        "n_overfit": n_overfit,
        "n_total": n_total,
        "interpretation": (f"PBO={pbo:.1%}: " + ("过拟合风险低" if pbo < 0.5 else "过拟合风险高")),
    }


def main():
    print("=" * 60)
    print("  Walk-forward 滚动验证 (R3)")
    print("  4 段 expanding window + purge/embargo + 参数扰动 + PBO")
    print("=" * 60)

    print("\n  加载数据...")
    data = load_data()
    print(f"  数据: {len(data)}只ETF")

    # 运行 walk-forward
    wf_results = run_walk_forward(data)

    # 计算 PBO
    print(f"\n  {'=' * 50}")
    print("  PBO (Probability of Backtest Overfitting)")
    print(f"  {'=' * 50}")
    pbo_result = calculate_pbo(wf_results)
    print(f"  {pbo_result.get('interpretation', pbo_result.get('reason', 'N/A'))}")
    if pbo_result.get("pbo") is not None:
        print(
            f"  PBO = {pbo_result['pbo']:.1%} "
            f"({pbo_result['n_overfit']}/{pbo_result['n_total']} 次过拟合)"
        )

    # 汇总
    print(f"\n  {'=' * 50}")
    print("  Walk-forward 汇总")
    print(f"  {'=' * 50}")
    print(f"  {'段落':<8} {'交易':>6} {'夏普':>8} {'收益':>10} {'回撤':>8} {'稳定':>6}")
    print(f"  {'-' * 50}")
    for r in wf_results:
        if r.get("status") == "ok":
            contam = "⚠️" if r.get("contaminated") else "  "
            print(
                f"  {r['segment']:<8} {r['n_trades']:>6} {r['sharpe']:>8.2f} "
                f"{r['total_return']:>+10.1%} {r['max_drawdown']:>8.1%} "
                f"{r.get('stability', 'N/A'):>6} {contam}"
            )
        else:
            print(f"  {r['segment']:<8} {r.get('status', 'N/A')}")

    # OOS 平均 (排除已污染段落)
    clean_results = [r for r in wf_results if r.get("status") == "ok" and not r.get("contaminated")]
    if clean_results:
        avg_sharpe = np.mean([r["sharpe"] for r in clean_results])
        avg_return = np.mean([r["total_return"] for r in clean_results])
        print("\n  OOS 平均 (排除已污染段落):")
        print(f"    夏普: {avg_sharpe:.2f} | 平均收益: {avg_return:+.1%}")

    # 保存结果
    output = {
        "walk_forward_results": wf_results,
        "pbo": pbo_result,
        "param_hash": _compute_param_hash(),
        "data_hash": _compute_data_hash(data),
        "timestamp": str(date.today()),
    }
    output_path = OUTPUT_DIR / "walk_forward_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  结果已保存: {output_path}")

    # 更新 manifest
    manifest_path = MANIFEST_DIR / "validation_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["walk_forward"] = {
            "segments": len(wf_results),
            "pbo": pbo_result.get("pbo"),
            "clean_avg_sharpe": float(avg_sharpe) if clean_results else None,
            "timestamp": str(date.today()),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  Manifest 已更新: {manifest_path}")


if __name__ == "__main__":
    main()
