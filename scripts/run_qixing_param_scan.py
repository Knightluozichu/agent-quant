"""七星V3参数扫描 — 反推原版动量公式.

测试不同动量周期/组合, 找最接近原版逐年收益的配置.
原版: 2020:+27% 2021:+33% 2022:+76% 2023:+8% 2024:+55% 2025:+238%
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cross_asset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "qixing_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ETF_POOL = {
    "518880": "黄金ETF",
    "159985": "豆粕ETF",
    "501018": "南方原油",
    "161226": "白银LOF",
    "513100": "纳指ETF",
    "159915": "创业板ETF",
    "511220": "城投债ETF",
}
DEFENSE = "511880"
FEE = 0.0005
SLIPPAGE = 0.001

ORIGINAL = {2020: 0.2716, 2021: 0.3278, 2022: 0.7605, 2023: 0.0810, 2024: 0.5473, 2025: 2.3803}


def load_data():
    data = {}
    for code in list(ETF_POOL.keys()) + [DEFENSE]:
        f = DATA_DIR / f"{code}.parquet"
        if f.exists():
            data[code] = pd.read_parquet(f).sort_values("trade_date").reset_index(drop=True)
    return data


def momentum_score(close, periods, weights, risk_adj=False):
    """计算动量得分."""
    score = 0.0
    for p, w in zip(periods, weights):
        if len(close) > p:
            ret = (close[-1] - close[-p - 1]) / close[-p - 1]
            score += ret * w
    if risk_adj and len(close) > 21:
        rets = np.diff(close[-21:]) / close[-21:-1]
        vol = np.std(rets)
        if vol > 0:
            score = score / vol
    return score


def run_backtest(
    data,
    periods,
    weights,
    rebalance=5,
    risk_adj=False,
    drop_filter=False,
    a_share_filter=True,
    switch_threshold=0.02,
    initial_capital=100_000.0,
):
    """通用回测."""
    common_dates = None
    for code in ETF_POOL:
        if code in data:
            dates = set(data[code]["trade_date"].tolist())
            common_dates = dates if common_dates is None else common_dates & dates
    if DEFENSE in data:
        common_dates &= set(data[DEFENSE]["trade_date"].tolist())

    all_dates = sorted(common_dates)
    trading_dates = all_dates[130:]
    rebalance_dates = trading_dates[::rebalance]

    cash = initial_capital
    holding = None
    holding_shares = 0
    equity_history = []

    for td in rebalance_dates:
        etf_idx = {}
        for code in list(ETF_POOL.keys()) + [DEFENSE]:
            if code in data:
                mask = data[code]["trade_date"] <= td
                if mask.sum() >= 130:
                    etf_idx[code] = mask.sum() - 1

        # A股走弱判断
        a_weak = False
        if a_share_filter and "159915" in etf_idx:
            c = data["159915"]["close"].values[: etf_idx["159915"] + 1].astype(float)
            if len(c) >= 20:
                a_weak = c[-1] < np.mean(c[-20:])

        candidates = []
        for code in ETF_POOL:
            if code not in etf_idx:
                continue
            if code == "159915" and a_weak:
                continue
            idx = etf_idx[code]
            close = data[code]["close"].values[: idx + 1].astype(float)
            if len(close) < max(periods) + 1:
                continue
            # 单日跌幅过滤
            if drop_filter and len(close) >= 4:
                skip = False
                for i in range(-3, 0):
                    if (close[i] - close[i - 1]) / close[i - 1] < -0.03:
                        skip = True
                        break
                if skip:
                    continue
            score = momentum_score(close, periods, weights, risk_adj)
            if score > 0:
                candidates.append((code, score))

        candidates.sort(key=lambda x: -x[1])
        best = candidates[0][0] if candidates else DEFENSE
        best_score = candidates[0][1] if candidates else 0

        # 换仓逻辑
        if holding and holding != DEFENSE:
            cur_score = dict(candidates).get(holding, -999)
            if cur_score > 0 and best_score <= cur_score + switch_threshold:
                target = holding
            else:
                target = best
        else:
            target = best

        # 交易
        if target != holding:
            if holding and holding in data:
                row = data[holding][data[holding]["trade_date"] == td]
                if not row.empty:
                    cash += holding_shares * row.iloc[0]["close"] * (1 - FEE - SLIPPAGE)
                    holding, holding_shares = None, 0
            if target in data:
                row = data[target][data[target]["trade_date"] == td]
                if not row.empty:
                    price = row.iloc[0]["close"]
                    shares = int(cash * 0.99 / price / 100) * 100
                    if shares > 0:
                        cash -= shares * price * (1 + FEE + SLIPPAGE)
                        holding, holding_shares = target, shares

        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity})

    if not equity_history:
        return None

    eq = pd.DataFrame(equity_history)
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])
    eq["year"] = eq["trade_date"].dt.year

    yearly = {}
    prev = initial_capital
    for year in sorted(eq["year"].unique()):
        ydf = eq[eq["year"] == year]
        if ydf.empty:
            continue
        end = ydf["equity"].iloc[-1]
        yearly[int(year)] = (end / prev) - 1
        prev = end

    total = (eq["equity"].iloc[-1] / initial_capital) - 1
    return {"yearly": yearly, "total": total}


def score_vs_original(yearly):
    """计算与原版的匹配度 (越小越好)."""
    err = 0.0
    n = 0
    for year, orig_ret in ORIGINAL.items():
        if year in yearly:
            err += abs(yearly[year] - orig_ret)
            n += 1
    return err / n if n > 0 else 999


def main():
    print("=" * 70)
    print("  七星V3参数扫描 — 反推原版动量公式")
    print("  原版: 2020:+27% 2021:+33% 2022:+76% 2023:+8% 2024:+55% 2025:+238%")
    print("=" * 70)

    data = load_data()
    print(f"\n  数据: {len(data)}只ETF")

    # 参数组合
    configs = [
        # (periods, weights, risk_adj, name)
        ((10,), (1.0,), False, "10日"),
        ((20,), (1.0,), False, "20日"),
        ((30,), (1.0,), False, "30日"),
        ((40,), (1.0,), False, "40日"),
        ((60,), (1.0,), False, "60日"),
        ((10, 20), (0.5, 0.5), False, "10+20"),
        ((20, 60), (0.5, 0.5), False, "20+60"),
        ((10, 30, 60), (0.4, 0.3, 0.3), False, "10+30+60"),
        ((20, 60, 120), (0.4, 0.3, 0.3), False, "20+60+120"),
        ((20,), (1.0,), True, "20日风险调整"),
        ((60,), (1.0,), True, "60日风险调整"),
    ]

    results = []
    for periods, weights, risk_adj, name in configs:
        for rebalance in [5, 10]:
            for drop_filter in [False, True]:
                res = run_backtest(
                    data,
                    periods,
                    weights,
                    rebalance=rebalance,
                    risk_adj=risk_adj,
                    drop_filter=drop_filter,
                )
                if res is None:
                    continue
                err = score_vs_original(res["yearly"])
                results.append(
                    {
                        "name": name,
                        "rebalance": rebalance,
                        "drop_filter": drop_filter,
                        "risk_adj": risk_adj,
                        "error": err,
                        "total": res["total"],
                        "yearly": res["yearly"],
                    }
                )

    # 按匹配度排序
    results.sort(key=lambda x: x["error"])

    print(
        f"\n  {'排名':<4} {'动量':<14} {'调仓':<5} {'跌幅过滤':<8} {'误差':>6} {'总收益':>9} | 2020 2021 2022 2023 2024 2025"
    )
    print(f"  {'-' * 100}")
    for i, r in enumerate(results[:15]):
        y = r["yearly"]
        yr_str = " ".join(f"{y.get(yr, 0):+.0%}" for yr in [2020, 2021, 2022, 2023, 2024, 2025])
        print(
            f"  {i + 1:<4} {r['name']:<14} {r['rebalance']:<5} "
            f"{'是' if r['drop_filter'] else '否':<8} {r['error']:>6.2f} {r['total']:>+9.0%} | {yr_str}"
        )

    print("\n  原版参考:                                        | +27% +33% +76% +8% +55% +238%")

    # 保存最佳结果
    best = results[0]
    with open(OUTPUT_DIR / "param_scan_results.json", "w") as f:
        json.dump(
            {
                "best": {k: v for k, v in best.items() if k != "yearly"},
                "best_yearly": {str(k): v for k, v in best["yearly"].items()},
                "all_results": [
                    {k: v for k, v in r.items() if k != "yearly"} for r in results[:20]
                ],
            },
            f,
            indent=2,
            default=str,
        )
    print(
        f"\n  最佳配置: {best['name']} rebalance={best['rebalance']} drop_filter={best['drop_filter']}"
    )
    print(f"  结果已保存: {OUTPUT_DIR / 'param_scan_results.json'}")


if __name__ == "__main__":
    main()
