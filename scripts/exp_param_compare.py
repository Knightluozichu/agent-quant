"""参数对比: 动量周期 × 调仓频率 × 暴跌过滤 多配置滚动回测对比.

测试不同参数组合, 滚动逐年回测, 对比全周期收益/夏普/回撤/交易次数。
用法: uv run python scripts/exp_param_compare.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq
from strategy_lab.engine import backtest

ORIG = {
    "MOM_PERIODS": rq.MOM_PERIODS,
    "MOM_WEIGHTS": rq.MOM_WEIGHTS,
    "USE_DROP_FILTER": rq.USE_DROP_FILTER,
}


def select_wrapper(data, idx_map, holding, params):
    target, _, _, _ = rq.select_target(data, idx_map, holding)
    return target


# (名称, 动量周期, 权重, 调仓天数, 暴跌过滤)
CONFIGS = [
    ("V3基线(10+20/5日/过滤开)", (10, 20), (0.5, 0.5), 5, True),
    ("配置1a(5+10/2日/过滤开)", (5, 10), (0.5, 0.5), 2, True),
    ("配置1b(5+10/3日/过滤开)", (5, 10), (0.5, 0.5), 3, True),
    ("配置2(10+20/10日/过滤开)", (10, 20), (0.5, 0.5), 10, True),
    ("配置3(10+20/10日/过滤关)", (10, 20), (0.5, 0.5), 10, False),
    ("配置4(10+20/20日/过滤关)", (10, 20), (0.5, 0.5), 20, False),
]


def run_config(data, cfg):
    name, periods, weights, rebal, drop = cfg
    rq.MOM_PERIODS = periods
    rq.MOM_WEIGHTS = weights
    rq.USE_DROP_FILTER = drop
    full = backtest(data, select_wrapper, {}, rebal)
    years = list(range(2020, 2027))
    yearly = {}
    for y in years:
        r = backtest(
            data, select_wrapper, {}, rebal, start_date=date(y, 1, 1), end_date=date(y, 12, 31)
        )
        yearly[y] = r["total_return"] * 100
    return {"full": full, "yearly": yearly}


def main() -> None:
    data = rq.load_data()
    results = {}
    for cfg in CONFIGS:
        print(f"  回测: {cfg[0]} ...", flush=True)
        results[cfg[0]] = run_config(data, cfg)
    # 还原
    rq.MOM_PERIODS = ORIG["MOM_PERIODS"]
    rq.MOM_WEIGHTS = ORIG["MOM_WEIGHTS"]
    rq.USE_DROP_FILTER = ORIG["USE_DROP_FILTER"]

    years = list(range(2020, 2027))
    base_yearly = results[CONFIGS[0][0]]["yearly"]

    print("\n" + "=" * 108)
    print("  参数对比 | 滚动逐年回测(每年fresh) + 全周期指标")
    print("=" * 108)
    hdr = (
        f"  {'配置':<28}"
        + "".join(f"{y:>8}" for y in years)
        + f"{'全周期':>10}{'年化':>9}{'夏普':>7}{'回撤':>8}{'交易':>7}{'胜V3':>6}"
    )
    print(hdr)
    print("  " + "-" * 104)
    for cfg in CONFIGS:
        name = cfg[0]
        r = results[name]
        full = r["full"]
        cells = "".join(f"{r['yearly'][y]:>+7.0f}%" for y in years)
        if name == CONFIGS[0][0]:
            win = "-"
        else:
            win = f"{sum(1 for y in years if r['yearly'][y] > base_yearly[y])}/7"
        print(
            f"  {name:<28}{cells}{full['total_return'] * 100:>+9.0f}%"
            f"{full['ann_return'] * 100:>+8.1f}%{full['sharpe']:>7.2f}"
            f"{full['max_drawdown'] * 100:>+7.1f}%{full['n_trades']:>7}{win:>6}"
        )
    print("=" * 108)
    print("  判读: '胜V3'=该配置在多少年跑赢V3基线。交易次数越多=手续费损耗越大。")


if __name__ == "__main__":
    main()
