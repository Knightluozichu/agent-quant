"""扩展跨资产池实验: 增加低相关资产 vs 当前策略对比.

对比池:
  A) Base-Full (8): 当前生产池 [黄金/豆粕/原油/白银/纳指/创业板/城投债/货币]
  B) Extended-10 (+标普500, +十年国债)
  C) Extended-11 (+标普500, +十年国债, +德国ETF)
  D) Extended-12 (+标普500, +十年国债, +德国ETF, +能源化工)

验证: IS/OOS + 3滚动窗口 + 全回测
公式: M0 (0.5*R10 + 0.5*R20), 上轮实验验证的最优公式
"""

from __future__ import annotations

import json
import warnings
from datetime import date as dt_date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cross_asset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "v8_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 交易参数 (与V3一致) ===
FEE = 0.0005
SLIPPAGE = 0.001
REBALANCE_DAYS = 5
WARMUP = 130
INITIAL_CAPITAL = 100_000.0

# === ETF池定义 ===
DEFENSE = "511880"
A_SHARE_ETF = "159915"

# A) 当前生产池 (Base-Full 8)
POOL_BASE = {
    "518880": "黄金ETF",
    "159985": "豆粕ETF",
    "501018": "南方原油",
    "161226": "白银LOF",
    "513100": "纳指ETF",
    "159915": "创业板ETF",
    "511220": "城投债ETF",
}

# B) +标普500 +十年国债 (10)
POOL_EXT_10 = {
    **POOL_BASE,
    "513500": "标普500ETF",
    "511260": "十年国债ETF",
}

# C) +标普500 +十年国债 +德国ETF (11)
POOL_EXT_11 = {
    **POOL_EXT_10,
    "513030": "德国ETF",
}

# D) +标普500 +十年国债 +德国ETF +能源化工 (12)
POOL_EXT_12 = {
    **POOL_EXT_11,
    "159981": "能源化工ETF",
}

POOLS = [
    ("A_Base(8)", POOL_BASE),
    ("B_Ext10(+标普+国债)", POOL_EXT_10),
    ("C_Ext11(+标普+国债+德国)", POOL_EXT_11),
    ("D_Ext12(+标普+国债+德国+能化)", POOL_EXT_12),
]


# ============================================================================ #
# 动量评分 (M0)
# ============================================================================ #


def score_m0(close: np.ndarray) -> float:
    """M0: 0.5*R(10) + 0.5*R(20)."""
    if len(close) <= 20:
        return 0.0
    r10 = (close[-1] - close[-11]) / close[-11]
    r20 = (close[-1] - close[-21]) / close[-21]
    return 0.5 * r10 + 0.5 * r20


# ============================================================================ #
# 回测引擎
# ============================================================================ #


def load_pool_data(pool: dict[str, str]) -> dict[str, pd.DataFrame]:
    """加载指定池的ETF数据."""
    data = {}
    for code in list(pool.keys()) + [DEFENSE]:
        f = DATA_DIR / f"{code}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            data[code] = df.sort_values("trade_date").reset_index(drop=True)
    return data


def check_single_day_drop(close: np.ndarray) -> bool:
    """近3日有单日跌>3% → 排除."""
    if len(close) < 4:
        return True
    for i in range(-3, 0):
        if (close[i] - close[i - 1]) / close[i - 1] < -0.03:
            return False
    return True


def check_a_share_weak(data: dict, as_of_idx: int) -> bool:
    """创业板<MA20 → A股走弱."""
    if A_SHARE_ETF not in data:
        return False
    df = data[A_SHARE_ETF]
    if as_of_idx < 20:
        return False
    close = df["close"].values[: as_of_idx + 1].astype(float)
    return close[-1] < np.mean(close[-20:])


def run_backtest(
    data: dict[str, pd.DataFrame],
    pool: dict[str, str],
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """回测引擎 (M0公式, Top1集中, 周频调仓)."""
    # 找公共日期
    common_dates = None
    for code in pool:
        if code not in data:
            continue
        dates = set(data[code]["trade_date"].tolist())
        common_dates = dates if common_dates is None else common_dates & dates
    if DEFENSE in data:
        common_dates &= set(data[DEFENSE]["trade_date"].tolist())

    if not common_dates:
        return {"error": "no common dates"}

    all_dates = sorted(common_dates)
    if start_date:
        all_dates = [d for d in all_dates if d >= dt_date.fromisoformat(start_date)]
    if end_date:
        all_dates = [d for d in all_dates if d <= dt_date.fromisoformat(end_date)]

    if len(all_dates) < WARMUP + REBALANCE_DAYS:
        return {"error": f"insufficient dates: {len(all_dates)}"}

    trading_dates = all_dates[WARMUP:]
    rebalance_dates = trading_dates[::REBALANCE_DAYS]

    cash = initial_capital
    holding: str | None = None
    holding_shares: int = 0
    equity_history = []
    n_trades = 0

    for td in rebalance_dates:
        etf_data_at_date = {}
        for code in list(pool.keys()) + [DEFENSE]:
            if code not in data:
                continue
            df = data[code]
            mask = df["trade_date"] <= td
            if mask.sum() < WARMUP:
                continue
            etf_data_at_date[code] = mask.sum() - 1

        # A股走弱判断
        a_share_weak = False
        if A_SHARE_ETF in etf_data_at_date:
            a_share_weak = check_a_share_weak(data, etf_data_at_date[A_SHARE_ETF])

        # 动量评分
        candidates = []
        for code in pool:
            if code not in etf_data_at_date:
                continue
            if code == A_SHARE_ETF and a_share_weak:
                continue
            idx = etf_data_at_date[code]
            close = data[code]["close"].values[: idx + 1].astype(float)
            if len(close) < WARMUP:
                continue
            if not check_single_day_drop(close):
                continue
            score = score_m0(close)
            if score > 0:
                candidates.append((code, score))

        candidates.sort(key=lambda x: -x[1])
        best_target = candidates[0][0] if candidates else DEFENSE
        best_score = candidates[0][1] if candidates else 0.0

        # 换仓逻辑
        threshold = 0.0 if best_score > 0.10 else 0.05
        if holding and holding != DEFENSE:
            cur_score = dict(candidates).get(holding, -999)
            if cur_score > 0:
                target = best_target if best_score > cur_score + threshold else holding
            else:
                target = best_target
        else:
            target = best_target

        # 交易执行
        if target != holding:
            if holding and holding in data:
                row = data[holding][data[holding]["trade_date"] == td]
                if not row.empty:
                    cash += holding_shares * row.iloc[0]["close"] * (1 - FEE - SLIPPAGE)
                    n_trades += 1
                    holding = None
                    holding_shares = 0
            if target in data:
                row = data[target][data[target]["trade_date"] == td]
                if not row.empty:
                    price = row.iloc[0]["close"]
                    shares = int(cash * 0.99 / price / 100) * 100
                    if shares > 0:
                        cash -= shares * price * (1 + FEE + SLIPPAGE)
                        holding = target
                        holding_shares = shares
                        n_trades += 1

        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

    if not equity_history:
        return {"error": "no trades"}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    return calc_metrics(eq_df, initial_capital, n_trades)


def calc_metrics(eq_df: pd.DataFrame, initial_capital: float, n_trades: int) -> dict:
    """计算绩效指标."""
    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    periods_per_year = 252 / REBALANCE_DAYS
    ann_vol = daily_rets.std() * np.sqrt(periods_per_year) if len(daily_rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # 年度收益
    eq_copy = eq_df.copy()
    eq_copy["year"] = eq_copy["trade_date"].dt.year
    yearly = {}
    prev_val = initial_capital
    for year in sorted(eq_copy["year"].unique()):
        ydf = eq_copy[eq_copy["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev_val) - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        yearly[int(year)] = {"return": round(yr, 4), "max_dd": round(dd, 4)}
        prev_val = end_val

    return {
        "total_return": round(total_return, 4),
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 3),
        "n_trades": n_trades,
        "turnover_ann": round(n_trades / span_years, 1),
        "span_years": round(span_years, 2),
        "yearly": yearly,
    }


# ============================================================================ #
# 验证框架
# ============================================================================ #


def run_is_oos(data: dict, pool: dict, is_end: str = "2021-12-31") -> dict:
    """IS/OOS分段回测."""
    is_r = run_backtest(data, pool, end_date=is_end)
    oos_r = run_backtest(data, pool, start_date="2022-01-01")
    passed = False
    if "error" not in is_r and "error" not in oos_r:
        passed = oos_r["ann_return"] > 0 and oos_r["sharpe"] > is_r["sharpe"] * 0.5
    return {"IS": is_r, "OOS": oos_r, "passed": passed}


def run_rolling(data: dict, pool: dict) -> dict:
    """3滚动窗口."""
    windows = [
        ("W1", "2019-08-01", "2022-07-31"),
        ("W2", "2022-08-01", "2025-07-31"),
        ("W3", "2025-08-01", "2026-07-31"),
    ]
    results = {}
    positive = 0
    for name, ws, we in windows:
        r = run_backtest(data, pool, start_date=ws, end_date=we)
        results[name] = r
        if "error" not in r and r["ann_return"] > 0:
            positive += 1
    return {"windows": results, "positive": positive, "passed": positive >= 2}


def run_full(data: dict, pool: dict) -> dict:
    """全周期回测."""
    return run_backtest(data, pool)


# ============================================================================ #
# 主实验
# ============================================================================ #


def main():
    print("=" * 70)
    print("  扩展跨资产池实验: 增加低相关资产 vs 当前策略")
    print("  新增: 标普500 / 十年国债 / 德国ETF / 能源化工")
    print("=" * 70)

    all_results = {}

    for pool_name, pool in POOLS:
        print(f"\n{'=' * 70}")
        print(f"  池: {pool_name}")
        print(f"  ETF: {', '.join(f'{v}' for v in pool.values())} + 货币(防御)")
        print(f"{'=' * 70}")

        data = load_pool_data(pool)
        loaded = [c for c in list(pool.keys()) + [DEFENSE] if c in data]
        print(f"  数据: {len(loaded)} 只ETF已加载")

        # 显示公共日期范围
        common = None
        for code in loaded:
            dates = set(data[code]["trade_date"].tolist())
            common = dates if common is None else common & dates
        if common:
            print(f"  公共日期: {min(common)} ~ {max(common)} ({len(common)}天)")

        # 1. IS/OOS
        print(f"\n  [1/3] IS/OOS...", end=" ", flush=True)
        is_oos = run_is_oos(data, pool)
        oos = is_oos["OOS"]
        if "error" not in oos:
            is_m = is_oos["IS"]
            print(
                f"IS年化={is_m.get('ann_return', 'ERR'):+.1%} "
                f"OOS年化={oos['ann_return']:+.1%} 夏普={oos['sharpe']:.2f} "
                f"{'PASS' if is_oos['passed'] else 'FAIL'}"
            )
        else:
            print(f"ERROR: {oos.get('error', 'unknown')}")

        # 2. 滚动窗口
        print(f"  [2/3] Rolling...", end=" ", flush=True)
        rolling = run_rolling(data, pool)
        print(f"正收益={rolling['positive']}/3 {'PASS' if rolling['passed'] else 'FAIL'}")
        for wname, wr in rolling["windows"].items():
            if "error" not in wr:
                print(
                    f"    {wname}: 年化={wr['ann_return']:+.1%} 夏普={wr['sharpe']:.2f} 回撤={wr['max_drawdown']:.1%}"
                )
            else:
                print(f"    {wname}: {wr.get('error', 'ERR')}")

        # 3. 全回测
        print(f"  [3/3] Full...", end=" ", flush=True)
        full = run_full(data, pool)
        if "error" not in full:
            print(
                f"年化={full['ann_return']:+.1%} 夏普={full['sharpe']:.2f} "
                f"回撤={full['max_drawdown']:.1%} Calmar={full['calmar']:.2f} "
                f"({full['span_years']}yr)"
            )
        else:
            print(f"ERROR: {full.get('error', 'unknown')}")

        all_results[pool_name] = {
            "pool": {k: v for k, v in pool.items()},
            "is_oos": is_oos,
            "rolling": rolling,
            "full": full,
        }

    # === 汇总对比 ===
    print(f"\n\n{'=' * 70}")
    print("  汇总对比")
    print(f"{'=' * 70}")
    print(
        f"  {'池':<32} {'Full年化':>8} {'夏普':>6} {'回撤':>8} {'Calmar':>7} "
        f"{'OOS年化':>8} {'OOS夏普':>7} {'IS/OOS':>7} {'Roll':>5}"
    )
    print(f"  {'-' * 95}")

    for pool_name, res in all_results.items():
        full = res["full"]
        oos = res["is_oos"]["OOS"]
        if "error" in full:
            print(f"  {pool_name:<32} ERROR")
            continue
        oos_ann = f"{oos['ann_return']:+.1%}" if "error" not in oos else "ERR"
        oos_shp = f"{oos['sharpe']:.2f}" if "error" not in oos else "ERR"
        print(
            f"  {pool_name:<32} {full['ann_return']:>+7.1%} {full['sharpe']:>6.2f} "
            f"{full['max_drawdown']:>7.1%} {full['calmar']:>7.2f} "
            f"{oos_ann:>8} {oos_shp:>7} "
            f"{'PASS' if res['is_oos']['passed'] else 'FAIL':>7} "
            f"{'PASS' if res['rolling']['passed'] else 'FAIL':>5}"
        )

    # 年度对比
    print(f"\n  年度收益对比:")
    header = f"  {'年份':<6}"
    for pool_name, _ in POOLS:
        short = pool_name.split("_")[0]
        header += f" {short:>10}"
    print(header)
    print(f"  {'-' * 50}")

    all_years = set()
    for res in all_results.values():
        if "error" not in res["full"]:
            all_years.update(res["full"].get("yearly", {}).keys())

    for year in sorted(all_years):
        row = f"  {year:<6}"
        for pool_name, _ in POOLS:
            res = all_results[pool_name]
            if "error" not in res["full"]:
                yr = res["full"]["yearly"].get(year, {}).get("return")
                row += f" {yr:>+9.1%}" if yr is not None else f" {'—':>10}"
            else:
                row += f" {'ERR':>10}"
        print(row)

    # === 保存 ===
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = OUTPUT_DIR / "extended_pool_results.json"
    with open(out_path, "w") as f:
        json.dump(clean(all_results), f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
