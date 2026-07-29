"""核心+卫星组合策略实验.

Core (70%): V3跨资产动量轮动 (M0公式, Top1集中)
Satellite (30%): 行业ETF动量轮动 (M0公式, Top1集中)

验证: IS/OOS + 3滚动窗口 + 10年全回测
对比: 纯Core vs Core+Satellite, 看卫星仓增量贡献
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
CROSS_ASSET_DIR = PROJECT_ROOT / "data" / "cross_asset"
SECTOR_DIR = PROJECT_ROOT / "data" / "sector_etf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "v8_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 交易参数 ===
FEE = 0.0005
SLIPPAGE = 0.001
REBALANCE_DAYS = 5
WARMUP = 130
INITIAL_CAPITAL = 100_000.0

# === 资金分配 ===
W_CORE = 0.70
W_SAT = 0.30

# === ETF池 ===
# Core: 跨资产 (使用Reduced池以覆盖10年)
CORE_POOL = {
    "518880": "黄金ETF",
    "161226": "白银LOF",
    "513100": "纳指ETF",
    "159915": "创业板ETF",
    "511220": "城投债ETF",
}
# Satellite: 行业ETF (排除159915避免与Core重叠)
SAT_POOL = {
    "510150": "消费ETF",
    "510300": "沪深300ETF",
    "512010": "医药ETF",
    "512660": "军工ETF",
    "512880": "证券ETF",
    "512800": "银行ETF",
}
DEFENSE = "511880"
A_SHARE_ETF = "159915"


# ============================================================================ #
# 动量评分 (M0基线, Step3验证胜出)
# ============================================================================ #

def score_m0(close: np.ndarray) -> float:
    """M0 基线: 0.5*R(10) + 0.5*R(20)."""
    if len(close) <= 20:
        return 0.0
    r10 = (close[-1] - close[-11]) / close[-11]
    r20 = (close[-1] - close[-21]) / close[-21]
    return 0.5 * r10 + 0.5 * r20


# ============================================================================ #
# 回测引擎 (单策略)
# ============================================================================ #

def load_pool_data(pool: dict[str, str], data_dir: Path) -> dict[str, pd.DataFrame]:
    """加载指定池的ETF数据."""
    data = {}
    for code in list(pool.keys()) + [DEFENSE]:
        f = data_dir / f"{code}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            data[code] = df.sort_values("trade_date").reset_index(drop=True)
    return data


def check_single_day_drop(close: np.ndarray) -> bool:
    """近3日有单日跌>3% → 排除."""
    if len(close) < 4:
        return True
    for i in range(-3, 0):
        daily_ret = (close[i] - close[i - 1]) / close[i - 1]
        if daily_ret < -0.03:
            return False
    return True


def check_a_share_weak(data: dict, as_of_idx: int) -> bool:
    """创业板<MA20 → A股走弱."""
    if A_SHARE_ETF not in data:
        return False
    df = data[A_SHARE_ETF]
    if as_of_idx < 20:
        return False
    close = df["close"].values[:as_of_idx + 1].astype(float)
    ma = np.mean(close[-20:])
    return close[-1] < ma


def run_single_backtest(
    data: dict[str, pd.DataFrame],
    pool: dict[str, str],
    score_fn: Callable[[np.ndarray], float],
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = INITIAL_CAPITAL,
    use_a_share_filter: bool = True,
    use_drop_filter: bool = True,
) -> pd.DataFrame:
    """单策略回测, 返回equity曲线DataFrame."""
    # 找公共日期
    common_dates = None
    for code in pool:
        if code not in data:
            continue
        dates = set(data[code]["trade_date"].tolist())
        if common_dates is None:
            common_dates = dates
        else:
            common_dates &= dates
    if DEFENSE in data:
        common_dates &= set(data[DEFENSE]["trade_date"].tolist())

    if not common_dates:
        return pd.DataFrame(columns=["trade_date", "equity", "holding"])

    all_dates = sorted(common_dates)

    # 日期过滤
    if start_date:
        sd = dt_date.fromisoformat(start_date)
        all_dates = [d for d in all_dates if d >= sd]
    if end_date:
        ed = dt_date.fromisoformat(end_date)
        all_dates = [d for d in all_dates if d <= ed]

    if len(all_dates) < WARMUP + REBALANCE_DAYS:
        return pd.DataFrame(columns=["trade_date", "equity", "holding"])

    trading_dates = all_dates[WARMUP:]
    rebalance_dates = trading_dates[::REBALANCE_DAYS]

    cash = initial_capital
    holding: str | None = None
    holding_shares: int = 0
    equity_history = []

    for td in rebalance_dates:
        # 获取各ETF在td的索引
        etf_data_at_date = {}
        for code in list(pool.keys()) + [DEFENSE]:
            if code not in data:
                continue
            df = data[code]
            mask = df["trade_date"] <= td
            if mask.sum() < WARMUP:
                continue
            etf_data_at_date[code] = mask.sum() - 1

        # A股走弱判断 (仅Core使用)
        a_share_weak = False
        if use_a_share_filter and A_SHARE_ETF in etf_data_at_date:
            a_share_weak = check_a_share_weak(data, etf_data_at_date[A_SHARE_ETF])

        # 动量评分
        candidates = []
        for code in pool:
            if code not in etf_data_at_date:
                continue
            if code == A_SHARE_ETF and a_share_weak:
                continue
            idx = etf_data_at_date[code]
            close = data[code]["close"].values[:idx + 1].astype(float)
            if len(close) < WARMUP:
                continue
            if use_drop_filter and not check_single_day_drop(close):
                continue
            score = score_fn(close)
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
                    price = row.iloc[0]["close"]
                    cash += holding_shares * price * (1 - FEE - SLIPPAGE)
                    holding = None
                    holding_shares = 0

            if target in data:
                row = data[target][data[target]["trade_date"] == td]
                if not row.empty:
                    price = row.iloc[0]["close"]
                    shares = int(cash * 0.99 / price / 100) * 100
                    if shares > 0:
                        cost = shares * price * (1 + FEE + SLIPPAGE)
                        cash -= cost
                        holding = target
                        holding_shares = shares

        # 记录equity
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

    if not equity_history:
        return pd.DataFrame(columns=["trade_date", "equity", "holding"])

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    return eq_df


# ============================================================================ #
# 组合回测
# ============================================================================ #

def run_combined_backtest(
    core_data: dict, sat_data: dict,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Core+Satellite组合回测.

    分别跑Core和Satellite的净值曲线, 按70/30加权合并.
    使用reindex对齐日期(前向填充), 避免因调仓日错位导致join失败.
    """
    core_capital = initial_capital * W_CORE
    sat_capital = initial_capital * W_SAT

    # Core: 跨资产轮动 (启用A股走弱过滤)
    core_eq = run_single_backtest(
        core_data, CORE_POOL, score_m0,
        start_date=start_date, end_date=end_date,
        initial_capital=core_capital,
        use_a_share_filter=True, use_drop_filter=True,
    )

    # Satellite: 行业ETF轮动 (不启用A股走弱过滤, 因为全是A股行业)
    sat_eq = run_single_backtest(
        sat_data, SAT_POOL, score_m0,
        start_date=start_date, end_date=end_date,
        initial_capital=sat_capital,
        use_a_share_filter=False, use_drop_filter=True,
    )

    if core_eq.empty or sat_eq.empty:
        return pd.DataFrame(columns=["trade_date", "equity", "holding"])

    # 用Core日期为基准, 将Satellite reindex对齐 (前向填充)
    core_eq = core_eq.set_index("trade_date")["equity"].rename("core_equity")
    sat_eq_s = sat_eq.set_index("trade_date")["equity"].rename("sat_equity")

    # 取Core日期范围内, Satellite有数据的交集
    # 先找Satellite的有效起始日 (第一个日期)
    sat_start = sat_eq_s.index.min()
    sat_end = sat_eq_s.index.max()

    # 筛选Core日期在Satellite有效范围内的
    valid_mask = (core_eq.index >= sat_start) & (core_eq.index <= sat_end)
    core_valid = core_eq[valid_mask]

    if core_valid.empty:
        return pd.DataFrame(columns=["trade_date", "equity", "holding"])

    # 将Satellite reindex到Core日期, 前向填充
    sat_aligned = sat_eq_s.reindex(core_valid.index, method="ffill")

    # 去掉Satellite还没有数据的行 (ffill产生NaN)
    valid = sat_aligned.notna()
    core_final = core_valid[valid]
    sat_final = sat_aligned[valid]

    if core_final.empty:
        return pd.DataFrame(columns=["trade_date", "equity", "holding"])

    combined = pd.DataFrame({
        "trade_date": core_final.index,
        "equity": core_final.values + sat_final.values,
        "holding": "combined",
    })
    return combined


# ============================================================================ #
# 指标计算
# ============================================================================ #

def calc_metrics(eq_df: pd.DataFrame, initial_capital: float) -> dict:
    """从equity曲线计算绩效指标."""
    if eq_df.empty or len(eq_df) < 2:
        return {"error": "insufficient data"}

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
        "span_years": round(span_years, 2),
        "yearly": yearly,
    }


# ============================================================================ #
# 验证框架
# ============================================================================ #

def validate_strategy(name: str, run_fn: Callable, initial_capital: float = INITIAL_CAPITAL):
    """对单个策略/组合跑3种验证."""
    print(f"\n  === {name} ===")

    # 1. IS/OOS
    print(f"    [1/3] IS/OOS...", end=" ", flush=True)
    is_eq = run_fn(start_date=None, end_date="2021-12-31")
    oos_eq = run_fn(start_date="2022-01-01", end_date=None)
    is_m = calc_metrics(is_eq, initial_capital)
    oos_m = calc_metrics(oos_eq, initial_capital)
    is_oos_pass = (
        "error" not in oos_m
        and oos_m["ann_return"] > 0
        and "error" not in is_m
        and oos_m["sharpe"] > is_m["sharpe"] * 0.5
    )
    if "error" not in oos_m:
        print(f"IS夏普={is_m.get('sharpe', 'N/A'):.2f} OOS年化={oos_m['ann_return']:+.1%} "
              f"OOS夏普={oos_m['sharpe']:.2f} {'PASS' if is_oos_pass else 'FAIL'}")
    else:
        print(f"ERROR: {oos_m}")

    # 2. 滚动窗口
    print(f"    [2/3] Rolling...", end=" ", flush=True)
    windows = [
        ("W1", "2019-08-01", "2022-07-31"),
        ("W2", "2022-08-01", "2025-07-31"),
        ("W3", "2025-08-01", "2026-07-31"),
    ]
    rolling_results = {}
    for wname, ws, we in windows:
        w_eq = run_fn(start_date=ws, end_date=we)
        w_m = calc_metrics(w_eq, initial_capital)
        rolling_results[wname] = w_m
    # 通过标准: 至少2/3窗口正收益
    positive_windows = sum(1 for r in rolling_results.values() if "error" not in r and r["ann_return"] > 0)
    rolling_pass = positive_windows >= 2
    print(f"正收益窗口={positive_windows}/3 {'PASS' if rolling_pass else 'FAIL'}")
    for wname, wm in rolling_results.items():
        if "error" not in wm:
            print(f"      {wname}: 年化={wm['ann_return']:+.1%} 夏普={wm['sharpe']:.2f} 回撤={wm['max_drawdown']:.1%}")

    # 3. 全回测
    print(f"    [3/3] Full...", end=" ", flush=True)
    full_eq = run_fn(start_date=None, end_date=None)
    full_m = calc_metrics(full_eq, initial_capital)
    if "error" not in full_m:
        print(f"年化={full_m['ann_return']:+.1%} 夏普={full_m['sharpe']:.2f} "
              f"回撤={full_m['max_drawdown']:.1%} Calmar={full_m['calmar']:.2f}")
    else:
        print(f"ERROR: {full_m}")

    return {
        "IS": is_m,
        "OOS": oos_m,
        "is_oos_pass": is_oos_pass,
        "rolling": rolling_results,
        "rolling_pass": rolling_pass,
        "full": full_m,
    }


# ============================================================================ #
# 主实验
# ============================================================================ #

def main():
    print("=" * 70)
    print("  核心+卫星组合策略实验")
    print(f"  Core ({W_CORE:.0%}): 跨资产动量轮动 [黄金/白银/纳指/创业板/城投债]")
    print(f"  Satellite ({W_SAT:.0%}): 行业ETF动量轮动 [消费/沪深300/医药/军工/证券/银行]")
    print("=" * 70)

    # 加载数据
    core_data = load_pool_data(CORE_POOL, CROSS_ASSET_DIR)
    sat_data = load_pool_data(SAT_POOL, SECTOR_DIR)

    print(f"\n  Core数据: {len(core_data)} 只ETF")
    for code, df in core_data.items():
        name = CORE_POOL.get(code, "货币基金")
        print(f"    {code} {name}: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

    print(f"\n  Satellite数据: {len(sat_data)} 只ETF")
    for code, df in sat_data.items():
        name = SAT_POOL.get(code, "货币基金")
        print(f"    {code} {name}: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

    # === 实验1: 纯Core ===
    core_results = validate_strategy(
        "纯Core (跨资产轮动)",
        lambda start_date, end_date: run_single_backtest(
            core_data, CORE_POOL, score_m0,
            start_date=start_date, end_date=end_date,
            initial_capital=INITIAL_CAPITAL,
            use_a_share_filter=True,
        ),
    )

    # === 实验2: 纯Satellite ===
    sat_results = validate_strategy(
        "纯Satellite (行业ETF轮动)",
        lambda start_date, end_date: run_single_backtest(
            sat_data, SAT_POOL, score_m0,
            start_date=start_date, end_date=end_date,
            initial_capital=INITIAL_CAPITAL,
            use_a_share_filter=False,
        ),
    )

    # === 实验3: Core + Satellite 组合 ===
    combined_results = validate_strategy(
        f"Core({W_CORE:.0%}) + Satellite({W_SAT:.0%}) 组合",
        lambda start_date, end_date: run_combined_backtest(
            core_data, sat_data,
            start_date=start_date, end_date=end_date,
            initial_capital=INITIAL_CAPITAL,
        ),
    )

    # === 对比汇总 ===
    print(f"\n\n{'=' * 70}")
    print("  对比汇总 (全周期)")
    print(f"{'=' * 70}")
    print(f"  {'策略':<28} {'年化':>8} {'夏普':>6} {'回撤':>8} {'Calmar':>8}")
    print(f"  {'-' * 60}")

    for label, res in [
        ("纯Core", core_results),
        ("纯Satellite", sat_results),
        (f"Core+Sat ({W_CORE:.0%}/{W_SAT:.0%})", combined_results),
    ]:
        full = res["full"]
        if "error" not in full:
            print(f"  {label:<28} {full['ann_return']:>+7.1%} {full['sharpe']:>6.2f} "
                  f"{full['max_drawdown']:>7.1%} {full['calmar']:>8.2f}")
        else:
            print(f"  {label:<28} ERROR")

    # 年度对比
    print(f"\n  年度收益对比:")
    print(f"  {'年份':<6} {'纯Core':>10} {'纯Sat':>10} {'组合':>10}")
    print(f"  {'-' * 40}")

    core_yearly = core_results["full"].get("yearly", {})
    sat_yearly = sat_results["full"].get("yearly", {})
    comb_yearly = combined_results["full"].get("yearly", {})
    all_years = sorted(set(list(core_yearly.keys()) + list(sat_yearly.keys()) + list(comb_yearly.keys())))

    for year in all_years:
        c = core_yearly.get(year, {}).get("return", None)
        s = sat_yearly.get(year, {}).get("return", None)
        m = comb_yearly.get(year, {}).get("return", None)
        c_str = f"{c:+.1%}" if c is not None else "—"
        s_str = f"{s:+.1%}" if s is not None else "—"
        m_str = f"{m:+.1%}" if m is not None else "—"
        print(f"  {year:<6} {c_str:>10} {s_str:>10} {m_str:>10}")

    # === 增量贡献分析 ===
    print(f"\n  卫星仓增量贡献:")
    if "error" not in core_results["full"] and "error" not in combined_results["full"]:
        delta_ret = combined_results["full"]["ann_return"] - core_results["full"]["ann_return"]
        delta_sharpe = combined_results["full"]["sharpe"] - core_results["full"]["sharpe"]
        delta_dd = combined_results["full"]["max_drawdown"] - core_results["full"]["max_drawdown"]
        print(f"    年化增量: {delta_ret:+.2%}")
        print(f"    夏普增量: {delta_sharpe:+.3f}")
        print(f"    回撤变化: {delta_dd:+.2%} (负=回撤扩大)")

    # === 保存 ===
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "config": {
            "W_CORE": W_CORE, "W_SAT": W_SAT,
            "core_pool": list(CORE_POOL.keys()),
            "sat_pool": list(SAT_POOL.keys()),
            "score_fn": "M0_baseline",
        },
        "core_only": clean(core_results),
        "satellite_only": clean(sat_results),
        "combined": clean(combined_results),
    }
    out_path = OUTPUT_DIR / "core_satellite_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
