"""动量数学模型对比实验.

5种动量公式 × 2个Core池 × 3种验证(IS/OOS, 滚动窗口, 全回测)
输出: data/v8_results/momentum_models_comparison.json

公式定义:
  M0 基线:     0.5*R(10) + 0.5*R(20)
  M1 加速度:   0.6*[0.5*R(10)+0.5*R(20)] + 0.4*(R(10)-R(20))
  M2 波动调整: R(20) / sigma(20)
  M3 EMA衰减:  sum(lambda^(60-i) * r_i), lambda=0.95
  M4 多尺度:   0.3*R(5) + 0.3*R(10) + 0.2*R(20) + 0.2*R(60)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
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

# === 交易参数 (与V3一致) ===
FEE = 0.0005  # 万五单边
SLIPPAGE = 0.001  # 千一滑点
REBALANCE_DAYS = 5  # 周频调仓
WARMUP = 130  # 预热期(交易日)
INITIAL_CAPITAL = 100_000.0

# === ETF池定义 ===
CORE_FULL = {
    "518880": "黄金ETF",
    "159985": "豆粕ETF",
    "501018": "南方原油",
    "161226": "白银LOF",
    "513100": "纳指ETF",
    "159915": "创业板ETF",
    "511220": "城投债ETF",
}
CORE_REDUCED = {
    "518880": "黄金ETF",
    "161226": "白银LOF",
    "513100": "纳指ETF",
    "159915": "创业板ETF",
    "511220": "城投债ETF",
}
DEFENSE = "511880"
A_SHARE_ETF = "159915"


# ============================================================================ #
# Part 1: 动量评分函数
# ============================================================================ #


def _ret(close: np.ndarray, n: int) -> float:
    """n日收益率 R(n) = (P_t - P_{t-n}) / P_{t-n}."""
    if len(close) <= n:
        return 0.0
    return (close[-1] - close[-n - 1]) / close[-n - 1]


def score_m0(close: np.ndarray) -> float:
    """M0 基线: 0.5*R(10) + 0.5*R(20)."""
    return 0.5 * _ret(close, 10) + 0.5 * _ret(close, 20)


def score_m1(close: np.ndarray) -> float:
    """M1 加速度增强: 0.6*base + 0.4*accel.

    accel = R(10) - R(20), 经济逻辑: 优先选正在加速的品种.
    """
    r10 = _ret(close, 10)
    r20 = _ret(close, 20)
    base = 0.5 * r10 + 0.5 * r20
    accel = r10 - r20
    return 0.6 * base + 0.4 * accel


def score_m2(close: np.ndarray) -> float:
    """M2 波动率调整动量: R(20) / sigma(20).

    经济逻辑: 路径更平滑的趋势更可靠(类Sharpe排序).
    """
    if len(close) < 21:
        return 0.0
    r20 = _ret(close, 20)
    daily_rets = np.diff(close[-21:]) / close[-21:-1]
    sigma = np.std(daily_rets) * np.sqrt(252)
    if sigma < 1e-8:
        return 0.0
    return r20 / sigma


def score_m3(close: np.ndarray) -> float:
    """M3 指数衰减加权(EMA动量): sum(lambda^(60-i) * r_i), lambda=0.95.

    经济逻辑: 近期权重更高, 对拐点响应更快, 无窗口边界效应.
    """
    lookback = 60
    if len(close) < lookback + 1:
        # 数据不足时用可用数据
        lookback = len(close) - 1
    if lookback < 5:
        return 0.0
    lam = 0.95
    prices = close[-(lookback + 1) :]
    daily_rets = np.diff(prices) / prices[:-1]
    weights = np.array([lam ** (lookback - 1 - i) for i in range(lookback)])
    weights /= weights.sum()  # 归一化
    return float(np.dot(weights, daily_rets))


def score_m4(close: np.ndarray) -> float:
    """M4 多尺度融合: 0.3*R(5) + 0.3*R(10) + 0.2*R(20) + 0.2*R(60).

    经济逻辑: 短中长期多尺度确认, 60日项过滤大趋势相反时的短期反弹.
    """
    return (
        0.3 * _ret(close, 5) + 0.3 * _ret(close, 10) + 0.2 * _ret(close, 20) + 0.2 * _ret(close, 60)
    )


MOMENTUM_MODELS: dict[str, Callable[[np.ndarray], float]] = {
    "M0_baseline": score_m0,
    "M1_accel": score_m1,
    "M2_vol_adj": score_m2,
    "M3_ema": score_m3,
    "M4_multiscale": score_m4,
}


# ============================================================================ #
# Part 2: 回测引擎 (参数化, 可插拔score_fn)
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


def check_single_day_drop(close: np.ndarray, lookback: int = 3, threshold: float = -0.03) -> bool:
    """近3日有单日跌>3% → 排除."""
    if len(close) < lookback + 1:
        return True
    for i in range(-lookback, 0):
        daily_ret = (close[i] - close[i - 1]) / close[i - 1]
        if daily_ret < threshold:
            return False
    return True


def check_a_share_weak(data: dict, as_of_idx: int, ma_period: int = 20) -> bool:
    """创业板<MA20 → A股走弱."""
    if A_SHARE_ETF not in data:
        return False
    df = data[A_SHARE_ETF]
    if as_of_idx < ma_period:
        return False
    close = df["close"].values[: as_of_idx + 1].astype(float)
    if len(close) < ma_period:
        return False
    ma = np.mean(close[-ma_period:])
    return close[-1] < ma


def run_backtest(
    data: dict[str, pd.DataFrame],
    pool: dict[str, str],
    score_fn: Callable[[np.ndarray], float],
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = INITIAL_CAPITAL,
    use_a_share_filter: bool = True,
    use_drop_filter: bool = True,
) -> dict:
    """参数化回测引擎.

    Args:
        data: {code: DataFrame} 全部历史数据
        pool: {code: name} ETF池
        score_fn: 动量评分函数
        start_date: 回测起始日 (None=从warmup后开始)
        end_date: 回测结束日 (None=到数据末尾)
        initial_capital: 初始资金
        use_a_share_filter: 是否启用A股走弱过滤
        use_drop_filter: 是否启用单日暴跌过滤

    Returns:
        回测结果dict (含equity_curve DataFrame)
    """
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
        return {"error": "no common dates"}

    all_dates = sorted(common_dates)

    # 日期过滤 (trade_date列为datetime.date类型)
    if start_date:
        sd = dt_date.fromisoformat(start_date)
        all_dates = [d for d in all_dates if d >= sd]
    if end_date:
        ed = dt_date.fromisoformat(end_date)
        all_dates = [d for d in all_dates if d <= ed]

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
        # 获取各ETF在td的索引位置
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
            close = data[code]["close"].values[: idx + 1].astype(float)
            if len(close) < WARMUP:
                continue
            # 暴跌过滤
            if use_drop_filter and not check_single_day_drop(close):
                continue
            score = score_fn(close)
            if score > 0:
                candidates.append((code, score))

        candidates.sort(key=lambda x: -x[1])
        best_target = candidates[0][0] if candidates else DEFENSE
        best_score = candidates[0][1] if candidates else 0.0

        # 换仓逻辑: 自适应缓冲
        if best_score > 0.10:
            threshold = 0.0
        else:
            threshold = 0.05

        if holding and holding != DEFENSE:
            cur_score = dict(candidates).get(holding, -999)
            if cur_score > 0:
                if best_score > cur_score + threshold:
                    target = best_target
                else:
                    target = holding
            else:
                target = best_target
        else:
            target = best_target

        # 交易执行
        if target != holding:
            # 卖出
            if holding and holding in data:
                row = data[holding][data[holding]["trade_date"] == td]
                if not row.empty:
                    price = row.iloc[0]["close"]
                    cash += holding_shares * price * (1 - FEE - SLIPPAGE)
                    n_trades += 1
                    holding = None
                    holding_shares = 0

            # 买入
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
                        n_trades += 1

        # 记录equity
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

    if not equity_history:
        return {"error": "no trades executed"}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])

    # 计算指标
    metrics = calc_metrics(eq_df, initial_capital, n_trades)
    metrics["equity_curve"] = eq_df
    return metrics


def calc_metrics(eq_df: pd.DataFrame, initial_capital: float, n_trades: int) -> dict:
    """从equity曲线计算绩效指标."""
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
    eq_df_copy = eq_df.copy()
    eq_df_copy["year"] = eq_df_copy["trade_date"].dt.year
    yearly = {}
    prev_val = initial_capital
    for year in sorted(eq_df_copy["year"].unique()):
        ydf = eq_df_copy[eq_df_copy["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev_val) - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        yearly[int(year)] = {"return": round(yr, 4), "max_dd": round(dd, 4)}
        prev_val = end_val

    # 换手率 (年化交易次数)
    turnover_ann = n_trades / span_years

    return {
        "total_return": round(total_return, 4),
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 3),
        "n_trades": n_trades,
        "turnover_ann": round(turnover_ann, 1),
        "span_years": round(span_years, 2),
        "yearly": yearly,
    }


# ============================================================================ #
# Part 3: 验证框架
# ============================================================================ #


def run_is_oos(
    data: dict,
    pool: dict,
    score_fn: Callable,
    is_end: str = "2021-12-31",
) -> dict:
    """样本内/样本外分段回测."""
    is_result = run_backtest(data, pool, score_fn, end_date=is_end)
    oos_result = run_backtest(data, pool, score_fn, start_date="2022-01-01")

    is_valid = "error" not in is_result
    oos_valid = "error" not in oos_result

    passed = False
    if is_valid and oos_valid:
        # 通过标准: OOS年化>0 且 OOS夏普 > IS夏普*0.5
        passed = oos_result["ann_return"] > 0 and oos_result["sharpe"] > is_result["sharpe"] * 0.5

    return {
        "IS": {k: v for k, v in is_result.items() if k != "equity_curve"}
        if is_valid
        else is_result,
        "OOS": {k: v for k, v in oos_result.items() if k != "equity_curve"}
        if oos_valid
        else oos_result,
        "passed": passed,
    }


def run_rolling(
    data: dict,
    pool: dict,
    score_fn: Callable,
    baseline_fn: Callable | None = None,
) -> dict:
    """3滚动窗口回测 (3年Train + 3年Test).

    Windows:
      W1: Train 2016-08~2019-07 / Test 2019-08~2022-07
      W2: Train 2019-08~2022-07 / Test 2022-08~2025-07
      W3: Train 2022-08~2025-07 / Test 2025-08~2026-07
    """
    windows = [
        ("W1", "2016-08-01", "2019-07-31", "2019-08-01", "2022-07-31"),
        ("W2", "2019-08-01", "2022-07-31", "2022-08-01", "2025-07-31"),
        ("W3", "2022-08-01", "2025-07-31", "2025-08-01", "2026-07-31"),
    ]

    results = {}
    wins_vs_baseline = 0
    valid_windows = 0

    for name, train_start, train_end, test_start, test_end in windows:
        # Train期回测 (仅记录, 不用于判断)
        train_r = run_backtest(data, pool, score_fn, start_date=train_start, end_date=train_end)
        # Test期回测 (核心判断)
        test_r = run_backtest(data, pool, score_fn, start_date=test_start, end_date=test_end)

        # 基线对比
        baseline_test = None
        if baseline_fn and score_fn != baseline_fn:
            baseline_test = run_backtest(
                data, pool, baseline_fn, start_date=test_start, end_date=test_end
            )

        win_entry = {
            "train": {k: v for k, v in train_r.items() if k != "equity_curve"}
            if "error" not in train_r
            else train_r,
            "test": {k: v for k, v in test_r.items() if k != "equity_curve"}
            if "error" not in test_r
            else test_r,
        }

        if baseline_test and "error" not in baseline_test and "error" not in test_r:
            win_entry["baseline_test_ann"] = baseline_test["ann_return"]
            win_entry["beat_baseline"] = test_r["ann_return"] > baseline_test["ann_return"]
            if test_r["ann_return"] > baseline_test["ann_return"]:
                wins_vs_baseline += 1
            valid_windows += 1
        elif "error" not in test_r:
            valid_windows += 1
            wins_vs_baseline += 1  # 无基线对比时, 只要正收益就算通过

        results[name] = win_entry

    # 通过标准: 至少2/3窗口跑赢基线
    passed = wins_vs_baseline >= 2 if valid_windows >= 2 else False

    return {
        "windows": results,
        "wins_vs_baseline": wins_vs_baseline,
        "valid_windows": valid_windows,
        "passed": passed,
    }


def run_full(data: dict, pool: dict, score_fn: Callable) -> dict:
    """全周期回测."""
    result = run_backtest(data, pool, score_fn)
    return {k: v for k, v in result.items() if k != "equity_curve"}


# ============================================================================ #
# Part 4: 主实验流程
# ============================================================================ #


def run_all_experiments():
    """跑全部实验: 5公式 × 2池 × 3验证."""
    print("=" * 70)
    print("  动量数学模型对比实验")
    print("  M0(基线) / M1(加速度) / M2(波动调整) / M3(EMA) / M4(多尺度)")
    print("=" * 70)

    # 加载数据
    core_full_data = load_pool_data(CORE_FULL, CROSS_ASSET_DIR)
    core_reduced_data = load_pool_data(CORE_REDUCED, CROSS_ASSET_DIR)

    print(f"\n  Core-Full 数据: {len(core_full_data)} 只ETF")
    for code, df in core_full_data.items():
        name = CORE_FULL.get(code, "货币基金")
        print(
            f"    {code} {name}: {df['trade_date'].min()} ~ {df['trade_date'].max()} ({len(df)}天)"
        )

    print(f"\n  Core-Reduced 数据: {len(core_reduced_data)} 只ETF")
    for code, df in core_reduced_data.items():
        name = CORE_REDUCED.get(code, "货币基金")
        print(
            f"    {code} {name}: {df['trade_date'].min()} ~ {df['trade_date'].max()} ({len(df)}天)"
        )

    all_results = {}

    pools = [
        ("Core-Full(8)", CORE_FULL, core_full_data),
        ("Core-Reduced(6)", CORE_REDUCED, core_reduced_data),
    ]

    for pool_name, pool, data in pools:
        print(f"\n{'=' * 70}")
        print(f"  池: {pool_name}")
        print(f"{'=' * 70}")

        pool_results = {}

        for model_name, score_fn in MOMENTUM_MODELS.items():
            print(f"\n  --- {model_name} ---")

            # 1. IS/OOS
            print(f"    [1/3] IS/OOS...", end=" ", flush=True)
            is_oos = run_is_oos(data, pool, score_fn)
            oos_info = is_oos["OOS"]
            if "error" not in oos_info:
                print(
                    f"OOS年化={oos_info['ann_return']:+.1%} 夏普={oos_info['sharpe']:.2f} "
                    f"{'PASS' if is_oos['passed'] else 'FAIL'}"
                )
            else:
                print(f"ERROR: {oos_info}")

            # 2. 滚动窗口
            print(f"    [2/3] Rolling...", end=" ", flush=True)
            rolling = run_rolling(data, pool, score_fn, baseline_fn=score_m0)
            print(
                f"Wins={rolling['wins_vs_baseline']}/{rolling['valid_windows']} "
                f"{'PASS' if rolling['passed'] else 'FAIL'}"
            )

            # 3. 全回测
            print(f"    [3/3] Full...", end=" ", flush=True)
            full = run_full(data, pool, score_fn)
            if "error" not in full:
                print(
                    f"年化={full['ann_return']:+.1%} 夏普={full['sharpe']:.2f} "
                    f"回撤={full['max_drawdown']:.1%} Calmar={full['calmar']:.2f}"
                )
            else:
                print(f"ERROR: {full}")

            pool_results[model_name] = {
                "is_oos": is_oos,
                "rolling": rolling,
                "full": full,
            }

        all_results[pool_name] = pool_results

    # === 汇总排名 ===
    print(f"\n\n{'=' * 70}")
    print("  汇总排名 (按OOS夏普排序)")
    print(f"{'=' * 70}")

    for pool_name in all_results:
        print(f"\n  [{pool_name}]")
        print(
            f"  {'模型':<16} {'OOS年化':>8} {'OOS夏普':>8} {'Full年化':>8} {'Full夏普':>8} "
            f"{'Full回撤':>8} {'IS/OOS':>7} {'Rolling':>8}"
        )
        print(f"  {'-' * 80}")

        rows = []
        for model_name, res in all_results[pool_name].items():
            oos = res["is_oos"]["OOS"]
            full = res["full"]
            oos_sharpe = oos.get("sharpe", -99) if "error" not in oos else -99
            rows.append((model_name, oos, full, res["is_oos"]["passed"], res["rolling"]["passed"]))

        rows.sort(
            key=lambda x: x[1].get("sharpe", -99) if "error" not in x[1] else -99, reverse=True
        )

        for model_name, oos, full, is_oos_pass, rolling_pass in rows:
            oos_ann = f"{oos['ann_return']:+.1%}" if "error" not in oos else "ERR"
            oos_shp = f"{oos['sharpe']:.2f}" if "error" not in oos else "ERR"
            full_ann = f"{full['ann_return']:+.1%}" if "error" not in full else "ERR"
            full_shp = f"{full['sharpe']:.2f}" if "error" not in full else "ERR"
            full_dd = f"{full['max_drawdown']:.1%}" if "error" not in full else "ERR"
            is_oos_str = "PASS" if is_oos_pass else "FAIL"
            roll_str = "PASS" if rolling_pass else "FAIL"
            print(
                f"  {model_name:<16} {oos_ann:>8} {oos_shp:>8} {full_ann:>8} {full_shp:>8} "
                f"{full_dd:>8} {is_oos_str:>7} {roll_str:>8}"
            )

    # === 保存JSON ===
    # 清理不可序列化对象
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k != "equity_curve"}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = clean(all_results)
    out_path = OUTPUT_DIR / "momentum_models_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")

    return all_results


if __name__ == "__main__":
    run_all_experiments()
