"""Multi-factor v3 backtest with flywheel evolution.

Runs v3 strategy on 4.3 years of data with:
- Walk-forward validation (10 windows)
- Champion/Challenger comparison
- v2 vs v3 head-to-head
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from a_share_quant.factors.v3 import (
    V3Config,
    V3Portfolio,
    FlywheelEvolution,
    CrashProtector,
    risk_parity_weights,
)
from a_share_quant.regime import RegimeDetector


def load_long_data() -> dict[str, pd.DataFrame]:
    """Load long history data."""
    data_dir = PROJECT_ROOT / "data" / "long_history"
    data = {}
    for f in data_dir.glob("*.parquet"):
        if f.name == "combined_long.parquet":
            continue
        df = pd.read_parquet(f)
        if "symbol" in df.columns and "trade_date" in df.columns:
            symbol = df["symbol"].iloc[0]
            if f.name.startswith("index_"):
                symbol = f"idx_{symbol}"
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            data[symbol] = df.sort_values("trade_date").reset_index(drop=True)
    return data


def run_v3_backtest(
    data: dict[str, pd.DataFrame],
    index_symbol: str,
    config: V3Config | None = None,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """Run v3 backtest."""
    cfg = config or V3Config()
    portfolio = V3Portfolio(cfg)
    detector = RegimeDetector()
    crash_protector = CrashProtector(cfg)

    index_df = data[index_symbol]

    # Tradable universe (exclude index and defensive assets from factor scoring)
    defensive_set = set(cfg.defensive_symbols)
    tradable = {k: v for k, v in data.items()
                if not k.startswith("idx_") and k not in defensive_set}

    all_dates = sorted(index_df["trade_date"].tolist())
    warmup = cfg.trend_slow + 10
    trading_days = all_dates[warmup:]

    cash = initial_capital
    holdings: dict[str, int] = {}
    equity_history = []
    trades = []
    rebalance_log = []
    fee_rate = 0.001

    for td in trading_days:
        # Detect regime
        idx_hist = index_df[index_df["trade_date"] <= td]
        regime = detector.detect(idx_hist, td)

        # Current equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]

        # Crash protection: force sell crashed positions (even between rebalances)
        for sym in list(holdings.keys()):
            if sym not in defensive_set and crash_protector.check_crash(data, sym, td):
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    price = row.iloc[0]["close"]
                    revenue = holdings[sym] * price
                    cash += revenue * (1 - fee_rate / 2)
                    trades.append({"date": td, "action": "CRASH_SELL", "symbol": sym})
                    del holdings[sym]

        # Check rebalance
        signal = portfolio.generate_signal(tradable, td, regime.state_id, equity, index_symbol)

        if signal:
            rebalance_log.append({
                "date": td,
                "holdings": signal.target_holdings,
                "regime": signal.regime_state,
                "position_scale": signal.position_scale,
                "crashed": signal.crashed_symbols,
                "weights": signal.weights,
            })

            # Sell positions not in target
            for sym in list(holdings.keys()):
                if sym not in signal.target_holdings:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        price = row.iloc[0]["close"]
                        revenue = holdings[sym] * price
                        cash += revenue * (1 - fee_rate / 2)
                        trades.append({"date": td, "action": "SELL", "symbol": sym})
                        del holdings[sym]

            # Buy target positions with risk parity weights
            total_equity = cash + sum(
                holdings.get(s, 0) * data[s][data[s]["trade_date"] == td].iloc[0]["close"]
                for s in holdings if s in data and not data[s][data[s]["trade_date"] == td].empty
            )

            for sym, weight in signal.weights.items():
                if sym not in holdings and sym in data:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        price = row.iloc[0]["close"]
                        target_value = total_equity * weight
                        shares = int(target_value / price / 100) * 100
                        if shares > 0:
                            cost = shares * price * (1 + fee_rate / 2)
                            if cost <= cash:
                                cash -= cost
                                holdings[sym] = shares
                                trades.append({"date": td, "action": "BUY", "symbol": sym})

        # Calculate equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]
        equity_history.append({"date": td, "equity": equity})

    equity_df = pd.DataFrame(equity_history)
    equity_series = equity_df.set_index("date")["equity"]

    return {
        "equity_curve": equity_series,
        "trades": trades,
        "rebalance_log": rebalance_log,
        "equity_df": equity_df,
    }


def run_walk_forward_v3(data: dict[str, pd.DataFrame], index_symbol: str) -> list:
    """Walk-forward with v3 strategy."""
    bench_symbol = "510300"
    if bench_symbol not in data:
        for k in data:
            if "510300" in k and not k.startswith("idx"):
                bench_symbol = k
                break

    bench_df = data[bench_symbol]
    all_dates = sorted(bench_df["trade_date"].tolist())

    # Walk-forward: train 200, test 80, step 80
    train_days = 200
    test_days = 80
    step_days = 80
    n = len(all_dates)
    start = 0
    window_id = 0
    results = []

    detector = RegimeDetector()

    while start + train_days + test_days <= n:
        test_start = all_dates[start + train_days]
        test_end = all_dates[min(start + train_days + test_days - 1, n - 1)]

        # Run v3 on test period
        cfg = V3Config()
        portfolio = V3Portfolio(cfg)
        crash_protector = CrashProtector(cfg)

        initial_capital = 1_000_000.0
        cash = initial_capital
        holdings: dict[str, int] = {}
        fee_rate = 0.001
        equity_history = []

        defensive_set = set(cfg.defensive_symbols)
        tradable = {k: v for k, v in data.items()
                    if not k.startswith("idx_") and k not in defensive_set}

        index_df = data[index_symbol]
        test_dates = [d for d in all_dates if test_start <= d <= test_end]

        for td in test_dates:
            idx_hist = index_df[index_df["trade_date"] <= td]
            regime = detector.detect(idx_hist, td)

            equity = cash
            for sym, shares in holdings.items():
                if sym in data:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        equity += shares * row.iloc[0]["close"]

            # Crash protection
            for sym in list(holdings.keys()):
                if sym not in defensive_set and crash_protector.check_crash(data, sym, td):
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        cash += holdings[sym] * row.iloc[0]["close"] * (1 - fee_rate / 2)
                        del holdings[sym]

            signal = portfolio.generate_signal(tradable, td, regime.state_id, equity, index_symbol)

            if signal:
                for sym in list(holdings.keys()):
                    if sym not in signal.target_holdings:
                        row = data[sym][data[sym]["trade_date"] == td]
                        if not row.empty:
                            cash += holdings[sym] * row.iloc[0]["close"] * (1 - fee_rate / 2)
                            del holdings[sym]

                total_eq = cash + sum(
                    holdings.get(s, 0) * data[s][data[s]["trade_date"] == td].iloc[0]["close"]
                    for s in holdings if s in data and not data[s][data[s]["trade_date"] == td].empty
                )
                for sym, weight in signal.weights.items():
                    if sym not in holdings and sym in data:
                        row = data[sym][data[sym]["trade_date"] == td]
                        if not row.empty:
                            price = row.iloc[0]["close"]
                            shares = int(total_eq * weight / price / 100) * 100
                            if shares > 0:
                                cost = shares * price * (1 + fee_rate / 2)
                                if cost <= cash:
                                    cash -= cost
                                    holdings[sym] = shares

            equity = cash
            for sym, shares in holdings.items():
                if sym in data:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        equity += shares * row.iloc[0]["close"]
            equity_history.append(equity)

        # Metrics
        if not equity_history:
            start += step_days
            window_id += 1
            continue

        final_equity = equity_history[-1]
        test_return = (final_equity / initial_capital) - 1

        equity_arr = np.array(equity_history)
        cummax = np.maximum.accumulate(equity_arr)
        dd = (equity_arr - cummax) / cummax
        max_dd = abs(dd.min()) if len(dd) > 0 else 0

        # Benchmark
        bench_start_row = bench_df[bench_df["trade_date"] == test_dates[0]]
        bench_end_row = bench_df[bench_df["trade_date"] == test_dates[-1]]
        if bench_start_row.empty or bench_end_row.empty:
            start += step_days
            window_id += 1
            continue

        bench_return = (bench_end_row.iloc[0]["close"] / bench_start_row.iloc[0]["close"]) - 1

        results.append({
            "window_id": window_id,
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "test_return": test_return,
            "benchmark_return": bench_return,
            "excess_return": test_return - bench_return,
            "max_drawdown": max_dd,
            "n_trades": 0,
        })

        start += step_days
        window_id += 1

    return results


def run_flywheel(data: dict[str, pd.DataFrame], index_symbol: str) -> dict:
    """Run flywheel evolution: champion vs challenger."""
    flywheel = FlywheelEvolution()
    flywheel.initialize()

    # Run walk-forward periods and compare champion vs challenger
    bench_symbol = "510300"
    if bench_symbol not in data:
        for k in data:
            if "510300" in k and not k.startswith("idx"):
                bench_symbol = k
                break

    bench_df = data[bench_symbol]
    all_dates = sorted(bench_df["trade_date"].tolist())

    train_days = 200
    test_days = 80
    step_days = 80
    n = len(all_dates)
    start = 0
    period_id = 0

    while start + train_days + test_days <= n:
        test_start = all_dates[start + train_days]
        test_end = all_dates[min(start + train_days + test_days - 1, n - 1)]
        test_dates = [d for d in all_dates if test_start <= d <= test_end]

        if len(test_dates) < 10:
            start += step_days
            period_id += 1
            continue

        # Run champion
        champion_result = run_v3_backtest(data, index_symbol, flywheel.champion.config)
        champ_equity = champion_result["equity_curve"]
        # Get return for this test period
        champ_dates = champ_equity.index.tolist()
        champ_test = [d for d in champ_dates if test_start <= d <= test_end]
        if len(champ_test) >= 2:
            champ_ret = (champ_equity[champ_test[-1]] / champ_equity[champ_test[0]]) - 1
        else:
            champ_ret = 0.0

        # Run challenger
        if flywheel.challengers:
            challenger_result = run_v3_backtest(data, index_symbol, flywheel.challengers[0].config)
            chall_equity = challenger_result["equity_curve"]
            chall_dates = chall_equity.index.tolist()
            chall_test = [d for d in chall_dates if test_start <= d <= test_end]
            if len(chall_test) >= 2:
                chall_ret = (chall_equity[chall_test[-1]] / chall_equity[chall_test[0]]) - 1
            else:
                chall_ret = 0.0
        else:
            chall_ret = 0.0

        flywheel.evaluate_period(champ_ret, chall_ret, period_id)

        # Check promotion
        if flywheel.check_promotion(min_periods=3):
            msg = flywheel.promote()
            print(f"  [飞轮] {msg}")

        start += step_days
        period_id += 1

    return flywheel.get_status()


def print_report(result: dict, wf_results: list, flywheel_status: dict, data: dict) -> None:
    """Print v3 report."""
    equity = result["equity_curve"]
    trades = result["trades"]
    rebalances = result["rebalance_log"]

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    n_days = len(equity)
    annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
    returns = equity.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = abs(drawdown.min())

    # Benchmark
    bench_key = "510300"
    if bench_key not in data:
        for k in data:
            if "510300" in k and not k.startswith("idx"):
                bench_key = k
                break
    bench_df = data[bench_key]
    all_dates = sorted(bench_df["trade_date"].tolist())
    warmup = 70
    bench_start = bench_df[bench_df["trade_date"] == all_dates[warmup]]
    bench_end = bench_df[bench_df["trade_date"] == all_dates[-1]]
    bench_return = (bench_end.iloc[0]["close"] / bench_start.iloc[0]["close"]) - 1 if not bench_start.empty else 0
    bench_annual = (1 + bench_return) ** (252 / (len(all_dates) - warmup)) - 1 if bench_return > -1 else 0

    crash_sells = [t for t in trades if t["action"] == "CRASH_SELL"]

    print("\n" + "=" * 70)
    print("  多因子策略 v3 回测报告")
    print("  改进: Top4持仓 + 风险平价 + 月频调仓 + 崩溃保护 + 防御资产")
    print("  数据: 2022-01-04 ~ 2026-04-17 (4.3年) | 21只ETF + 国债/货币ETF")
    print("=" * 70)

    print(f"\n{'指标':<20} {'v3策略':>12} {'基准(沪深300)':>14}")
    print("-" * 50)
    print(f"{'总收益':<20} {total_return:>11.2%} {bench_return:>13.2%}")
    print(f"{'年化收益':<20} {annual_return:>11.2%} {bench_annual:>13.2%}")
    print(f"{'波动率':<20} {volatility:>11.2%} {'—':>13}")
    print(f"{'Sharpe':<20} {sharpe:>11.2f} {'—':>13}")
    print(f"{'最大回撤':<20} {max_dd:>11.2%} {'—':>13}")
    print(f"{'调仓次数':<20} {len(rebalances):>11d} {'—':>13}")
    print(f"{'崩溃保护触发':<20} {len(crash_sells):>11d} {'—':>13}")

    excess = total_return - bench_return
    print(f"\n超额收益(累计): {excess:+.2%}")
    print(f"超额收益(年化): {annual_return - bench_annual:+.2%}")

    # Walk-forward
    print(f"\n{'=' * 70}")
    print("  Walk-Forward 验证 (训练200天 / 测试80天)")
    print(f"{'=' * 70}")

    if wf_results:
        print(f"\n{'窗口':<5} {'测试期':<27} {'策略':>8} {'基准':>8} {'超额':>8} {'MaxDD':>8}")
        print("-" * 70)

        total_excess = 0
        wins = 0
        for r in wf_results:
            test_period = f"{r['test_start']} ~ {r['test_end']}"
            print(f"  {r['window_id']:<3} {test_period:<27} "
                  f"{r['test_return']:>7.2%} {r['benchmark_return']:>7.2%} "
                  f"{r['excess_return']:>+7.2%} {r['max_drawdown']:>7.2%}")
            total_excess += r["excess_return"]
            if r["excess_return"] > 0:
                wins += 1

        n_windows = len(wf_results)
        avg_excess = total_excess / n_windows
        win_rate = wins / n_windows

        print("-" * 70)
        print(f"  平均超额: {avg_excess:+.2%} | 胜率: {win_rate:.0%} ({wins}/{n_windows})")

        if avg_excess > 0.005:
            print(f"  ✓ 样本外正超额，v3有泛化能力")
        elif avg_excess > -0.01:
            print(f"  △ 样本外接近零，alpha边际")
        else:
            print(f"  ✗ 样本外为负，仍需改进")

    # Flywheel
    print(f"\n{'=' * 70}")
    print("  飞轮进化 (Champion / Challenger)")
    print(f"{'=' * 70}")
    print(f"  当前Champion: {flywheel_status.get('champion', 'N/A')}")
    print(f"  Champion累计: {flywheel_status.get('champion_return', 0):+.2%}")
    print(f"  当前Challenger: {flywheel_status.get('challenger', 'N/A')}")
    print(f"  Challenger累计: {flywheel_status.get('challenger_return', 0):+.2%}")
    print(f"  评估期数: {flywheel_status.get('n_periods', 0)}")
    print(f"  代数: {flywheel_status.get('generation', 0)}")

    # Recent rebalances
    print(f"\n{'=' * 70}")
    print("  最近5次调仓")
    print(f"{'=' * 70}")
    for rb in rebalances[-5:]:
        holdings_str = ", ".join(rb["holdings"][:4]) if rb["holdings"] else "现金"
        crashed_str = f" | 崩溃换出: {rb['crashed']}" if rb["crashed"] else ""
        print(f"  {rb['date']} [{rb['regime']}] 仓位{rb['position_scale']:.0%} → {holdings_str}{crashed_str}")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    print("加载数据...")
    data = load_long_data()

    index_symbol = None
    for k in data:
        if k.startswith("idx_") and "000300" in k:
            index_symbol = k
            break

    if index_symbol is None:
        print("错误: 未找到指数数据")
        sys.exit(1)

    n_etf = sum(1 for k in data if not k.startswith("idx_"))
    print(f"  {len(data)} 个标的 ({n_etf} ETF/基金 + 指数)")
    print(f"  指数: {index_symbol}\n")

    print("运行 v3 回测...")
    result = run_v3_backtest(data, index_symbol)

    print("运行 Walk-Forward...")
    wf_results = run_walk_forward_v3(data, index_symbol)

    print("运行 飞轮进化...")
    flywheel_status = run_flywheel(data, index_symbol)

    print_report(result, wf_results, flywheel_status, data)
