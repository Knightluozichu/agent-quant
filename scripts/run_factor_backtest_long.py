"""Multi-factor backtest v2 with LONG history (4.3 years).

Data: 2022-01-04 ~ 2026-04-17 (akshare sina source)
This gives us enough data for 5+ walk-forward windows with statistical significance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from a_share_quant.factors import (
    FactorConfig,
    FactorPortfolio,
    WalkForwardValidator,
)
from a_share_quant.regime import RegimeDetector


def load_long_data() -> dict[str, pd.DataFrame]:
    """Load long history data from akshare."""
    data_dir = PROJECT_ROOT / "data" / "long_history"
    data = {}
    for f in data_dir.glob("*.parquet"):
        if f.name == "combined_long.parquet":
            continue
        df = pd.read_parquet(f)
        if "symbol" in df.columns and "trade_date" in df.columns:
            symbol = df["symbol"].iloc[0]
            # Handle index files (prefixed with "index_")
            if f.name.startswith("index_"):
                symbol = f"idx_{symbol}"
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            data[symbol] = df.sort_values("trade_date").reset_index(drop=True)
    return data


def run_factor_backtest_v2(
    data: dict[str, pd.DataFrame],
    index_symbol: str,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """Run multi-factor backtest v2 over full period."""
    cfg = FactorConfig()
    portfolio = FactorPortfolio(cfg)
    detector = RegimeDetector()

    index_df = data[index_symbol]

    # Tradable universe (exclude index)
    tradable = {k: v for k, v in data.items() if not k.startswith("idx_")}

    # Trading days (skip warmup)
    all_dates = sorted(index_df["trade_date"].tolist())
    warmup = cfg.trend_slow + 10
    trading_days = all_dates[warmup:]

    # Portfolio state
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

        # Check rebalance
        signal = portfolio.generate_signal(tradable, td, regime.state_id, equity)

        if signal:
            rebalance_log.append(
                {
                    "date": td,
                    "holdings": signal.target_holdings,
                    "regime": signal.regime_state,
                    "position_scale": signal.position_scale,
                    "active_factors": signal.active_factors,
                }
            )

            # Sell positions not in target
            for sym in list(holdings.keys()):
                if sym not in signal.target_holdings:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        price = row.iloc[0]["close"]
                        revenue = holdings[sym] * price
                        fee = revenue * fee_rate / 2
                        cash += revenue - fee
                        trades.append({"date": td, "action": "SELL", "symbol": sym, "fee": fee})
                        del holdings[sym]

            # Buy target positions
            total_equity = cash + sum(
                holdings.get(s, 0) * data[s][data[s]["trade_date"] == td].iloc[0]["close"]
                for s in holdings
                if s in data and not data[s][data[s]["trade_date"] == td].empty
            )

            for sym, weight in signal.weights.items():
                if sym not in holdings and sym in data:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        price = row.iloc[0]["close"]
                        target_value = total_equity * weight
                        shares = int(target_value / price / 100) * 100
                        if shares > 0:
                            cost = shares * price
                            fee = cost * fee_rate / 2
                            if cost + fee <= cash:
                                cash -= cost + fee
                                holdings[sym] = shares
                                trades.append(
                                    {"date": td, "action": "BUY", "symbol": sym, "fee": fee}
                                )

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


def run_walk_forward_long(data: dict[str, pd.DataFrame], index_symbol: str) -> list:
    """Run walk-forward with longer windows for statistical significance."""
    # With 4.3 years (~1037 days), use:
    # train=200 days (~10 months), test=80 days (~4 months), step=80
    # This gives us ~10 windows
    validator = WalkForwardValidator(
        train_days=200,
        test_days=80,
        step_days=80,
    )

    # Use the ETF benchmark for walk-forward
    bench_symbol = "510300"
    if bench_symbol not in data:
        # Try to find it
        for k in data:
            if "510300" in k:
                bench_symbol = k
                break

    results = validator.validate(data, bench_symbol)
    return results


def print_report(
    result: dict, data: dict[str, pd.DataFrame], wf_results: list, index_symbol: str
) -> None:
    """Print comprehensive report."""
    equity = result["equity_curve"]
    trades = result["trades"]
    rebalances = result["rebalance_log"]

    # Metrics
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    n_days = len(equity)
    annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
    returns = equity.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = abs(drawdown.min())

    # Benchmark (510300)
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
    bench_return = (
        (bench_end.iloc[0]["close"] / bench_start.iloc[0]["close"]) - 1
        if not bench_start.empty
        else 0
    )

    # Benchmark annualized
    bench_annual = (
        (1 + bench_return) ** (252 / (len(all_dates) - warmup)) - 1 if bench_return > -1 else 0
    )

    total_fees = sum(t.get("fee", 0) for t in trades)

    print("\n" + "=" * 70)
    print("  多因子策略 v2 回测报告 (长历史 4.3年)")
    print("  数据: 2022-01-04 ~ 2026-04-17 | 标的池: 21只ETF | 源: akshare(新浪)")
    print("  因子: 动量/反转/低波/趋势/量能 (Regime条件激活 + IC调权)")
    print("  风控: 回撤>12%降仓 + 动量崩溃清仓 + Regime仓位缩放")
    print("=" * 70)

    print(f"\n{'指标':<20} {'v2策略':>12} {'基准(沪深300ETF)':>16}")
    print("-" * 55)
    print(f"{'总收益':<20} {total_return:>11.2%} {bench_return:>15.2%}")
    print(f"{'年化收益':<20} {annual_return:>11.2%} {bench_annual:>15.2%}")
    print(f"{'波动率':<20} {volatility:>11.2%} {'—':>15}")
    print(f"{'Sharpe':<20} {sharpe:>11.2f} {'—':>15}")
    print(f"{'最大回撤':<20} {max_dd:>11.2%} {'—':>15}")
    print(f"{'调仓次数':<20} {len(rebalances):>11d} {'—':>15}")
    print(f"{'交易笔数':<20} {len(trades):>11d} {'—':>15}")
    print(f"{'总费用':<20} {'¥' + f'{total_fees:,.0f}':>11} {'—':>15}")

    excess = total_return - bench_return
    print(f"\n超额收益(累计): {excess:+.2%}")
    print(f"超额收益(年化): {annual_return - bench_annual:+.2%}")

    # Walk-forward
    print(f"\n{'=' * 70}")
    print("  Walk-Forward 验证 (训练200天 / 测试80天 / 步进80天)")
    print(f"{'=' * 70}")

    if wf_results:
        print(
            f"\n{'窗口':<5} {'测试期':<27} {'策略':>8} {'基准':>8} {'超额':>8} {'MaxDD':>8} {'交易':>5}"
        )
        print("-" * 78)

        total_excess = 0
        wins = 0
        for r in wf_results:
            test_period = f"{r.test_start} ~ {r.test_end}"
            print(
                f"  {r.window_id:<3} {test_period:<27} "
                f"{r.test_return:>7.2%} {r.benchmark_return:>7.2%} "
                f"{r.excess_return:>+7.2%} {r.max_drawdown:>7.2%} {r.n_trades:>4}"
            )
            total_excess += r.excess_return
            if r.excess_return > 0:
                wins += 1

        n_windows = len(wf_results)
        avg_excess = total_excess / n_windows
        win_rate = wins / n_windows

        # Annualized excess
        avg_test_days = 80
        annual_excess = avg_excess * (252 / avg_test_days)

        print("-" * 78)
        print(f"  窗口数: {n_windows}")
        print(f"  平均超额(每期): {avg_excess:+.2%}")
        print(f"  年化超额: {annual_excess:+.2%}")
        print(f"  胜率: {win_rate:.0%} ({wins}/{n_windows})")

        print("\n  过拟合检验:")
        if avg_excess > 0.01:
            print(f"  ✓ 样本外正超额 ({avg_excess:+.2%})，策略有泛化能力")
        elif avg_excess > -0.005:
            print(f"  △ 样本外接近零 ({avg_excess:+.2%})，alpha边际")
        else:
            print(f"  ✗ 样本外为负 ({avg_excess:+.2%})，策略可能过拟合")

        # Consistency check
        if n_windows >= 5:
            positive_windows = sum(1 for r in wf_results if r.excess_return > 0)
            print(f"  一致性: {positive_windows}/{n_windows} 个窗口为正")
            if positive_windows >= n_windows * 0.6:
                print("  ✓ 多数窗口为正，策略一致性可接受")
            else:
                print("  △ 正窗口不足60%，策略稳定性待观察")

    # Recent rebalances
    print(f"\n{'=' * 70}")
    print("  最近5次调仓")
    print(f"{'=' * 70}")
    for rb in rebalances[-5:]:
        holdings_str = ", ".join(rb["holdings"][:3]) if rb["holdings"] else "现金"
        factors_str = "+".join(rb["active_factors"][:3])
        print(
            f"  {rb['date']} [{rb['regime']}] 仓位{rb['position_scale']:.0%} → {holdings_str} ({factors_str})"
        )

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    print("加载长历史数据...")
    data = load_long_data()

    # Find index symbol
    index_symbol = None
    for k in data:
        if k.startswith("idx_") and "000300" in k:
            index_symbol = k
            break

    if index_symbol is None:
        print("错误: 未找到沪深300指数数据")
        sys.exit(1)

    n_etf = sum(1 for k in data if not k.startswith("idx_"))
    n_days = max(len(df) for df in data.values())
    print(f"  {len(data)} 个标的 ({n_etf} ETF + 指数)")
    print(f"  数据长度: ~{n_days} 天")
    print(f"  指数: {index_symbol}\n")

    print("运行多因子 v2 回测 (4.3年)...")
    result = run_factor_backtest_v2(data, index_symbol)

    print("\n运行 Walk-Forward 验证 (5+窗口)...")
    wf_results = run_walk_forward_long(data, index_symbol)

    print_report(result, data, wf_results, index_symbol)
