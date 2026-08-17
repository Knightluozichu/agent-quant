"""Multi-factor backtest v2 with improvements.

Changes from v1:
- Regime-conditional factor activation (no contradictory factors)
- Dynamic IC weighting (rolling Spearman IC)
- Risk overlay: drawdown control + momentum crash detection
- Position scaling based on regime + risk state
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

# =============================================================================
# Data Loading
# =============================================================================


def load_data() -> dict[str, pd.DataFrame]:
    """Load all cached real data."""
    data_dir = PROJECT_ROOT / "data" / "real"
    data = {}
    for f in data_dir.glob("*.parquet"):
        if f.name == "combined_daily.parquet":
            continue
        df = pd.read_parquet(f)
        if "symbol" in df.columns and "trade_date" in df.columns:
            symbol = df["symbol"].iloc[0]
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            data[symbol] = df.sort_values("trade_date").reset_index(drop=True)
    return data


# =============================================================================
# Full Period Backtest v2
# =============================================================================


def run_factor_backtest_v2(
    data: dict[str, pd.DataFrame],
    initial_capital: float = 1_000_000.0,
) -> dict:
    """Run multi-factor backtest v2 over full period."""
    cfg = FactorConfig()
    portfolio = FactorPortfolio(cfg)
    detector = RegimeDetector()

    index_symbol = "000300.XSHG"
    index_df = data[index_symbol]

    # Tradable universe (exclude index)
    tradable = {k: v for k, v in data.items() if not k.startswith("000")}

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
    regime_history = []

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

        regime_history.append(
            {
                "date": td,
                "state": regime.state_id,
                "equity": equity,
            }
        )

        if signal:
            rebalance_log.append(
                {
                    "date": td,
                    "holdings": signal.target_holdings,
                    "regime": signal.regime_state,
                    "position_scale": signal.position_scale,
                    "active_factors": signal.active_factors,
                    "ic_weights": signal.ic_weights,
                    "top_scores": signal.factor_scores.head(5)[
                        ["symbol", "composite_score"]
                    ].to_dict("records"),
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
                        trades.append(
                            {
                                "date": td,
                                "action": "SELL",
                                "symbol": sym,
                                "price": price,
                                "quantity": holdings[sym],
                                "fee": fee,
                                "regime": signal.regime_state,
                            }
                        )
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
                                    {
                                        "date": td,
                                        "action": "BUY",
                                        "symbol": sym,
                                        "price": price,
                                        "quantity": shares,
                                        "fee": fee,
                                        "regime": signal.regime_state,
                                    }
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
        "regime_history": regime_history,
    }


# =============================================================================
# Walk-Forward Validation
# =============================================================================


def run_walk_forward(data: dict[str, pd.DataFrame]) -> list:
    """Run walk-forward validation."""
    validator = WalkForwardValidator(
        train_days=100,
        test_days=40,
        step_days=40,
    )

    results = validator.validate(data, "510300.XSHG")
    return results


# =============================================================================
# Report
# =============================================================================


def print_report(result: dict, data: dict[str, pd.DataFrame], wf_results: list) -> None:
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

    # Benchmark
    bench_df = data["510300.XSHG"]
    all_dates = sorted(bench_df["trade_date"].tolist())
    warmup = 70
    bench_start = bench_df[bench_df["trade_date"] == all_dates[warmup]]
    bench_end = bench_df[bench_df["trade_date"] == all_dates[-1]]
    bench_return = (
        (bench_end.iloc[0]["close"] / bench_start.iloc[0]["close"]) - 1
        if not bench_start.empty
        else 0
    )

    total_fees = sum(t.get("fee", 0) for t in trades)

    print("\n" + "=" * 65)
    print("  多因子策略 v2 回测报告")
    print("  改进: 因子正交化 + IC动态调权 + 风控叠加 + 动量崩溃检测")
    print("  数据: 2025-04-11 ~ 2026-04-18 | 标的池: 21只ETF")
    print("=" * 65)

    print(f"\n{'指标':<20} {'v2策略':>12} {'基准(沪深300)':>14}")
    print("-" * 50)
    print(f"{'总收益':<20} {total_return:>11.2%} {bench_return:>13.2%}")
    print(f"{'年化收益':<20} {annual_return:>11.2%} {'—':>13}")
    print(f"{'波动率':<20} {volatility:>11.2%} {'—':>13}")
    print(f"{'Sharpe':<20} {sharpe:>11.2f} {'—':>13}")
    print(f"{'最大回撤':<20} {max_dd:>11.2%} {'—':>13}")
    print(f"{'调仓次数':<20} {len(rebalances):>11d} {'—':>13}")
    print(f"{'交易笔数':<20} {len(trades):>11d} {'—':>13}")
    print(f"{'总费用':<20} {'¥' + f'{total_fees:,.0f}':>11} {'—':>13}")

    excess = total_return - bench_return
    print(f"\n超额收益: {excess:+.2%}")

    # Walk-forward
    print(f"\n{'=' * 65}")
    print("  Walk-Forward 验证 (防过拟合)")
    print(f"{'=' * 65}")

    if wf_results:
        print(
            f"\n{'窗口':<6} {'测试期':<25} {'策略':>8} {'基准':>8} {'超额':>8} {'MaxDD':>8} {'交易':>5}"
        )
        print("-" * 75)

        total_excess = 0
        wins = 0
        for r in wf_results:
            test_period = f"{r.test_start} ~ {r.test_end}"
            print(
                f"  {r.window_id:<4} {test_period:<25} "
                f"{r.test_return:>7.2%} {r.benchmark_return:>7.2%} "
                f"{r.excess_return:>+7.2%} {r.max_drawdown:>7.2%} {r.n_trades:>4}"
            )
            total_excess += r.excess_return
            if r.excess_return > 0:
                wins += 1

        n_windows = len(wf_results)
        avg_excess = total_excess / n_windows
        win_rate = wins / n_windows

        print("-" * 75)
        print(f"  平均超额: {avg_excess:+.2%} | 胜率: {win_rate:.0%} ({wins}/{n_windows})")

        print("\n  过拟合检验:")
        if avg_excess > 0.01:
            print(f"  ✓ 样本外正超额 ({avg_excess:+.2%})，策略有泛化能力")
        elif avg_excess > -0.01:
            print(f"  △ 样本外接近零 ({avg_excess:+.2%})，alpha不显著")
        else:
            print(f"  ✗ 样本外为负 ({avg_excess:+.2%})，可能过拟合")

    # Recent rebalances
    print(f"\n{'=' * 65}")
    print("  最近5次调仓")
    print(f"{'=' * 65}")
    for rb in rebalances[-5:]:
        holdings_str = ", ".join(rb["holdings"][:3]) if rb["holdings"] else "现金"
        factors_str = "+".join(rb["active_factors"][:3])
        print(f"  {rb['date']} [{rb['regime']}] 仓位{rb['position_scale']:.0%} → {holdings_str}")
        print(f"         因子: {factors_str} | IC权重: {rb['ic_weights']}")

    print(f"\n{'=' * 65}\n")

    return {
        "total_return": total_return,
        "bench_return": bench_return,
        "excess": excess,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_trades": len(trades),
        "n_rebalances": len(rebalances),
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("加载数据...")
    data = load_data()
    print(f"  {len(data)} 个标的")

    tradable = {k: v for k, v in data.items() if not k.startswith("000")}
    print(f"  可交易: {len(tradable)} 只ETF")
    print("  指数: 000300.XSHG, 000905.XSHG\n")

    print("运行多因子 v2 回测...")
    result = run_factor_backtest_v2(data)

    print("\n运行 Walk-Forward 验证...")
    wf_results = run_walk_forward(data)

    metrics = print_report(result, data, wf_results)
