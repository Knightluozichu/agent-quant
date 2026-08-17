"""Run backtest with real JQData.

Compares strategy performance against buy-and-hold benchmark.
Uses regime detection + strategy routing on real ETF data.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from a_share_quant.backtest import (
    MetricsCalculator,
    PerformanceMetrics,
)
from a_share_quant.regime import RegimeDetector

# =============================================================================
# Data Loading
# =============================================================================


def load_real_data() -> dict[str, pd.DataFrame]:
    """Load cached real data from Parquet files."""
    data_dir = PROJECT_ROOT / "data" / "real"
    data = {}

    for f in data_dir.glob("*.parquet"):
        if f.name == "combined_daily.parquet":
            continue
        df = pd.read_parquet(f)
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else f.stem
        # Ensure trade_date is date type
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        data[symbol] = df.sort_values("trade_date").reset_index(drop=True)

    return data


# =============================================================================
# Simple Strategy: Regime-based ETF rotation
# =============================================================================


class RegimeRotationStrategy:
    """Simple regime-based ETF rotation strategy.

    Rules:
    - UP regime: Hold 510300 (沪深300ETF)
    - FLAT regime: Hold 510500 (中证500ETF) or cash
    - DOWN regime: Cash (or 159915 for rebound)
    """

    def __init__(self):
        self.detector = RegimeDetector()
        self.position: str | None = None  # Current holding symbol
        self.entry_date: date | None = None
        self.entry_price: float = 0.0

    def get_signal(
        self,
        trade_date: date,
        index_df: pd.DataFrame,
        current_price: float,
    ) -> tuple[str, str]:
        """Get trading signal.

        Returns:
            (action, target_symbol)
            action: BUY, SELL, HOLD
        """
        # Detect regime using index data
        regime = self.detector.detect(index_df, trade_date)
        state = regime.state_id

        # Determine target based on regime
        if state.startswith("UP"):
            target = "510300.XSHG"  # 沪深300ETF
        elif state.startswith("FLAT"):
            target = "510500.XSHG"  # 中证500ETF
        else:  # DOWN
            target = None  # Cash

        # Generate signal
        if self.position is None and target is not None:
            return "BUY", target
        if self.position is not None and target is None:
            return "SELL", self.position
        if self.position is not None and target != self.position:
            return "SWITCH", target
        return "HOLD", self.position or ""


# =============================================================================
# Backtest Runner
# =============================================================================


def run_backtest(
    data: dict[str, pd.DataFrame],
    initial_capital: float = 1_000_000.0,
) -> dict:
    """Run regime rotation backtest on real data.

    Returns:
        Dict with equity curve, trades, and metrics.
    """
    # Use 000300.XSHG as regime detection index
    index_symbol = "000300.XSHG"
    index_df = data[index_symbol]

    # Tradable ETFs
    etf_symbols = ["510300.XSHG", "510500.XSHG", "159915.XSHE"]

    # Get trading days (skip first 60 for warmup)
    all_dates = sorted(index_df["trade_date"].tolist())
    warmup = 60
    trading_days = all_dates[warmup:]

    strategy = RegimeRotationStrategy()

    # Portfolio state
    cash = initial_capital
    shares = 0
    holding_symbol: str | None = None
    entry_price = 0.0
    entry_date = None

    equity_history = []
    trades = []
    regime_history = []

    # Fee rates
    commission_rate = 0.00025  # 万2.5
    min_commission = 5.0
    stamp_tax_rate = 0.0005  # 印花税 (卖出)

    for td in trading_days:
        # Get index data up to today
        idx_data = index_df[index_df["trade_date"] <= td]

        # Detect regime
        regime = strategy.detector.detect(idx_data, td)
        regime_history.append({"date": td, "state": regime.state_id})

        # Get current prices
        prices = {}
        for sym in etf_symbols:
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    prices[sym] = row.iloc[0]["close"]

        # Get signal
        action, target = strategy.get_signal(td, idx_data, prices.get("510300.XSHG", 0))

        # Execute
        if action == "BUY" and target in prices and holding_symbol is None:
            price = prices[target]
            qty = int(cash * 0.95 / price / 100) * 100  # 95% position, round to 100
            if qty > 0:
                cost = qty * price
                fee = max(cost * commission_rate, min_commission)
                cash -= cost + fee
                shares = qty
                holding_symbol = target
                entry_price = price
                entry_date = td
                trades.append(
                    {
                        "date": td,
                        "action": "BUY",
                        "symbol": target,
                        "price": price,
                        "quantity": qty,
                        "fee": fee,
                        "regime": regime.state_id,
                    }
                )
                strategy.position = target

        elif action == "SELL" and holding_symbol is not None and holding_symbol in prices:
            price = prices[holding_symbol]
            revenue = shares * price
            fee = max(revenue * commission_rate, min_commission)
            tax = revenue * stamp_tax_rate
            cash += revenue - fee - tax

            pnl = (price - entry_price) * shares - fee - tax
            trades.append(
                {
                    "date": td,
                    "action": "SELL",
                    "symbol": holding_symbol,
                    "price": price,
                    "quantity": shares,
                    "fee": fee + tax,
                    "pnl": pnl,
                    "regime": regime.state_id,
                    "holding_days": (td - entry_date).days if entry_date else 0,
                }
            )
            shares = 0
            holding_symbol = None
            entry_price = 0.0
            strategy.position = None

        elif action == "SWITCH" and holding_symbol is not None and target in prices:
            # Sell current
            if holding_symbol in prices:
                price = prices[holding_symbol]
                revenue = shares * price
                fee = max(revenue * commission_rate, min_commission)
                tax = revenue * stamp_tax_rate
                cash += revenue - fee - tax
                pnl = (price - entry_price) * shares - fee - tax
                trades.append(
                    {
                        "date": td,
                        "action": "SELL",
                        "symbol": holding_symbol,
                        "price": price,
                        "quantity": shares,
                        "fee": fee + tax,
                        "pnl": pnl,
                        "regime": regime.state_id,
                        "holding_days": (td - entry_date).days if entry_date else 0,
                    }
                )

            # Buy new
            new_price = prices[target]
            qty = int(cash * 0.95 / new_price / 100) * 100
            if qty > 0:
                cost = qty * new_price
                fee = max(cost * commission_rate, min_commission)
                cash -= cost + fee
                shares = qty
                holding_symbol = target
                entry_price = new_price
                entry_date = td
                trades.append(
                    {
                        "date": td,
                        "action": "BUY",
                        "symbol": target,
                        "price": new_price,
                        "quantity": qty,
                        "fee": fee,
                        "regime": regime.state_id,
                    }
                )
                strategy.position = target
            else:
                shares = 0
                holding_symbol = None
                strategy.position = None

        # Calculate equity
        position_value = 0
        if holding_symbol and holding_symbol in prices:
            position_value = shares * prices[holding_symbol]
        equity = cash + position_value
        equity_history.append({"date": td, "equity": equity})

    # Build results
    equity_df = pd.DataFrame(equity_history)
    equity_series = equity_df.set_index("date")["equity"]

    # Calculate metrics
    metrics_calc = MetricsCalculator()
    sell_trades = [t for t in trades if t["action"] == "SELL"]
    metrics = metrics_calc.calculate(equity_series, sell_trades)

    return {
        "equity_curve": equity_series,
        "trades": trades,
        "metrics": metrics,
        "regime_history": regime_history,
        "equity_df": equity_df,
    }


def run_buy_and_hold(
    data: dict[str, pd.DataFrame],
    symbol: str = "510300.XSHG",
    initial_capital: float = 1_000_000.0,
    warmup: int = 60,
) -> pd.Series:
    """Run buy-and-hold benchmark."""
    df = data[symbol]
    all_dates = sorted(df["trade_date"].tolist())
    trading_days = all_dates[warmup:]

    # Buy at first day's close
    first_row = df[df["trade_date"] == trading_days[0]]
    if first_row.empty:
        return pd.Series(dtype=float)

    entry_price = first_row.iloc[0]["close"]
    shares = int(initial_capital / entry_price / 100) * 100
    cost = shares * entry_price
    cash = initial_capital - cost

    equity_history = []
    for td in trading_days:
        row = df[df["trade_date"] == td]
        if not row.empty:
            equity = cash + shares * row.iloc[0]["close"]
            equity_history.append({"date": td, "equity": equity})

    equity_df = pd.DataFrame(equity_history)
    return equity_df.set_index("date")["equity"]


# =============================================================================
# Report
# =============================================================================


def print_report(result: dict, benchmark_equity: pd.Series) -> None:
    """Print backtest report."""
    metrics: PerformanceMetrics = result["metrics"]
    trades = result["trades"]
    regime_history = result["regime_history"]

    # Benchmark metrics
    bench_calc = MetricsCalculator()
    bench_metrics = bench_calc.calculate(benchmark_equity, [])

    print("\n" + "=" * 60)
    print("  真实数据回测报告")
    print("  数据范围: 2025-04-11 ~ 2026-04-18 (JQData)")
    print("  策略: 9状态市场轮动 (沪深300/中证500/现金)")
    print("=" * 60)

    print(f"\n{'指标':<20} {'策略':>12} {'基准(沪深300)':>14}")
    print("-" * 50)
    print(f"{'总收益':<20} {metrics.total_return:>11.2%} {bench_metrics.total_return:>13.2%}")
    print(f"{'年化收益':<20} {metrics.annual_return:>11.2%} {bench_metrics.annual_return:>13.2%}")
    print(f"{'波动率':<20} {metrics.volatility:>11.2%} {bench_metrics.volatility:>13.2%}")
    print(f"{'Sharpe':<20} {metrics.sharpe_ratio:>11.2f} {bench_metrics.sharpe_ratio:>13.2f}")
    print(f"{'最大回撤':<20} {metrics.max_drawdown:>11.2%} {bench_metrics.max_drawdown:>13.2%}")
    print(f"{'胜率':<20} {metrics.win_rate:>11.1%} {'N/A':>13}")
    print(f"{'盈亏比':<20} {metrics.profit_factor:>11.2f} {'N/A':>13}")
    print(f"{'交易次数':<20} {metrics.total_trades:>11d} {'1':>13}")

    # Excess return
    excess = metrics.total_return - bench_metrics.total_return
    print(f"\n超额收益: {excess:+.2%}")

    # Trade summary
    sell_trades = [t for t in trades if t["action"] == "SELL"]
    if sell_trades:
        print(f"\n--- 交易明细 (共 {len(sell_trades)} 笔平仓) ---")
        for t in sell_trades[:10]:
            pnl_str = f"{t.get('pnl', 0):+,.0f}"
            print(
                f"  {t['date']} {t['symbol'][:6]} "
                f"盈亏:{pnl_str:>10} "
                f"持有:{t.get('holding_days', 0):>3}天 "
                f"状态:{t['regime']}"
            )

    # Regime distribution
    print("\n--- 市场状态分布 ---")
    state_counts = {}
    for r in regime_history:
        state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1
    for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
        pct = count / len(regime_history) * 100
        print(f"  {state:<15} {count:>4}天 ({pct:.1f}%)")

    print("\n" + "=" * 60)

    # Verdict
    if excess > 0.02:
        print("  结论: 策略跑赢基准，值得继续优化")
    elif excess > -0.02:
        print("  结论: 策略接近基准，需要改进alpha来源")
    else:
        print("  结论: 策略跑输基准，需要重新设计策略逻辑")
    print("=" * 60 + "\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("加载真实数据...")
    data = load_real_data()
    print(f"  已加载 {len(data)} 个标的")
    for sym, df in data.items():
        print(f"    {sym}: {len(df)} 天 ({df['trade_date'].min()} ~ {df['trade_date'].max()})")

    print("\n运行策略回测...")
    result = run_backtest(data)

    print("运行基准 (买入持有沪深300ETF)...")
    benchmark = run_buy_and_hold(data, "510300.XSHG")

    print_report(result, benchmark)
