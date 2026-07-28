"""V5.1: 纯动量 + 波动率管理 (Moreira & Muir 2017).

设计原则:
- 选股: 纯动量Top4 (v4.3审计证明+64%, 最简单最有效)
- 仓位: 波动率缩放 (单参数target_vol, 不预测崩溃)
- 剩余资金: 配国债ETF
- 不加: 专家模型/HRP/动态退出/Conviction (全部是噪声)

公式:
  scale = target_vol² / realized_vol²
  scale ∈ [0.2, 1.0]  (不加杠杆, 最低20%仓位)
  实际股票仓位 = scale × 100%
  债券仓位 = (1 - scale) × 100%

为什么不过拟合:
- 只有1个参数 (target_vol = 15%年化)
- 不改变选股 (动量alpha完整保留)
- 机械规则 (无主观判断)
- 学术验证 (多市场多时段有效)
"""

from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "long_history"
OUTPUT_DIR = PROJECT_ROOT / "data" / "v51_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFENSIVE = {"511010": "国债ETF", "511880": "货币ETF"}
FEE = 0.001
SLIPPAGE = 0.0005

# === 唯一参数 ===
TARGET_VOL = 0.12  # 12% annualized target (conservative for momentum)
VOL_LOOKBACK = 20  # 20 trading days for realized vol
SCALE_MIN = 0.20   # Minimum 20% in stocks
SCALE_MAX = 1.00   # No leverage


# =============================================================================
# Momentum Factor (simple, proven)
# =============================================================================

def calc_momentum(data: dict[str, pd.DataFrame], as_of: date) -> pd.DataFrame:
    """Calculate 20-80 day momentum for all ETFs. The proven +64% factor."""
    records = []
    for symbol, df in data.items():
        hist = df[df["trade_date"] <= as_of].sort_values("trade_date")
        if len(hist) < 80:
            continue
        close = hist["close"].values.astype(float)
        # 20-80 day momentum (skip recent 20 days to avoid reversal)
        mom = (close[-20] - close[-80]) / close[-80]
        records.append({"symbol": symbol, "momentum": mom})

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("momentum", ascending=False)


# =============================================================================
# Volatility Scaling (Moreira & Muir 2017)
# =============================================================================

def calc_vol_scale(
    data: dict[str, pd.DataFrame],
    selected: list[str],
    index_df: pd.DataFrame,
    as_of: date,
    target_vol: float = TARGET_VOL,
) -> float:
    """Calculate volatility scaling using PORTFOLIO's own realized vol.

    Key fix: use the actual momentum ETFs' vol, not the index vol.
    When momentum stocks get volatile (crash precursor), scale down.

    scale = target_vol² / portfolio_realized_vol²
    Clamped to [SCALE_MIN, SCALE_MAX]
    """
    # Calculate realized vol of the SELECTED momentum stocks (portfolio-level)
    vols = []
    for sym in selected:
        if sym not in data:
            continue
        hist = data[sym][data[sym]["trade_date"] <= as_of].sort_values("trade_date")
        if len(hist) < VOL_LOOKBACK + 1:
            continue
        close = hist["close"].values.astype(float)
        rets = np.diff(close[-VOL_LOOKBACK - 1:]) / close[-VOL_LOOKBACK - 1:-1]
        vols.append(np.std(rets) * np.sqrt(252))

    if not vols:
        # Fallback to index vol
        idx = index_df[index_df["trade_date"] <= as_of].sort_values("trade_date")
        if len(idx) < VOL_LOOKBACK + 1:
            return 1.0
        close = idx["close"].values.astype(float)
        rets = np.diff(close[-VOL_LOOKBACK - 1:]) / close[-VOL_LOOKBACK - 1:-1]
        vols = [np.std(rets) * np.sqrt(252)]

    # Portfolio vol = average of individual vols (simplified, ignores correlation)
    portfolio_vol = np.mean(vols)

    if portfolio_vol < 1e-8:
        return SCALE_MAX

    scale = (target_vol ** 2) / (portfolio_vol ** 2)
    return float(np.clip(scale, SCALE_MIN, SCALE_MAX))


# =============================================================================
# Backtest
# =============================================================================

def run_v51_backtest(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    start_date: date | None = None,
    end_date: date | None = None,
    initial_capital: float = 100_000.0,
    rebalance_days: int = 20,
    top_n: int = 4,
    target_vol: float = TARGET_VOL,
) -> dict:
    """V5.1 backtest: momentum Top4 + vol scaling + bonds for remainder."""
    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {k: v for k, v in data.items()
                if k not in DEFENSIVE and k not in index_symbols}

    all_dates = sorted(index_df["trade_date"].tolist())
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]

    warmup = 80
    trading_days = all_dates[warmup:]

    cash = initial_capital
    holdings: dict[str, int] = {}
    equity_history = []
    scale_history = []
    n_trades = 0
    days_since = rebalance_days  # Force first rebalance

    for td in trading_days:
        # Current equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]

        days_since += 1
        if days_since >= rebalance_days:
            days_since = 0

            # 1. Momentum selection
            mom_df = calc_momentum(tradable, td)
            if mom_df.empty or len(mom_df) < top_n:
                continue
            selected = mom_df.head(top_n)["symbol"].tolist()

            # 2. Volatility scaling (use PORTFOLIO's own vol, not index)
            vol_scale = calc_vol_scale(data, selected, index_df, td, target_vol)
            scale_history.append({"date": str(td), "scale": vol_scale})

            # 3. Target allocation
            # Stocks: vol_scale × equity, split equally among Top4
            # Bonds: (1 - vol_scale) × equity
            stock_budget = equity * vol_scale
            bond_budget = equity * (1 - vol_scale)
            per_stock = stock_budget / top_n

            # 4. Sell non-target stocks
            for sym in list(holdings.keys()):
                if sym not in selected and sym not in DEFENSIVE:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        cash += holdings[sym] * row.iloc[0]["close"] * (1 - FEE - SLIPPAGE)
                        n_trades += 1
                        del holdings[sym]

            # 5. Adjust bond position
            bond_sym = "511010"
            if bond_sym in data:
                row = data[bond_sym][data[bond_sym]["trade_date"] == td]
                if not row.empty:
                    price = row.iloc[0]["close"]
                    cur_bond = holdings.get(bond_sym, 0)
                    target_bond_shares = int(bond_budget / price / 10) * 10
                    diff = target_bond_shares - cur_bond
                    if abs(diff) >= 10:
                        if diff > 0:
                            cost = diff * price * (1 + FEE / 2)
                            if cost <= cash:
                                cash -= cost
                                holdings[bond_sym] = cur_bond + diff
                                n_trades += 1
                        else:
                            sell = min(-diff, cur_bond)
                            cash += sell * price * (1 - FEE / 2)
                            holdings[bond_sym] = cur_bond - sell
                            if holdings[bond_sym] <= 0:
                                del holdings[bond_sym]
                            n_trades += 1

            # 6. Buy/adjust stock positions
            for sym in selected:
                if sym not in data:
                    continue
                row = data[sym][data[sym]["trade_date"] == td]
                if row.empty:
                    continue
                price = row.iloc[0]["close"]
                cur = holdings.get(sym, 0)
                target_shares = int(per_stock / price / 100) * 100
                diff = target_shares - cur

                if diff > 0:
                    cost = diff * price * (1 + FEE + SLIPPAGE)
                    if cost <= cash:
                        cash -= cost
                        holdings[sym] = cur + diff
                        n_trades += 1
                elif diff < -100:
                    sell = min(-diff, cur)
                    sell = int(sell / 100) * 100
                    if sell > 0:
                        cash += sell * price * (1 - FEE - SLIPPAGE)
                        holdings[sym] = cur - sell
                        if holdings[sym] <= 0:
                            del holdings[sym]
                        n_trades += 1

        # Record equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity})

    if not equity_history:
        return {"total_return": 0.0}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    n_days = len(eq_df)
    ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "total_return": total_return, "ann_return": ann_return,
        "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd,
        "calmar": calmar, "n_trades": n_trades, "n_days": n_days,
        "equity_curve": eq_df, "scale_history": scale_history,
    }


# =============================================================================
# Main
# =============================================================================

def load_data():
    data = {}
    for f in DATA_DIR.glob("*.parquet"):
        if f.name in ("combined_long.parquet", "northbound.parquet",
                      "pe_percentile.parquet", "margin_sentiment.parquet"):
            continue
        df = pd.read_parquet(f)
        if "symbol" not in df.columns or "trade_date" not in df.columns:
            continue
        symbol = df["symbol"].iloc[0]
        if f.name.startswith("index_"):
            symbol = f"idx_{symbol}"
            df["symbol"] = symbol
        data[symbol] = df.sort_values("trade_date").reset_index(drop=True)
    index_df = data.get("idx_000300")
    if index_df is None:
        for k, v in data.items():
            if "000300" in k:
                index_df = v
                break
    return data, index_df


def main():
    print("=" * 70)
    print("  V5.1: 纯动量Top4 + 波动率管理 (Moreira & Muir 2017)")
    print(f"  唯一参数: target_vol = {TARGET_VOL:.0%} | vol_lookback = {VOL_LOOKBACK}天")
    print(f"  仓位范围: [{SCALE_MIN:.0%}, {SCALE_MAX:.0%}] | 剩余配国债")
    print("=" * 70)

    data, index_df = load_data()
    if index_df is None:
        print("ERROR: 无指数数据")
        return

    n_etf = len([k for k in data if not k.startswith("idx_")])
    print(f"\n  数据: {n_etf} 标的, {len(index_df)} 天")
    print(f"  范围: {index_df['trade_date'].min()} ~ {index_df['trade_date'].max()}")

    # === Full period ===
    print(f"\n[1/3] 全周期回测 (本金10万)...")
    result = run_v51_backtest(data, index_df, initial_capital=100_000)

    eq = result["equity_curve"]
    print(f"\n  年度收益:")
    print(f"  {'年份':<6} {'年初':>10} {'年末':>10} {'收益':>8} {'回撤':>8} {'平均仓位':>8}")
    print(f"  {'-' * 56}")
    prev = 100_000.0
    scale_df = pd.DataFrame(result["scale_history"])
    if not scale_df.empty:
        scale_df["year"] = pd.to_datetime(scale_df["date"]).dt.year

    for year in sorted(eq["year"].unique()):
        ydf = eq[eq["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev) - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        # Average vol scale for this year
        avg_scale = 1.0
        if not scale_df.empty:
            ys = scale_df[scale_df["year"] == year]
            if not ys.empty:
                avg_scale = ys["scale"].mean()
        print(f"  {year:<6} {prev:>10,.0f} {end_val:>10,.0f} {yr:>+8.2%} {dd:>8.2%} {avg_scale:>7.0%}")
        prev = end_val

    final = eq["equity"].iloc[-1]
    total_ret = (final / 100_000) - 1
    n_years = result["n_days"] / 252
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1

    print(f"  {'-' * 56}")
    print(f"\n  10万 → {final:,.0f} ({total_ret:+.1%}, {final / 100_000:.2f}x)")
    print(f"  年化: {ann_ret:+.1%} | 夏普: {result['sharpe']:.2f} | "
          f"回撤: {result['max_drawdown']:.1%} | Calmar: {result['calmar']:.2f}")
    print(f"  交易次数: {result['n_trades']}")

    # === 2023 stress test ===
    print(f"\n[2/3] 2023压力测试...")
    from datetime import date as dt_date
    r2023 = run_v51_backtest(data, index_df,
                             start_date=dt_date(2023, 1, 1),
                             end_date=dt_date(2023, 12, 31),
                             initial_capital=100_000)
    if r2023.get("equity_curve") is not None:
        eq23 = r2023["equity_curve"]
        ret_2023 = (eq23["equity"].iloc[-1] / 100_000) - 1
        cm23 = eq23["equity"].cummax()
        dd23 = ((eq23["equity"] - cm23) / cm23).min()
        print(f"  2023: {ret_2023:+.2%} | 回撤: {dd23:.1%}")
        # Show vol scale in 2023
        s23 = pd.DataFrame(r2023["scale_history"])
        if not s23.empty:
            print(f"  2023平均仓位: {s23['scale'].mean():.0%} | "
                  f"最低: {s23['scale'].min():.0%} | 最高: {s23['scale'].max():.0%}")

    # === Benchmark comparison ===
    print(f"\n[3/3] 基准对比...")
    # Pure momentum (no vol scaling)
    r_pure = run_v51_backtest(data, index_df, initial_capital=100_000)
    # Override: run with scale always = 1.0 (pure momentum)
    # We'll compute this differently
    idx_c = index_df["close"].values
    bench_300 = (idx_c[-1] - idx_c[0]) / idx_c[0]

    # Run pure momentum by setting target_vol very high (no scaling)
    r_nomgmt = run_v51_backtest(data, index_df, initial_capital=100_000, target_vol=999.0)

    # 2023 for pure momentum
    r_nomgmt_2023 = run_v51_backtest(data, index_df,
                                      start_date=dt_date(2023, 1, 1),
                                      end_date=dt_date(2023, 12, 31),
                                      initial_capital=100_000, target_vol=999.0)

    print(f"\n  {'策略':<24} {'全周期':>8} {'年化':>7} {'夏普':>6} {'回撤':>7} {'2023':>7}")
    print(f"  {'-' * 62}")

    nomgmt_ret = r_nomgmt["total_return"]
    nomgmt_ann = r_nomgmt["ann_return"]
    nomgmt_2023 = (r_nomgmt_2023["equity_curve"]["equity"].iloc[-1] / 100_000 - 1) if r_nomgmt_2023.get("equity_curve") is not None else 0

    print(f"  {'纯动量Top4(无管理)':<24} {nomgmt_ret:>+8.1%} {nomgmt_ann:>+7.1%} "
          f"{r_nomgmt['sharpe']:>6.2f} {r_nomgmt['max_drawdown']:>7.1%} {nomgmt_2023:>+7.2%}")
    print(f"  {'V5.1(动量+波动率管理)':<24} {total_ret:>+8.1%} {ann_ret:>+7.1%} "
          f"{result['sharpe']:>6.2f} {result['max_drawdown']:>7.1%} {ret_2023:>+7.2%}")
    print(f"  {'沪深300':<24} {bench_300:>+8.1%} {'':>7} {'':>6} {'':>7} {'-11.4%':>7}")

    # Improvement
    dd_improve = r_nomgmt["max_drawdown"] - result["max_drawdown"]
    print(f"\n  波动率管理贡献:")
    print(f"    回撤改善: {dd_improve:+.1%} (从{r_nomgmt['max_drawdown']:.1%}→{result['max_drawdown']:.1%})")
    print(f"    2023改善: {ret_2023 - nomgmt_2023:+.2%}")
    print(f"    夏普变化: {result['sharpe'] - r_nomgmt['sharpe']:+.2f}")
    print(f"    收益代价: {total_ret - nomgmt_ret:+.1%}")

    # Save
    summary = {
        "version": "v5.1",
        "method": "momentum_top4 + vol_scaling",
        "params": {"target_vol": TARGET_VOL, "vol_lookback": VOL_LOOKBACK,
                   "scale_min": SCALE_MIN, "scale_max": SCALE_MAX, "top_n": 4},
        "full_period": {"return": total_ret, "ann_return": ann_ret,
                        "sharpe": result["sharpe"], "max_dd": result["max_drawdown"]},
        "2023": {"return": ret_2023, "max_dd": dd23},
    }
    with open(OUTPUT_DIR / "v51_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
