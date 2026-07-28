"""七星跨资产ETF动量轮动 — 本地复现.

ETF池 (7只风险资产 + 1只防御):
  518880 黄金ETF | 159985 豆粕ETF | 501018 南方原油
  161226 白银LOF | 513100 纳指ETF | 159915 创业板ETF
  511220 城投债ETF | 511880 货币基金(防御)

策略逻辑:
  1. 计算每只ETF的加权动量得分 (20日×0.4 + 60日×0.3 + 120日×0.3)
  2. 选得分最高的1只持有
  3. 如果所有ETF动量<0, 切货币基金(防御)
  4. 月频调仓(20天)

数据: akshare拉取, 尽可能长的历史
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "qixing_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data" / "cross_asset"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ETF池
ETF_POOL = {
    "518880": "黄金ETF",
    "159985": "豆粕ETF",
    "501018": "南方原油",
    "161226": "白银LOF",
    "513100": "纳指ETF",
    "159915": "创业板ETF",
    "511220": "城投债ETF",
}
DEFENSE_ETF = "511880"  # 货币基金

FEE = 0.0005  # 万五单边(ETF免印花税)
SLIPPAGE = 0.001


def fetch_etf_data():
    """用akshare新浪源拉取ETF历史数据."""
    import akshare as ak
    import time

    all_data = {}
    all_codes = list(ETF_POOL.keys()) + [DEFENSE_ETF]

    for code in all_codes:
        name = ETF_POOL.get(code, "货币基金")
        cache_file = DATA_DIR / f"{code}.parquet"

        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            print(f"  {code} {name}: 缓存 {len(df)}天 ({df['trade_date'].min()} ~ {df['trade_date'].max()})")
            all_data[code] = df
            continue

        print(f"  {code} {name}: 拉取中...", end=" ", flush=True)
        try:
            # 新浪源: 需要加市场前缀
            prefix = "sh" if code.startswith(("5", "6")) else "sz"
            symbol = f"{prefix}{code}"
            raw = ak.fund_etf_hist_sina(symbol=symbol)
            if raw is None or raw.empty:
                print("无数据")
                continue

            # 标准化列名 (新浪返回: date, open, high, low, close, volume, amount, ...)
            df = raw.rename(columns={"date": "trade_date"})
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            df["symbol"] = code
            # 确保有需要的列
            for col in ["open", "close", "high", "low", "volume"]:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[["trade_date", "open", "close", "high", "low", "volume", "symbol"]]
            df = df.sort_values("trade_date").reset_index(drop=True)

            # 缓存
            df.to_parquet(cache_file, index=False)
            print(f"{len(df)}天 ({df['trade_date'].min()} ~ {df['trade_date'].max()})")
            all_data[code] = df
            time.sleep(1)  # 避免频率限制

        except Exception as e:
            print(f"失败: {e}")

    return all_data


def calc_momentum_score(close: np.ndarray) -> float:
    """加权动量: 20日×0.4 + 60日×0.3 + 120日×0.3."""
    if len(close) < 121:
        # 数据不足时用可用的
        if len(close) >= 61:
            m20 = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0
            m60 = (close[-1] - close[-60]) / close[-60]
            return m20 * 0.5 + m60 * 0.5
        elif len(close) >= 21:
            return (close[-1] - close[-20]) / close[-20]
        return 0.0

    m20 = (close[-1] - close[-20]) / close[-20]
    m60 = (close[-1] - close[-60]) / close[-60]
    m120 = (close[-1] - close[-120]) / close[-120]
    return m20 * 0.4 + m60 * 0.3 + m120 * 0.3


def run_qixing_backtest(data: dict, rebalance: int = 20, top_n: int = 1,
                        initial_capital: float = 100_000.0) -> dict:
    """七星动量轮动回测."""
    # 找公共日期范围
    common_dates = None
    for code in ETF_POOL:
        if code not in data:
            continue
        dates = set(data[code]["trade_date"].tolist())
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates & dates

    if DEFENSE_ETF in data:
        defense_dates = set(data[DEFENSE_ETF]["trade_date"].tolist())
        common_dates = common_dates & defense_dates

    if not common_dates:
        return {"error": "no common dates"}

    all_dates = sorted(common_dates)
    warmup = 130  # 需要120天warmup
    trading_dates = all_dates[warmup:]
    rebalance_dates = trading_dates[::rebalance]

    cash = initial_capital
    holdings: dict[str, int] = {}
    equity_history = []
    n_trades = 0
    decision_log = []

    for td in rebalance_dates:
        # 计算每只ETF的动量得分
        scores = {}
        for code in ETF_POOL:
            if code not in data:
                continue
            df = data[code]
            hist = df[df["trade_date"] <= td].sort_values("trade_date")
            if len(hist) < 20:
                continue
            close = hist["close"].values.astype(float)
            score = calc_momentum_score(close)
            scores[code] = score

        if not scores:
            continue

        # 选最好的top_n只, 如果全部<0则切防御
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        positive = [(c, s) for c, s in sorted_scores if s > 0]

        if positive:
            selected = [c for c, s in positive[:top_n]]
        else:
            selected = [DEFENSE_ETF]  # 全部动量<0, 切货币基金

        decision_log.append({
            "date": str(td),
            "selected": selected,
            "scores": {c: round(s, 4) for c, s in sorted_scores[:3]},
            "all_negative": len(positive) == 0,
        })

        # 交易执行
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]

        per_target = equity / len(selected)

        # 卖出非目标
        for sym in list(holdings.keys()):
            if sym not in selected:
                if sym in data:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        cash += holdings[sym] * row.iloc[0]["close"] * (1 - FEE - SLIPPAGE)
                        n_trades += 1
                        del holdings[sym]

        # 买入目标
        for sym in selected:
            if sym not in data:
                continue
            row = data[sym][data[sym]["trade_date"] == td]
            if row.empty:
                continue
            price = row.iloc[0]["close"]
            cur = holdings.get(sym, 0)
            target_shares = int(per_target / price / 100) * 100
            diff = target_shares - cur
            if diff > 0:
                cost = diff * price * (1 + FEE + SLIPPAGE)
                if cost <= cash:
                    cash -= cost
                    holdings[sym] = cur + diff
                    n_trades += 1
            elif diff < -100:
                sell = int(min(-diff, cur) / 100) * 100
                if sell > 0:
                    cash += sell * price * (1 - FEE - SLIPPAGE)
                    holdings[sym] = cur - sell
                    if holdings[sym] <= 0:
                        holdings.pop(sym, None)
                    n_trades += 1

        # 记录equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity})

    if not equity_history:
        return {"error": "no trades"}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    n_days = len(eq_df)
    ann_ret = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    yearly = {}
    prev_val = initial_capital
    for year in sorted(eq_df["year"].unique()):
        ydf = eq_df[eq_df["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev_val) - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        yearly[int(year)] = {"return": yr, "max_dd": dd}
        prev_val = end_val

    return {
        "total_return": total_return, "ann_return": ann_ret,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "yearly": yearly, "n_trades": n_trades,
        "equity_curve": eq_df, "decision_log": decision_log,
    }


def main():
    print("=" * 70)
    print("  七星跨资产ETF动量轮动 — 本地复现")
    print(f"  ETF池: {list(ETF_POOL.values())}")
    print(f"  防御: 货币基金(511880) | 调仓: 月频(20天) | 持仓: Top1")
    print(f"  动量: 20日×0.4 + 60日×0.3 + 120日×0.3 | 全<0→货币基金")
    print("=" * 70)

    # 拉取数据
    print(f"\n[1/3] 拉取ETF数据...")
    data = fetch_etf_data()
    print(f"  成功: {len(data)}/{len(ETF_POOL)+1} 只")

    if len(data) < 4:
        print("ERROR: 数据不足")
        return

    # 回测
    print(f"\n[2/3] 回测...")
    result = run_qixing_backtest(data, rebalance=20, top_n=1)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    eq = result["equity_curve"]
    print(f"\n  年度收益:")
    print(f"  {'年份':<6} {'年初':>10} {'年末':>10} {'收益':>8} {'回撤':>8}")
    print(f"  {'-' * 46}")
    prev = 100_000.0
    for year in sorted(eq["year"].unique()):
        ydf = eq[eq["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev) - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        print(f"  {year:<6} {prev:>10,.0f} {end_val:>10,.0f} {yr:>+8.2%} {dd:>8.2%}")
        prev = end_val

    final = eq["equity"].iloc[-1]
    total_ret = (final / 100_000) - 1
    n_years = len(eq) / (252 / 20)  # 调仓次数→年数
    print(f"  {'-' * 46}")
    print(f"\n  10万 → {final:,.0f} ({total_ret:+.1%})")
    print(f"  年化: {result['ann_return']:+.1%} | 夏普: {result['sharpe']:.2f} | "
          f"回撤: {result['max_drawdown']:.1%}")
    print(f"  交易次数: {result['n_trades']}")

    # 持仓统计
    decisions = result["decision_log"]
    defense_days = sum(1 for d in decisions if d["all_negative"])
    print(f"  防御(货币基金)期数: {defense_days}/{len(decisions)} ({defense_days/len(decisions):.0%})")

    # 各ETF被选中次数
    from collections import Counter
    selected_counts = Counter()
    for d in decisions:
        for s in d["selected"]:
            name = ETF_POOL.get(s, "货币基金")
            selected_counts[name] += 1
    print(f"\n  持仓分布:")
    for name, count in selected_counts.most_common():
        print(f"    {name:<10} {count}期 ({count/len(decisions):.0%})")

    # 对比基准
    print(f"\n[3/3] 基准对比...")
    # 沪深300
    if "159915" in data:
        cyb = data["159915"]
        cyb_years = {}
        for year in sorted(eq["year"].unique()):
            ydf = cyb[cyb["trade_date"].apply(lambda x: x.year == year)]
            if len(ydf) >= 2:
                cyb_years[year] = (ydf["close"].iloc[-1] - ydf["close"].iloc[0]) / ydf["close"].iloc[0]

    print(f"\n  {'年份':<6} {'七星策略':>8} {'创业板ETF':>9} {'超额':>8}")
    print(f"  {'-' * 34}")
    for year in sorted(result["yearly"].keys()):
        strat_ret = result["yearly"][year]["return"]
        bench_ret = cyb_years.get(year, 0)
        print(f"  {year:<6} {strat_ret:>+8.1%} {bench_ret:>+9.1%} {strat_ret - bench_ret:>+8.1%}")

    # 保存
    summary = {
        "strategy": "七星跨资产动量轮动",
        "params": {"rebalance": 20, "top_n": 1, "momentum": "20×0.4+60×0.3+120×0.3"},
        "etf_pool": ETF_POOL,
        "total_return": total_ret,
        "ann_return": result["ann_return"],
        "sharpe": result["sharpe"],
        "max_drawdown": result["max_drawdown"],
        "yearly": {str(k): v for k, v in result["yearly"].items()},
    }
    with open(OUTPUT_DIR / "qixing_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  结果已保存: {OUTPUT_DIR / 'qixing_results.json'}")


if __name__ == "__main__":
    main()
