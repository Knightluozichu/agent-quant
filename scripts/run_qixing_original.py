"""七星ETF轮动 — 聚宽原版(任侠) 一比一复刻.

目的: 用与 V3 完全相同的数据缓存, 只把"规则"换回原版, 隔离出
      "规则差异" vs "数据差异" 各自对收益缺口的贡献.

原版规则 (聚宽 任侠):
  1. 加权动量: 20日×0.4 + 60日×0.3 + 120日×0.3
  2. 短期动量过滤: 近10日年化<0 → 排除
  3. 放量过滤: 年化>100%时, 当日量>5日均量×2.5 → 排除
  4. 单日跌幅过滤: 近3日有单日跌>3% → 排除
  5. 盈利保护: 持仓从买入后最高点回撤>5% → 卖出切货币
  6. A股走弱回避: 创业板<MA20时, 排除创业板ETF
  7. 全部不通过 → 切货币基金(511880)
  8. 日频调仓, 持仓Top1 (纯每日轮动, 无换仓缓冲阈值)

与 V3 的唯一区别 = 规则; 数据/费用/滑点/起始日完全一致 → 公平对比.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_qixing_v3 import (  # noqa: E402
    A_SHARE_ETF,
    CATEGORIES,
    DEFENSE,
    ETF_POOL,
    FEE,
    SLIPPAGE,
    check_a_share_weak,
    check_short_momentum,
    check_single_day_drop,
    check_volume_spike,
    load_data,
)

# 是否启用"类别轮动"(代码注释称其为原版核心): 先选最强资产类别, 再在类别内选Top1
USE_CATEGORY_SWITCH = True

# === 原版动量参数 (20/60/120) ===
ORIG_MOM_PERIODS = (20, 60, 120)
ORIG_MOM_WEIGHTS = (0.4, 0.3, 0.3)
PROFIT_PROTECTION_DD = 0.05  # 盈利保护: 回撤5%切货币
WARMUP = 130

# 聚宽原版逐年收益 (评论区用户数据) — 对比基准
ORIGINAL = {2020: 0.2716, 2021: 0.3278, 2022: 0.7605,
            2023: 0.0810, 2024: 0.5473, 2025: 2.3803}


def orig_momentum_score(close: np.ndarray) -> float:
    """原版加权动量: 20日×0.4 + 60日×0.3 + 120日×0.3."""
    score = 0.0
    for period, weight in zip(ORIG_MOM_PERIODS, ORIG_MOM_WEIGHTS):
        if len(close) > period:
            ret = (close[-1] - close[-period - 1]) / close[-period - 1]
            score += ret * weight
    return score


def select_target_original(data: dict, etf_data_at_date: dict) -> str:
    """原版选股: 全部过滤开启 + 纯Top1 (无换仓缓冲阈值)."""
    a_share_weak = check_a_share_weak(data, etf_data_at_date.get(A_SHARE_ETF, 0))
    candidates = []
    for code in ETF_POOL:
        if code not in etf_data_at_date:
            continue
        if code == A_SHARE_ETF and a_share_weak:
            continue
        idx = etf_data_at_date[code]
        df = data[code]
        close = df["close"].values[:idx + 1].astype(float)
        volume = df["volume"].values[:idx + 1].astype(float)
        if len(close) < 121:
            continue
        # 原版三大过滤全开
        if not check_short_momentum(close):
            continue
        if not check_volume_spike(volume, close):
            continue
        if not check_single_day_drop(close):
            continue
        score = orig_momentum_score(close)
        if score > 0:
            candidates.append((code, score))
    candidates.sort(key=lambda x: -x[1])

    # 类别轮动(原版核心): 先选平均动量最强的资产类别, 再在该类别内选Top1
    if USE_CATEGORY_SWITCH and candidates:
        score_map = dict(candidates)
        cat_scores = {}
        for cat_name, cat_codes in CATEGORIES.items():
            cat_moms = [score_map[c] for c in cat_codes if c in score_map]
            if cat_moms:
                cat_scores[cat_name] = np.mean(cat_moms)
        if cat_scores:
            best_cat = max(cat_scores, key=cat_scores.get)
            best_cat_codes = set(CATEGORIES[best_cat])
            cat_candidates = [(c, s) for c, s in candidates if c in best_cat_codes]
            if cat_candidates:
                candidates = cat_candidates

    return candidates[0][0] if candidates else DEFENSE


def run_original(data: dict, initial_capital: float = 100_000.0) -> dict:
    """原版回测: 日频调仓 + 盈利保护."""
    common_dates = None
    for code in list(ETF_POOL.keys()) + [DEFENSE]:
        if code not in data:
            continue
        dates = set(data[code]["trade_date"].tolist())
        common_dates = dates if common_dates is None else common_dates & dates
    all_dates = sorted(common_dates)
    trading_dates = all_dates[WARMUP:]

    cash = initial_capital
    holding: str | None = None
    holding_shares = 0
    holding_peak = 0.0
    equity_history = []
    n_trades = 0

    for td in trading_dates:  # 日频 (原版每日调仓)
        etf_data_at_date = {}
        for code in list(ETF_POOL.keys()) + [DEFENSE]:
            if code not in data:
                continue
            df = data[code]
            mask = df["trade_date"] <= td
            if mask.sum() < WARMUP:
                continue
            etf_data_at_date[code] = mask.sum() - 1

        # 盈利保护: 持仓从买入后最高点回撤>5% → 切货币
        profit_prot = False
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                price = float(row.iloc[0]["close"])
                if price > holding_peak:
                    holding_peak = price
                if holding_peak > 0 and (price - holding_peak) / holding_peak < -PROFIT_PROTECTION_DD:
                    profit_prot = True

        target = select_target_original(data, etf_data_at_date)
        if profit_prot and holding and holding != DEFENSE:
            target = DEFENSE

        # 交易执行
        if target != holding:
            if holding and holding in data:
                row = data[holding][data[holding]["trade_date"] == td]
                if not row.empty:
                    price = float(row.iloc[0]["close"])
                    cash += holding_shares * price * (1 - FEE - SLIPPAGE)
                    n_trades += 1
                    holding = None
                    holding_shares = 0
                    holding_peak = 0.0
            if target in data:
                row = data[target][data[target]["trade_date"] == td]
                if not row.empty:
                    price = float(row.iloc[0]["close"])
                    shares = int(cash * 0.99 / price / 100) * 100
                    if shares > 0:
                        cash -= shares * price * (1 + FEE + SLIPPAGE)
                        holding = target
                        holding_shares = shares
                        holding_peak = price
                        n_trades += 1

        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * float(row.iloc[0]["close"])
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = eq_df["equity"].iloc[-1] / initial_capital - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    span_years = max((eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    yearly = {}
    prev = initial_capital
    for year in sorted(eq_df["year"].unique()):
        ydf = eq_df[eq_df["year"] == year]
        if ydf.empty:
            continue
        end = ydf["equity"].iloc[-1]
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        yearly[int(year)] = {"return": end / prev - 1, "max_dd": dd}
        prev = end

    return {"total_return": total_return, "ann_return": ann_ret, "sharpe": sharpe,
            "max_drawdown": max_dd, "yearly": yearly, "n_trades": n_trades, "eq_df": eq_df}


def main() -> None:
    print("=" * 70)
    print("  七星ETF轮动 — 聚宽原版(任侠) 一比一复刻")
    print("  规则: 20/60/120动量 + 全过滤 + 盈利保护5% + 日频Top1")
    print("=" * 70)
    data = load_data()
    print(f"\n  数据: {len(data)}只ETF (与V3同一份缓存, 隔离规则差异)")
    print(f"\n  回测中 (日频)...")
    res = run_original(data)
    eq = res["eq_df"]

    print(f"\n  {'年份':<6} {'年初':>10} {'年末':>10} {'本地原版':>9} {'聚宽原版':>9} {'差异':>8}")
    print(f"  {'-' * 60}")
    prev = 100_000.0
    for year in sorted(eq["year"].unique()):
        ydf = eq[eq["year"] == year]
        if ydf.empty:
            continue
        end = ydf["equity"].iloc[-1]
        local = end / prev - 1
        orig = ORIGINAL.get(year)
        orig_s = f"{orig:+.1%}" if orig is not None else "—"
        diff_s = f"{local - orig:+.1%}" if orig is not None else "—"
        print(f"  {year:<6} {prev:>10,.0f} {end:>10,.0f} {local:>+9.1%} {orig_s:>9} {diff_s:>8}")
        prev = end

    final = eq["equity"].iloc[-1]
    print(f"  {'-' * 60}")
    print(f"\n  10万 → {final:,.0f} ({final / 100_000 - 1:+.1%})")
    print(f"  年化: {res['ann_return']:+.1%} | 夏普: {res['sharpe']:.2f} | "
          f"回撤: {res['max_drawdown']:.1%} | 交易: {res['n_trades']}次")

    # 持仓分布
    from collections import Counter
    counts = Counter(eq["holding"].tolist())
    total_days = len(eq)
    print(f"\n  持仓分布 (共{total_days}天):")
    for code, cnt in counts.most_common():
        name = ETF_POOL.get(code, "货币基金")
        print(f"    {name:<10} {cnt}天 ({cnt / total_days:.0%})")


if __name__ == "__main__":
    main()
