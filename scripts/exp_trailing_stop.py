"""实验: 分层追踪止损对七星V3策略的影响.

对比:
  A) 原版V3 (无止损, 仅周频调仓)
  B) V3 + 分层追踪止损 (每日检查)
  C) V3 + 分层追踪止损 + MA退出
  D) V3 + 分层追踪止损 + MA退出 + 冷却期

结论: 验证止损是否改善最大回撤/夏普, 同时不过度损害收益.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_qixing_v3 as rq  # noqa: E402

# === 止损参数 ===
TRAILING_STOP = {
    # 商品类: 波动大, 给更宽的空间
    "518880": 0.15,   # 黄金
    "159985": 0.15,   # 豆粕
    "501018": 0.15,   # 原油
    "161226": 0.15,   # 白银
    # 股票类: 中等波动
    "513100": 0.12,   # 纳指
    "159915": 0.12,   # 创业板
    # 债券类: 极低波动
    "511220": 0.05,   # 城投债
}
DEFAULT_STOP = 0.12
MA_EXIT_PERIOD = 20       # MA退出周期
COOLDOWN_DAYS = 10        # 止损后冷却交易日数


def run_with_stop_loss(
    data: dict,
    use_trailing_stop: bool = True,
    use_ma_exit: bool = True,
    use_cooldown: bool = True,
    use_equity_breaker: bool = False,
    equity_breaker_pct: float = 0.10,
    equity_breaker_cooldown: int = 10,
    initial_capital: float = 100_000.0,
) -> dict:
    """V3回测 + 每日止损检查.

    与原版的核心区别: 遍历所有交易日(而非仅调仓日), 每日检查止损.
    equity_breaker: 账户级熔断, 从高点回撤>X% → 全切货币+冷却N天.
    """
    # 找公共日期
    common_dates = None
    for code in rq.ETF_POOL:
        if code not in data:
            continue
        dates = data[code]["trade_date"].tolist()
        if common_dates is None:
            common_dates = set(dates)
        else:
            common_dates &= set(dates)
    if rq.DEFENSE in data:
        common_dates &= set(data[rq.DEFENSE]["trade_date"].tolist())

    all_dates = sorted(common_dates)
    warmup = 130
    trading_dates = all_dates[warmup:]

    cash = initial_capital
    holding: str | None = None
    holding_shares: int = 0
    holding_peak: float = 0.0
    equity_history = []
    n_trades = 0
    stop_events = []       # 记录止损事件
    cooldown_until = {}    # {code: 冷却截止日期}
    equity_peak = initial_capital   # 账户级高点
    breaker_until = -1              # 熔断冷却截止的日索引

    # 调仓日集合
    rebalance_set = set(trading_dates[::rq.REBALANCE_DAYS])
    rebalance_counter = 0

    for di, td in enumerate(trading_dates):
        # === 每日: 更新持仓价格 & 检查止损 ===
        cur_price = None
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                cur_price = float(row.iloc[0]["close"])
                if cur_price > holding_peak:
                    holding_peak = cur_price

        # 更新账户高点
        cur_equity = cash
        if holding and cur_price:
            cur_equity += holding_shares * cur_price
        if cur_equity > equity_peak:
            equity_peak = cur_equity

        # --- 账户级熔断检查 ---
        breaker_triggered = False
        if use_equity_breaker and di >= breaker_until and holding and holding != rq.DEFENSE:
            acct_dd = (cur_equity - equity_peak) / equity_peak
            if acct_dd < -equity_breaker_pct:
                breaker_triggered = True
                stop_events.append({
                    "date": str(td), "code": holding,
                    "name": rq.ETF_POOL.get(holding, ""),
                    "dd": acct_dd, "peak": equity_peak,
                    "price": cur_price or 0, "type": "breaker",
                })
                # 立即卖出切货币
                if cur_price:
                    cash += holding_shares * cur_price * (1 - rq.FEE - rq.SLIPPAGE)
                    n_trades += 1
                holding = None
                holding_shares = 0
                holding_peak = 0.0
                breaker_until = di + equity_breaker_cooldown

        # 熔断冷却期内: 跳过所有交易逻辑
        if di < breaker_until:
            equity = cash
            if holding and holding in data:
                row = data[holding][data[holding]["trade_date"] == td]
                if not row.empty:
                    equity += holding_shares * float(row.iloc[0]["close"])
            equity_history.append({"trade_date": td, "equity": equity,
                                   "holding": holding or rq.DEFENSE})
            continue

        # 熔断冷却期刚结束: 重置高点为当前值 (给策略一个新鲜起点)
        if use_equity_breaker and breaker_until >= 0 and di == breaker_until:
            equity_peak = cur_equity
            breaker_until = -1

        # --- 追踪止损检查 ---
        stop_triggered = False
        if use_trailing_stop and holding and holding != rq.DEFENSE and cur_price and holding_peak > 0:
            stop_pct = TRAILING_STOP.get(holding, DEFAULT_STOP)
            dd_from_peak = (cur_price - holding_peak) / holding_peak
            if dd_from_peak < -stop_pct:
                stop_triggered = True
                stop_events.append({
                    "date": str(td), "code": holding,
                    "name": rq.ETF_POOL.get(holding, ""),
                    "dd": dd_from_peak, "peak": holding_peak,
                    "price": cur_price, "type": "trailing",
                })

        # --- MA退出检查 ---
        ma_triggered = False
        if use_ma_exit and not stop_triggered and holding and holding != rq.DEFENSE and cur_price:
            df = data[holding]
            hist = df[df["trade_date"] <= td]["close"].astype(float)
            if len(hist) >= MA_EXIT_PERIOD:
                ma = hist.iloc[-MA_EXIT_PERIOD:].mean()
                if cur_price < ma:
                    ma_triggered = True
                    stop_events.append({
                        "date": str(td), "code": holding,
                        "name": rq.ETF_POOL.get(holding, ""),
                        "dd": (cur_price - holding_peak) / holding_peak if holding_peak > 0 else 0,
                        "peak": holding_peak, "price": cur_price, "type": "ma_exit",
                    })

        # --- 执行止损/MA退出: 卖出切货币 ---
        if (stop_triggered or ma_triggered) and holding and holding_shares > 0:
            cash += holding_shares * cur_price * (1 - rq.FEE - rq.SLIPPAGE)
            n_trades += 1
            # 冷却期
            if use_cooldown:
                cooldown_idx = di + COOLDOWN_DAYS
                cooldown_until[holding] = cooldown_idx
            holding = None
            holding_shares = 0
            holding_peak = 0.0

        # === 调仓日: 正常选股逻辑 ===
        if td in rebalance_set:
            # 构建索引
            etf_data_at_date = {}
            for code in list(rq.ETF_POOL.keys()) + [rq.DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() < warmup:
                    continue
                etf_data_at_date[code] = mask.sum() - 1

            # 选股
            target, candidates, best_score, a_share_weak = rq.select_target(
                data, etf_data_at_date, holding)

            # 冷却期过滤: 被止损的ETF在冷却期内不买
            if use_cooldown and target in cooldown_until:
                if di < cooldown_until[target]:
                    # 目标被冷却, 选次优
                    for c, s in candidates:
                        if c not in cooldown_until or di >= cooldown_until[c]:
                            target = c
                            break
                    else:
                        target = rq.DEFENSE

            # 交易执行
            if target != holding:
                # 卖出当前
                if holding and holding in data:
                    row = data[holding][data[holding]["trade_date"] == td]
                    if not row.empty:
                        price = float(row.iloc[0]["close"])
                        cash += holding_shares * price * (1 - rq.FEE - rq.SLIPPAGE)
                        n_trades += 1
                        holding = None
                        holding_shares = 0
                        holding_peak = 0.0

                # 买入目标
                if target in data:
                    row = data[target][data[target]["trade_date"] == td]
                    if not row.empty:
                        price = float(row.iloc[0]["close"])
                        shares = int(cash * 0.99 / price / 100) * 100
                        if shares > 0:
                            cost = shares * price * (1 + rq.FEE + rq.SLIPPAGE)
                            cash -= cost
                            holding = target
                            holding_shares = shares
                            holding_peak = price
                            n_trades += 1

        # === 记录每日equity ===
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * float(row.iloc[0]["close"])
        equity_history.append({"trade_date": td, "equity": equity,
                               "holding": holding or rq.DEFENSE})

    if not equity_history:
        return {"error": "no data"}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = eq_df["equity"].iloc[-1] / initial_capital - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    # 年度
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
        yearly[int(year)] = {"return": round(yr, 4), "max_dd": round(dd, 4)}
        prev_val = end_val

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "total_return": round(total_return, 4),
        "ann_return": round(ann_ret, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 3),
        "ann_vol": round(ann_vol, 4),
        "yearly": yearly,
        "n_trades": n_trades,
        "n_stops": len(stop_events),
        "stop_events": stop_events,
        "equity_curve": eq_df,
    }


def run_baseline(data: dict, initial_capital: float = 100_000.0) -> dict:
    """原版V3回测 (无止损, 仅调仓日采样) — 作为基准."""
    result = rq.run_qixing_v3(data, initial_capital)
    if "error" in result:
        return result
    # 补充 calmar
    ann_ret = result["ann_return"]
    max_dd = result["max_drawdown"]
    result["calmar"] = round(ann_ret / abs(max_dd), 3) if max_dd != 0 else 0.0
    result["n_stops"] = 0
    result["stop_events"] = []
    return result


def run_no_stop_daily(data: dict, initial_capital: float = 100_000.0) -> dict:
    """无止损但每日采样 — 公平基准 (与止损版同采样频率)."""
    return run_with_stop_loss(data, use_trailing_stop=False,
                              use_ma_exit=False, use_cooldown=False,
                              initial_capital=initial_capital)


def print_comparison(baseline: dict, variants: dict[str, dict]) -> None:
    """打印对比表."""
    print("\n" + "=" * 80)
    print("  分层追踪止损实验 — 对比结果")
    print("=" * 80)

    header = f"  {'指标':<12} {'原版V3':>10}"
    for name in variants:
        header += f" {name:>12}"
    print(header)
    print("  " + "-" * (14 + 10 + 13 * len(variants)))

    metrics = [
        ("总收益", "total_return", "{:+.1%}"),
        ("年化收益", "ann_return", "{:+.1%}"),
        ("夏普比率", "sharpe", "{:.2f}"),
        ("最大回撤", "max_drawdown", "{:.1%}"),
        ("Calmar", "calmar", "{:.2f}"),
        ("交易次数", "n_trades", "{:d}"),
        ("止损次数", "n_stops", "{:d}"),
    ]

    for label, key, fmt in metrics:
        row = f"  {label:<12}"
        val = baseline.get(key, 0)
        row += f" {fmt.format(val):>10}"
        for name, res in variants.items():
            val = res.get(key, 0)
            row += f" {fmt.format(val):>12}"
        print(row)

    # 年度对比
    print(f"\n  {'年度收益对比':}")
    print(f"  {'年份':<6} {'原版V3':>8}", end="")
    for name in variants:
        print(f" {name:>12}", end="")
    print()
    print("  " + "-" * (6 + 8 + 13 * len(variants)))

    all_years = sorted(set(baseline.get("yearly", {}).keys()) |
                       set().union(*(set(r.get("yearly", {}).keys()) for r in variants.values())))
    for year in all_years:
        row = f"  {year:<6}"
        by = baseline.get("yearly", {}).get(year, {})
        row += f" {by.get('return', 0):>+8.1%}"
        for name, res in variants.items():
            vy = res.get("yearly", {}).get(year, {})
            row += f" {vy.get('return', 0):>+12.1%}"
        print(row)

    # 年度最大回撤
    print(f"\n  {'年度最大回撤':}")
    print(f"  {'年份':<6} {'原版V3':>8}", end="")
    for name in variants:
        print(f" {name:>12}", end="")
    print()
    print("  " + "-" * (6 + 8 + 13 * len(variants)))
    for year in all_years:
        row = f"  {year:<6}"
        by = baseline.get("yearly", {}).get(year, {})
        row += f" {by.get('max_dd', 0):>8.1%}"
        for name, res in variants.items():
            vy = res.get("yearly", {}).get(year, {})
            row += f" {vy.get('max_dd', 0):>12.1%}"
        print(row)


def print_stop_events(events: list, title: str) -> None:
    """打印止损事件明细."""
    if not events:
        print(f"\n  [{title}] 无止损事件")
        return
    print(f"\n  [{title}] 止损事件 ({len(events)}次):")
    print(f"  {'日期':<12} {'ETF':<10} {'类型':<10} {'回撤':>8} {'峰值':>8} {'触发价':>8}")
    print("  " + "-" * 60)
    for e in events:
        print(f"  {e['date']:<12} {e['name']:<10} {e['type']:<10} "
              f"{e['dd']:>+8.1%} {e['peak']:>8.3f} {e['price']:>8.3f}")


def main():
    print("  加载数据...")
    data = rq.load_data()
    if not data:
        print("  ❌ 无数据")
        return
    print(f"  数据: {len(data)}只ETF")

    # A) 原版基准 (调仓日采样)
    print("\n  [A] 运行原版V3 (无止损, 调仓日采样)...")
    baseline = run_baseline(data)
    if "error" in baseline:
        print(f"  ERROR: {baseline['error']}")
        return

    # A2) 公平基准: 无止损但每日采样
    print("  [A2] 运行无止损 (每日采样, 公平对比)...")
    res_a2 = run_no_stop_daily(data)

    # B) 仅追踪止损
    print("  [B] 运行 V3 + 分层追踪止损...")
    res_b = run_with_stop_loss(data, use_trailing_stop=True,
                               use_ma_exit=False, use_cooldown=False)

    # E) 追踪止损 + 账户熔断10%
    print("  [E] 运行 V3 + 追踪止损 + 账户熔断10%...")
    res_e = run_with_stop_loss(data, use_trailing_stop=True,
                               use_ma_exit=False, use_cooldown=False,
                               use_equity_breaker=True,
                               equity_breaker_pct=0.10,
                               equity_breaker_cooldown=10)

    # F) 追踪止损 + 账户熔断15%
    print("  [F] 运行 V3 + 追踪止损 + 账户熔断15%...")
    res_f = run_with_stop_loss(data, use_trailing_stop=True,
                               use_ma_exit=False, use_cooldown=False,
                               use_equity_breaker=True,
                               equity_breaker_pct=0.15,
                               equity_breaker_cooldown=10)

    # G) 追踪止损 + 账户熔断10% + 冷却20天
    print("  [G] 运行 V3 + 追踪止损 + 账户熔断10% + 冷協20天...")
    res_g = run_with_stop_loss(data, use_trailing_stop=True,
                               use_ma_exit=False, use_cooldown=False,
                               use_equity_breaker=True,
                               equity_breaker_pct=0.10,
                               equity_breaker_cooldown=20)

    variants = {
        "A2:无止损": res_a2,
        "B:追踪止损": res_b,
        "E:熔断10%": res_e,
        "F:熔断15%": res_f,
        "G:熔断10%+冷20": res_g,
    }

    print_comparison(baseline, variants)

    # 止损事件明细
    print_stop_events(res_e.get("stop_events", []), "E:熔断10%+冷却10天")
    print_stop_events(res_b.get("stop_events", []), "B:仅追踪止损")

    # 结论: 以A2(每日采样无止损)为公平基准
    print("\n" + "=" * 80)
    print("  结论 (以 A2:无止损每日采样 为公平基准)")
    print("=" * 80)
    base_dd = abs(res_a2["max_drawdown"])
    for name, res in variants.items():
        if name.startswith("A2"):
            continue
        dd_improve = (base_dd - abs(res["max_drawdown"])) / base_dd * 100
        ret_cost = (res_a2["ann_return"] - res["ann_return"]) * 100
        sharpe_diff = res["sharpe"] - res_a2["sharpe"]
        print(f"  {name}: 回撤改善 {dd_improve:+.0f}% | "
              f"年化代价 {ret_cost:+.1f}pp | 夏普变化 {sharpe_diff:+.2f}")


if __name__ == "__main__":
    main()
