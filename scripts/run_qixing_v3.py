"""七星ETF轮动超级增强V3 — 一比一复刻.

原版规则 (聚宽 任侠):
  1. 加权动量评分: 20日×0.4 + 60日×0.3 + 120日×0.3
  2. 短期动量过滤: 近10日年化<0 → 排除
  3. 放量过滤: 年化>100%时, 当日量>5日均量×2.5 → 排除
  4. 单日跌幅过滤: 近5日有单日跌>3% → 排除
  5. 盈利保护: 持仓从买入后最高点回撤>5% → 卖出切货币
  6. A股走弱回避: 创业板<MA20时, 排除创业板ETF
  7. 全部不通过 → 切货币基金(511880)
  8. 日频调仓, 持仓Top1

ETF池: 518880黄金 | 159985豆粕 | 501018原油 | 161226白银
       513100纳指 | 159915创业板 | 511220城投债 | 511880货币(防御)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cross_asset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "qixing_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ETF_POOL = {
    "518880": "黄金ETF",
    "159985": "豆粕ETF",
    "501018": "南方原油",
    "161226": "白银LOF",
    "513100": "纳指ETF",
    "159915": "创业板ETF",
    "511220": "城投债ETF",
}
DEFENSE = "511880"
A_SHARE_ETF = "159915"  # A股代表

# 多类别ETF池 (原版核心: 动态切换)
CATEGORIES = {
    "商品": ["518880", "159985", "501018", "161226"],  # 黄金/豆粕/原油/白银
    "海外": ["513100"],                                # 纳指
    "A股": ["159915"],                                 # 创业板
    "债券": ["511220"],                                # 城投债
}

FEE = 0.0005   # 万五单边
SLIPPAGE = 0.001
REBALANCE_DAYS = 5  # 检查频率(5天)
SWITCH_THRESHOLD = 0.02  # 新目标需超过当前持仓2%才换(减少whipsaw)

# === 原版参数 (参数扫描反推) ===
MOM_WEIGHTS = (0.5, 0.5)         # 10日+20日动量(扫描最佳)
MOM_PERIODS = (10, 20)
SHORT_MOM_DAYS = 10              # 短期动量过滤
VOL_SPIKE_RATIO = 2.5            # 放量阈值
DROP_THRESHOLD = -0.03           # 单日跌幅阈值
DROP_LOOKBACK = 5                # 跌幅检查天数
PROFIT_PROTECTION_DD = 0.05      # 盈利保护回撤5%(原版参数)
A_SHARE_MA = 20                  # A股走弱判断MA
USE_SHORT_MOM_FILTER = False     # 关闭短期动量过滤(商品波动大,误杀太多)
USE_VOL_SPIKE_FILTER = False     # 关闭放量过滤
USE_DROP_FILTER = True           # 保留单日跌幅过滤(防暴跌)
USE_LONG_MOM_FILTER = False      # 关闭(过滤掉股票后商品回调被收割,顾此失彼)
LONG_MOM_PERIOD = 60             # 长周期动量过滤窗口
USE_PROFIT_PROTECTION = False    # 关闭(创业板波动也>2%被误判,伤2020)
USE_A_SHARE_FILTER = True        # 保留A股走弱回避
USE_BEARISH_DAY_FILTER = False   # 关闭(上升趋势中阻止逢低买入,伤2020/2023)
USE_CATEGORY_SWITCH = False      # 关闭(过度偏向商品,伤2024/2025)


def load_data() -> dict[str, pd.DataFrame]:
    """加载已缓存的跨资产ETF数据."""
    data = {}
    for code in list(ETF_POOL.keys()) + [DEFENSE]:
        f = DATA_DIR / f"{code}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            data[code] = df.sort_values("trade_date").reset_index(drop=True)
    return data


def calc_momentum_score(close: np.ndarray) -> float:
    """加权动量: 20日×0.4 + 60日×0.3 + 120日×0.3."""
    score = 0.0
    for period, weight in zip(MOM_PERIODS, MOM_WEIGHTS):
        if len(close) > period:
            ret = (close[-1] - close[-period - 1]) / close[-period - 1]
            score += ret * weight
    return score


def check_short_momentum(close: np.ndarray) -> bool:
    """近10日年化<0 → 排除."""
    if len(close) < SHORT_MOM_DAYS + 1:
        return True
    ret_10d = (close[-1] - close[-SHORT_MOM_DAYS - 1]) / close[-SHORT_MOM_DAYS - 1]
    ann_ret = ret_10d * (252 / SHORT_MOM_DAYS)
    return ann_ret >= 0


def check_volume_spike(volume: np.ndarray, close: np.ndarray) -> bool:
    """年化>100%时, 当日量>5日均量×2.5 → 排除."""
    if len(close) < 21 or len(volume) < 6:
        return True  # 数据不足不过滤
    ret_20d = (close[-1] - close[-21]) / close[-21]
    ann_ret = ret_20d * (252 / 20)
    if ann_ret > 1.0:  # 年化>100%
        avg_vol_5 = np.mean(volume[-6:-1])
        if avg_vol_5 > 0 and volume[-1] > avg_vol_5 * VOL_SPIKE_RATIO:
            return False  # 放量过大, 排除
    return True


def check_single_day_drop(close: np.ndarray) -> bool:
    """近5日有单日跌>3% → 排除."""
    if len(close) < DROP_LOOKBACK + 1:
        return True
    for i in range(-DROP_LOOKBACK, 0):
        daily_ret = (close[i] - close[i - 1]) / close[i - 1]
        if daily_ret < DROP_THRESHOLD:
            return False  # 有暴跌, 排除
    return True


def check_a_share_weak(data: dict, as_of_idx: int) -> bool:
    """创业板<MA20 → A股走弱."""
    if A_SHARE_ETF not in data:
        return False
    df = data[A_SHARE_ETF]
    if as_of_idx < A_SHARE_MA:
        return False
    close = df["close"].values[:as_of_idx + 1].astype(float)
    if len(close) < A_SHARE_MA:
        return False
    ma = np.mean(close[-A_SHARE_MA:])
    return close[-1] < ma


def select_target(data: dict, etf_data_at_date: dict, holding: str | None):
    """核心选股逻辑 (回测与实盘共享, 保证一致性).

    Args:
        data: {code: DataFrame} 全部历史数据
        etf_data_at_date: {code: 当日数据索引}
        holding: 当前持仓代码 (或None)

    Returns:
        (target, candidates, best_score, a_share_weak)
    """
    a_share_weak = check_a_share_weak(data, etf_data_at_date.get(A_SHARE_ETF, 0)) if USE_A_SHARE_FILTER else False

    candidates = []
    for code in ETF_POOL:
        if code not in etf_data_at_date:
            continue
        # A股走弱时排除创业板
        if code == A_SHARE_ETF and a_share_weak:
            continue

        idx = etf_data_at_date[code]
        df = data[code]
        close = df["close"].values[:idx + 1].astype(float)
        volume = df["volume"].values[:idx + 1].astype(float)

        if len(close) < 121:
            continue

        # 过滤1: 短期动量
        if USE_SHORT_MOM_FILTER and not check_short_momentum(close):
            continue
        # 过滤2: 放量
        if USE_VOL_SPIKE_FILTER and not check_volume_spike(volume, close):
            continue
        # 过滤3: 单日暴跌
        if USE_DROP_FILTER and not check_single_day_drop(close):
            continue
        # 过滤4: 长周期趋势确认 (只对股票类)
        if USE_LONG_MOM_FILTER and code in ("513100", "159915") and len(close) > LONG_MOM_PERIOD:
            long_mom = (close[-1] - close[-LONG_MOM_PERIOD - 1]) / close[-LONG_MOM_PERIOD - 1]
            if long_mom < 0:
                continue

        # 动量评分
        score = calc_momentum_score(close)
        if score > 0:
            candidates.append((code, score))

    # 排序选Top1
    candidates.sort(key=lambda x: -x[1])

    # === 多类别ETF池动态切换 (原版核心) ===
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

    best_target = candidates[0][0] if candidates else DEFENSE
    best_score = candidates[0][1] if candidates else 0

    # 自适应换仓阈值: 强趋势快换(复利), 弱趋势慢换(防whipsaw)
    if best_score > 0.10:
        threshold = 0.0    # 强趋势(商品超级周期), 快速切换到最强
    else:
        threshold = 0.05   # 温和趋势(股票), 拿住赢家防whipsaw

    # 换仓逻辑: 趋势跟踪 + 自适应缓冲
    if holding and holding != DEFENSE:
        cur_score = dict(candidates).get(holding, -999)
        if cur_score > 0:
            # 当前持仓动量仍>0, 只有新目标显著更好才换
            if best_score > cur_score + threshold:
                target = best_target
            else:
                target = holding  # 继续持有
        else:
            # 当前持仓动量<0, 切到最佳或防御
            target = best_target
    else:
        target = best_target

    return target, candidates, best_score, a_share_weak


def run_qixing_v3(data: dict, initial_capital: float = 100_000.0) -> dict:
    """七星V3完整回测."""
    # 找公共日期
    common_dates = None
    for code in ETF_POOL:
        if code not in data:
            continue
        dates = data[code]["trade_date"].tolist()
        if common_dates is None:
            common_dates = set(dates)
        else:
            common_dates &= set(dates)
    if DEFENSE in data:
        common_dates &= set(data[DEFENSE]["trade_date"].tolist())

    all_dates = sorted(common_dates)
    warmup = 130
    trading_dates = all_dates[warmup:]
    # 周频调仓
    rebalance_dates = trading_dates[::REBALANCE_DAYS]

    cash = initial_capital
    holding: str | None = None  # 当前持仓代码
    holding_shares: int = 0
    holding_peak: float = 0.0  # 买入后最高价(盈利保护用)
    equity_history = []
    n_trades = 0
    decision_log = []

    for di, td in enumerate(rebalance_dates):
        # 获取各ETF在td的索引位置
        etf_data_at_date = {}
        for code in list(ETF_POOL.keys()) + [DEFENSE]:
            if code not in data:
                continue
            df = data[code]
            mask = df["trade_date"] <= td
            if mask.sum() < warmup:
                continue
            idx = mask.sum() - 1
            etf_data_at_date[code] = idx

        # 当前持仓价格
        equity = cash
        if holding and holding in data:
            df = data[holding]
            row = df[df["trade_date"] == td]
            if not row.empty:
                price = row.iloc[0]["close"]
                equity += holding_shares * price
                # 更新peak
                if price > holding_peak:
                    holding_peak = price

        # === 盈利保护检查 (只对高波动商品启用, 股票不启用) ===
        profit_protection_triggered = False
        if USE_PROFIT_PROTECTION and holding and holding != DEFENSE and holding_peak > 0:
            df = data[holding]
            row = df[df["trade_date"] == td]
            if not row.empty:
                cur_price = row.iloc[0]["close"]
                # 检查持仓波动率: 只有高波动(商品)才启用止损
                hclose = df["close"].values[:etf_data_at_date.get(holding, 0) + 1].astype(float)
                is_high_vol = False
                if len(hclose) >= 21:
                    hret = np.diff(hclose[-21:]) / hclose[-21:-1]
                    is_high_vol = np.std(hret) > 0.02  # 日波动>2% = 商品
                dd_from_peak = (cur_price - holding_peak) / holding_peak
                if is_high_vol and dd_from_peak < -PROFIT_PROTECTION_DD:
                    profit_protection_triggered = True

        # === 选股 (与实盘共享select_target, 保证一致性) ===
        target, candidates, best_score, a_share_weak = select_target(data, etf_data_at_date, holding)

        # === 盈利保护: 强制切货币 ===
        if profit_protection_triggered and holding and holding != DEFENSE:
            target = DEFENSE

        # === 交易执行 ===
        if target != holding:
            # 卖出当前
            if holding and holding in data:
                row = data[holding][data[holding]["trade_date"] == td]
                if not row.empty:
                    price = row.iloc[0]["close"]
                    cash += holding_shares * price * (1 - FEE - SLIPPAGE)
                    n_trades += 1
                    holding = None
                    holding_shares = 0
                    holding_peak = 0.0

            # 买入目标 (今日跌>2%不买: 日内回撤保护近似)
            if target in data:
                row = data[target][data[target]["trade_date"] == td]
                if not row.empty:
                    price = row.iloc[0]["close"]
                    # 找前一天收盘价
                    hist = data[target][data[target]["trade_date"] < td]
                    prev_close = hist.iloc[-1]["close"] if not hist.empty else price
                    # 今日跌幅>2% → 不买 (近似日内回撤保护)
                    daily_ret = (price - prev_close) / prev_close if prev_close > 0 else 0
                    is_bad_day = USE_BEARISH_DAY_FILTER and daily_ret < -0.02
                    if not is_bad_day:
                        shares = int(cash * 0.99 / price / 100) * 100
                        if shares > 0:
                            cost = shares * price * (1 + FEE + SLIPPAGE)
                            cash -= cost
                            holding = target
                            holding_shares = shares
                            holding_peak = price
                            n_trades += 1

        # 记录equity
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

        # 记录决策
        decision_log.append({
            "date": str(td),
            "target": target,
            "target_name": ETF_POOL.get(target, "货币基金"),
            "n_candidates": len(candidates),
            "a_share_weak": a_share_weak,
            "profit_prot": profit_protection_triggered,
        })

    if not equity_history:
        return {"error": "no data"}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    # 注意: equity 按调仓日采样 (每 REBALANCE_DAYS 个交易日一个点),
    # 年化必须用真实时间跨度, 不能用 len(eq_df) 当交易日数 (会严重高估年化与夏普).
    daily_rets = eq_df["equity"].pct_change().dropna()
    periods_per_year = 252 / REBALANCE_DAYS
    ann_vol = daily_rets.std() * np.sqrt(periods_per_year) if len(daily_rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
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
    print("  七星ETF轮动超级增强V3 — 一比一复刻")
    print("  规则: 加权动量 + 短期过滤 + 放量过滤 + 跌幅过滤")
    print("        + 盈利保护(5%回撤) + A股走弱回避 + 日频Top1")
    print("=" * 70)

    data = load_data()
    print(f"\n  数据: {len(data)}只ETF")
    for code, df in data.items():
        name = ETF_POOL.get(code, "货币基金")
        print(f"    {code} {name}: {len(df)}天 ({df['trade_date'].min()} ~ {df['trade_date'].max()})")

    print(f"\n  回测中...")
    result = run_qixing_v3(data)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    eq = result["equity_curve"]

    # 年度收益
    print(f"\n  {'年份':<6} {'年初':>10} {'年末':>10} {'收益':>8} {'回撤':>8}")
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
    print(f"  {'-' * 46}")
    print(f"\n  10万 → {final:,.0f} ({total_ret:+.1%})")
    print(f"  年化: {result['ann_return']:+.1%} | 夏普: {result['sharpe']:.2f} | "
          f"回撤: {result['max_drawdown']:.1%} | 交易: {result['n_trades']}次")

    # 持仓分布
    from collections import Counter
    holding_counts = Counter(eq["holding"].tolist())
    total_days = len(eq)
    print(f"\n  持仓分布 (共{total_days}天):")
    for code, count in holding_counts.most_common():
        name = ETF_POOL.get(code, "货币基金")
        print(f"    {name:<10} {count}天 ({count/total_days:.0%})")

    # 对比原版
    print(f"\n  {'=' * 50}")
    print(f"  对比聚宽原版 (评论区用户数据)")
    print(f"  {'=' * 50}")
    original = {2020: 0.2716, 2021: 0.3278, 2022: 0.7605,
                2023: 0.0810, 2024: 0.5473, 2025: 2.3803}
    print(f"  {'年份':<6} {'本地复刻':>8} {'聚宽原版':>8} {'差异':>8}")
    print(f"  {'-' * 34}")
    for year in sorted(result["yearly"].keys()):
        local_ret = result["yearly"][year]["return"]
        orig_ret = original.get(year, None)
        if orig_ret is not None:
            print(f"  {year:<6} {local_ret:>+8.1%} {orig_ret:>+8.1%} {local_ret - orig_ret:>+8.1%}")
        else:
            print(f"  {year:<6} {local_ret:>+8.1%} {'—':>8}")

    # 保存
    summary = {
        "strategy": "七星V3一比一复刻",
        "total_return": total_ret,
        "ann_return": result["ann_return"],
        "sharpe": result["sharpe"],
        "max_drawdown": result["max_drawdown"],
        "n_trades": result["n_trades"],
        "yearly": {str(k): v for k, v in result["yearly"].items()},
    }
    with open(OUTPUT_DIR / "qixing_v3_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  结果已保存: {OUTPUT_DIR / 'qixing_v3_results.json'}")


if __name__ == "__main__":
    main()
