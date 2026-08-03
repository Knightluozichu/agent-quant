"""七星ETF轮动超级增强V3 — 一比一复刻.

原版规则 (聚宽 任侠), 经参数扫描+稳健性验证后的实际配置:
  1. 加权动量评分: 10日×0.5 + 20日×0.5 (扫描最佳, 非原版20/60/120)
  2. 短期动量过滤: [关闭] 商品波动大, 误杀太多
  3. 放量过滤: [关闭]
  4. 单日跌幅过滤: 近5日有单日跌>3% → 排除 (保留, 防暴跌)
  5. 盈利保护: [关闭] 创业板波动也>2%被误判, 伤2020
  6. A股走弱回避: 创业板<MA15时, 排除创业板ETF (保守优化: MA20→MA15)
  7. 全部不通过 → 切货币基金(511880)
  8. 每5个交易日调仓, 持仓Top1 (非日频调仓)
  9. 自适应换仓阈值: 强趋势快换(复利), 弱趋势慢换(防whipsaw)

ETF池: 518880黄金 | 159985豆粕 | 501018原油 | 161226白银
       513100纳指 | 159915创业板 | 511220城投债 | 511880货币(防御)
"""

from __future__ import annotations

import hashlib
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
SWITCH_THRESHOLD = 0.02  # 参考: 原版换仓阈值 (实际使用自适应阈值, 见 select_target)

# === 原版参数 (参数扫描反推) ===
MOM_WEIGHTS = (0.5, 0.5)         # 10日+20日动量(扫描最佳)
MOM_PERIODS = (10, 20)
SHORT_MOM_DAYS = 10              # 短期动量过滤
VOL_SPIKE_RATIO = 2.5            # 放量阈值
DROP_THRESHOLD = -0.03           # 单日跌幅阈值
DROP_LOOKBACK = 5                # 跌幅检查天数
PROFIT_PROTECTION_DD = 0.05      # 盈利保护回撤5%(原版参数)
A_SHARE_MA = 15                  # A股走弱判断MA(保守优化: 更早回避A股走弱)
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
    for code in [*list(ETF_POOL.keys()), DEFENSE]:
        f = DATA_DIR / f"{code}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            data[code] = df.sort_values("trade_date").reset_index(drop=True)
    return data


def calc_momentum_score(close: np.ndarray) -> float:
    """加权动量: 20日×0.4 + 60日×0.3 + 120日×0.3."""
    score = 0.0
    for period, weight in zip(MOM_PERIODS, MOM_WEIGHTS, strict=False):
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
    return bool(ann_ret >= 0)


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
    return bool(close[-1] < ma)


def select_target(data: dict, etf_data_at_date: dict, holding: str | None):
    """核心选股逻辑 (回测与实盘共享, 保证一致性).

    Args:
        data: {code: DataFrame} 全部历史数据
        etf_data_at_date: {code: 当日数据索引}
        holding: 当前持仓代码 (或None)

    Returns:
        (target, candidates, best_score, a_share_weak)
    """
    a_share_weak = (
        check_a_share_weak(data, etf_data_at_date.get(A_SHARE_ETF, 0))
        if USE_A_SHARE_FILTER
        else False
    )

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
            best_cat = max(cat_scores, key=lambda k: cat_scores[k])
            best_cat_codes = set(CATEGORIES[best_cat])
            cat_candidates = [(c, s) for c, s in candidates if c in best_cat_codes]
            if cat_candidates:
                candidates = cat_candidates

    best_target = candidates[0][0] if candidates else DEFENSE
    best_score = candidates[0][1] if candidates else 0

    # 自适应换仓阈值: 强趋势快换(复利), 弱趋势慢换(防whipsaw)
    threshold = 0.0 if best_score > 0.10 else 0.05

    # 换仓逻辑: 趋势跟踪 + 自适应缓冲
    if holding and holding != DEFENSE:
        cur_score = dict(candidates).get(holding, -999)
        if cur_score > 0:
            # 当前持仓动量仍>0, 只有新目标显著更好才换
            target = best_target if best_score > cur_score + threshold else holding  # 继续持有
        else:
            # 当前持仓动量<0, 切到最佳或防御
            target = best_target
    else:
        target = best_target

    return target, candidates, best_score, a_share_weak


def run_qixing_v3(data: dict, initial_capital: float = 100_000.0) -> dict:
    """七星V3完整回测."""
    # 找公共日期
    common_dates: set = set()
    for code in ETF_POOL:
        if code not in data:
            continue
        dates = data[code]["trade_date"].tolist()
        if not common_dates:
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

    for _di, td in enumerate(rebalance_dates):
        # 获取各ETF在td的索引位置
        etf_data_at_date = {}
        for code in [*list(ETF_POOL.keys()), DEFENSE]:
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
        target, candidates, _best_score, a_share_weak = select_target(
            data, etf_data_at_date, holding
        )

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


# --------------------------------------------------------------------------- #
# R1: 无未来函数回测 (T日收盘信号 → T+1开盘成交)
# --------------------------------------------------------------------------- #
def _compute_param_hash() -> str:
    """计算策略参数 hash (审计追溯用)."""
    params = {
        "MOM_PERIODS": MOM_PERIODS, "MOM_WEIGHTS": MOM_WEIGHTS,
        "DROP_LOOKBACK": DROP_LOOKBACK, "DROP_THRESHOLD": DROP_THRESHOLD,
        "A_SHARE_MA": A_SHARE_MA, "REBALANCE_DAYS": REBALANCE_DAYS,
        "FEE": FEE, "SLIPPAGE": SLIPPAGE,
        "USE_SHORT_MOM_FILTER": USE_SHORT_MOM_FILTER,
        "USE_VOL_SPIKE_FILTER": USE_VOL_SPIKE_FILTER,
        "USE_DROP_FILTER": USE_DROP_FILTER,
        "USE_LONG_MOM_FILTER": USE_LONG_MOM_FILTER,
        "USE_PROFIT_PROTECTION": USE_PROFIT_PROTECTION,
        "USE_A_SHARE_FILTER": USE_A_SHARE_FILTER,
        "USE_BEARISH_DAY_FILTER": USE_BEARISH_DAY_FILTER,
        "USE_CATEGORY_SWITCH": USE_CATEGORY_SWITCH,
    }
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _compute_data_hash(data: dict) -> str:
    """计算数据 hash (审计追溯用)."""
    parts = []
    for code in sorted(data.keys()):
        df = data[code]
        parts.append(f"{code}:{len(df)}:{df['trade_date'].min()}:{df['trade_date'].max()}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _check_tradable(data: dict, code: str, td) -> tuple[bool, str]:
    """检查 code 在 td 日是否可交易 (非停牌/有开盘价/未涨跌停).

    Returns:
        (can_trade, reason) - can_trade=True 表示可交易
    """
    if code not in data:
        return (False, f"{code} 不在数据中")
    df = data[code]
    row = df[df["trade_date"] == td]
    if row.empty:
        return (False, f"{code} 在 {td} 停牌或无数据")
    open_price = float(row.iloc[0]["open"])
    if open_price <= 0:
        return (False, f"{code} 在 {td} 无开盘价")
    # 涨跌停检查: 与前一日收盘价对比
    hist = data[code][data[code]["trade_date"] < td]
    if not hist.empty:
        prev_close = float(hist.iloc[-1]["close"])
        if prev_close > 0:
            change = abs(open_price / prev_close - 1)
            if change >= 0.099:  # ETF 涨跌停 10% (近似)
                return (False, f"{code} 在 {td} 开盘涨跌停 ({change:+.1%})")
    return (True, "")


def run_qixing_v3_no_lookahead(
    data: dict, initial_capital: float = 100_000.0, cost_multiplier: float = 1.0
) -> dict:
    """无未来函数回测: T日收盘信号 → T+1开盘成交.

    修复 lookahead bias:
      - 信号生成: T日收盘数据
      - 信号进入 pending execution queue (带唯一 ID)
      - 交易执行: T+1日开盘价
      - T+1停牌/无开盘价/涨跌停/数据缺失 → 不成交
      - 卖出失败 → 不继续买入
      - 最后交易日未执行信号不计入
      - 净值曲线: 每日采样 (含非调仓日)

    Args:
        data: {code: DataFrame} 全部历史数据
        initial_capital: 初始资金
        cost_multiplier: 成本倍数 (1=基础, 2=2倍, 3=3倍), 用于压力测试
    """
    # 找公共日期
    common_dates: set = set()
    for code in ETF_POOL:
        if code not in data:
            continue
        dates = data[code]["trade_date"].tolist()
        if not common_dates:
            common_dates = set(dates)
        else:
            common_dates &= set(dates)
    if DEFENSE in data:
        common_dates &= set(data[DEFENSE]["trade_date"].tolist())

    all_dates = sorted(common_dates)
    warmup = 130
    trading_dates = all_dates[warmup:]
    rebalance_dates = trading_dates[::REBALANCE_DAYS]
    rebalance_set = set(rebalance_dates)

    # 审计 hash
    param_hash = _compute_param_hash()
    data_hash = _compute_data_hash(data)

    # 成本 (支持压力测试倍数)
    fee = FEE * cost_multiplier
    slippage = SLIPPAGE * cost_multiplier

    # 状态
    cash = initial_capital
    holding: str | None = None
    holding_shares = 0
    holding_peak = 0.0
    pending_signal: dict | None = None  # T日信号, 等待T+1执行

    equity_history: list[dict] = []
    trade_log: list[dict] = []
    decision_log: list[dict] = []
    signal_counter = 0

    for i, td in enumerate(trading_dates):
        # === 1. 执行昨日 pending 信号 (T-1 信号 → T 开盘成交) ===
        if pending_signal:
            sig_id = pending_signal["signal_id"]
            sig_date = pending_signal["signal_date"]
            target = pending_signal["target"]

            if target != holding:
                sell_ok = True
                # 卖出当前持仓
                if holding and holding in data:
                    can_sell, reason = _check_tradable(data, holding, td)
                    if not can_sell:
                        sell_ok = False
                        trade_log.append({
                            "signal_id": sig_id, "signal_date": str(sig_date),
                            "execution_date": str(td), "action": "sell",
                            "code": holding, "status": "cancelled",
                            "reason": f"卖出失败: {reason}",
                            "data_hash": data_hash, "param_hash": param_hash,
                        })
                    else:
                        row = data[holding][data[holding]["trade_date"] == td]
                        exec_price = float(row.iloc[0]["open"])
                        amount = holding_shares * exec_price * (1 - fee - slippage)
                        cash += amount
                        trade_log.append({
                            "signal_id": sig_id, "signal_date": str(sig_date),
                            "execution_date": str(td), "action": "sell",
                            "code": holding, "shares": holding_shares,
                            "price": exec_price, "amount": round(amount, 2),
                            "status": "executed", "reason": "",
                            "data_hash": data_hash, "param_hash": param_hash,
                        })
                        holding = None
                        holding_shares = 0
                        holding_peak = 0.0

                # 买入目标 (卖出失败则不买入)
                if sell_ok and target and target in data:
                    can_buy, reason = _check_tradable(data, target, td)
                    if not can_buy:
                        trade_log.append({
                            "signal_id": sig_id, "signal_date": str(sig_date),
                            "execution_date": str(td), "action": "buy",
                            "code": target, "status": "cancelled",
                            "reason": f"买入失败: {reason}",
                            "data_hash": data_hash, "param_hash": param_hash,
                        })
                    else:
                        row = data[target][data[target]["trade_date"] == td]
                        exec_price = float(row.iloc[0]["open"])
                        shares = int(cash * 0.99 / exec_price / 100) * 100
                        if shares > 0:
                            cost = shares * exec_price * (1 + fee + slippage)
                            cash -= cost
                            holding = target
                            holding_shares = shares
                            holding_peak = exec_price
                            trade_log.append({
                                "signal_id": sig_id, "signal_date": str(sig_date),
                                "execution_date": str(td), "action": "buy",
                                "code": target, "shares": shares,
                                "price": exec_price, "amount": round(cost, 2),
                                "status": "executed", "reason": "",
                                "data_hash": data_hash, "param_hash": param_hash,
                            })
                        else:
                            trade_log.append({
                                "signal_id": sig_id, "signal_date": str(sig_date),
                                "execution_date": str(td), "action": "buy",
                                "code": target, "status": "cancelled",
                                "reason": "买入数量为0 (现金不足)",
                                "data_hash": data_hash, "param_hash": param_hash,
                            })

            pending_signal = None

        # === 2. 记录每日净值 (含非调仓日) ===
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                price = float(row.iloc[0]["close"])
                equity += holding_shares * price
                if price > holding_peak:
                    holding_peak = price
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

        # === 3. 调仓日: 生成信号 (用 T 日收盘数据) ===
        if td in rebalance_set:
            etf_data_at_date = {}
            for code in [*list(ETF_POOL.keys()), DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() < warmup:
                    continue
                etf_data_at_date[code] = mask.sum() - 1

            # 盈利保护检查
            profit_protection_triggered = False
            if USE_PROFIT_PROTECTION and holding and holding != DEFENSE and holding_peak > 0:
                df = data[holding]
                row = df[df["trade_date"] == td]
                if not row.empty:
                    cur_price = float(row.iloc[0]["close"])
                    hclose = df["close"].values[:etf_data_at_date.get(holding, 0) + 1].astype(float)
                    is_high_vol = False
                    if len(hclose) >= 21:
                        hret = np.diff(hclose[-21:]) / hclose[-21:-1]
                        is_high_vol = np.std(hret) > 0.02
                    dd_from_peak = (cur_price - holding_peak) / holding_peak
                    if is_high_vol and dd_from_peak < -PROFIT_PROTECTION_DD:
                        profit_protection_triggered = True

            # 选股 (与实盘共享 select_target)
            target, candidates, _best_score, a_share_weak = select_target(
                data, etf_data_at_date, holding
            )

            if profit_protection_triggered and holding and holding != DEFENSE:
                target = DEFENSE

            # 信号进入 pending queue
            signal_counter += 1
            next_td = trading_dates[i + 1] if i + 1 < len(trading_dates) else None
            pending_signal = {
                "signal_id": f"SIG-{signal_counter:06d}",
                "signal_date": td,
                "target": target,
                "holding": holding,
            }

            decision_log.append({
                "date": str(td),
                "signal_id": pending_signal["signal_id"],
                "target": target,
                "target_name": ETF_POOL.get(target, "货币基金"),
                "n_candidates": len(candidates),
                "a_share_weak": a_share_weak,
                "profit_prot": profit_protection_triggered,
                "execution_date": str(next_td) if next_td else None,
            })

    # 最后一个 pending 信号未执行 → 不计入成交
    if pending_signal:
        trade_log.append({
            "signal_id": pending_signal["signal_id"],
            "signal_date": str(pending_signal["signal_date"]),
            "execution_date": None,
            "action": "none", "code": pending_signal["target"],
            "status": "unexecuted",
            "reason": "最后一个交易日, 无 T+1 可执行",
            "data_hash": data_hash, "param_hash": param_hash,
        })

    if not equity_history:
        return {"error": "no data"}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    # 每日采样 → 年化用 252 交易日
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
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

    n_executed = sum(1 for t in trade_log if t.get("status") == "executed")
    n_cancelled = sum(1 for t in trade_log if t.get("status") in ("cancelled", "unexecuted"))

    return {
        "total_return": total_return, "ann_return": ann_ret,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "yearly": yearly, "n_trades": n_executed,
        "n_cancelled": n_cancelled,
        "equity_curve": eq_df, "decision_log": decision_log,
        "trade_log": trade_log,
        "param_hash": param_hash, "data_hash": data_hash,
        "cost_multiplier": cost_multiplier,
    }


def run_qixing_v3_same_day(
    data: dict, initial_capital: float = 100_000.0, cost_multiplier: float = 1.0,
    live_mirror: bool = False,
) -> dict:
    """同日收盘成交回测 (对齐实盘 14:50 执行口径).

    实盘流程: 调仓日 14:30 实时快照生成信号 → 14:50~15:00 执行成交。
    信号时刻早于成交时刻, 无未来函数; 日线数据下用 T 日收盘价同时近似
    14:30 信号价与 14:50 成交价。

    Args:
        live_mirror: 实盘镜像模式 - 额外复刻 live_signal.py 的实时急跌保护
            (信号日盘中跌>3%的候选剔除后重选目标, 全剔除切货币)。
            日线近似: 信号日 收盘/昨收 - 1 < -3%。

    与 run_qixing_v3_no_lookahead (T+1开盘成交) 的唯一区别是成交时点,
    其余规则(涨跌停检查/卖出失败不买/每日净值采样)保持一致。
    """
    common_dates: set = set()
    for code in ETF_POOL:
        if code not in data:
            continue
        dates = data[code]["trade_date"].tolist()
        if not common_dates:
            common_dates = set(dates)
        else:
            common_dates &= set(dates)
    if DEFENSE in data:
        common_dates &= set(data[DEFENSE]["trade_date"].tolist())

    all_dates = sorted(common_dates)
    warmup = 130
    trading_dates = all_dates[warmup:]
    rebalance_dates = trading_dates[::REBALANCE_DAYS]
    rebalance_set = set(rebalance_dates)

    fee = FEE * cost_multiplier
    slippage = SLIPPAGE * cost_multiplier

    cash = initial_capital
    holding: str | None = None
    holding_shares = 0
    equity_history: list[dict] = []
    trade_log: list[dict] = []
    decision_log: list[dict] = []
    rt_filter_log: list[dict] = []  # 实时急跌保护剔除记录 (live_mirror)
    signal_counter = 0

    def _check_close_tradable(code: str, td) -> tuple[bool, str]:
        """收盘口径可交易检查: 有数据 + 未涨跌停(收盘价 vs 昨收)."""
        df = data[code]
        row = df[df["trade_date"] == td]
        if row.empty:
            return (False, f"{code} 在 {td} 无数据")
        price = float(row.iloc[0]["close"])
        if price <= 0:
            return (False, f"{code} 在 {td} 收盘价无效")
        hist = df[df["trade_date"] < td]
        if not hist.empty:
            prev_close = float(hist.iloc[-1]["close"])
            if prev_close > 0 and abs(price / prev_close - 1) >= 0.099:
                return (False, f"{code} 在 {td} 收盘涨跌停")
        return (True, "")

    for td in trading_dates:
        # === 调仓日: T日信号 → T日收盘成交 ===
        if td in rebalance_set:
            etf_data_at_date = {}
            for code in [*list(ETF_POOL.keys()), DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() < warmup:
                    continue
                etf_data_at_date[code] = mask.sum() - 1

            target, candidates, _best_score, a_share_weak = select_target(
                data, etf_data_at_date, holding
            )

            # === 实盘镜像: 实时急跌保护 (1:1 复刻 live_signal.py run() 语义) ===
            # 实盘: 候选当日盘中跌>3% (实时价 vs 昨收) → 剔除后重选Top1, 全剔除切货币
            # 日线近似: 信号日收盘/昨收 - 1 < -3%
            if live_mirror and candidates:
                dropped = set()
                for code, _s in candidates:
                    idx = etf_data_at_date[code]
                    close = data[code]["close"].values[: idx + 1].astype(float)
                    if len(close) >= 2 and close[-2] > 0:
                        intraday = (close[-1] - close[-2]) / close[-2]
                        if intraday < -0.03:
                            dropped.add(code)
                            rt_filter_log.append({
                                "date": str(td), "code": code,
                                "name": ETF_POOL.get(code, "货币基金"),
                                "intraday_ret": round(float(intraday), 4),
                            })
                if dropped:
                    candidates = [(c, s) for c, s in candidates if c not in dropped]
                    target = candidates[0][0] if candidates else DEFENSE

            signal_counter += 1
            sig_id = f"SIG-{signal_counter:06d}"
            decision_log.append({
                "date": str(td), "signal_id": sig_id,
                "target": target,
                "target_name": ETF_POOL.get(target, "货币基金"),
                "n_candidates": len(candidates),
                "a_share_weak": a_share_weak,
            })

            if target != holding:
                sell_ok = True
                if holding and holding in data:
                    can_sell, reason = _check_close_tradable(holding, td)
                    if not can_sell:
                        sell_ok = False
                        trade_log.append({
                            "signal_id": sig_id, "date": str(td),
                            "action": "sell", "code": holding,
                            "status": "cancelled", "reason": f"卖出失败: {reason}",
                        })
                    else:
                        price = float(
                            data[holding][data[holding]["trade_date"] == td].iloc[0]["close"]
                        )
                        amount = holding_shares * price * (1 - fee - slippage)
                        cash += amount
                        trade_log.append({
                            "signal_id": sig_id, "date": str(td),
                            "action": "sell", "code": holding,
                            "shares": holding_shares, "price": price,
                            "amount": round(amount, 2),
                            "status": "executed", "reason": "",
                        })
                        holding = None
                        holding_shares = 0

                if sell_ok and target and target in data:
                    can_buy, reason = _check_close_tradable(target, td)
                    if not can_buy:
                        trade_log.append({
                            "signal_id": sig_id, "date": str(td),
                            "action": "buy", "code": target,
                            "status": "cancelled", "reason": f"买入失败: {reason}",
                        })
                    else:
                        price = float(
                            data[target][data[target]["trade_date"] == td].iloc[0]["close"]
                        )
                        shares = int(cash * 0.99 / price / 100) * 100
                        if shares > 0:
                            cost = shares * price * (1 + fee + slippage)
                            cash -= cost
                            holding = target
                            holding_shares = shares
                            trade_log.append({
                                "signal_id": sig_id, "date": str(td),
                                "action": "buy", "code": target,
                                "shares": shares, "price": price,
                                "amount": round(cost, 2),
                                "status": "executed", "reason": "",
                            })

        # === 每日净值 ===
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * float(row.iloc[0]["close"])
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

    if not equity_history:
        return {"error": "no data"}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
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

    n_executed = sum(1 for t in trade_log if t.get("status") == "executed")
    n_cancelled = sum(1 for t in trade_log if t.get("status") == "cancelled")

    return {
        "total_return": total_return, "ann_return": ann_ret,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "yearly": yearly, "n_trades": n_executed,
        "n_cancelled": n_cancelled,
        "equity_curve": eq_df, "decision_log": decision_log,
        "trade_log": trade_log,
        "rt_filter_log": rt_filter_log,
    }


def run_cost_stress_test(data: dict, initial_capital: float = 100_000.0) -> dict:
    """成本压力测试: 基础/2x/3x 三档成本对比."""
    results = {}
    for multiplier in [1.0, 2.0, 3.0]:
        r = run_qixing_v3_no_lookahead(data, initial_capital, cost_multiplier=multiplier)
        results[f"{multiplier:.0f}x"] = {
            "total_return": r["total_return"],
            "ann_return": r["ann_return"],
            "sharpe": r["sharpe"],
            "max_drawdown": r["max_drawdown"],
            "n_trades": r["n_trades"],
            "n_cancelled": r["n_cancelled"],
        }
    return results


def main():  # pragma: no cover
    print("=" * 70)
    print("  七星ETF轮动超级增强V3 — 无未来函数回测 (R1)")
    print("  规则: 加权动量 + 跌幅过滤 + A股走弱回避")
    print("        T日收盘信号 → T+1开盘成交 (消除 lookahead bias)")
    print("=" * 70)

    data = load_data()
    print(f"\n  数据: {len(data)}只ETF")
    for code, df in data.items():
        name = ETF_POOL.get(code, "货币基金")
        print(
            f"    {code} {name}: {len(df)}天 "
            f"({df['trade_date'].min()} ~ {df['trade_date'].max()})"
        )

    # === 无未来函数回测 (R1 修复版) ===
    print("\n  回测中 (T日信号 → T+1开盘成交, 无 lookahead bias)...")
    result = run_qixing_v3_no_lookahead(data)

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
          f"回撤: {result['max_drawdown']:.1%} | 交易: {result['n_trades']}次 "
          f"(取消: {result['n_cancelled']}次)")

    # 审计信息
    print(f"\n  审计: param_hash={result['param_hash']} data_hash={result['data_hash']}")

    # 持仓分布
    from collections import Counter
    holding_counts = Counter(eq["holding"].tolist())
    total_days = len(eq)
    print(f"\n  持仓分布 (共{total_days}天):")
    for code, count in holding_counts.most_common():
        name = ETF_POOL.get(code, "货币基金")
        print(f"    {name:<10} {count}天 ({count/total_days:.0%})")

    # === 对比旧版 (lookahead bias 诊断基线) ===
    print(f"\n  {'=' * 50}")
    print("  对比: 无未来函数 vs 旧版(lookahead bias)")
    print(f"  {'=' * 50}")
    old_result = run_qixing_v3(data)
    print(f"  {'指标':<12} {'无未来函数':>12} {'旧版(有bias)':>12} {'差异':>12}")
    print(f"  {'-' * 50}")
    print(f"  {'总收益':<12} {result['total_return']:>+12.1%} {old_result['total_return']:>+12.1%} "
          f"{result['total_return'] - old_result['total_return']:>+12.1%}")
    print(f"  {'年化':<12} {result['ann_return']:>+12.1%} {old_result['ann_return']:>+12.1%} "
          f"{result['ann_return'] - old_result['ann_return']:>+12.1%}")
    print(f"  {'夏普':<12} {result['sharpe']:>12.2f} {old_result['sharpe']:>12.2f} "
          f"{result['sharpe'] - old_result['sharpe']:>+12.2f}")
    print(f"  {'最大回撤':<12} {result['max_drawdown']:>12.1%} {old_result['max_drawdown']:>12.1%} "
          f"{result['max_drawdown'] - old_result['max_drawdown']:>+12.1%}")
    print(f"  {'交易次数':<12} {result['n_trades']:>12} {old_result['n_trades']:>12} "
          f"{result['n_trades'] - old_result['n_trades']:>+12}")

    # === 成本压力测试 ===
    print(f"\n  {'=' * 50}")
    print("  成本压力测试 (基础 / 2x / 3x)")
    print(f"  {'=' * 50}")
    stress = run_cost_stress_test(data)
    print(f"  {'成本倍数':<10} {'总收益':>10} {'年化':>10} {'夏普':>8} {'回撤':>8} {'交易':>6}")
    print(f"  {'-' * 56}")
    for label, r in stress.items():
        print(f"  {label:<10} {r['total_return']:>+10.1%} {r['ann_return']:>+10.1%} "
              f"{r['sharpe']:>8.2f} {r['max_drawdown']:>8.1%} {r['n_trades']:>6}")

    # === 成交记录摘要 ===
    trade_log = result.get("trade_log", [])
    if trade_log:
        print(f"\n  {'=' * 50}")
        print(f"  成交记录摘要 (共{len(trade_log)}条)")
        print(f"  {'=' * 50}")
        executed = [t for t in trade_log if t.get("status") == "executed"]
        cancelled = [t for t in trade_log if t.get("status") in ("cancelled", "unexecuted")]
        print(f"  已执行: {len(executed)}笔 | 取消/未执行: {len(cancelled)}笔")
        if cancelled:
            print("\n  取消记录 (前5条):")
            for t in cancelled[:5]:
                print(f"    {t.get('signal_id')} {t.get('action')} {t.get('code')} "
                      f"{t.get('status')} - {t.get('reason', '')}")

    # 保存
    summary = {
        "strategy": "七星V3 无未来函数回测 (R1)",
        "total_return": total_ret,
        "ann_return": result["ann_return"],
        "sharpe": result["sharpe"],
        "max_drawdown": result["max_drawdown"],
        "n_trades": result["n_trades"],
        "n_cancelled": result["n_cancelled"],
        "yearly": {str(k): v for k, v in result["yearly"].items()},
        "param_hash": result["param_hash"],
        "data_hash": result["data_hash"],
        "cost_stress_test": stress,
        "lookahead_bias_comparison": {
            "no_lookahead": {
                "total_return": result["total_return"],
                "sharpe": result["sharpe"],
                "max_drawdown": result["max_drawdown"],
            },
            "old_lookahead": {
                "total_return": old_result["total_return"],
                "sharpe": old_result["sharpe"],
                "max_drawdown": old_result["max_drawdown"],
            },
        },
    }
    with open(OUTPUT_DIR / "qixing_v3_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  结果已保存: {OUTPUT_DIR / 'qixing_v3_results.json'}")


if __name__ == "__main__":
    main()
