"""七星V3 实盘信号生成器 (生产级).

核心保证: 直接 import run_qixing_v3 的 select_target, 实盘与回测逻辑100%一致.

用法:
  uv run python scripts/live_signal.py --init 100000   # 首次: 初始化账户(10万本金)
  uv run python scripts/live_signal.py                 # 每日: 更新数据+生成信号
  uv run python scripts/live_signal.py --status        # 查看当前持仓(不更新数据)
  uv run python scripts/live_signal.py --dry-run       # 预演今日信号(不改动状态)
  uv run python scripts/live_signal.py --sync-only     # 仅更新行情(夜间补齐当日K线, 不出信号)

推荐工作流:
  每个交易日 14:50 后运行 → 若为调仓日且有换仓信号 → 15:00 收盘前执行
  (或次日开盘执行, 周频策略隔日影响很小)

调仓频率: 每5个交易日检查一次 (与回测一致).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# === 保证与回测100%一致: 直接复用回测的核心逻辑 ===
sys.path.insert(0, str(Path(__file__).parent))
from run_qixing_v3 import (  # noqa: E402
    A_SHARE_ETF,
    A_SHARE_MA,
    DATA_DIR,
    DEFENSE,
    DROP_LOOKBACK,
    DROP_THRESHOLD,
    ETF_POOL,
    FEE,
    REBALANCE_DAYS,
    SLIPPAGE,
    calc_momentum_score,
    load_data,
    select_target,
)
from notify import load_config, push_bark, save_config, set_bark_key, test_push  # noqa: E402

LIVE_DIR = DATA_DIR.parent / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = LIVE_DIR / "state.json"

ALL_CODES = list(ETF_POOL.keys()) + [DEFENSE]


def name_of(code: str) -> str:
    return ETF_POOL.get(code, "货币基金")


def sina_symbol(code: str) -> str:
    prefix = "sh" if code.startswith(("5", "6")) else "sz"
    return f"{prefix}{code}"


# --------------------------------------------------------------------------- #
# 数据更新 (增量追加, 不碰已复权的历史数据 → 保证实盘==回测)
# --------------------------------------------------------------------------- #
def update_data(data: dict) -> dict:
    """增量拉取最新行情, 追加到已复权的历史缓存.

    历史数据保持不变(与回测一致), 只追加新交易日.
    若检测到疑似份额拆分(单日跳变>40%)会告警.
    """
    import akshare as ak

    today = date.today()
    updated = []

    for code in ALL_CODES:
        if code not in data:
            continue
        df = data[code]
        last_date = df["trade_date"].max()
        if last_date >= today:
            continue  # 已是最新

        try:
            raw = ak.fund_etf_hist_sina(symbol=sina_symbol(code))
            if raw is None or raw.empty:
                continue
            raw = raw.rename(columns={"date": "trade_date"})
            raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.date
            new = raw[raw["trade_date"] > last_date].copy()
            if new.empty:
                continue

            # 份额拆分安全检查: 新数据首日 vs 缓存末日 收盘价跳变>40%
            last_close = float(df.iloc[-1]["close"])
            first_new_close = float(new.iloc[0]["close"])
            jump = (first_new_close - last_close) / last_close
            if abs(jump) > 0.40:
                print(f"  ⚠️  {code} {name_of(code)} 检测到异常跳变 {jump:+.0%}, "
                      f"疑似份额拆分! 请人工复权后再用 (切勿直接交易)")
                continue

            new["symbol"] = code
            for col in ["open", "close", "high", "low", "volume"]:
                if col not in new.columns:
                    new[col] = 0.0
            new = new[["trade_date", "open", "close", "high", "low", "volume", "symbol"]]

            df = pd.concat([df, new], ignore_index=True)
            df = df.sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)
            df.to_parquet(DATA_DIR / f"{code}.parquet", index=False)
            data[code] = df
            updated.append(f"{code}+{len(new)}天")
            time.sleep(1.0)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️  {code} 更新失败: {e}")

    if updated:
        print(f"  数据更新: {', '.join(updated)}")
    else:
        _td = get_trading_dates(data)
        print(f"  数据已是最新 (最新数据: {_td[-1] if _td else '无'})")
    check_data_freshness(data)
    return data


def check_data_freshness(data: dict) -> None:
    """数据新鲜度检查: 若缺失上一个已完成交易日的数据则告警.

    新浪等数据源的当日K线有发布延迟。若上一交易日数据缺失,
    信号将基于过期数据, 此处打印告警(不阻断)供日志排查。
    """
    try:
        import akshare as ak
        cal = ak.tool_trade_date_hist_sina()
        cal_dates = sorted(pd.to_datetime(cal["trade_date"]).dt.date)
        today = date.today()
        past = [d for d in cal_dates if d < today]
        if not past:
            return
        expected = past[-1]  # 上一个已完成交易日
        trading_dates = get_trading_dates(data)
        if not trading_dates:
            return
        latest = trading_dates[-1]
        if latest < expected:
            print(f"  ⚠️  数据滞后: 最新数据 {latest}, 缺失上一交易日 {expected} 的数据!")
            print(f"      信号将基于过期数据, 请检查数据源 (可能为发布延迟)")
    except Exception as e:  # noqa: BLE001
        print(f"  (数据新鲜度检查跳过: {e})")


# 已知份额拆分 (前复权: 拆分前 OHLC 除以比例, 与回测缓存一致)
KNOWN_SPLITS = {
    "513100": [("2022-01-14", 5.1153)],   # 纳指ETF 1:5.1153
    "511220": [("2023-01-16", 10.0663)],  # 城投债ETF 1:10.0663
}


def bootstrap_data() -> None:
    """全量拉取历史并前复权 (服务器首次部署 / 灾难恢复用).

    从新浪源拉取全部历史, 对已知拆分做前复权, 重建缓存.
    注意: 只应在空缓存时运行; 已有缓存请用增量 update_data.
    """
    import akshare as ak

    print("  全量数据初始化 (拉取完整历史 + 份额拆分复权)...")
    for code in ALL_CODES:
        print(f"  拉取 {code} {name_of(code)}...", end=" ", flush=True)
        try:
            raw = ak.fund_etf_hist_sina(symbol=sina_symbol(code))
            if raw is None or raw.empty:
                print("无数据")
                continue
            df = raw.rename(columns={"date": "trade_date"})
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            df["symbol"] = code
            for col in ["open", "close", "high", "low", "volume"]:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[["trade_date", "open", "close", "high", "low", "volume", "symbol"]]
            df = df.sort_values("trade_date").reset_index(drop=True)

            # 数值列统一转 float64 (避免拆分复权后与 int64 schema 冲突)
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = df[col].astype("float64")

            # 份额拆分前复权
            for split_date, ratio in KNOWN_SPLITS.get(code, []):
                split_d = pd.to_datetime(split_date).date()
                mask = df["trade_date"] < split_d
                for col in ["open", "close", "high", "low"]:
                    df.loc[mask, col] = df.loc[mask, col].astype(float) / ratio
                df.loc[mask, "volume"] = df.loc[mask, "volume"].astype(float) * ratio

            df.to_parquet(DATA_DIR / f"{code}.parquet", index=False)
            print(f"{len(df)}天 ({df['trade_date'].min()} ~ {df['trade_date'].max()})")
            time.sleep(1.0)
        except Exception as e:  # noqa: BLE001
            print(f"失败: {e}")
    print("  ✓ 数据初始化完成")


# --------------------------------------------------------------------------- #
# 状态管理
# --------------------------------------------------------------------------- #
def default_state(capital: float) -> dict:
    return {
        "initial_capital": capital,
        "cash": capital,
        "holding": None,
        "shares": 0,
        "entry_date": None,
        "entry_price": 0.0,
        "last_rebalance_date": None,
        "last_run_date": None,
        "pending_order": None,
        "trade_log": [],
    }


def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def is_paper_mode() -> bool:
    """模拟记账模式: 信号按理论价自动记账 (用于尚无法真实下单的阶段)."""
    return bool(load_config().get("paper_mode", False))


def set_paper_mode(on: bool) -> None:
    cfg = load_config()
    cfg["paper_mode"] = bool(on)
    save_config(cfg)


# --------------------------------------------------------------------------- #
# 交易日历
# --------------------------------------------------------------------------- #
def get_trading_dates(data: dict) -> list:
    """公共交易日历 (与回测一致)."""
    common = None
    for code in ALL_CODES:
        if code not in data:
            continue
        dates = set(data[code]["trade_date"].tolist())
        common = dates if common is None else common & dates
    return sorted(common)


def price_on(data: dict, code: str, td) -> float | None:
    if code not in data:
        return None
    row = data[code][data[code]["trade_date"] == td]
    if row.empty:
        return None
    return float(row.iloc[0]["close"])


_REALTIME_CACHE = {"time": 0.0, "spot": {}}


def _fetch_tencent_spot() -> dict[str, dict]:
    """腾讯实时行情 (qt.gtimg.cn), 覆盖全部场内ETF+LOF+QDII, 缓存1分钟。

    东方财富 fund_etf_spot_em 不覆盖 LOF (如501018南方原油),
    腾讯源无此限制。返回 {code: {"price": float, "prev_close": float}}。
    """
    import requests
    now = time.time()
    if _REALTIME_CACHE["spot"] and (now - _REALTIME_CACHE["time"]) <= 60:
        return _REALTIME_CACHE["spot"]

    tencent_codes = [
        f"{'sh' if c.startswith(('5', '6')) else 'sz'}{c}" for c in ALL_CODES
    ]
    url = f"http://qt.gtimg.cn/q={','.join(tencent_codes)}"
    try:
        resp = requests.get(url, timeout=10)
        resp.encoding = "gbk"
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  腾讯行情获取失败: {e}")
        return _REALTIME_CACHE["spot"]

    spot: dict[str, dict] = {}
    for line in resp.text.strip().split(";"):
        if "~" not in line:
            continue
        parts = line.split("~")
        if len(parts) < 5:
            continue
        code = parts[2]
        price = float(parts[3]) if parts[3] else 0
        prev_close = float(parts[4]) if parts[4] else 0
        if price > 0:
            spot[code] = {"price": price, "prev_close": prev_close}

    if spot:
        _REALTIME_CACHE["spot"] = spot
        _REALTIME_CACHE["time"] = now
    return spot


def get_realtime_price(code: str) -> float | None:
    """获取ETF/LOF当日实时价 (腾讯源, 覆盖全部品种含LOF, 缓存1分钟)。

    用于实时急跌保护: 盘中急跌>3%时排除候选。
    失败返回None(回退到历史价)。
    """
    info = _fetch_tencent_spot().get(code)
    return info["price"] if info else None


def account_value(state: dict, data: dict, td) -> float:
    """账户总值 = 现金 + 持仓市值."""
    total = state["cash"]
    if state["holding"]:
        p = price_on(data, state["holding"], td)
        if p:
            total += state["shares"] * p
    return total


def build_etf_data_at_date(data: dict, td, warmup: int = 130) -> dict:
    """构造 select_target 需要的 {code: 当日索引} (与回测一致)."""
    idx_map = {}
    for code in ALL_CODES:
        if code not in data:
            continue
        df = data[code]
        mask = df["trade_date"] <= td
        if mask.sum() < warmup:
            continue
        idx_map[code] = mask.sum() - 1
    return idx_map


# --------------------------------------------------------------------------- #
# 报告输出
# --------------------------------------------------------------------------- #
def fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def print_header(td) -> None:
    print("\n" + "=" * 56)
    print(f"  七星V3 实盘信号  |  {td}")
    print("=" * 56)


def print_account(state: dict, data: dict, td) -> None:
    holding = state["holding"]
    cash = state["cash"]
    mv = 0.0
    if holding:
        p = price_on(data, holding, td)
        if p:
            mv = state["shares"] * p
    total = cash + mv
    init = state["initial_capital"]
    print(f"\n  【账户】 总值 {fmt_money(total)} 元 "
          f"(累计 {(total / init - 1) * 100:+.1f}%)")
    if holding:
        p = price_on(data, holding, td)
        cost = state["entry_price"]
        pnl = (p / cost - 1) * 100 if cost else 0
        print(f"    持仓: {name_of(holding)} ({holding}) "
              f"{state['shares']}股 @ {cost:.3f} → 现价 {p:.3f} ({pnl:+.1f}%)")
        print(f"    现金: {fmt_money(cash)} 元")
    else:
        print(f"    持仓: 空仓 (全部现金 {fmt_money(cash)} 元)")


def print_momentum_board(data: dict, td, holding: str | None, target: str) -> None:
    """动量排行 + 每个ETF的状态."""
    idx_map = build_etf_data_at_date(data, td)
    rows = []
    for code in ETF_POOL:
        if code not in idx_map:
            continue
        df = data[code]
        close = df["close"].values[: idx_map[code] + 1].astype(float)
        if len(close) < 121:
            continue
        score = calc_momentum_score(close)
        # 单日跌幅过滤状态
        drop_flag = ""
        for i in range(-DROP_LOOKBACK, 0):
            r = (close[i] - close[i - 1]) / close[i - 1]
            if r < DROP_THRESHOLD:
                drop_flag = f"⛔近{DROP_LOOKBACK}日暴跌"
                break
        rows.append((code, score, drop_flag))
    rows.sort(key=lambda x: -x[1])

    print(f"\n  【动量排行】(10日×0.5 + 20日×0.5)")
    for rank, (code, score, flag) in enumerate(rows, 1):
        tag = ""
        if code == holding:
            tag += " ◀持仓"
        if code == target:
            tag += " ◀目标"
        status = flag if flag else ("入选" if score > 0 else "动量<0")
        print(f"    {rank}. {name_of(code):<8} {score * 100:+6.2f}%  [{status}]{tag}")


def print_orders(sell: tuple | None, buy: tuple | None) -> None:
    print(f"\n  {'─' * 56}")
    print("  📋 今日操作指令")
    print(f"  {'─' * 56}")
    if sell:
        code, shares, price, amount = sell
        print(f"    ① 卖出  {name_of(code)} ({code})")
        print(f"       {shares} 股 @ 约 {price:.3f} 元 ≈ {fmt_money(amount)} 元")
    if buy:
        code, shares, price, amount = buy
        print(f"    ② 买入  {name_of(code)} ({code})")
        print(f"       {shares} 股 @ 约 {price:.3f} 元 ≈ {fmt_money(amount)} 元")
    if not sell and not buy:
        print("    无操作")
    print(f"  {'─' * 56}")


def notify_hold(td, holding: str | None, state: dict, data: dict) -> None:
    """调仓日继续持有 → 推送一条平安通知."""
    total = account_value(state, data, td)
    ret = (total / state["initial_capital"] - 1) * 100
    if holding:
        holding_disp = f"【{holding}】{name_of(holding)}"
    else:
        holding_disp = "现金"
    body = (
        f"📅 {td} 调仓日 · 无操作\n"
        f"继续持有 {holding_disp}\n"
        f"💰 账户 {fmt_money(total)} 元 ({ret:+.1f}%)"
    )
    push_bark("😴 七星V3 继续持有", body, level="active")


def notify_trade(td, sell_order, buy_order, reason: str, state: dict, data: dict) -> None:
    """调仓日换仓 → 推送紧急买卖指令 (突破勿扰)."""
    total = account_value(state, data, td)
    ret = (total / state["initial_capital"] - 1) * 100
    lines = [f"📅 {td} 调仓日", ""]
    if sell_order:
        code, shares, price, amount = sell_order
        lines.append(f"① 卖出 【{code}】{name_of(code)}  {shares}股 @{price:.3f} ≈ {fmt_money(amount)}元")
    if buy_order:
        code, shares, price, amount = buy_order
        lines.append(f"② 买入 【{code}】{name_of(code)}  {shares}股 @{price:.3f} ≈ {fmt_money(amount)}元")
    lines += ["", f"💡 {reason}", f"💰 账户 {fmt_money(total)} 元 ({ret:+.1f}%)", ""]
    lines.append("👉 易淘金App按代码下单:")
    if sell_order:
        lines.append(f"   卖出 {sell_order[0]} ({name_of(sell_order[0])}) {sell_order[1]}股")
    if buy_order:
        lines.append(f"   买入 {buy_order[0]} ({name_of(buy_order[0])}) {buy_order[1]}股")
    lines.append("完成后打开记账网页确认成交")
    push_bark("🔄 七星V3 换仓信号", "\n".join(lines), level="timeSensitive", sound="alarm")


# --------------------------------------------------------------------------- #
# 实时行情注入 (解决14:50信号看不到当天数据的问题)
# --------------------------------------------------------------------------- #
def inject_realtime(data: dict) -> dict:
    """将当日实时行情注入内存数据 (不写parquet, 仅用于信号计算).

    数据源: 腾讯行情接口 qt.gtimg.cn (免费/无认证/覆盖全部场内ETF+LOF+QDII).
    复用 _fetch_tencent_spot(), 与实时急跌保护同源, 保证一致性。
    """
    today = date.today()

    # 检查是否已有今天数据
    sample_code = next(iter(data))
    if data[sample_code]["trade_date"].max() >= today:
        return data  # 已有今天数据, 无需注入

    # 腾讯实时行情 (复用 _fetch_tencent_spot, 与急跌保护同源)
    spot_map = _fetch_tencent_spot()
    code_list = list(data.keys())
    if not spot_map:
        print("  ⚠️  腾讯行情获取失败, 回退到历史数据")
        return data

    injected = []
    for code in code_list:
        df = data[code]
        if code in spot_map:
            s = spot_map[code]
            new_row = {
                "trade_date": today,
                "open": s["prev_close"],  # 用昨收近似开盘
                "close": s["price"],       # 实时价作为当天收盘
                "high": max(s["price"], s["prev_close"]),
                "low": min(s["price"], s["prev_close"]),
                "volume": 0.0,
            }
        else:
            # 拉不到的用昨日收盘价填充
            last_row = df.iloc[-1]
            new_row = {
                "trade_date": today,
                "open": float(last_row["close"]),
                "close": float(last_row["close"]),
                "high": float(last_row["close"]),
                "low": float(last_row["close"]),
                "volume": 0.0,
            }
        if "symbol" in df.columns:
            new_row["symbol"] = code
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = df.sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)
        data[code] = df
        injected.append(code)

    if injected:
        print(f"  ⚡ 已注入当日实时行情 ({today}): {len(injected)}只ETF [腾讯源]")
        print(f"     注: 使用实时价作为当日收盘, 与回测(真实收盘)有微小差异")
    return data


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(dry_run: bool = False) -> None:
    state = load_state()
    if state is None:
        print("  ❌ 尚未初始化账户, 请先运行:")
        print("     uv run python scripts/live_signal.py --init 100000")
        return

    print("  加载数据...")
    data = load_data()
    if not data:
        print("  ❌ 无数据缓存, 请先运行: uv run python scripts/live_signal.py --bootstrap")
        return
    if not dry_run:
        data = update_data(data)

    # 注入当日实时行情 (解决14:50拿不到当天数据的问题)
    data = inject_realtime(data)

    trading_dates = get_trading_dates(data)
    td = trading_dates[-1]  # 最新交易日 (注入后=今天)

    print_header(td)

    # 幂等: 今日已处理过
    if not dry_run and state.get("last_run_date") == str(td):
        print("\n  ✓ 今日已生成过信号, 以下为当前状态:")
        print_account(state, data, td)
        print("\n  (如需强制重算, 使用 --dry-run 预演)")
        return

    # 是否为调仓日: 绝对网格 (与回测一致: trading_dates[warmup:][::5])
    # 回测逻辑: 从公共日期第130天起, 每隔5天为一个调仓日
    # 实盘必须用同一网格, 否则调仓日会与回测逐渐错位
    WARMUP = 130
    all_trading = trading_dates  # 公共交易日历
    if len(all_trading) > WARMUP:
        post_warmup = all_trading[WARMUP:]  # 与回测 trading_dates 一致
        rebalance_grid = set(post_warmup[::REBALANCE_DAYS])  # 绝对网格
        is_rebalance = td in rebalance_grid
        # 计算距下次调仓还有几天
        if is_rebalance:
            days_since = REBALANCE_DAYS
        else:
            try:
                td_idx = post_warmup.index(td)
                days_since = td_idx % REBALANCE_DAYS
            except ValueError:
                days_since = 0
                is_rebalance = True  # 找不到就当作调仓日
    else:
        is_rebalance = True
        days_since = REBALANCE_DAYS

    holding = state["holding"]

    # === 核心决策 (与回测同一函数) ===
    idx_map = build_etf_data_at_date(data, td)
    target, candidates, best_score, a_share_weak = select_target(data, idx_map, holding)

    # === 实时急跌保护: 14:50信号时检查当天盘中是否已暴跌>3% ===
    # 新浪日K在收盘后才发布, 14:50拿不到当天数据, 但实时行情能拿到
    realtime_dropped = []
    for code, score in candidates:
        rt_price = get_realtime_price(code)
        if rt_price is None:
            continue
        # 用昨天收盘价作为基准
        prev_close = price_on(data, code, td)
        if prev_close and prev_close > 0:
            intraday_ret = (rt_price - prev_close) / prev_close
            if intraday_ret < -0.03:
                realtime_dropped.append((code, intraday_ret))
    if realtime_dropped:
        dropped_codes = {c for c, _ in realtime_dropped}
        for code, ret in realtime_dropped:
            print(f"  ⚡ 实时急跌: {name_of(code)} 当日{ret:+.1%}, 已排除")
        print(f"     注: 此过滤为实盘保护, 回测中不存在(回测用真实收盘价自然触发)")
        # 从candidates中移除, 重新选目标
        candidates = [(c, s) for c, s in candidates if c not in dropped_codes]
        if candidates:
            target = candidates[0][0]
            best_score = candidates[0][1]
        else:
            target = DEFENSE
            best_score = 0

    print_account(state, data, td)
    print_momentum_board(data, td, holding, target)

    if a_share_weak:
        print(f"\n  ⚠️  A股走弱 (创业板<MA{A_SHARE_MA}), 已排除创业板ETF")

    if not is_rebalance:
        wait = REBALANCE_DAYS - days_since
        print(f"\n  😴 今日非调仓日 (距下次还有 {wait} 个交易日)")
        print(f"     继续持有 {name_of(holding) if holding else '现金'}, 无操作")
        if not dry_run:
            state["last_run_date"] = str(td)
            save_state(state)
        return

    # === 调仓日 ===
    print(f"\n  🔄 今日为调仓日 (每{REBALANCE_DAYS}个交易日)")

    if target == holding:
        print(f"     信号: 继续持有 {name_of(target) if target else '现金'}, 无操作")
        if not dry_run:
            state["last_rebalance_date"] = str(td)
            state["last_run_date"] = str(td)
            save_state(state)
            notify_hold(td, target, state, data)
        return

    # === 需要换仓: 生成订单 ===
    sell_order = None
    buy_order = None
    cash = state["cash"]

    # 1. 卖出当前持仓
    if holding:
        p_sell = price_on(data, holding, td)
        if p_sell:
            amount = state["shares"] * p_sell * (1 - FEE - SLIPPAGE)
            sell_order = (holding, state["shares"], p_sell, amount)
            cash += amount

    # 2. 买入目标 (预留1%现金, 按100股整数)
    if target:
        p_buy = price_on(data, target, td)
        if p_buy:
            shares = int(cash * 0.99 / p_buy / 100) * 100
            if shares > 0:
                cost = shares * p_buy * (1 + FEE + SLIPPAGE)
                buy_order = (target, shares, p_buy, cost)

    print_orders(sell_order, buy_order)

    why = dict(candidates).get(target, 0)
    cur = dict(candidates).get(holding, None) if holding else None
    if target == DEFENSE:
        reason = "所有ETF动量转弱, 切入货币基金防御"
    else:
        reason = (f"{name_of(target)} 动量 {why * 100:+.1f}% 为最强"
                  + (f" (当前持仓 {cur * 100:+.1f}%)" if cur is not None else ""))
    print(f"  原因: {reason}")

    if dry_run:
        print("\n  [预演模式] 以上为模拟信号, 未改动账户状态")
        return

    # === 模拟记账模式: 信号按理论价自动成交 (尚无法真实下单的阶段) ===
    if is_paper_mode():
        if sell_order:
            state["cash"] += sell_order[3]
            state["trade_log"].append({
                "date": str(td), "action": "sell", "code": holding,
                "name": name_of(holding), "shares": state["shares"],
                "price": sell_order[2], "amount": sell_order[3],
            })
            state["holding"] = None
            state["shares"] = 0
            state["entry_price"] = 0.0
        if buy_order:
            state["cash"] -= buy_order[3]
            state["holding"] = target
            state["shares"] = buy_order[1]
            state["entry_price"] = buy_order[2]
            state["entry_date"] = str(td)
            state["trade_log"].append({
                "date": str(td), "action": "buy", "code": target,
                "name": name_of(target), "shares": buy_order[1],
                "price": buy_order[2], "amount": buy_order[3],
            })
        state["pending_order"] = None
        state["last_rebalance_date"] = str(td)
        state["last_run_date"] = str(td)
        save_state(state)
        print(f"\n  ✓ [模拟记账] 已按理论价自动记录: {STATE_FILE}")
        notify_trade(td, sell_order, buy_order, reason + " [模拟记账]", state, data)
        return

    # === 保存为【待确认】订单 (不自动成交, 以用户在网页填入的真实成交为准) ===
    state["pending_order"] = {
        "date": str(td),
        "sell": {
            "code": sell_order[0], "name": name_of(sell_order[0]),
            "shares": sell_order[1], "price": sell_order[2], "amount": sell_order[3],
        } if sell_order else None,
        "buy": {
            "code": buy_order[0], "name": name_of(buy_order[0]),
            "shares": buy_order[1], "price": buy_order[2], "amount": buy_order[3],
        } if buy_order else None,
        "reason": reason,
        "status": "pending",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    state["last_rebalance_date"] = str(td)
    state["last_run_date"] = str(td)
    save_state(state)
    print(f"\n  ✓ 信号已保存为【待确认】: {STATE_FILE}")
    print("    👉 请在网页填入真实成交后确认 (持仓状态以网页确认为准)")

    # === 推送换仓指令到手机 ===
    notify_trade(td, sell_order, buy_order, reason, state, data)


def show_status() -> None:
    state = load_state()
    if state is None:
        print("  ❌ 尚未初始化账户")
        return
    data = load_data()
    trading_dates = get_trading_dates(data)
    td = trading_dates[-1]
    print_header(td)
    print_account(state, data, td)
    log = state.get("trade_log", [])
    if log:
        print(f"\n  【最近交易】(共{len(log)}笔)")
        for t in log[-6:]:
            arrow = "卖出" if t["action"] == "sell" else "买入"
            print(f"    {t['date']}  {arrow} {t['name']} "
                  f"{t['shares']}股 @ {t['price']:.3f} ≈ {fmt_money(t['amount'])}元")


def sync_only() -> None:
    """仅更新行情数据 (不生成信号/不改账户状态), 供夜间定时任务补齐当日K线.

    新浪当日K线在收盘后才陆续发布, 14:50/16:30 的定时任务拿不到当日数据。
    本函数幂等 (只追加新交易日), 可由夜间 cron 安全重跑。
    """
    print("  加载数据...")
    data = load_data()
    if not data:
        print("  ❌ 无数据缓存, 请先运行: uv run python scripts/live_signal.py --bootstrap")
        return
    update_data(data)  # 内部已写回 parquet
    print("  ✓ 数据同步完成")


# --------------------------------------------------------------------------- #
# 网页确认成交 (实盘以用户填入的真实成交为准)
# --------------------------------------------------------------------------- #
def confirm_order(real_sell: dict | None, real_buy: dict | None) -> dict:
    """确认待确认订单, 用真实成交数据更新持仓状态.

    real_sell: {"shares": int, "price": float} 或 None (卖出当前持仓)
    real_buy:  {"code": str, "shares": int, "price": float} 或 None
    """
    state = load_state()
    if state is None:
        raise ValueError("账户未初始化")
    pending = state.get("pending_order")
    td = pending["date"] if pending else str(date.today())

    if real_sell and state["holding"]:
        code = state["holding"]
        shares = int(real_sell["shares"])
        price = float(real_sell["price"])
        amount = shares * price * (1 - FEE - SLIPPAGE)
        state["cash"] += amount
        state["trade_log"].append({
            "date": td, "action": "sell", "code": code, "name": name_of(code),
            "shares": shares, "price": price, "amount": amount,
        })
        state["holding"] = None
        state["shares"] = 0
        state["entry_price"] = 0.0

    if real_buy:
        code = real_buy["code"]
        shares = int(real_buy["shares"])
        price = float(real_buy["price"])
        amount = shares * price * (1 + FEE + SLIPPAGE)
        state["cash"] -= amount
        state["holding"] = code
        state["shares"] = shares
        state["entry_price"] = price
        state["entry_date"] = td
        state["trade_log"].append({
            "date": td, "action": "buy", "code": code, "name": name_of(code),
            "shares": shares, "price": price, "amount": amount,
        })

    if pending:
        pending["status"] = "confirmed"
        pending["confirmed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    return state


def skip_pending() -> dict:
    """标记待确认订单为'本次不操作'."""
    state = load_state()
    if state is None:
        raise ValueError("账户未初始化")
    pending = state.get("pending_order")
    if pending:
        pending["status"] = "skipped"
    save_state(state)
    return state


def record_manual_trade(action: str, code: str, shares: int, price: float,
                        td: str | None = None) -> dict:
    """手动记录一笔成交 (用于修正或非信号交易)."""
    state = load_state()
    if state is None:
        raise ValueError("账户未初始化")
    td = td or str(date.today())
    shares = int(shares)
    price = float(price)
    if action == "sell":
        amount = shares * price * (1 - FEE - SLIPPAGE)
        state["cash"] += amount
        state["trade_log"].append({
            "date": td, "action": "sell", "code": code, "name": name_of(code),
            "shares": shares, "price": price, "amount": amount,
        })
        state["holding"] = None
        state["shares"] = 0
        state["entry_price"] = 0.0
    elif action == "buy":
        amount = shares * price * (1 + FEE + SLIPPAGE)
        state["cash"] -= amount
        state["holding"] = code
        state["shares"] = shares
        state["entry_price"] = price
        state["entry_date"] = td
        state["trade_log"].append({
            "date": td, "action": "buy", "code": code, "name": name_of(code),
            "shares": shares, "price": price, "amount": amount,
        })
    else:
        raise ValueError(f"未知操作: {action}")
    save_state(state)
    return state


def momentum_board_data(data: dict, td, holding: str | None, target: str) -> list[dict]:
    """动量排行 (返回结构化数据, 供网页显示)."""
    idx_map = build_etf_data_at_date(data, td)
    rows = []
    for code in ETF_POOL:
        if code not in idx_map:
            continue
        df = data[code]
        close = df["close"].values[: idx_map[code] + 1].astype(float)
        if len(close) < 121:
            continue
        score = calc_momentum_score(close)
        dropped = False
        for i in range(-DROP_LOOKBACK, 0):
            r = (close[i] - close[i - 1]) / close[i - 1]
            if r < DROP_THRESHOLD:
                dropped = True
                break
        rows.append({
            "code": code, "name": name_of(code), "score": round(float(score) * 100, 2),
            "dropped": dropped, "is_holding": code == holding, "is_target": code == target,
            "eligible": bool(score > 0) and not dropped,
        })
    rows.sort(key=lambda x: -x["score"])
    return rows


def build_equity_curve(state: dict, data: dict) -> list[dict]:
    """根据已确认成交记录, 重建账户净值曲线."""
    log = state.get("trade_log", [])
    if not log:
        return []
    trading_dates = get_trading_dates(data)
    start_date = log[0]["date"]
    trades_by_date: dict[str, list] = {}
    for t in log:
        trades_by_date.setdefault(t["date"], []).append(t)
    cash = state["initial_capital"]
    holding = None
    shares = 0
    curve = []
    for td in trading_dates:
        td_str = str(td)
        if td_str < start_date:
            continue
        for t in trades_by_date.get(td_str, []):
            if t["action"] == "sell":
                cash += t["amount"]
                holding = None
                shares = 0
            elif t["action"] == "buy":
                cash -= t["amount"]
                holding = t["code"]
                shares = t["shares"]
        value = cash
        if holding and holding in data:
            p = price_on(data, holding, td)
            if p:
                value += shares * p
        curve.append({"date": td_str, "value": round(value, 2)})
    return curve


def init_account(capital: float) -> None:
    if STATE_FILE.exists():
        print(f"  ⚠️  账户已存在: {STATE_FILE}")
        print("     如需重置, 请先手动删除该文件")
        return
    state = default_state(capital)
    save_state(state)
    print(f"  ✓ 账户已初始化: 本金 {fmt_money(capital)} 元")
    print(f"    状态文件: {STATE_FILE}")
    print(f"\n  下一步: 运行 uv run python scripts/live_signal.py 生成首个信号")


def main() -> None:
    parser = argparse.ArgumentParser(description="七星V3 实盘信号生成器")
    parser.add_argument("--init", type=float, metavar="CAPITAL", help="初始化账户本金")
    parser.add_argument("--status", action="store_true", help="查看当前持仓状态")
    parser.add_argument("--dry-run", action="store_true", help="预演信号(不改动状态)")
    parser.add_argument("--set-bark", metavar="KEY", help="设置 Bark 设备 Key (开启手机推送)")
    parser.add_argument("--notify-test", action="store_true", help="发送一条测试推送")
    parser.add_argument("--bootstrap", action="store_true", help="全量拉取历史数据 (服务器首次部署)")
    parser.add_argument("--sync-only", action="store_true", help="仅更新行情数据 (夜间补齐当日K线, 不生成信号)")
    parser.add_argument("--paper-mode", metavar="on/off",
                        help="模拟记账: on=信号按理论价自动记账, off=网页确认真实成交")
    args = parser.parse_args()

    if args.bootstrap:
        bootstrap_data()
    elif args.set_bark:
        set_bark_key(args.set_bark)
        print("  ✓ Bark Key 已保存")
        print("    下一步: uv run python scripts/live_signal.py --notify-test 验证能否收到")
    elif args.notify_test:
        test_push()
    elif args.paper_mode:
        on = args.paper_mode.strip().lower() in ("on", "1", "true", "yes")
        set_paper_mode(on)
        print(f"  ✓ 模拟记账模式已{'开启 (信号按理论价自动记账)' if on else '关闭 (网页确认真实成交)'}")
    elif args.init:
        init_account(args.init)
    elif args.status:
        show_status()
    elif args.sync_only:
        sync_only()
    else:
        run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
