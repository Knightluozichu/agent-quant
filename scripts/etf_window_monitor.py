"""短期窗口涨幅监控器 — 多资产短周期价格变动系统性对比.

在相同时间段内比较不同资产的短期涨幅, 覆盖四个视图:
  1. 逐日视图: 最近 N 个交易日各资产每日涨跌 (同一天横向对比)
  2. 窗口视图: 截至基准日的 R1/R2/R3/R5/R10 多窗口收益矩阵
     (2-3日窗口为重点, 对应 '上周白银+8% / 今天原油+2.75%' 这类观察)
  3. 周视图:   本周/上周/上上周 自然周累计收益对比
     (识别 '上上周石油+12%, 上周原油-5%' 这种周际反转)
  4. 叙事摘要: 自然语言总结今日领涨领跌、3日窗口最强最弱、周际动能变化

数据源: data/cross_asset/*.parquet (前复权缓存, 与回测一致), 默认池与
exp_momentum_models.CORE_FULL 一致 (7只风险资产 + 货币基金防御).

用法:
  uv run python scripts/etf_window_monitor.py              # 默认: 逐日5天+窗口+最近3周+摘要
  uv run python scripts/etf_window_monitor.py --days 10    # 逐日视图看10个交易日
  uv run python scripts/etf_window_monitor.py --weeks 4    # 周视图看4个自然周
  uv run python scripts/etf_window_monitor.py --narrative  # 只输出叙事摘要
  uv run python scripts/etf_window_monitor.py --json       # 结构化JSON (供程序消费)
  uv run python scripts/etf_window_monitor.py --all        # 扩展到 cross_asset 全部资产
  uv run python scripts/etf_window_monitor.py --asof 2026-07-28   # 指定基准日
  uv run python scripts/etf_window_monitor.py --update     # 先增量更新行情再分析
  uv run python scripts/etf_window_monitor.py --notify     # Bark推送摘要
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import date as dt_date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cross_asset"

# === ETF池 (与 exp_momentum_models.CORE_FULL / run_qixing_v3.ETF_POOL 一致) ===
POOL_CORE = {
    "518880": "黄金ETF",
    "159985": "豆粕ETF",
    "501018": "南方原油",
    "161226": "白银LOF",
    "513100": "纳指ETF",
    "159915": "创业板ETF",
    "511220": "城投债ETF",
}
DEFENSE = "511880"
DEFENSE_NAME = "货币基金"

# cross_asset 中核心池之外的跨资产品种 (exp_extended_pool 扩展池成员)
EXTRA_NAMES = {
    "513500": "标普500ETF",
    "511260": "十年国债ETF",
    "513030": "德国ETF",
    "159981": "能源化工ETF",
    "510900": "H股ETF",
    "511880": DEFENSE_NAME,
}

WINDOWS = (1, 2, 3, 5, 10)  # 窗口收益: 1日/2日/3日/5日/10日
COL_NAME_W = 20  # 资产列显示宽度


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _disp_width(s: str) -> int:
    """显示宽度: 中文按2字符计 (用于对齐)."""
    return sum(2 if ord(c) > 127 else 1 for c in str(s))


def _pad(s, width: int) -> str:
    """按显示宽度填充/截断到 width (中文按2字符)."""
    s = str(s)
    if _disp_width(s) <= width:
        return s + " " * (width - _disp_width(s))
    # 超宽截断, 避免破坏中文字符
    out = ""
    w = 0
    for c in s:
        w += 2 if ord(c) > 127 else 1
        if w > width - 1:
            break
        out += c
    return out + "…"


def fmt_ret(x: float | None, width: int = 8) -> str:
    """收益率格式化: +8.00% / -5.00% / '-'."""
    if x is None or not np.isfinite(x):
        return _pad("-", width)
    return _pad(f"{x * 100:+7.2f}%", width)


def iso_week(d) -> tuple[int, int]:
    """ISO 周 (year, week)."""
    ts = pd.Timestamp(d)
    return (ts.isocalendar().year, ts.isocalendar().week)


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_pool_data(pool: dict[str, str], data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """加载池内全部资产的 parquet 缓存 (与回测同源, 前复权)."""
    data = {}
    for code in [*list(pool.keys()), DEFENSE]:
        f = data_dir / f"{code}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            df = df.sort_values("trade_date").reset_index(drop=True)
            data[code] = df
    return data


def truncate_asof(df: pd.DataFrame, asof: dt_date) -> pd.DataFrame:
    """截断到基准日 (无未来数据)."""
    return df[df["trade_date"] <= asof].reset_index(drop=True)


def ref_calendar(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """参考交易日历: 取覆盖最长的资产序列 (通常是货币基金/黄金)."""
    return max(data.values(), key=len)


# --------------------------------------------------------------------------- #
# 收益计算
# --------------------------------------------------------------------------- #
def daily_rets(df: pd.DataFrame, n: int) -> list[tuple[dt_date, float]]:
    """最近 n 个交易日的每日涨跌幅 [(date, ret), ...], 时间正序."""
    close = df["close"].astype(float).values
    dates = df["trade_date"].tolist()
    out = []
    for i in range(max(len(close) - n, 1), len(close)):
        if close[i - 1] > 0:
            out.append((dates[i], close[i] / close[i - 1] - 1.0))
    return out


def window_rets(df: pd.DataFrame) -> dict[int, float | None]:
    """截至最新交易日的多窗口收益 {n: R(n)}, R(n) = P_t/P_{t-n} - 1."""
    close = df["close"].astype(float).values
    out: dict[int, float | None] = {}
    for n in WINDOWS:
        if len(close) > n and close[-n - 1] > 0:
            out[n] = close[-1] / close[-n - 1] - 1.0
        else:
            out[n] = None
    return out


def week_returns(
    df: pd.DataFrame, week_map: list[tuple[tuple[int, int], dt_date, dt_date, int]]
) -> dict[str, float | None]:
    """各自然周累计收益.

    Args:
        df: 资产序列 (截断到基准日)
        week_map: [(week_key, first_date, last_date, weekday_of_last), ...]
            由参考日历生成, 每个元素描述一个交易周的起止交易日.

    Returns:
        {week_label: ret}; week_label 为 '本周'/'上周'/'上上周'/...
        周收益 = 周内最后收盘 / 周首日前一收盘 - 1 (包含周末跳空).
    """
    if df.empty:
        return {}
    close_series = df.set_index("trade_date")["close"].astype(float)
    out: dict[str, float | None] = {}
    for label, first, last, _weekday in week_map:
        sub = close_series.loc[(close_series.index >= first) & (close_series.index <= last)]
        before = close_series.loc[close_series.index < first]
        if sub.empty or before.empty:
            out[label] = None
        else:
            out[label] = float(sub.iloc[-1]) / float(before.iloc[-1]) - 1.0
    return out


def build_week_map(
    cal: pd.DataFrame, n_weeks: int, asof: dt_date
) -> list[tuple[str, dt_date, dt_date, int]]:
    """从参考日历生成最近 n_weeks 个自然周区间.

    Returns:
        [(label, first_date, last_date, last_weekday), ...] 时间倒序 (最近在前).
        label: 最新周='本周', 依次 '上周'/'上上周'/'上上上周'...
    """
    cal = truncate_asof(cal, asof)
    if cal.empty:
        return []
    by_week: dict[tuple[int, int], list[dt_date]] = {}
    for d in cal["trade_date"]:
        by_week.setdefault(iso_week(d), []).append(d)
    week_keys = sorted(by_week.keys())  # 时间正序
    recent = week_keys[-n_weeks:]

    names = ["本周", "上周", "上上周", "上上上周", "上上上上周", "上上上上上周"]
    out = []
    for i, key in enumerate(reversed(recent)):
        dates = by_week[key]
        label = names[i] if i < len(names) else f"W{key[0]}-{key[1]:02d}"
        out.append((label, dates[0], dates[-1], pd.Timestamp(dates[-1]).dayofweek))
    return out


# --------------------------------------------------------------------------- #
# 输出: 表格
# --------------------------------------------------------------------------- #
def print_daily_table(data: dict, pool: dict, n_days: int, cal: pd.DataFrame) -> None:
    """逐日涨跌矩阵: 资产 × 参考日历最近N个交易日 (缺失显示'-')."""
    max_dates = cal["trade_date"].tolist()[-n_days:]
    if not max_dates:
        return

    rows = []
    for code, df in data.items():
        if code not in pool and code != DEFENSE:
            continue
        ret_map = dict(daily_rets(df, n_days + 1))
        rows.append((code, ret_map))

    print(f"\n【逐日涨跌】(最近{len(max_dates)}个交易日, 单位 %), 时间正序")
    header = _pad("资产", COL_NAME_W) + "".join(_pad(d.strftime("%m-%d"), 9) for d in max_dates)
    print("  " + header)
    print("  " + "-" * (COL_NAME_W + 9 * len(max_dates)))

    for code, ret_map in rows:
        name = pool.get(code, DEFENSE_NAME)
        line = _pad(f"{name}({code})", COL_NAME_W)
        for d in max_dates:
            line += fmt_ret(ret_map.get(d))
        print("  " + line)


def print_window_table(data: dict, pool: dict) -> None:
    """窗口收益矩阵: 资产 × R(1)/R(2)/R(3)/R(5)/R(10)."""
    print("\n【窗口收益】(截至最新交易日, 单位 %), R(n)=n日累计涨幅")
    header = _pad("资产", COL_NAME_W) + "".join(_pad(f"R{n}", 9) for n in WINDOWS)
    print("  " + header)
    print("  " + "-" * (COL_NAME_W + 9 * len(WINDOWS)))
    for code, df in data.items():
        if code not in pool and code != DEFENSE:
            continue
        name = pool.get(code, DEFENSE_NAME)
        wr = window_rets(df)
        line = _pad(f"{name}({code})", COL_NAME_W)
        for n in WINDOWS:
            line += fmt_ret(wr[n])
        print("  " + line)


def print_week_table(data: dict, pool: dict, week_map: list) -> None:
    """周收益矩阵: 资产 × 本周/上周/上上周..."""
    labels = [lbl for lbl, *_ in week_map]
    note = "(本周=周初至今)" if week_map and week_map[0][3] < 4 else ""
    print(f"\n【周收益对比】({', '.join(labels)}, 单位 %) {note}")
    header = _pad("资产", COL_NAME_W) + "".join(_pad(lbl, 9) for lbl in labels)
    print("  " + header)
    print("  " + "-" * (COL_NAME_W + 9 * len(labels)))
    for code, df in data.items():
        if code not in pool and code != DEFENSE:
            continue
        name = pool.get(code, DEFENSE_NAME)
        wr = week_returns(df, week_map)
        line = _pad(f"{name}({code})", COL_NAME_W)
        for lbl in labels:
            line += fmt_ret(wr.get(lbl))
        print("  " + line)


# --------------------------------------------------------------------------- #
# 输出: 叙事摘要
# --------------------------------------------------------------------------- #
def build_summary(
    data: dict, pool: dict, week_map: list, asof: dt_date
) -> str:
    """自然语言摘要 (对应'上周白银+8% / 今天原油+2.75%'这类观察)."""
    active = {c: df for c, df in data.items() if c in pool or c == DEFENSE}
    lines: list[str] = []
    lines.append(f"【短期窗口观察】基准日 {asof}")

    # 1. 今日领涨/领跌 (最新一根K线单日收益)
    today_rets = {}
    for code, df in active.items():
        r = daily_rets(df, 1)
        if r:
            today_rets[code] = r[-1][1]
    if today_rets:
        ranked = sorted(today_rets.items(), key=lambda kv: -kv[1])
        best, worst = ranked[0], ranked[-1]
        b_name, w_name = pool.get(best[0], DEFENSE_NAME), pool.get(worst[0], DEFENSE_NAME)
        lines.append(
            f"• 今日: {b_name} {best[1] * 100:+.2f}% 领涨"
            f" | {w_name} {worst[1] * 100:+.2f}% 领跌"
            f" (差距 {(best[1] - worst[1]) * 100:.2f}pp)"
        )

    # 2. 3日窗口最强/最弱 (2-3日窗口重点)
    r3 = {}
    for code, df in active.items():
        wr = window_rets(df)
        if wr[3] is not None:
            r3[code] = wr[3]
    if r3:
        ranked = sorted(r3.items(), key=lambda kv: -kv[1])
        best, worst = ranked[0], ranked[-1]
        b_name, w_name = pool.get(best[0], DEFENSE_NAME), pool.get(worst[0], DEFENSE_NAME)
        spread = (best[1] - worst[1]) * 100
        lines.append(
            f"• 3日窗口: {b_name} {best[1] * 100:+.2f}% 最强"
            f" vs {w_name} {worst[1] * 100:+.2f}% 最弱 (分歧 {spread:.2f}pp)"
        )

    # 3. 周际动能对比: 每只资产 本周 vs 上周 vs 上上周, 突出反转/加速
    if week_map:
        labels = [lbl for lbl, *_ in week_map]
        has_prev = len(labels) >= 2
        movers = []
        for code, df in active.items():
            wr = week_returns(df, week_map)
            cur = wr.get(labels[0])
            if cur is None:
                continue
            prev = wr.get(labels[1]) if has_prev else None
            prev2 = wr.get(labels[2]) if len(labels) >= 3 else None
            name = pool.get(code, DEFENSE_NAME)
            # 周际变化: 本周 vs 上周, 或上周 vs 上上周
            pairs = [(labels[0], cur, labels[1], prev), (labels[1], prev, labels[2], prev2)]
            for lbl_a, a, lbl_b, b in pairs:
                if a is not None and b is not None and abs((a - b) * 100) >= 3.0:
                    direction = "反转" if a * b < 0 else ("加速" if abs(a) > abs(b) else "降温")
                    movers.append(
                        f"{name} {lbl_b}{b * 100:+.2f}% → "
                        f"{lbl_a}{a * 100:+.2f}% ({direction})"
                    )
        if movers:
            lines.append("• 周际动能变化:")
            for m in movers[:6]:
                lines.append(f"    - {m}")

    # 4. 5日窗口排名 Top3
    r5 = {}
    for code, df in active.items():
        wr = window_rets(df)
        if wr[5] is not None:
            r5[code] = wr[5]
    if r5:
        top3 = sorted(r5.items(), key=lambda kv: -kv[1])[:3]
        names = [f"{pool.get(c, DEFENSE_NAME)}{v * 100:+.2f}%" for c, v in top3]
        lines.append("• 5日(周)窗口 Top3: " + ", ".join(names))

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(args) -> tuple[dict, str]:
    pool = POOL_CORE
    if args.all:
        # 扩展池: cross_asset 目录全部资产, 名称取 EXTRA_NAMES / 未知用代码
        discovered = {}
        for f in sorted(DATA_DIR.glob("*.parquet")):
            code = f.stem
            if code in POOL_CORE:
                discovered[code] = POOL_CORE[code]
            elif code == DEFENSE:
                continue
            else:
                discovered[code] = EXTRA_NAMES.get(code, f"ETF {code}")
        pool = discovered

    data = load_pool_data(pool)
    if not data:
        raise SystemExit(f"❌ 无数据缓存: {DATA_DIR}, 请先运行 run_qixing.py 拉取")

    if args.update:
        from live_signal import update_data
        print("  增量更新行情...")
        data = update_data(data)
        data = load_pool_data(pool)  # 重新加载 (update_data 已写回 parquet)

    # 基准日
    asof = dt_date.fromisoformat(args.asof) if args.asof else None
    data = {c: truncate_asof(df, asof) for c, df in data.items()} if asof else data
    if asof is None:
        asof = ref_calendar(data)["trade_date"].max()
        data = {c: truncate_asof(df, asof) for c, df in data.items()}

    cal = ref_calendar(data)
    week_map = build_week_map(cal, args.weeks, asof)

    summary = build_summary(data, pool, week_map, asof)

    result: dict = {
        "asof": str(asof),
        "pool": dict(pool.items()),
        "assets": [],
    }
    active = {c: df for c, df in data.items() if c in pool or c == DEFENSE}
    for code, df in active.items():
        result["assets"].append({
            "code": code,
            "name": pool.get(code, DEFENSE_NAME),
            "daily": {str(d): round(r, 6) for d, r in daily_rets(df, args.days)},
            "windows": {f"r{n}": (round(v, 6) if v is not None else None)
                        for n, v in window_rets(df).items()},
            "weeks": {lbl: (round(v, 6) if v is not None else None)
                      for lbl, v in week_returns(df, week_map).items()},
        })
    result["summary"] = summary

    # === 终端输出 ===
    if not args.json:
        print("=" * 72)
        print(f"  短期窗口涨幅监控 | 基准日 {asof} | 资产 {len(active)} 只")
        print(f"  池: {', '.join(pool.values())}")
        print("=" * 72)
        if not args.narrative:
            print_daily_table(data, pool, args.days, cal)
            print_window_table(data, pool)
            if week_map:
                print_week_table(data, pool, week_map)
        print()
        print(summary)

    if args.notify:
        from notify import push_bark
        push_bark(f"📊 短期窗口监控 {asof}", summary, level="active")
        print("\n  ✓ 已推送 Bark")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="短期窗口涨幅监控器 (多资产短周期对比)")
    parser.add_argument("--days", type=int, default=5, help="逐日视图: 最近N个交易日 (默认5)")
    parser.add_argument("--weeks", type=int, default=3, help="周视图: 最近N个自然周 (默认3)")
    parser.add_argument("--asof", default=None, help="基准日 YYYY-MM-DD (默认最新交易日)")
    parser.add_argument("--all", action="store_true", help="扩展到 cross_asset 全部资产")
    parser.add_argument("--update", action="store_true", help="先增量更新行情再分析")
    parser.add_argument("--narrative", action="store_true", help="只输出叙事摘要")
    parser.add_argument("--json", action="store_true", help="输出结构化JSON")
    parser.add_argument("--notify", action="store_true", help="Bark推送摘要")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
