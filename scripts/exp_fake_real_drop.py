"""真假跌识别实验: 测试网上常见区分因子(成交量/趋势位置/波动率)能否识别真跌vs假摔.

方法: 收集池内所有ETF的"单日大跌>5%"事件(大样本), 计算各因子,
看哪些因子能显著提高"触发后继续跌(真跌)"的比例。
用法: uv run python scripts/exp_fake_real_drop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
from strategy_lab.engine import WARMUP, get_common_dates  # noqa: E402

DROP_THR = -0.05   # 单日大跌阈值
HORIZON = 5        # 观察后续5日


def collect_drop_events(data: dict) -> pd.DataFrame:
    """收集所有ETF的单日大跌事件及其因子与后续走势."""
    events = []
    for code in rq.ETF_POOL:
        if code not in data:
            continue
        df = data[code].sort_values("trade_date").reset_index(drop=True)
        close = df["close"].astype(float).values
        vol = df["volume"].astype(float).values
        dates = df["trade_date"].tolist()
        for i in range(61, len(close) - HORIZON):
            day_ret = (close[i] - close[i - 1]) / close[i - 1]
            if day_ret < DROP_THR:
                avg_vol = vol[i - 21:i - 1].mean()
                vol_ratio = vol[i] / avg_vol if avg_vol > 0 else 1.0
                ret60 = (close[i] - close[i - 61]) / close[i - 61]
                vol20 = float(np.std(np.diff(close[i - 20:i]) / close[i - 20:i - 1]))
                fwd5 = (close[i + HORIZON] - close[i]) / close[i]
                events.append({
                    "code": code, "date": dates[i], "vol_ratio": vol_ratio,
                    "ret60": ret60, "vol20": vol20, "fwd5": fwd5,
                })
    return pd.DataFrame(events)


def split_report(df: pd.DataFrame, col: str, lo_label: str, hi_label: str) -> None:
    """按某因子中位数分高低组, 对比继续跌比例."""
    med = df[col].median()
    lo = df[df[col] <= med]
    hi = df[df[col] > med]
    def cont(d):
        return (d["fwd5"] < 0).mean() * 100 if len(d) else 0
    def avg(d):
        return d["fwd5"].mean() * 100 if len(d) else 0
    print(f"    {lo_label}({len(lo)}次): 继续跌 {cont(lo):.0f}% | 后5日均 {avg(lo):+.2f}%")
    print(f"    {hi_label}({len(hi)}次): 继续跌 {cont(hi):.0f}% | 后5日均 {avg(hi):+.2f}%")
    print(f"    → 区分度: {abs(cont(hi)-cont(lo)):.0f}百分点")


def main() -> None:
    data = rq.load_data()
    df = collect_drop_events(data)
    total = len(df)
    base_cont = (df["fwd5"] < 0).mean() * 100
    base_avg = df["fwd5"].mean() * 100

    print("=" * 64)
    print(f"  真假跌识别实验 | 单日大跌>{abs(DROP_THR)*100:.0f}%事件 共{total}次")
    print("=" * 64)
    print(f"\n  【基准】 大跌后5日: 继续跌(真跌) {base_cont:.0f}% | 反弹(假摔) {100-base_cont:.0f}% | 平均 {base_avg:+.2f}%")
    print(f"\n  【因子1: 成交量】 放量=真跌? 缩量=假摔?")
    split_report(df, "vol_ratio", "缩量组", "放量组")
    print(f"\n  【因子2: 趋势位置】 大涨后跌=真跌? 平淡跌=假摔?")
    split_report(df, "ret60", "前期平淡组", "前期大涨组")
    print(f"\n  【因子3: 波动率环境】 高波动=假摔? 低波动=真跌?")
    split_report(df, "vol20", "低波动组", "高波动组")
    print("=" * 64)
    print("  判读: 若某因子高低组'继续跌%'差距大(>10百分点), 说明它有区分力;")
    print("        若差距小(<5百分点), 说明它分不清真假跌(≈抛硬币)。")


if __name__ == "__main__":
    main()
