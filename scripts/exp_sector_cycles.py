"""行业ETF周期观察: 抓取主要行业ETF约10年数据, 绘制价格周期曲线 + 年度收益热力图.

目的: 直观看到每个行业的涨跌周期, 以及"行业轮动"的本质(没有永远涨的行业)。
用法: uv run python scripts/exp_sector_cycles.py
输出: data/sector_etf/sector_price_cycles.png + sector_annual_heatmap.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "SimHei",
    "Noto Sans CJK SC",
]
matplotlib.rcParams["axes.unicode_minus"] = False

SECTOR_DIR = Path(__file__).parent.parent / "data" / "sector_etf"
SECTOR_DIR.mkdir(parents=True, exist_ok=True)

# 行业ETF池 (代码: 名称), 含两个宽基做对照
SECTORS = {
    "510150": "消费",
    "512010": "医药",
    "512880": "证券",
    "512660": "军工",
    "512800": "银行",
    "512400": "有色",
    "512200": "房地产",
    "512720": "计算机",
    "515030": "新能源车",
    "512760": "芯片",
    "515220": "煤炭",
    "512690": "酒",
    "510300": "沪深300",
    "159915": "创业板",
}


def sina_symbol(code: str) -> str:
    return ("sh" if code.startswith(("5", "6")) else "sz") + code


def fetch_all() -> None:
    import akshare as ak

    for code, name in SECTORS.items():
        f = SECTOR_DIR / f"{code}.parquet"
        if f.exists():
            print(f"  {code} {name} 已缓存, 跳过")
            continue
        print(f"  拉取 {code} {name}...", end=" ", flush=True)
        try:
            raw = ak.fund_etf_hist_sina(symbol=sina_symbol(code))
            if raw is None or raw.empty:
                print("无数据")
                continue
            df = raw.rename(columns={"date": "trade_date"})
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            for col in ["open", "close", "high", "low", "volume"]:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[["trade_date", "open", "close", "high", "low", "volume"]]
            df = df.sort_values("trade_date").reset_index(drop=True)
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = df[col].astype("float64")
            df.to_parquet(f, index=False)
            print(f"{len(df)}天 ({df['trade_date'].min()}~{df['trade_date'].max()})")
            time.sleep(1.0)
        except Exception as e:  # noqa: BLE001
            print(f"失败: {e}")


def load_all() -> dict:
    data = {}
    for code in SECTORS:
        f = SECTOR_DIR / f"{code}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = adjust_splits(df)
            data[code] = df.sort_values("trade_date").reset_index(drop=True)
    return data


def adjust_splits(df: pd.DataFrame, threshold: float = -0.30) -> pd.DataFrame:
    """自动检测份额拆分并前复权.

    新浪数据不复权: ETF拆分当日价格骟降(如1拆10变十分之一), 误似暴跌。
    单日跌幅>30%对行业ETF几乎必然是拆分(非真实下跌), 据此检测并前复权。
    """
    df = df.copy().sort_values("trade_date").reset_index(drop=True)
    close = df["close"].values.astype(float)
    n = len(close)
    if n < 2:
        return df
    factor = np.ones(n)
    cum = 1.0
    for i in range(n - 1, 0, -1):
        factor[i] = cum
        prev = close[i - 1]
        if prev > 0:
            ret = (close[i] - prev) / prev
            if ret < threshold:  # 疑似拆分
                cum *= prev / close[i]
    factor[0] = cum
    for col in ["open", "close", "high", "low"]:
        df[col] = df[col].astype(float) / factor
    df["volume"] = df["volume"].astype(float) * factor
    return df


def yearly_returns(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["year"] = df["trade_date"].dt.year
    out = {}
    for y, g in df.groupby("year"):
        if len(g) >= 2:
            out[int(y)] = float(g["close"].iloc[-1] / g["close"].iloc[0] - 1)
    return out


def plot_price_cycles(data: dict) -> None:
    """归一化价格曲线 (2016年起=100, 对数坐标)."""
    fig, ax = plt.subplots(figsize=(14, 8))
    start = pd.Timestamp("2016-01-01")
    for code, name in SECTORS.items():
        if code not in data:
            continue
        df = data[code]
        sub = df[df["trade_date"] >= start]
        if len(sub) < 100:
            continue
        base = sub["close"].iloc[0]
        ax.plot(
            sub["trade_date"], sub["close"] / base * 100, label=f"{name}({code})", linewidth=1.3
        )
    ax.set_yscale("log")
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("行业ETF价格周期曲线 (2016年起归一化=100, 对数坐标)", fontsize=15)
    ax.set_ylabel("归一化价格 (起点=100)")
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    out = SECTOR_DIR / "sector_price_cycles.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 价格周期图: {out}")


def plot_heatmap(data: dict) -> None:
    """年度收益热力图 (行业×年份)."""
    years = list(range(2016, 2027))
    codes = [c for c in SECTORS if c in data]
    matrix = np.full((len(codes), len(years)), np.nan)
    for i, code in enumerate(codes):
        yr = yearly_returns(data[code])
        for j, y in enumerate(years):
            if y in yr:
                matrix[i, j] = yr[y] * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 1)
    vmax = min(vmax, 100)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([f"{SECTORS[c]}({c})" for c in codes])
    for i in range(len(codes)):
        for j in range(len(years)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(v) < vmax * 0.6 else "white",
                )
    ax.set_title("行业ETF年度收益热力图 (红涨绿跌, 看轮动)", fontsize=15)
    fig.colorbar(im, ax=ax, label="年度收益 %", shrink=0.8)
    out = SECTOR_DIR / "sector_annual_heatmap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 年度热力图: {out}")


def main() -> None:
    print("=== 抓取行业ETF数据 ===")
    fetch_all()
    data = load_all()
    print(f"\n=== 已加载 {len(data)} 只, 绘图 ===")
    plot_price_cycles(data)
    plot_heatmap(data)
    # 打印年度收益表
    years = list(range(2016, 2027))
    print("\n=== 年度收益表 (%) ===")
    header = "行业      " + "".join(f"{y:>7}" for y in years)
    print(header)
    for code in SECTORS:
        if code not in data:
            continue
        yr = yearly_returns(data[code])
        row = f"{SECTORS[code]:<8}" + "".join(
            f"{yr[y] * 100:>+6.0f} " if y in yr else "    -  " for y in years
        )
        print(row)


if __name__ == "__main__":
    main()
