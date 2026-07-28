"""Fetch long-term ETF history via akshare (sina source, no rate limit).

Uses fund_etf_hist_sina (新浪源) instead of fund_etf_hist_em (东财源).
Sina source is more stable and doesn't have aggressive rate limiting.

Symbol format: "sh510300" for Shanghai, "sz159915" for Shenzhen.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "long_history"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Full ETF universe with sina symbol format
ETF_UNIVERSE = {
    # Broad market
    "sh510300": "沪深300ETF",
    "sh510500": "中证500ETF",
    "sz159915": "创业板ETF",
    "sh510050": "上证50ETF",
    "sh512100": "中证1000ETF",
    "sz159901": "深证100ETF",
    # Sector
    "sh512010": "医药ETF",
    "sh512660": "军工ETF",
    "sh512880": "证券ETF",
    "sh515030": "新能源车ETF",
    "sh512690": "酒ETF",
    "sh512480": "半导体ETF",
    "sh515790": "光伏ETF",
    "sh512200": "房地产ETF",
    "sh515220": "煤炭ETF",
    "sh512800": "银行ETF",
    # Style
    "sh510880": "红利ETF",
    "sz159949": "创业板50ETF",
    "sh512170": "医疗ETF",
    "sh515050": "5GETF",
    "sz159869": "游戏ETF",
}

# Index for regime detection (use sina index interface)
INDEX_UNIVERSE = {
    "sh000300": "沪深300指数",
    "sh000905": "中证500指数",
}

# Filter date range
START_DATE = "2022-01-01"
END_DATE = "2026-04-18"


def fetch_etf_history() -> None:
    """Fetch long-term ETF history via akshare sina source."""
    import akshare as ak

    print("=" * 60)
    print("  akshare 长历史数据拉取 (新浪源)")
    print(f"  时间范围: {START_DATE} ~ {END_DATE}")
    print(f"  标的数: {len(ETF_UNIVERSE)} ETF + {len(INDEX_UNIVERSE)} 指数")
    print("=" * 60)

    fetched = 0
    failed = []

    # Fetch ETFs via fund_etf_hist_sina
    for sina_symbol, name in ETF_UNIVERSE.items():
        code = sina_symbol[2:]  # Remove sh/sz prefix
        cache_file = DATA_DIR / f"{code}.parquet"

        if cache_file.exists():
            df_check = pd.read_parquet(cache_file)
            print(f"  [缓存] {name} ({code}) → {len(df_check)} 条")
            fetched += 1
            continue

        print(f"  [拉取] {name} ({sina_symbol})...", end=" ", flush=True)
        try:
            df = ak.fund_etf_hist_sina(symbol=sina_symbol)

            if df is not None and not df.empty:
                # Standardize columns
                df = df.rename(columns={"date": "trade_date"})
                df["symbol"] = code
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

                # Filter date range
                start_d = pd.to_datetime(START_DATE).date()
                end_d = pd.to_datetime(END_DATE).date()
                df = df[(df["trade_date"] >= start_d) & (df["trade_date"] <= end_d)]

                cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
                df = df[[c for c in cols if c in df.columns]]
                df = df.reset_index(drop=True)

                df.to_parquet(cache_file, index=False)
                fetched += 1
                print(f"→ {len(df)} 条 ({df['trade_date'].min()} ~ {df['trade_date'].max()})")
            else:
                print("→ 无数据")
                failed.append((code, name, "无数据"))

        except Exception as e:
            print(f"→ 错误: {e}")
            failed.append((code, name, str(e)))

        # Sina is more lenient, 1s is enough
        time.sleep(1.0)

    # Fetch indices via stock_zh_index_daily (sina source)
    print(f"\n  --- 指数数据 ---")
    for sina_symbol, name in INDEX_UNIVERSE.items():
        code = sina_symbol[2:]
        cache_file = DATA_DIR / f"index_{code}.parquet"

        if cache_file.exists():
            df_check = pd.read_parquet(cache_file)
            print(f"  [缓存] {name} ({code}) → {len(df_check)} 条")
            fetched += 1
            continue

        print(f"  [拉取] {name} ({sina_symbol})...", end=" ", flush=True)
        try:
            # stock_zh_index_daily uses sina source
            df = ak.stock_zh_index_daily(symbol=sina_symbol)

            if df is not None and not df.empty:
                df = df.rename(columns={"date": "trade_date"})
                df["symbol"] = code
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

                # Filter date range
                start_d = pd.to_datetime(START_DATE).date()
                end_d = pd.to_datetime(END_DATE).date()
                df = df[(df["trade_date"] >= start_d) & (df["trade_date"] <= end_d)]

                cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
                df = df[[c for c in cols if c in df.columns]]
                df = df.reset_index(drop=True)

                df.to_parquet(cache_file, index=False)
                fetched += 1
                print(f"→ {len(df)} 条 ({df['trade_date'].min()} ~ {df['trade_date'].max()})")
            else:
                print("→ 无数据")
                failed.append((code, name, "无数据"))

        except Exception as e:
            print(f"→ 错误: {e}")
            failed.append((code, name, str(e)))

        time.sleep(1.0)

    # Build combined dataset
    print(f"\n  --- 合并数据集 ---")
    all_files = list(DATA_DIR.glob("*.parquet"))
    all_dfs = []
    for f in all_files:
        if f.name == "combined_long.parquet":
            continue
        all_dfs.append(pd.read_parquet(f))

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(DATA_DIR / "combined_long.parquet", index=False)
        n_symbols = combined["symbol"].nunique()
        date_range = f"{combined['trade_date'].min()} ~ {combined['trade_date'].max()}"
        print(f"  合并: {len(combined)} 条, {n_symbols} 个标的, {date_range}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  完成: {fetched} 个标的")
    if failed:
        print(f"  失败: {len(failed)} 个")
        for code, name, err in failed:
            print(f"    {name} ({code}): {err}")
    print(f"  数据目录: {DATA_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    fetch_etf_history()
