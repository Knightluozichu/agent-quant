"""Fetch real data from JQData and cache locally.

Quota-aware: minimizes API calls, caches to Parquet.
Data range: 2025-04-11 to 2026-04-18 (trial account limit).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "real"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Data range (trial account limit)
START_DATE = "2025-04-11"
END_DATE = "2026-04-18"

# Symbols to fetch (minimize quota usage)
SYMBOLS = {
    # ETFs (low cost, high liquidity, good for strategy testing)
    "510300.XSHG": "沪深300ETF",
    "510500.XSHG": "中证500ETF",
    "159915.XSHE": "创业板ETF",
    # Index for regime detection
    "000300.XSHG": "沪深300指数",
    "000905.XSHG": "中证500指数",
}


def fetch_and_cache():
    """Fetch data from JQData and cache to Parquet."""
    import jqdatasdk

    from a_share_quant.settings import get_settings

    settings = get_settings()
    print("认证 JQData...")
    jqdatasdk.auth(settings.joinquant_username, settings.joinquant_password)
    print("认证成功\n")

    # Check quota
    quota = jqdatasdk.get_query_count()
    print(f"配额: {quota}")

    all_data = []

    for jq_symbol, name in SYMBOLS.items():
        cache_file = DATA_DIR / f"{jq_symbol.replace('.', '_')}.parquet"

        if cache_file.exists():
            print(f"  [缓存] {name} ({jq_symbol})")
            df = pd.read_parquet(cache_file)
            all_data.append(df)
            continue

        print(f"  [拉取] {name} ({jq_symbol}) {START_DATE} ~ {END_DATE}")

        try:
            df = jqdatasdk.get_price(
                jq_symbol,
                start_date=START_DATE,
                end_date=END_DATE,
                frequency="daily",
                panel=False,
                fq="pre",  # 前复权
            )

            if df is not None and not df.empty:
                # JQData returns datetime as index, columns: open/close/high/low/volume/money
                df = df.reset_index()
                df = df.rename(columns={"index": "trade_date", "money": "amount"})
                # If index was unnamed
                if "trade_date" not in df.columns and "time" in df.columns:
                    df = df.rename(columns={"time": "trade_date"})
                df["symbol"] = jq_symbol

                # Keep essential columns
                cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
                if "amount" in df.columns:
                    cols.append("amount")
                df = df[[c for c in cols if c in df.columns]]

                # Cache
                df.to_parquet(cache_file, index=False)
                all_data.append(df)
                print(f"         → {len(df)} 条, 已缓存")
            else:
                print("         → 无数据")

        except Exception as e:
            print(f"         → 错误: {e}")

    # Save combined dataset
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined_file = DATA_DIR / "combined_daily.parquet"
        combined.to_parquet(combined_file, index=False)
        print(f"\n合并数据集: {len(combined)} 条 → {combined_file}")

        # Summary
        print(f"\n{'=' * 50}")
        print("数据摘要:")
        print(f"  标的数: {combined['symbol'].nunique()}")
        print(f"  日期范围: {combined['trade_date'].min()} ~ {combined['trade_date'].max()}")
        print(f"  总行数: {len(combined)}")
        print(f"{'=' * 50}")

    # Final quota check
    quota_after = jqdatasdk.get_query_count()
    print(f"\n配额使用: {quota}")
    print(f"配额剩余: {quota_after}")


if __name__ == "__main__":
    fetch_and_cache()
