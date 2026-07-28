"""Fetch extended ETF universe for factor screening.

Adds sector/style ETFs to the existing 5 symbols.
Quota cost: ~20 symbols × 247 days ≈ 5000 rows (negligible).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "real"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2025-04-11"
END_DATE = "2026-04-18"

# Extended ETF universe (sector + style + size)
EXTENDED_SYMBOLS = {
    # Broad market (already fetched)
    # "510300.XSHG": "沪深300ETF",
    # "510500.XSHG": "中证500ETF",
    # "159915.XSHE": "创业板ETF",
    # Size factors
    "510050.XSHG": "上证50ETF",
    "512100.XSHG": "中证1000ETF",
    "159901.XSHE": "深证100ETF",
    # Sector factors
    "512010.XSHG": "医药ETF",
    "512660.XSHG": "军工ETF",
    "512880.XSHG": "证券ETF",
    "515030.XSHG": "新能源车ETF",
    "512690.XSHG": "酒ETF",
    "512480.XSHG": "半导体ETF",
    "515790.XSHG": "光伏ETF",
    "512200.XSHG": "房地产ETF",
    "515220.XSHG": "煤炭ETF",
    "512800.XSHG": "银行ETF",
    # Style factors
    "510880.XSHG": "红利ETF",
    "159949.XSHE": "创业板50ETF",
    "512170.XSHG": "医疗ETF",
    "515050.XSHG": "5GETF",
    "159869.XSHE": "游戏ETF",
}


def fetch_extended():
    """Fetch extended ETF data."""
    import jqdatasdk
    from a_share_quant.settings import get_settings

    settings = get_settings()
    print("认证 JQData...")
    jqdatasdk.auth(settings.joinquant_username, settings.joinquant_password)

    quota_before = jqdatasdk.get_query_count()
    print(f"配额: {quota_before}\n")

    fetched = 0
    skipped = 0

    for jq_symbol, name in EXTENDED_SYMBOLS.items():
        cache_file = DATA_DIR / f"{jq_symbol.replace('.', '_')}.parquet"

        if cache_file.exists():
            print(f"  [缓存] {name} ({jq_symbol})")
            skipped += 1
            continue

        print(f"  [拉取] {name} ({jq_symbol})")
        try:
            df = jqdatasdk.get_price(
                jq_symbol,
                start_date=START_DATE,
                end_date=END_DATE,
                frequency="daily",
                panel=False,
                fq="pre",
            )

            if df is not None and not df.empty:
                df = df.reset_index()
                df = df.rename(columns={"index": "trade_date", "money": "amount"})
                if "trade_date" not in df.columns and "time" in df.columns:
                    df = df.rename(columns={"time": "trade_date"})
                df["symbol"] = jq_symbol

                cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
                if "amount" in df.columns:
                    cols.append("amount")
                df = df[[c for c in cols if c in df.columns]]

                df.to_parquet(cache_file, index=False)
                fetched += 1
                print(f"         → {len(df)} 条")
            else:
                print(f"         → 无数据")

        except Exception as e:
            print(f"         → 错误: {e}")

    # Rebuild combined dataset
    all_files = list(DATA_DIR.glob("*.parquet"))
    all_dfs = []
    for f in all_files:
        if f.name == "combined_daily.parquet":
            continue
        all_dfs.append(pd.read_parquet(f))

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(DATA_DIR / "combined_daily.parquet", index=False)
        print(f"\n合并数据集: {len(combined)} 条, {combined['symbol'].nunique()} 个标的")

    quota_after = jqdatasdk.get_query_count()
    print(f"\n本次拉取: {fetched} 个, 跳过: {skipped} 个")
    print(f"配额变化: {quota_before} → {quota_after}")


if __name__ == "__main__":
    fetch_extended()
