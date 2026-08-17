"""Fetch external factor data: northbound capital flow + PE percentile.

Northbound: ak.stock_hsgt_hist_em (note: data may be incomplete after 2024-08)
PE percentile: ak.stock_a_indicator_lg (乐咕乐股, index-level PE)

Fallback: if external data unavailable, factors default to 0 (neutral).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "long_history"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2022-01-01"
END_DATE = "2026-04-18"


def fetch_northbound():
    """Fetch northbound capital net flow history."""
    import akshare as ak

    cache_file = DATA_DIR / "northbound.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        print(
            f"  [缓存] 北向资金: {len(df)} 条 ({df['trade_date'].min()} ~ {df['trade_date'].max()})"
        )
        return

    print("  [拉取] 北向资金历史数据...")
    try:
        # stock_hsgt_hist_em: 北向资金历史
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if df is not None and not df.empty:
            print(f"    原始列: {df.columns.tolist()}")
            print(f"    原始行数: {len(df)}")

            # Standardize
            # Typical columns: 日期, 当日资金流入, 当日余额, ...
            # or: date, net_flow, ...
            rename_map = {}
            for col in df.columns:
                if "日期" in col or "date" in col.lower():
                    rename_map[col] = "trade_date"
                elif "净流入" in col or "net" in col.lower() or "资金流入" in col:
                    rename_map[col] = "net_flow"

            if rename_map:
                df = df.rename(columns=rename_map)

            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

                # Filter date range
                start_d = pd.to_datetime(START_DATE).date()
                end_d = pd.to_datetime(END_DATE).date()
                df = df[(df["trade_date"] >= start_d) & (df["trade_date"] <= end_d)]

                # Keep relevant columns
                keep_cols = ["trade_date"]
                if "net_flow" in df.columns:
                    keep_cols.append("net_flow")
                else:
                    # Try to find numeric columns
                    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    keep_cols.extend(numeric_cols[:3])
                    if numeric_cols:
                        df = df.rename(columns={numeric_cols[0]: "net_flow"})
                        keep_cols = ["trade_date", "net_flow"]

                df = df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)
                df.to_parquet(cache_file, index=False)
                print(
                    f"    保存: {len(df)} 条 ({df['trade_date'].min()} ~ {df['trade_date'].max()})"
                )
            else:
                print(f"    警告: 未找到日期列, 列名: {df.columns.tolist()}")
        else:
            print("    无数据返回")
    except Exception as e:
        print(f"    错误: {e}")
        # Create empty placeholder
        print("    创建空占位文件 (因子将默认为0)")
        pd.DataFrame({"trade_date": [], "net_flow": []}).to_parquet(cache_file, index=False)


def fetch_pe_percentile():
    """Fetch PE data for major indices (for PE percentile calculation)."""
    import akshare as ak

    cache_file = DATA_DIR / "pe_percentile.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        print(f"  [缓存] PE百分位: {len(df)} 条, {df['index_code'].nunique()} 个指数")
        return

    # Major indices that our ETFs track
    indices = {
        "000300": "沪深300",
        "000905": "中证500",
        "000016": "上证50",
        "399006": "创业板指",
        "000852": "中证1000",
    }

    all_dfs = []
    for code, name in indices.items():
        print(f"  [拉取] {name} ({code}) PE数据...", end=" ", flush=True)
        try:
            # Try index_value_hist_funddb (韭圈儿)
            df = ak.index_value_hist_funddb(symbol=name, indicator="市盈率")
            if df is not None and not df.empty:
                df = df.rename(
                    columns={
                        df.columns[0]: "trade_date",
                        df.columns[1]: "pe",
                    }
                )
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
                df["index_code"] = code

                start_d = pd.to_datetime(START_DATE).date()
                end_d = pd.to_datetime(END_DATE).date()
                df = df[(df["trade_date"] >= start_d) & (df["trade_date"] <= end_d)]

                df = df[["index_code", "trade_date", "pe"]].reset_index(drop=True)
                all_dfs.append(df)
                print(f"→ {len(df)} 条")
            else:
                print("→ 无数据")
        except Exception as e:
            print(f"→ 错误: {e}")
            # Fallback: try stock_a_indicator_lg
            try:
                df2 = ak.stock_a_indicator_lg(symbol=code)
                if df2 is not None and not df2.empty:
                    # Has pe, pe_ttm, pb columns
                    if "trade_date" in df2.columns:
                        df2["trade_date"] = pd.to_datetime(df2["trade_date"]).dt.date
                    elif "日期" in df2.columns:
                        df2 = df2.rename(columns={"日期": "trade_date"})
                        df2["trade_date"] = pd.to_datetime(df2["trade_date"]).dt.date

                    pe_col = "pe" if "pe" in df2.columns else "pe_ttm"
                    if pe_col in df2.columns:
                        df2 = df2.rename(columns={pe_col: "pe"})
                        df2["index_code"] = code
                        start_d = pd.to_datetime(START_DATE).date()
                        end_d = pd.to_datetime(END_DATE).date()
                        df2 = df2[(df2["trade_date"] >= start_d) & (df2["trade_date"] <= end_d)]
                        df2 = df2[["index_code", "trade_date", "pe"]].reset_index(drop=True)
                        all_dfs.append(df2)
                        print(f"    (fallback) → {len(df2)} 条")
            except Exception as e2:
                print(f"    (fallback也失败: {e2})")

        time.sleep(1.5)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(cache_file, index=False)
        print(f"  合并PE数据: {len(combined)} 条, {combined['index_code'].nunique()} 个指数")
    else:
        print("  警告: 无PE数据, 创建空占位")
        pd.DataFrame({"index_code": [], "trade_date": [], "pe": []}).to_parquet(
            cache_file, index=False
        )


def main():
    print("=" * 60)
    print("  外部因子数据拉取: 北向资金 + PE百分位")
    print("=" * 60)

    print("\n[1/2] 北向资金...")
    fetch_northbound()

    print("\n[2/2] PE百分位...")
    fetch_pe_percentile()

    print("\n完成!")


if __name__ == "__main__":
    main()
