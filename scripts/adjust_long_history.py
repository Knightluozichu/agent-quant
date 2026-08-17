"""复权修正 long_history 数据: 自动检测ETF份额拆分并前复权.

新浪原始数据不复权, ETF拆分当日价格骤降(如1拆10)会被误判为暴跌。
本脚本检测单日跌幅>30%的异常(对ETF几乎必然是拆分), 做前复权修正。
只处理ETF价格文件(文件名为纯数字代码), 跳过指数/因子文件。
用法: uv run python scripts/adjust_long_history.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "long_history"


def adjust_splits(df: pd.DataFrame, threshold: float = -0.30):
    """检测拆分并前复权, 返回(修正后df, 拆分记录列表)."""
    df = df.copy().sort_values("trade_date").reset_index(drop=True)
    if "close" not in df.columns:
        return df, []
    close = df["close"].values.astype(float)
    n = len(close)
    if n < 2:
        return df, []
    factor = np.ones(n)
    cum = 1.0
    splits = []
    for i in range(n - 1, 0, -1):
        factor[i] = cum
        prev = close[i - 1]
        if prev > 0:
            ret = (close[i] - prev) / prev
            if ret < threshold:  # 疑似拆分
                ratio = prev / close[i]
                cum *= ratio
                splits.append((str(df["trade_date"].iloc[i]), ratio, ret * 100))
    factor[0] = cum
    for col in ["open", "close", "high", "low"]:
        if col in df.columns:
            df[col] = df[col].astype(float) / factor
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float) * factor
    return df, splits


def main() -> None:
    files = sorted(DATA_DIR.glob("*.parquet"))
    total_splits = 0
    adjusted_files = 0
    print("=== 扫描并复权修正 long_history ETF数据 ===")
    for f in files:
        if not f.stem.isdigit():  # 只处理纯数字代码的ETF文件
            continue
        df = pd.read_parquet(f)
        df_adj, splits = adjust_splits(df)
        if splits:
            adjusted_files += 1
            total_splits += len(splits)
            sym = df["symbol"].iloc[0] if "symbol" in df.columns else f.stem
            print(f"  {f.stem} ({sym}): 检测到 {len(splits)} 次拆分")
            for d, ratio, ret in splits:
                print(f"      {d}: 拆分比例≈1:{ratio:.2f} (当日{ret:+.1f}%)")
            df_adj.to_parquet(f, index=False)
    print(f"\n=== 完成: {adjusted_files} 个文件含拆分, 共修正 {total_splits} 处 ===")


if __name__ == "__main__":
    main()
