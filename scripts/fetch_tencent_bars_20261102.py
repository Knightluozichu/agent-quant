"""Fetch Tencent qfq daily bars for the V4/V5 eight-asset pool.

Source: web.ifzq.gtimg.cn/appstock/app/fqkline/get.  ETFs use ``qfqday``;
LOFs without qfq fall back to ``day`` (Tencent returns no qfq block for them).
Each series is scaled by a constant so its final close equals the Sina frozen
close on 2026-10-30; a constant scale does not change percentage returns but
keeps 100-share lot sizing comparable.

Output: artifacts/tencent_v4_v5_20261102/inputs + manifest + comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SINA = ROOT / "artifacts/v4_since_entry_20261102/inputs"
OUT = ROOT / "artifacts/tencent_v4_v5_20261102"
CODES = ["518880", "159985", "501018", "161226", "513100", "159915", "511220", "511880"]
END = date(2026, 10, 30)
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
COLUMNS = ["trade_date", "open", "close", "high", "low", "volume"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tencent_symbol(code: str) -> str:
    return ("sh" if code.startswith(("5", "6")) else "sz") + code


def fetch_range(code: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    symbol = tencent_symbol(code)
    query = f"{symbol},day,{start},{end},640,qfq"
    url = f"{URL}?param={query}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    block = payload.get("data", {}).get(symbol, {})
    source = "qfqday" if block.get("qfqday") else "day"
    rows = block.get(source) or []
    return pd.DataFrame(rows, columns=COLUMNS), source


def fetch_code(code: str) -> tuple[pd.DataFrame, set[str]]:
    frames: list[pd.DataFrame] = []
    sources: set[str] = set()
    for year in range(2019, 2027):
        start = f"{year}-01-01"
        end = f"{year}-12-31" if year < 2026 else END.isoformat()
        frame, source = fetch_range(code, start, end)
        sources.add(source)
        frames.append(frame)
        time.sleep(0.2)
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError(f"No Tencent data for {code}")
    combined = combined.drop_duplicates("trade_date").sort_values("trade_date")
    combined["trade_date"] = pd.to_datetime(combined["trade_date"]).dt.date
    for column in ("open", "high", "low", "close", "volume"):
        combined[column] = combined[column].astype(float)
    combined["symbol"] = code
    if (combined[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"Non-positive price in {code}")
    if combined.trade_date.duplicated().any():
        raise ValueError(f"Duplicate dates in {code}")
    return combined, sources


def compare_to_sina(code: str, tencent: pd.DataFrame, normalization: float) -> dict[str, Any]:
    sina = pd.read_parquet(SINA / f"{code}.parquet")
    sina["trade_date"] = pd.to_datetime(sina.trade_date).dt.date
    merged = tencent.merge(sina, on="trade_date", suffixes=("_t", "_s")).sort_values("trade_date")
    if merged.empty:
        return {"overlap_rows": 0}
    diffs = {
        column: float((merged[f"{column}_t"] - merged[f"{column}_s"]).abs().max())
        for column in ("open", "close", "high", "low")
    }
    return_diff = float((merged.close_t.pct_change() - merged.close_s.pct_change()).abs().max())
    ratio = merged.close_t / merged.close_s
    return {
        "overlap_rows": len(merged),
        "max_abs_price_diff": diffs,
        "max_abs_return_diff": return_diff,
        "ratio_min": float(ratio.min()),
        "ratio_median": float(ratio.median()),
        "ratio_max": float(ratio.max()),
        "normalization": normalization,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.output
    if out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out}; pass --overwrite")
    (out / "inputs").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source": "Tencent web.ifzq.gtimg.cn fqkline (ETF qfq; LOF day fallback)",
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "end": str(END),
        "normalization": "scale each series so its final close equals Sina 2026-10-30 close",
        "files": {},
        "comparison": {},
    }
    for code in CODES:
        frame, sources = fetch_code(code)
        sina = pd.read_parquet(SINA / f"{code}.parquet")
        sina["trade_date"] = pd.to_datetime(sina.trade_date).dt.date
        last_tencent = float(frame.close.iloc[-1])
        last_sina = float(sina.loc[sina.trade_date == END, "close"].iloc[0])
        normalization = last_sina / last_tencent
        for column in ("open", "high", "low", "close"):
            frame[column] = frame[column] * normalization
        frame = frame[frame.trade_date <= END].reset_index(drop=True)
        path = out / "inputs" / f"{code}.parquet"
        frame.to_parquet(path, index=False)
        comparison = compare_to_sina(code, frame, normalization)
        manifest["files"][code] = {
            "rows": len(frame),
            "start": str(frame.trade_date.min()),
            "end": str(frame.trade_date.max()),
            "tencent_fields": sorted(sources),
            "normalization": normalization,
            "sha256": sha256(path),
        }
        manifest["comparison"][code] = comparison
        print(
            code,
            sorted(sources),
            len(frame),
            frame.trade_date.min(),
            frame.trade_date.max(),
            comparison,
        )
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str)
    )
    print(f"written {out}")


if __name__ == "__main__":
    main()
