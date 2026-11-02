"""Feasibility gate for a 20%/month compounded target. Research only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/high_return_alpha_20261102"
CURVES = ROOT / "artifacts/tencent_v4_v5_20261102/curves.parquet"
TARGET_MONTHLY = 0.20


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves = pd.read_parquet(CURVES)
    v5 = curves[curves.config == "tencent_v5_1x"].copy()
    v5["trade_date"] = pd.to_datetime(v5.trade_date)
    v5 = v5.set_index("trade_date").sort_index()
    monthly = v5.equity.resample("ME").last().pct_change().dropna()

    monthly_ge_20 = monthly >= TARGET_MONTHLY
    longest = current = 0
    for hit in monthly_ge_20:
        current = current + 1 if hit else 0
        longest = max(longest, current)

    best_12m = monthly.add(1).rolling(12).apply(np.prod, raw=True).sub(1).max()
    best_24m = monthly.add(1).rolling(24).apply(np.prod, raw=True).sub(1).max()
    required_12m = (1 + TARGET_MONTHLY) ** 12 - 1
    required_24m = (1 + TARGET_MONTHLY) ** 24 - 1

    annual_target = required_12m
    required_sharpe = {str(vol): round(annual_target / vol, 1) for vol in (0.20, 0.30, 0.40, 0.60)}
    per_trade = {
        str(n): round(((1 + TARGET_MONTHLY) ** (1 / n) - 1) * 100, 2) for n in (5, 10, 20, 40)
    }
    commission_drag = {str(n): round((n * 2 * 5) / 10_000 * 100, 2) for n in (5, 10, 20, 40)}

    payload = {
        "target": {
            "monthly": TARGET_MONTHLY,
            "annual_compounded": round(required_12m * 100, 1),
            "two_year_multiple": round((1 + TARGET_MONTHLY) ** 24, 2),
            "three_year_multiple": round((1 + TARGET_MONTHLY) ** 36, 2),
            "10k_after_12m": round(10_000 * (1 + TARGET_MONTHLY) ** 12, 0),
            "10k_after_24m": round(10_000 * (1 + TARGET_MONTHLY) ** 24, 0),
            "10k_after_36m": round(10_000 * (1 + TARGET_MONTHLY) ** 36, 0),
        },
        "v5_base_rate": {
            "months": len(monthly),
            "months_ge_20pct": int(monthly_ge_20.sum()),
            "share_ge_20pct": round(float(monthly_ge_20.mean()), 3),
            "longest_consecutive_ge_20pct": int(longest),
            "median_month": round(float(monthly.median()) * 100, 1),
            "worst_month": round(float(monthly.min()) * 100, 1),
            "best_month": round(float(monthly.max()) * 100, 1),
            "best_12m": round(float(best_12m) * 100, 1),
            "best_24m": round(float(best_24m) * 100, 1),
            "required_12m": round(required_12m * 100, 1),
            "required_24m": round(required_24m * 100, 1),
        },
        "required_edge": {
            "required_sharpe_at_annual_vol": required_sharpe,
            "required_net_pct_per_trade": per_trade,
            "commission_drag_pct_of_10k": commission_drag,
            "note": (
                "Assumes full capital, no leverage, independent monthly returns; not a forecast."
            ),
        },
        "verdict": (
            "20%/month is not an investable acceptance target for a 10k retail account; "
            "use as stress target only."
        ),
    }
    (OUT / "feasibility.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
