"""Feasibility gate: 12-30%/month, max drawdown -10%, A-share/ETF long-only.

Research only. No broker, no orders. The de-risking table is an empirical
upper bound from scaling V5 daily returns, not a tradable strategy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/high_return_alpha_20261102"
CURVES = ROOT / "artifacts/tencent_v4_v5_20261102/curves.parquet"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves = pd.read_parquet(CURVES)
    v5 = curves[curves.config == "tencent_v5_1x"].copy()
    v5["trade_date"] = pd.to_datetime(v5.trade_date)
    v5 = v5.set_index("trade_date").sort_index()
    returns = v5.equity.pct_change().fillna(0).to_numpy()
    years = (v5.index[-1] - v5.index[0]).days / 365.25
    monthly = v5.equity.resample("ME").last().pct_change().dropna()

    base_equity = np.cumprod(1 + returns)
    base_cagr = float(base_equity[-1] ** (1 / years) - 1)
    base_mdd = float((base_equity / np.maximum.accumulate(base_equity) - 1).min())

    scale_rows = []
    for scale in (0.20, 0.25, 0.27, 0.30, 0.35, 0.50, 1.00):
        equity = np.cumprod(1 + scale * returns)
        mdd = float((equity / np.maximum.accumulate(equity) - 1).min())
        cagr = float(equity[-1] ** (1 / years) - 1)
        scale_rows.append(
            {
                "v5_return_scale": scale,
                "cagr_pct": round(cagr * 100, 1),
                "max_drawdown_pct": round(mdd * 100, 1),
                "monthly_cagr_equivalent_pct": round(((1 + cagr) ** (1 / 12) - 1) * 100, 2),
            }
        )

    # Largest scale that still keeps the empirical max drawdown <= -10%.
    safe_scale = 0.0
    for candidate in np.linspace(0.0, 1.0, 1001):
        equity = np.cumprod(1 + candidate * returns)
        if float((equity / np.maximum.accumulate(equity) - 1).min()) >= -0.10:
            safe_scale = float(candidate)
        else:
            break
    safe_equity = np.cumprod(1 + safe_scale * returns)
    safe_cagr = float(safe_equity[-1] ** (1 / years) - 1)
    safe_mdd = float((safe_equity / np.maximum.accumulate(safe_equity) - 1).min())

    targets = {}
    for monthly_target in (0.12, 0.20, 0.30):
        annual = (1 + monthly_target) ** 12 - 1
        targets[str(monthly_target)] = {
            "annual_compounded_pct": round(annual * 100, 1),
            "required_calmar_with_10pct_mdd": round(annual / 0.10, 1),
            "v5_calmar": round(base_cagr / abs(base_mdd), 2),
            "required_monthly_sharpe_at_3pct_vol": round(monthly_target / 0.03, 2),
        }

    payload = {
        "constraints": {
            "capital": 10_000,
            "instruments": "A-share stocks and ETFs only",
            "leverage": "none",
            "monthly_target_range": [0.12, 0.30],
            "max_drawdown": -0.10,
        },
        "v5_reference": {
            "cagr_pct": round(base_cagr * 100, 1),
            "max_drawdown_pct": round(base_mdd * 100, 1),
            "best_12m_pct": round(
                float(monthly.add(1).rolling(12).apply(np.prod, raw=True).sub(1).max()) * 100,
                1,
            ),
            "months_ge_12pct": int((monthly >= 0.12).sum()),
            "months_ge_20pct": int((monthly >= 0.20).sum()),
            "months_ge_30pct": int((monthly >= 0.30).sum()),
            "months_total": len(monthly),
        },
        "derisk_upper_bound": {
            "safe_scale": round(safe_scale, 3),
            "cagr_pct": round(safe_cagr * 100, 1),
            "max_drawdown_pct": round(safe_mdd * 100, 1),
            "monthly_equivalent_pct": round(((1 + safe_cagr) ** (1 / 12) - 1) * 100, 2),
            "note": (
                "Scaling V5 daily returns is an empirical upper bound, "
                "not an implementable strategy."
            ),
        },
        "target_gap": targets,
        "verdict": (
            "12-30%/month with a -10% max drawdown is not achievable by scaling the "
            "current V5 engine; the drawdown budget caps the monthly equivalent near 1-2%."
        ),
        "scale_table": scale_rows,
    }
    (OUT / "c_constraints.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
