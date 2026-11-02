"""V5: V4 full-pool signal plus a circuit-breaker high-water-mark reset.

V5 is deliberately narrow.  It does not add a factor, change the pool, or
re-tune V4 parameters.  The only behavioral change is: after the -30%
portfolio circuit breaker fires, the risk high-water mark is rebased to the
de-risked equity, so the same stale peak cannot immediately re-trigger on the
next rebalance.

This is a shadow/research strategy until the 252-trading-day forward gate is
met.  Production mode remains V4.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Support both ``python -m scripts.qixing_v5`` and direct ``python scripts/live_signal.py``.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import qixing_v4 as v4  # noqa: E402
import risk_overrides as ro  # noqa: E402
import run_qixing_v3 as rq  # noqa: E402

STRATEGY_NAME = "V5"
STRATEGY_ID = "QIXING_V5_FULL_POOL_CONSENSUS_CB_RESET_20261102"
STATE_SCHEMA_VERSION = 1
V5_PARAMS = v4.V4_PARAMS
RESET_AFTER_CIRCUIT_BREAKER = True


def _config_payload() -> dict[str, Any]:
    return {
        "strategy_id": STRATEGY_ID,
        "signal": asdict(V5_PARAMS),
        "risk": {
            "circuit_breaker": "portfolio_drawdown_-30pct",
            "after_trigger": "rebase_peak_to_derisked_equity",
        },
        "candidate_pool": list(rq.ETF_POOL),
        "defense": rq.DEFENSE,
        "rebalance_days": rq.REBALANCE_DAYS,
        "warmup": 130,
    }


def config_hash() -> str:
    payload = json.dumps(_config_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


CONFIG_HASH = config_hash()


def assess_v5(**kwargs: Any) -> ro.RiskDecision:
    """Run the canonical risk layer, then rebase the peak after a CB exit.

    The caller supplies the same kwargs as ``risk_overrides.assess``.  The
    rebase uses only information observable at T (cash, shares, T close) and
    only changes ``state["peak_equity"]``.  It never changes the target,
    exposure, action, cooldown, or any signal input.
    """
    decision = ro.assess(**kwargs)
    if decision.action == ro.ACTION_EMERGENCY:
        state = kwargs["state"]
        holding = state.get("holding")
        equity = float(state.get("cash", 0.0))
        if holding:
            data = kwargs["data"]
            idx_map = kwargs["idx_map"]
            equity += float(state.get("shares", 0.0)) * float(
                data[holding].iloc[idx_map[holding]]["close"]
            )
        state["peak_equity"] = equity
    return decision
