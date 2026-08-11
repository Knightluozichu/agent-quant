"""Pure decision tests for full-pool V3-G fast/slow handoffs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_qixing_v3 as rq  # noqa: E402
from exp_v3g_full_pool_fast_slow import (  # noqa: E402
    FullPoolParams,
    decide_full_pool_handoff,
    run_full_pool_strategy,
)
from exp_v3g_pair_multifactor import PairAssetFactors  # noqa: E402
from exp_v3g_relative_rotation import RotationParams, run_strategy  # noqa: E402


def asset(
    code: str,
    *,
    eligible: bool = True,
    slow: float = 0.05,
    ret3: float = 0.01,
    ret5: float = 0.02,
    trend: float = 0.01,
) -> PairAssetFactors:
    return PairAssetFactors(
        code=code,
        eligible=eligible,
        slow_momentum=slow,
        return_3d=ret3,
        return_5d=ret5,
        acceleration_5d=0.0,
        trend_strength=trend,
        vol_adjusted_5d=0.0,
        drawdown_5d=0.0,
    )


def test_full_pool_slow_mode_switches_to_slow_leader() -> None:
    factors = {
        "held": asset("held", slow=0.04),
        "slow": asset("slow", slow=0.06, ret3=0.0, ret5=0.0),
        "fast": asset("fast", slow=0.05, ret3=0.05, ret5=0.06),
    }

    decision = decide_full_pool_handoff(
        holding="held",
        factors=factors,
        params=FullPoolParams(mode="slow"),
        signal_hits=1,
        days_since_rotation=99,
    )

    assert decision.triggered
    assert decision.target == "slow"
    assert decision.reasons == ("slow",)


def test_full_pool_fast_mode_switches_to_fast_leader() -> None:
    factors = {
        "held": asset("held", slow=0.04, ret3=-0.01, ret5=-0.01),
        "slow": asset("slow", slow=0.06, ret3=0.0, ret5=0.0),
        "fast": asset("fast", slow=0.05, ret3=0.04, ret5=0.05),
    }

    decision = decide_full_pool_handoff(
        holding="held",
        factors=factors,
        params=FullPoolParams(mode="fast"),
        signal_hits=1,
        days_since_rotation=99,
    )

    assert decision.triggered
    assert decision.target == "fast"
    assert decision.reasons == ("fast",)


def test_consensus_requires_same_fast_and_slow_leader() -> None:
    disagreement = {
        "held": asset("held", slow=0.02, ret3=-0.02, ret5=-0.02),
        "slow": asset("slow", slow=0.08, ret3=0.01, ret5=0.02),
        "fast": asset("fast", slow=0.06, ret3=0.05, ret5=0.06),
    }
    agreement = {
        **disagreement,
        "slow": asset("slow", slow=0.08, ret3=0.06, ret5=0.07),
        "fast": asset("fast", slow=0.06, ret3=0.01, ret5=0.02),
    }

    blocked = decide_full_pool_handoff(
        holding="held",
        factors=disagreement,
        params=FullPoolParams(mode="consensus"),
        signal_hits=1,
        days_since_rotation=99,
    )
    accepted = decide_full_pool_handoff(
        holding="held",
        factors=agreement,
        params=FullPoolParams(mode="consensus"),
        signal_hits=1,
        days_since_rotation=99,
    )

    assert not blocked.triggered
    assert accepted.triggered
    assert accepted.target == "slow"
    assert accepted.reasons == ("slow", "fast")


def test_or_mode_uses_slow_first_and_fast_as_fallback() -> None:
    factors = {
        "held": asset("held", slow=0.04, ret3=-0.01, ret5=-0.01),
        "slow": asset("slow", slow=0.043, ret3=0.0, ret5=0.0),
        "fast": asset("fast", slow=0.05, ret3=0.04, ret5=0.05),
    }

    decision = decide_full_pool_handoff(
        holding="held",
        factors=factors,
        params=FullPoolParams(mode="or"),
        signal_hits=1,
        days_since_rotation=99,
    )

    assert decision.triggered
    assert decision.target == "fast"


def test_gate_lock_and_confirmation_remain_mandatory() -> None:
    factors = {
        "held": asset("held", slow=0.01, ret3=-0.02, ret5=-0.02),
        "leader": asset("leader", slow=0.08, ret3=0.05, ret5=0.06),
    }
    params = FullPoolParams(mode="slow", confirmation_hits=2)

    assert not decide_full_pool_handoff(
        holding="held", factors=factors, params=params,
        signal_hits=1, days_since_rotation=99,
    ).triggered
    assert not decide_full_pool_handoff(
        holding="held", factors=factors, params=params,
        signal_hits=2, days_since_rotation=2,
    ).triggered

    gated = dict(factors)
    gated["leader"] = asset("leader", eligible=False, slow=0.08, ret3=0.05, ret5=0.06)
    assert not decide_full_pool_handoff(
        holding="held", factors=gated, params=FullPoolParams(mode="slow"),
        signal_hits=1, days_since_rotation=99,
    ).triggered


def test_disabled_overlay_matches_canonical_server_replay() -> None:
    data = rq.load_data()
    canonical = run_strategy(data, RotationParams(enabled=False))["equity_curve"]
    full_pool = run_full_pool_strategy(data, FullPoolParams.disabled())["equity_curve"]

    assert canonical["trade_date"].tolist() == full_pool["trade_date"].tolist()
    assert np.allclose(canonical["equity"], full_pool["equity"])
    assert canonical["holding"].tolist() == full_pool["holding"].tolist()
