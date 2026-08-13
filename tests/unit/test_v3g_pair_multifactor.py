"""Pure-signal tests for the V3-G gold/silver multifactor handoff."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_qixing_v3 as rq  # noqa: E402
from exp_v3g_pair_multifactor import (  # noqa: E402
    PairAssetFactors,
    PairFactorParams,
    compute_pair_factors,
    decide_pair_handoff,
    effective_signal_hits,
    enrich_pair_rotation_events,
    run_pair_strategy,
)
from exp_v3g_relative_rotation import RotationParams, run_strategy  # noqa: E402


def factors(
    code: str,
    *,
    eligible: bool = True,
    slow: float = 0.05,
    ret3: float = 0.01,
    ret5: float = 0.02,
    acceleration: float = 0.01,
    trend: float = 0.01,
    vol_adjusted: float = 0.5,
    drawdown: float = -0.01,
) -> PairAssetFactors:
    return PairAssetFactors(
        code=code,
        eligible=eligible,
        slow_momentum=slow,
        return_3d=ret3,
        return_5d=ret5,
        acceleration_5d=acceleration,
        trend_strength=trend,
        vol_adjusted_5d=vol_adjusted,
        drawdown_5d=drawdown,
    )


def test_slow_relative_leadership_triggers_pair_handoff() -> None:
    decision = decide_pair_handoff(
        holding="161226",
        held=factors("161226", slow=0.04),
        peer=factors("518880", slow=0.047),
        params=PairFactorParams.slow_only(),
        signal_hits=1,
        days_since_rotation=99,
    )

    assert decision.triggered
    assert decision.target == "518880"
    assert decision.reasons == ("slow",)


def test_same_rule_can_be_declared_for_another_asset_pair() -> None:
    decision = decide_pair_handoff(
        holding="159985",
        held=factors("159985", slow=0.04),
        peer=factors("501018", slow=0.047),
        params=PairFactorParams.slow_only(),
        signal_hits=1,
        days_since_rotation=99,
        pair=("159985", "501018"),
    )

    assert decision.triggered
    assert decision.target == "501018"


def test_fast_consensus_can_trigger_before_slow_rank_flip() -> None:
    decision = decide_pair_handoff(
        holding="161226",
        held=factors("161226", slow=0.06, ret3=-0.01, ret5=-0.005),
        peer=factors("518880", slow=0.058, ret3=0.01, ret5=0.015),
        params=PairFactorParams.slow_fast(),
        signal_hits=1,
        days_since_rotation=99,
    )

    assert decision.triggered
    assert "fast" in decision.reasons


def test_multifactor_drawdown_divergence_adds_a_signal() -> None:
    decision = decide_pair_handoff(
        holding="518880",
        held=factors("518880", slow=0.04, ret5=-0.005, drawdown=-0.04),
        peer=factors("161226", slow=0.042, ret5=0.004, trend=0.005),
        params=PairFactorParams.multifactor(),
        signal_hits=1,
        days_since_rotation=99,
    )

    assert decision.triggered
    assert "drawdown" in decision.reasons


def test_negative_or_gated_peer_is_never_bought_early() -> None:
    for peer in (
        factors("518880", slow=-0.01, ret3=0.05, ret5=0.06),
        factors("518880", eligible=False, slow=0.08, ret3=0.05, ret5=0.06),
    ):
        decision = decide_pair_handoff(
            holding="161226",
            held=factors("161226", slow=0.01, ret3=-0.03, ret5=-0.04),
            peer=peer,
            params=PairFactorParams.multifactor(),
            signal_hits=2,
            days_since_rotation=99,
        )
        assert not decision.triggered


def test_lock_and_confirmation_both_apply() -> None:
    params = PairFactorParams.multifactor(confirmation_hits=2)
    held = factors("161226", slow=0.01, ret3=-0.03, ret5=-0.04)
    peer = factors("518880", slow=0.08, ret3=0.03, ret5=0.04)

    assert not decide_pair_handoff(
        holding="161226",
        held=held,
        peer=peer,
        params=params,
        signal_hits=1,
        days_since_rotation=99,
    ).triggered
    assert not decide_pair_handoff(
        holding="161226",
        held=held,
        peer=peer,
        params=params,
        signal_hits=2,
        days_since_rotation=2,
    ).triggered
    assert decide_pair_handoff(
        holding="161226",
        held=held,
        peer=peer,
        params=params,
        signal_hits=2,
        days_since_rotation=3,
    ).triggered


def test_global_confirmation_can_bypass_internal_two_day_confirmation() -> None:
    assert (
        effective_signal_hits(
            observed_hits=1,
            required_hits=2,
            globally_confirmed=True,
            allow_global_immediate=True,
        )
        == 2
    )
    assert (
        effective_signal_hits(
            observed_hits=1,
            required_hits=2,
            globally_confirmed=False,
            allow_global_immediate=True,
        )
        == 1
    )


def test_factor_snapshot_does_not_change_when_future_rows_are_appended() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D").date
    gold = pd.DataFrame({"trade_date": dates, "close": range(100, 130)})
    silver = pd.DataFrame({"trade_date": dates, "close": range(200, 230)})
    data = {"518880": gold.copy(), "161226": silver.copy()}
    idx_map = {"518880": 24, "161226": 24}
    before = compute_pair_factors(data, idx_map)

    data["518880"].loc[30] = {"trade_date": pd.Timestamp("2025-01-01").date(), "close": 9999}
    data["161226"].loc[30] = {"trade_date": pd.Timestamp("2025-01-01").date(), "close": 1}
    after = compute_pair_factors(data, idx_map)

    assert before == after


def test_redirected_event_keeps_intended_pair_diagnostic_separate() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D").date.tolist()
    data = {
        "518880": pd.DataFrame({"trade_date": dates, "close": [100] * 6}),
        "161226": pd.DataFrame({"trade_date": dates, "close": [100, 100, 100, 100, 100, 110]}),
        "511220": pd.DataFrame({"trade_date": dates, "close": [100, 100, 100, 100, 100, 90]}),
    }
    index_maps = {td: dict.fromkeys(data, i) for i, td in enumerate(dates)}
    events = [
        {
            "trade_date_raw": dates[0],
            "date": str(dates[0]),
            "from": "518880",
            "to": "511220",
            "intended_to": "161226",
            "risk_redirected": True,
        }
    ]

    enriched = enrich_pair_rotation_events(events, data, dates, index_maps)

    assert enriched[0]["ex_post_actual_relative_5d"] == pytest.approx(-0.1)
    assert enriched[0]["ex_post_intended_relative_5d"] == pytest.approx(0.1)


def test_disabled_overlay_matches_canonical_server_replay() -> None:
    data = rq.load_data()
    canonical = run_strategy(data, RotationParams(enabled=False))["equity_curve"]
    pair_runner = run_pair_strategy(data, PairFactorParams.disabled())["equity_curve"]

    assert canonical["trade_date"].tolist() == pair_runner["trade_date"].tolist()
    assert np.allclose(canonical["equity"], pair_runner["equity"])
    assert canonical["holding"].tolist() == pair_runner["holding"].tolist()
