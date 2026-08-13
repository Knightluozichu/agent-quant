"""Pure tests for the V4 post-entry failure guard."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import qixing_v4 as v4  # noqa: E402
from exp_v4_stop import entry_guard_decision  # noqa: E402


def asset(
    code: str,
    *,
    eligible: bool = True,
    slow: float = 0.05,
    ret3: float = 0.01,
    ret5: float = 0.02,
    trend: float = 0.01,
) -> v4.AssetFactors:
    return v4.AssetFactors(
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


def test_entry_guard_cash_triggers_on_first_day_shock() -> None:
    decision = entry_guard_decision(
        mode="entry_guard_cash",
        holding="silver",
        holding_age=1,
        entry_return=-0.05,
        one_day=-0.05,
        candidates=[("silver", 0.20), ("gold", 0.05)],
        factors={"silver": asset("silver"), "gold": asset("gold")},
        first_v4_target=None,
    )

    assert decision.triggered
    assert decision.exit_to_cash
    assert decision.target is None
    assert "entry_1d<=-5%" in decision.reasons


def test_entry_guard_is_limited_to_first_three_trading_days() -> None:
    decision = entry_guard_decision(
        mode="entry_guard_cash",
        holding="silver",
        holding_age=4,
        entry_return=-0.20,
        one_day=-0.08,
        candidates=[("silver", 0.20), ("gold", 0.05)],
        factors={"silver": asset("silver"), "gold": asset("gold")},
        first_v4_target=None,
    )

    assert not decision.triggered


def test_entry_guard_uses_entry_loss_even_without_single_day_shock() -> None:
    decision = entry_guard_decision(
        mode="entry_guard_cash",
        holding="silver",
        holding_age=2,
        entry_return=-0.10,
        one_day=-0.04,
        candidates=[("silver", 0.20), ("gold", 0.05)],
        factors={"silver": asset("silver"), "gold": asset("gold")},
        first_v4_target=None,
    )

    assert decision.triggered
    assert "entry_return<=-10%" in decision.reasons


def test_entry_guard_v4_or_cash_prefers_first_consensus_target() -> None:
    decision = entry_guard_decision(
        mode="entry_guard_v4_or_cash",
        holding="silver",
        holding_age=1,
        entry_return=-0.06,
        one_day=-0.06,
        candidates=[("silver", 0.14), ("gold", 0.06)],
        factors={"silver": asset("silver"), "gold": asset("gold")},
        first_v4_target="gold",
    )

    assert decision.triggered
    assert decision.target == "gold"
    assert not decision.exit_to_cash


def test_entry_guard_v4_or_cash_defends_without_consensus() -> None:
    decision = entry_guard_decision(
        mode="entry_guard_v4_or_cash",
        holding="silver",
        holding_age=1,
        entry_return=-0.06,
        one_day=-0.06,
        candidates=[("silver", 0.14), ("gold", 0.06)],
        factors={"silver": asset("silver"), "gold": asset("gold")},
        first_v4_target=None,
    )

    assert decision.triggered
    assert decision.exit_to_cash


def test_entry_guard_best_or_cash_requires_positive_replacement() -> None:
    accepted = entry_guard_decision(
        mode="entry_guard_best_or_cash",
        holding="silver",
        holding_age=1,
        entry_return=-0.06,
        one_day=-0.06,
        candidates=[("silver", 0.14), ("gold", 0.06)],
        factors={
            "silver": asset("silver", ret3=-0.05, ret5=-0.05),
            "gold": asset("gold", slow=0.06, ret3=0.01, ret5=0.02, trend=0.01),
        },
        first_v4_target=None,
    )
    rejected = entry_guard_decision(
        mode="entry_guard_best_or_cash",
        holding="silver",
        holding_age=1,
        entry_return=-0.06,
        one_day=-0.06,
        candidates=[("silver", 0.14), ("gold", 0.06)],
        factors={
            "silver": asset("silver", ret3=-0.05, ret5=-0.05),
            "gold": asset("gold", slow=0.06, ret3=0.01, ret5=0.02, trend=-0.01),
        },
        first_v4_target=None,
    )

    assert accepted.target == "gold"
    assert not accepted.exit_to_cash
    assert rejected.target is None
    assert rejected.exit_to_cash


def test_selective_guard_ignores_day_shock_without_replacement() -> None:
    decision = entry_guard_decision(
        mode="entry_guard_selective",
        holding="silver",
        holding_age=1,
        entry_return=-0.06,
        one_day=-0.06,
        candidates=[("silver", 0.14), ("gold", 0.06)],
        factors={
            "silver": asset("silver", ret3=-0.05, ret5=-0.05),
            "gold": asset("gold", slow=0.06, ret3=0.01, ret5=0.02, trend=-0.01),
        },
        first_v4_target=None,
    )

    assert not decision.triggered


def test_selective_guard_switches_only_to_positive_relative_replacement() -> None:
    decision = entry_guard_decision(
        mode="entry_guard_selective",
        holding="silver",
        holding_age=1,
        entry_return=-0.06,
        one_day=-0.06,
        candidates=[("silver", 0.14), ("gold", 0.06)],
        factors={
            "silver": asset("silver", ret3=-0.05, ret5=-0.05),
            "gold": asset("gold", slow=0.06, ret3=0.01, ret5=0.02, trend=0.01),
        },
        first_v4_target=None,
    )

    assert decision.triggered
    assert decision.target == "gold"
    assert not decision.exit_to_cash


def test_selective_guard_defends_on_catastrophic_entry_loss() -> None:
    decision = entry_guard_decision(
        mode="entry_guard_selective",
        holding="silver",
        holding_age=2,
        entry_return=-0.10,
        one_day=-0.04,
        candidates=[("silver", 0.08), ("gold", 0.06)],
        factors={
            "silver": asset("silver", ret3=-0.08, ret5=-0.04),
            "gold": asset("gold", slow=0.06, ret3=0.01, ret5=0.02, trend=0.01),
        },
        first_v4_target=None,
    )

    assert decision.triggered
    assert decision.exit_to_cash
    assert decision.target is None


def test_entry_guard_thresholds_are_explicitly_perturbable() -> None:
    decision = entry_guard_decision(
        mode="entry_guard_selective",
        holding="silver",
        holding_age=1,
        entry_return=-0.06,
        one_day=-0.06,
        candidates=[("silver", 0.14), ("gold", 0.06)],
        factors={
            "silver": asset("silver", ret3=-0.05, ret5=-0.05),
            "gold": asset("gold", slow=0.06, ret3=0.01, ret5=0.02, trend=0.01),
        },
        first_v4_target=None,
        day_threshold=-0.07,
        entry_loss_threshold=-0.12,
    )

    assert not decision.triggered
