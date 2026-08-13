"""Boundary tests for the research-only strict V3.5 replay."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import exp_strict_v35 as audit  # noqa: E402


def shock_path() -> np.ndarray:
    close = np.full(80, 110.0)
    close[-21:] = np.linspace(100.0, 104.0, 21)
    close[-5] = 99.0
    close[-4:] = (100.5, 102.0, 103.0, 104.0)
    return close


def test_strict_v35_rejects_shock_that_v3g_gate_releases() -> None:
    close = shock_path()

    assert audit.gate_allows(close, audit.GateParams())
    assert not audit.strict_drop_allows(close)


def test_strict_v35_definition_excludes_every_v3g_component() -> None:
    components = audit.strict_v35_components()

    assert components == {
        "strict_drop_filter": True,
        "v4_full_pool_fast_slow": True,
        "v4_confirmation_hits": 2,
        "v4_confirmation_window": 2,
        "v4_minimum_hold_days": 3,
        "v3g_ret60_gate": False,
        "v3g_momentum_release": False,
        "v3g_holding_buffer_exemption": False,
        "v3g_realtime_filter_approximation": False,
    }
    assert audit.STRICT_V3_PARAMS.strict_drop_filter
    assert not audit.STRICT_V3_PARAMS.exempt_buffer
    assert audit.STRICT_V35_PARAMS == audit.v4.V4_PARAMS


def test_strict_v35_does_not_fix_the_three_day_shock_delay() -> None:
    scenario = audit.replay_documented_shock_scenario()

    assert [row["blocked_by"] for row in scenario["decisions"]] == [
        "no_signal",
        "confirmation",
        "",
    ]
    assert not scenario["decisions"][0]["triggered"]
    assert not scenario["decisions"][1]["triggered"]
    assert scenario["decisions"][2]["triggered"]
    assert scenario["decisions"][2]["target"] == "518880"
    assert np.isclose(scenario["silver_cumulative_return"], -0.15222)
    assert scenario["known_defect_fixed"] is False


def test_strict_runner_applies_and_restores_all_research_patches(
    monkeypatch: Any,
) -> None:
    original_select = audit.rq.select_target
    original_drop = audit.rq.check_single_day_drop
    original_realtime = audit.rr.apply_server_realtime_filter
    observed: dict[str, Any] = {}

    def fake_run(
        _data: dict[str, Any],
        params: audit.v4.FullPoolParams,
        *,
        cost_multiplier: float,
    ) -> dict[str, Any]:
        observed["select_patched"] = audit.rq.select_target is not original_select
        observed["drop_rejects"] = not audit.rq.check_single_day_drop(shock_path())
        observed["realtime"] = audit.rr.apply_server_realtime_filter(
            {}, {}, [("161226", 1.0)]
        )
        return {
            "params": {"enabled": params.enabled},
            "metrics": {"cost_multiplier": cost_multiplier},
        }

    monkeypatch.setattr(audit, "run_full_pool_strategy", fake_run)

    result = audit.run_strict_strategy({}, audit.STRICT_V35_PARAMS, cost_multiplier=2.0)

    assert observed == {
        "select_patched": True,
        "drop_rejects": True,
        "realtime": ([("161226", 1.0)], []),
    }
    assert result["strategy_id"] == "STRICT_V3_5"
    assert result["strict_components"] == audit.strict_v35_components()
    assert result["strict_gate_params"] == {
        "drop_threshold": 0.03,
        "drop_lookback": 5,
        "strict_drop_filter": True,
    }
    assert result["metrics"]["cost_multiplier"] == 2.0
    assert audit.rq.select_target is original_select
    assert audit.rq.check_single_day_drop is original_drop
    assert audit.rr.apply_server_realtime_filter is original_realtime
