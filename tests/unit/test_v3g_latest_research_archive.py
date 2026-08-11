"""Regression tests for the frozen 2026-08-11 V3-G research archive."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from archive_v3g_latest_research import (  # noqa: E402
    SOURCE,
    build_archive,
    sha256_file,
)


def load_source() -> dict[str, object]:
    return json.loads(SOURCE.read_text())


def test_archive_freezes_names_params_results_and_production_status() -> None:
    source = load_source()
    archive = build_archive(
        source,
        source_sha256=sha256_file(SOURCE),
        artifact_hashes={"source_result": sha256_file(SOURCE)},
        archived_at="2026-08-11T12:00:00+08:00",
    )

    assert archive["archive_id"] == "V3G-LATEST-20260811-FULL-POOL-FAST-SLOW"
    assert archive["production_status"] == {
        "active_variant": "V3-G_BASELINE",
        "changed_by_this_archive": False,
        "research_variants_deployed": False,
    }
    assert list(archive["variants"]) == [
        "V3-G_BASELINE",
        "V3-G_FULL_POOL_SLOW_1PP_CONFIRM2",
        "V4",
    ]

    baseline = archive["variants"]["V3-G_BASELINE"]
    slow = archive["variants"]["V3-G_FULL_POOL_SLOW_1PP_CONFIRM2"]
    strict = archive["variants"]["V4"]

    assert baseline["classification"] == "production_reference"
    assert slow["classification"] == "research_only_not_approved"
    assert strict["classification"] == "research_only_not_approved"
    assert strict["display_name"] == "V4（V3-G + 全池严格快慢共振）"
    assert strict["lineage"] == "V3-G + full-pool strict fast/slow consensus"
    assert slow["params"]["mode"] == "slow"
    assert slow["params"]["slow_gap"] == pytest.approx(0.01)
    assert slow["params"]["confirmation_hits"] == 2
    assert strict["params"]["mode"] == "consensus"
    assert strict["params"]["slow_gap"] == pytest.approx(0.0075)
    assert strict["params"]["fast_5d_gap"] == pytest.approx(0.0225)
    assert strict["params"]["fast_3d_gap"] == pytest.approx(0.01125)
    assert strict["params"]["confirmation_hits"] == 2

    assert baseline["metrics"] == source["named_variants"]["baseline"]["metrics"]
    assert slow["yearly"] == source["named_variants"]["slow_gap1_confirm2"]["yearly"]
    assert strict["segments"]["IS"] == source["segments"]["IS"]["consensus_strict"]
    assert strict["cost_pressure"]["3x"]["final_value"] == pytest.approx(15405.809976987504)
    assert slow["cost_pressure"]["3x"]["max_drawdown"] == pytest.approx(
        -0.9782324085396407
    )
    assert archive["source"]["sha256"] == sha256_file(SOURCE)


def test_archive_rejects_parameter_drift_under_same_name() -> None:
    source = copy.deepcopy(load_source())
    source["named_variants"]["consensus_strict"]["params"]["slow_gap"] = 0.008

    with pytest.raises(ValueError, match=r"parameter drift.*consensus_strict"):
        build_archive(
            source,
            source_sha256="test-digest",
            artifact_hashes={},
            archived_at="2026-08-11T12:00:00+08:00",
        )
