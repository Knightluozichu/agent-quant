"""上线闸门 gate_check 测试 (I-V4A-02)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import gate_check  # noqa: E402


def _passing_audit() -> dict:
    """构造一份全部阻断规则通过的审计 JSON."""
    return {
        "assessment": {
            "classification": "acceptable",
            "statistical_increment_pass": True,
        },
        "incremental": {
            "white_reality_check": {"20": {"p_value": 0.01}},
            "moving_block_bootstrap": {"20": {"ci95": [0.01, 0.12]}},
            "annual": {"post_2024_share_of_total_relative_log_return": 0.3},
        },
        "meta": {
            "live_oos_observations": 300,
            "trial_count_is_lower_bound": False,
        },
    }


@pytest.mark.unit
def test_passing_audit_allows():
    result = gate_check.evaluate(_passing_audit())
    assert result.verdict == "allow"
    assert result.failed_blocking == []
    assert result.warnings == []


@pytest.mark.unit
def test_high_overfit_risk_blocks():
    audit = _passing_audit()
    audit["assessment"]["classification"] = "high_overfit_risk"
    result = gate_check.evaluate(audit)
    assert result.verdict == "block"
    assert any(r.rule.startswith("R2") for r in result.failed_blocking)


@pytest.mark.unit
def test_nonsignificant_white_p_blocks():
    audit = _passing_audit()
    audit["incremental"]["white_reality_check"]["20"]["p_value"] = 0.755
    result = gate_check.evaluate(audit)
    assert result.verdict == "block"


@pytest.mark.unit
def test_bootstrap_ci_crossing_zero_blocks():
    audit = _passing_audit()
    audit["incremental"]["moving_block_bootstrap"]["20"]["ci95"] = [-0.03, 0.145]
    result = gate_check.evaluate(audit)
    assert result.verdict == "block"


@pytest.mark.unit
def test_insufficient_live_oos_blocks():
    audit = _passing_audit()
    audit["meta"]["live_oos_observations"] = 0
    result = gate_check.evaluate(audit)
    assert result.verdict == "block"


@pytest.mark.unit
def test_lower_bound_trial_count_warns_but_does_not_block():
    audit = _passing_audit()
    audit["meta"]["trial_count_is_lower_bound"] = True
    result = gate_check.evaluate(audit)
    assert result.verdict == "allow"
    assert any(r.rule.startswith("W1") for r in result.warnings)


@pytest.mark.unit
def test_regime_concentration_warns_but_does_not_block():
    audit = _passing_audit()
    audit["incremental"]["annual"]["post_2024_share_of_total_relative_log_return"] = 0.991
    result = gate_check.evaluate(audit)
    assert result.verdict == "allow"
    assert any(r.rule.startswith("W2") for r in result.warnings)


@pytest.mark.unit
def test_missing_field_raises_fail_closed():
    with pytest.raises(gate_check.AuditInputError):
        gate_check.evaluate({"assessment": {}})


@pytest.mark.unit
def test_real_v4_audit_blocks(tmp_path):
    """用仓库真实归档的 V4 审计结果: 必须 block (high_overfit_risk, OOS=0)."""
    real = PROJECT_ROOT / "data" / "v9_results" / "v4_overfit_audit.json"
    if not real.is_file():
        pytest.skip("归档审计 JSON 不在本地")
    audit = json.loads(real.read_text(encoding="utf-8"))
    result = gate_check.evaluate(audit)
    assert result.verdict == "block"
    failed = {r.rule.split()[0] for r in result.failed_blocking}
    assert {"R1", "R2", "R3", "R4", "R5"} <= failed


@pytest.mark.unit
def test_cli_missing_file_returns_2(tmp_path):
    assert gate_check.main([str(tmp_path / "nope.json")]) == 2


@pytest.mark.unit
def test_cli_block_exit_1(tmp_path, capsys):
    p = tmp_path / "audit.json"
    audit = _passing_audit()
    audit["meta"]["live_oos_observations"] = 0
    p.write_text(json.dumps(audit), encoding="utf-8")
    assert gate_check.main([str(p)]) == 1
    out = capsys.readouterr().out
    assert "BLOCK" in out and "影子模式" in out
