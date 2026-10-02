"""Safety boundaries for a research-only frozen bridge."""

import json

import pytest
from scripts import review_strategy_bridge_20260916 as bridge


def test_bridge_refuses_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        bridge.run_bridge(tmp_path)


def test_bridge_rejects_changed_frozen_input(monkeypatch, tmp_path):
    frozen = tmp_path / "frozen"
    (frozen / "inputs").mkdir(parents=True)
    (frozen / "inputs/518880.parquet").write_bytes(b"modified")
    (frozen / "review.json").write_text(json.dumps({"input_sha256": {"518880": "expected"}}))
    monkeypatch.setattr(bridge, "FROZEN", frozen)
    with pytest.raises(ValueError, match="Frozen input hash mismatch: 518880"):
        bridge.run_bridge(tmp_path / "output")


def test_buggy_diagnostic_does_not_persist_valid_decision():
    state = bridge.live.default_state(100000)
    overlay = {"history": [{"date": "2026-09-01", "target": "159985"}]}
    bridge.buggy_record(state, overlay, {"pool_version": "should-not-be-saved"})
    assert state["v4_state"]["candidate_history"] == overlay["history"]
    assert state["last_decision"] is None
