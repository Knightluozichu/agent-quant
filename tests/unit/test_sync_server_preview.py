"""服务器实盘状态同步与本地盘中预演测试。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import qixing_v4  # noqa: E402
import sync_server_preview as sync_preview  # noqa: E402


def _state() -> dict:
    return {
        "initial_capital": 100_000.0,
        "cash": 167.98,
        "holding": "161226",
        "shares": 53_000,
        "entry_date": "2026-07-30",
        "entry_price": 1.861,
        "last_rebalance_date": "2026-08-05",
        "last_run_date": "2026-08-11",
        "pending_order": None,
        "trade_log": [],
        "peak_equity": 110_000.0,
        "risk_exposure": 1.0,
        "risk_log": [],
        "_version": 8,
    }


def _release(project_root: Path) -> dict:
    files: dict[str, str] = {}
    for relative in sync_preview.PREVIEW_CORE_FILES:
        path = project_root / relative
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "release_id": "test-v4",
        "strategy_id": qixing_v4.STRATEGY_ID,
        "strategy_mode": "V4",
        "config_hash": qixing_v4.CONFIG_HASH,
        "files": files,
    }


def _project(tmp_path: Path) -> Path:
    for relative in sync_preview.PREVIEW_CORE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"test fixture: {relative}\n", encoding="utf-8")
    return tmp_path


@pytest.mark.unit
def test_validate_snapshot_accepts_legacy_state_without_v4_runtime(tmp_path):
    project_root = _project(tmp_path)
    snapshot = sync_preview.RemoteSnapshot(
        state=_state(),
        mode={"mode": "V4"},
        release=_release(project_root),
    )

    normalized = sync_preview.validate_snapshot(snapshot, project_root=project_root)

    assert normalized["holding"] == "161226"
    assert normalized["cash"] == pytest.approx(167.98)
    assert normalized["v4_state"]["strategy_id"] == qixing_v4.STRATEGY_ID
    assert normalized["v4_state"]["candidate_history"] == []


@pytest.mark.unit
def test_validate_snapshot_rejects_unknown_holding(tmp_path):
    project_root = _project(tmp_path)
    state = _state()
    state["holding"] = "NOT_AN_ETF"
    snapshot = sync_preview.RemoteSnapshot(
        state=state,
        mode={"mode": "V4"},
        release=_release(project_root),
    )

    with pytest.raises(ValueError, match="未知持仓"):
        sync_preview.validate_snapshot(snapshot, project_root=project_root)


@pytest.mark.unit
def test_validate_snapshot_rejects_release_config_mismatch(tmp_path):
    project_root = _project(tmp_path)
    release = _release(project_root)
    release["config_hash"] = "wrong"
    snapshot = sync_preview.RemoteSnapshot(state=_state(), mode={"mode": "V4"}, release=release)

    with pytest.raises(ValueError, match="配置哈希"):
        sync_preview.validate_snapshot(snapshot, project_root=project_root)


@pytest.mark.unit
def test_install_snapshot_backs_up_and_atomically_replaces_local_state(tmp_path):
    live_dir = tmp_path / "data" / "live"
    live_dir.mkdir(parents=True)
    old_state = {"initial_capital": 100_000, "cash": 100_000, "holding": None}
    (live_dir / "state.json").write_text(json.dumps(old_state), encoding="utf-8")
    (live_dir / "strategy_mode.json").write_text(json.dumps({"mode": "V3-G"}), encoding="utf-8")
    snapshot = sync_preview.RemoteSnapshot(
        state=_state(), mode={"mode": "V4"}, release={"release_id": "test-v4"}
    )

    backup = sync_preview.install_snapshot(snapshot, live_dir=live_dir)

    installed = json.loads((live_dir / "state.json").read_text(encoding="utf-8"))
    installed_mode = json.loads((live_dir / "strategy_mode.json").read_text(encoding="utf-8"))
    backed_up = json.loads((backup / "state.json").read_text(encoding="utf-8"))
    assert installed["holding"] == "161226"
    assert installed["_version"] == 8
    assert installed_mode == {"mode": "V4"}
    assert backed_up == old_state
    assert not list(live_dir.glob("*.server-sync.tmp"))
