"""通知配置持久化测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import notify  # noqa: E402


@pytest.mark.unit
def test_save_config_atomically_replaces_json_without_temp_file(monkeypatch, tmp_path):
    live_dir = tmp_path / "live"
    config_file = live_dir / "config.json"
    monkeypatch.setattr(notify, "LIVE_DIR", live_dir)
    monkeypatch.setattr(notify, "CONFIG_FILE", config_file)

    payload = {"web_password": "hash", "web_salt": "salt", "web_tokens": []}
    notify.save_config(payload)

    assert json.loads(config_file.read_text(encoding="utf-8")) == payload
    assert not list(live_dir.glob(".config.json.*.tmp"))
