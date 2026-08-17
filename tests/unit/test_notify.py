"""通知配置持久化与 Bark 推送重试测试。"""

from __future__ import annotations

import json
import sys
import urllib.error
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


class _FakeResp:
    """模拟 urllib.request.urlopen 的响应上下文管理器."""

    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *args) -> bool:
        return False


def _make_http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.day.app", code, "error", hdrs=None, fp=None)


def _isolate_notify(monkeypatch, tmp_path) -> tuple[list, list]:
    """隔离 push_bark 的外部依赖: BARK_KEY、失败日志、urlopen、sleep.

    返回 (urlopen 的 timeout 记录, sleep 的间隔记录).
    """
    monkeypatch.setenv("BARK_KEY", "testkey123")
    monkeypatch.setattr(notify, "LIVE_DIR", tmp_path / "live")
    monkeypatch.setattr(notify, "CONFIG_FILE", tmp_path / "live" / "config.json")
    monkeypatch.setattr(notify, "FAILURE_LOG", tmp_path / "live" / "notify_failures.log")
    timeouts: list[float] = []
    sleeps: list[float] = []
    monkeypatch.setattr(notify.time, "sleep", lambda s: sleeps.append(s))
    return timeouts, sleeps


@pytest.mark.unit
def test_push_bark_retries_network_errors_with_exponential_backoff(monkeypatch, tmp_path):
    timeouts, sleeps = _isolate_notify(monkeypatch, tmp_path)

    def fake_urlopen(req, timeout):
        timeouts.append(timeout)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    assert notify.push_bark("测试标题", "正文") is False
    assert len(timeouts) == 3  # 网络错误重试满 3 次
    assert timeouts == [10, 10, 10]  # 每次固定超时 10s
    assert sleeps == [2, 4]  # 指数退避 2s → 4s


@pytest.mark.unit
def test_push_bark_does_not_retry_http_4xx(monkeypatch, tmp_path):
    timeouts, sleeps = _isolate_notify(monkeypatch, tmp_path)

    def fake_urlopen(req, timeout):
        timeouts.append(timeout)
        raise _make_http_error(400)

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    assert notify.push_bark("测试标题", "正文") is False
    assert len(timeouts) == 1  # 4xx 配置错误不重试
    assert sleeps == []
    # 不重试即不是"重试耗尽", 不写失败日志
    assert not notify.FAILURE_LOG.exists()


@pytest.mark.unit
def test_push_bark_does_not_retry_bark_business_failure(monkeypatch, tmp_path):
    timeouts, sleeps = _isolate_notify(monkeypatch, tmp_path)

    def fake_urlopen(req, timeout):
        timeouts.append(timeout)
        return _FakeResp({"code": 400, "message": "failed to get device token"})

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    assert notify.push_bark("测试标题", "正文") is False
    assert len(timeouts) == 1  # HTTP 200 但 body code != 200, 不重试
    assert sleeps == []


@pytest.mark.unit
def test_push_bark_retries_http_5xx(monkeypatch, tmp_path):
    timeouts, sleeps = _isolate_notify(monkeypatch, tmp_path)

    def fake_urlopen(req, timeout):
        timeouts.append(timeout)
        raise _make_http_error(502)

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    assert notify.push_bark("测试标题", "正文") is False
    assert len(timeouts) == 3  # 5xx 退避重试
    assert sleeps == [2, 4]


@pytest.mark.unit
def test_push_bark_logs_failure_after_retries_exhausted(monkeypatch, tmp_path):
    _isolate_notify(monkeypatch, tmp_path)

    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    assert notify.push_bark("调仓指令", "正文") is False

    log = notify.FAILURE_LOG.read_text(encoding="utf-8")
    lines = log.strip().splitlines()
    assert len(lines) == 1
    assert "调仓指令" in lines[0]
    assert "URLError" in lines[0]
    assert "testkey123" not in lines[0]  # 失败记录不得含 Bark key
    assert "api.day.app" not in lines[0]  # 不得含完整 URL
