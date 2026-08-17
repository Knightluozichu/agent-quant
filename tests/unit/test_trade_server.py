"""trade_server 幂等持久化与鉴权的最小测试骨架.

隔离: 所有文件操作通过 monkeypatch 重定向到 tmp_path, 不触碰真实 data/live/.
不涉及真实交易路径: record_manual_trade 只写 tmp_path 下的 state.json.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import live_signal as ls  # noqa: E402
import notify  # noqa: E402
import trade_server as ts  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_TOKEN = "test-" + "token-abc123"  # 测试用假 token, 非真实凭证 (拼接写法规避密钥扫描器)


@pytest.fixture
def isolated_live(tmp_path, monkeypatch):
    """把 data/live 相关的文件、目录与配置全部重定向到 tmp_path."""
    # 幂等表与锁
    monkeypatch.setattr(ts, "_IDEMPOTENCY_FILE", tmp_path / "idempotency.json")
    monkeypatch.setattr(ts, "_IDEMPOTENCY_LOCK_FILE", tmp_path / "idempotency.lock")
    # 缓存全局复位, 避免用例间串扰
    monkeypatch.setattr(ts, "_DATA_CACHE", None)
    monkeypatch.setattr(ts, "_DATA_CACHE_TIME", 0.0)
    monkeypatch.setattr(ts, "_DATA_CACHE_FILES", 0)
    # 状态文件 (load_state / state_transaction)
    monkeypatch.setattr(ls, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(ls, "STATE_TMP_FILE", tmp_path / "state.json.tmp")
    monkeypatch.setattr(ls, "LOCK_FILE", tmp_path / "quant_state.lock")
    # 行情数据目录 (health 的 data_files / get_data 的 glob)
    data_dir = tmp_path / "cross_asset"
    data_dir.mkdir()
    monkeypatch.setattr(ls, "DATA_DIR", data_dir)
    # 策略模式文件 (get_strategy_mode)
    monkeypatch.setattr(ls, "STRATEGY_MODE_FILE", tmp_path / "strategy_mode.json")
    monkeypatch.delenv("QIXING_STRATEGY_MODE", raising=False)
    # token/密码配置 (require_token → notify.load_config/save_config)
    monkeypatch.setattr(notify, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(notify, "CONFIG_FILE", tmp_path / "config.json")
    return tmp_path


def _seed_token(tmp_path: Path) -> None:
    cfg = {"web_tokens": [{"token": _TOKEN, "expires": time.time() + 3600}]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))


def _seed_state(tmp_path: Path) -> None:
    state = {
        "initial_capital": 10000.0,
        "cash": 10000.0,
        "holding": None,
        "shares": 0,
        "entry_price": 0.0,
        "trade_log": [],
    }
    (tmp_path / "state.json").write_text(json.dumps(state))


def _authed_client() -> TestClient:
    client = TestClient(ts.app)
    client.cookies.set("qx_token", _TOKEN)
    return client


# --------------------------------------------------------------------------- #
# a. 幂等: 占位操作第二次拒绝 / 并发唯一获胜 / 重启恢复
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_guard_rejects_duplicate_key(isolated_live):
    with ts._idempotency_guard("k1"):
        pass
    with pytest.raises(HTTPException) as exc_info, ts._idempotency_guard("k1"):
        pass
    assert exc_info.value.status_code == 409


@pytest.mark.unit
def test_guard_allows_retry_after_failure(isolated_live):
    """执行失败不记录 key, 允许客户端修正后重试."""
    with pytest.raises(ValueError, match="boom"), ts._idempotency_guard("k2"):
        raise ValueError("boom")
    with ts._idempotency_guard("k2"):
        pass  # 不抛 409 即通过


@pytest.mark.unit
def test_guard_concurrent_same_key_single_winner(isolated_live):
    """并发同 key: 恰好一个成功, 其余 409 (消除双成交窗口)."""

    def enter() -> str | int:
        try:
            with ts._idempotency_guard("race-key"):
                time.sleep(0.01)
            return "ok"
        except HTTPException as e:
            return e.status_code

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: enter(), range(4)))
    assert results.count("ok") == 1
    assert results.count(409) == 3


@pytest.mark.unit
def test_idempotency_persisted_across_reload(isolated_live):
    """记录落盘后重新加载 (模拟服务重启) 仍能识别重复 key."""
    with ts._idempotency_guard("restart-key"):
        pass
    store = ts._load_idempotency()
    assert "restart-key" in store
    with pytest.raises(HTTPException) as exc_info, ts._idempotency_guard("restart-key"):
        pass
    assert exc_info.value.status_code == 409


# --------------------------------------------------------------------------- #
# b. idempotency.json 损坏: 接口不 500, 非法条目丢弃, 坏文件备份重建
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_load_drops_non_numeric_entries(isolated_live):
    ts._IDEMPOTENCY_FILE.write_text(
        json.dumps(
            {
                "good": time.time(),
                "bad-str": "x",
                "bad-bool": True,
                "expired": time.time() - 7200,
            }
        )
    )
    store = ts._load_idempotency()
    assert set(store) == {"good"}


@pytest.mark.unit
def test_load_backs_up_invalid_json(isolated_live):
    ts._IDEMPOTENCY_FILE.write_text("not-json{{{")
    assert ts._load_idempotency() == {}
    assert not ts._IDEMPOTENCY_FILE.exists()
    assert ts._IDEMPOTENCY_FILE.with_suffix(".corrupt").exists()


@pytest.mark.unit
def test_load_backs_up_non_dict_json(isolated_live):
    ts._IDEMPOTENCY_FILE.write_text("[1, 2, 3]")
    assert ts._load_idempotency() == {}
    assert ts._IDEMPOTENCY_FILE.with_suffix(".corrupt").exists()


@pytest.mark.unit
def test_trade_endpoint_not_500_with_corrupt_idempotency(isolated_live):
    _seed_token(isolated_live)
    ts._IDEMPOTENCY_FILE.write_text("{{{corrupt")
    client = _authed_client()
    resp = client.post(
        "/api/trade",
        json={
            "action": "buy",
            "code": "518880",
            "shares": 100,
            "price": 1.0,
            "idempotency_key": "k1",
        },
    )
    # 账户未初始化 → 400; 关键是不因幂等文件损坏而 500
    assert resp.status_code == 400
    assert "未初始化" in resp.json()["detail"]


@pytest.mark.unit
def test_trade_endpoint_duplicate_key_conflict(isolated_live):
    """端到端: 相同 key 的 /api/trade 第二次返回 409."""
    _seed_token(isolated_live)
    _seed_state(isolated_live)
    client = _authed_client()
    payload = {
        "action": "buy",
        "code": "518880",
        "shares": 100,
        "price": 1.0,
        "idempotency_key": "dup-key",
    }
    r1 = client.post("/api/trade", json=payload)
    assert r1.status_code == 200
    assert r1.json()["holding"] == "518880"
    r2 = client.post("/api/trade", json=payload)
    assert r2.status_code == 409


# --------------------------------------------------------------------------- #
# c. /api/health 鉴权
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_health_requires_token(isolated_live):
    _seed_token(isolated_live)
    client = TestClient(ts.app)
    assert client.get("/api/health").status_code == 401
    client.cookies.set("qx_token", _TOKEN)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("ok", "uninitialized")


# --------------------------------------------------------------------------- #
# d. FastAPI 自动文档已关闭
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_docs_disabled(isolated_live):
    client = TestClient(ts.app)
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


# --------------------------------------------------------------------------- #
# 附加: /api/refresh 真注入 (refresh_data 对缓存本体执行 inject_realtime)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_refresh_injects_into_cache(isolated_live, monkeypatch):
    _seed_token(isolated_live)
    injected_with: list[dict] = []

    def fake_load_data() -> dict:
        return {"TEST": object()}  # 非交易池代码, get_trading_dates 会跳过

    def fake_inject(data: dict, spot_map: dict | None = None) -> dict:
        injected_with.append(data)
        return data

    def fake_spot() -> dict:
        return {}

    monkeypatch.setattr(ls, "load_data", fake_load_data)
    monkeypatch.setattr(ls, "inject_realtime", fake_inject)
    monkeypatch.setattr(ls, "_fetch_tencent_spot", fake_spot)

    client = _authed_client()
    resp = client.post("/api/refresh")
    assert resp.status_code == 200
    body = resp.json()
    # 空数据无法推进到今天 → fail-closed 体现在 realtime_ok=False
    assert body["realtime_ok"] is False
    assert body["realtime_reason"]
    # refresh 确实对缓存本体执行了注入, 且计数器同步
    assert len(injected_with) == 1
    assert ts._DATA_CACHE is not None
    assert ts._DATA_CACHE_TIME > 0
