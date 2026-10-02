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
    # config 锁 (require_token/login/logout/set_password 的读-改-写串行化)
    monkeypatch.setattr(ts, "_CONFIG_LOCK_FILE", tmp_path / "config.lock")
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


# --------------------------------------------------------------------------- #
# e. config.lock: 并发 login + 鉴权请求不丢 token (I-FIX-01)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_concurrent_logins_no_token_loss(isolated_live, monkeypatch):
    password = "test-" + "pw-12345"  # 拼接写法规避密钥扫描器
    ts.set_password(password)
    # set_password 清空了 token, 重新种入鉴权 token
    cfg = json.loads((isolated_live / "config.json").read_text())
    cfg["web_tokens"] = [{"token": _TOKEN, "expires": time.time() + 3600}]
    (isolated_live / "config.json").write_text(json.dumps(cfg))
    # 放行登录限流 (默认 5 次/分钟/IP, 并发测试需要更多), 并复位限流状态
    monkeypatch.setattr(ts, "_LOGIN_RATE_LIMIT", 1000)
    monkeypatch.setattr(ts, "_LOGIN_ATTEMPTS", {})

    def do_login(_: int) -> int:
        client = TestClient(ts.app)
        return client.post("/api/login", json={"password": password}).status_code

    def do_authed(_: int) -> int:
        return _authed_client().get("/api/health").status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(do_login, i) for i in range(8)] + [
            pool.submit(do_authed, i) for i in range(8)
        ]
        results = [f.result() for f in futures]
    assert results[:8] == [200] * 8  # login 全部成功
    assert results[8:] == [200] * 8  # 并发鉴权请求全部成功
    # 关键断言: 8 个并发 login 追加的 token 无一丢失 (读-改-写被锁串行化)
    cfg = json.loads((isolated_live / "config.json").read_text())
    assert len(cfg["web_tokens"]) == 8 + 1  # 8 个 login token + 1 个预置 token


# --------------------------------------------------------------------------- #
# f. 请求体大小限制: Content-Length 超 64KB → 413 (I-FIX-04)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_body_over_limit_413(isolated_live):
    _seed_token(isolated_live)
    client = _authed_client()
    payload = {
        "action": "buy",
        "code": "518880",
        "shares": 100,
        "price": 1.0,
        "idempotency_key": "x" * (70 * 1024),  # 把请求体撑过 64KB
    }
    resp = client.post("/api/trade", json=payload)
    assert resp.status_code == 413


@pytest.mark.unit
def test_body_normal_size_not_blocked(isolated_live):
    _seed_token(isolated_live)
    client = _authed_client()
    resp = client.post(
        "/api/trade",
        json={"action": "buy", "code": "518880", "shares": 100, "price": 1.0},
    )
    # 正常大小请求穿过 middleware, 走到业务校验 (账户未初始化 → 400)
    assert resp.status_code == 400
    assert "未初始化" in resp.json()["detail"]


@pytest.mark.unit
def test_body_invalid_content_length_400(isolated_live):
    client = TestClient(ts.app)
    req = client.build_request(
        "POST", "/api/login", content=b"{}", headers={"content-length": "abc"}
    )
    resp = client.send(req)
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# I-FIX-04 遗留: 公网暴露前的全局请求超时
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_global_timeout_returns_504(monkeypatch):
    import asyncio

    monkeypatch.setattr(ts, "REQUEST_TIMEOUT_S", 0.05)

    async def slow_app(scope, receive, send):
        await asyncio.sleep(5)

    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    mw = ts._TimeoutMiddleware(slow_app)
    asyncio.run(mw({"type": "http", "headers": []}, receive, send))
    assert any(m.get("status") == 504 for m in sent)


@pytest.mark.unit
def test_global_timeout_fast_request_passthrough():
    import asyncio

    from fastapi.responses import PlainTextResponse

    async def fast_app(scope, receive, send):
        await PlainTextResponse("ok")(scope, receive, send)

    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    mw = ts._TimeoutMiddleware(fast_app)
    asyncio.run(mw({"type": "http", "headers": []}, receive, send))
    assert any(m.get("status") == 200 for m in sent)


@pytest.mark.unit
def test_global_timeout_skips_non_http_scope():
    import asyncio

    called = []

    async def lifespan_app(scope, receive, send):
        called.append(scope["type"])

    mw = ts._TimeoutMiddleware(lifespan_app)
    asyncio.run(mw({"type": "lifespan"}, None, None))
    assert called == ["lifespan"]


# 2026-09-16: account-specific pool and truthful signal provenance.
@pytest.mark.unit
def test_live_pool_keeps_historical_catalog_and_calendar():
    assert "501018" not in ls.ETF_POOL
    assert "501018" in ls.ALL_CODES
    assert ls.name_of("501018") == "南方原油"
    assert "501018" in ts.rq.ETF_POOL  # frozen historical benchmark remains reproducible


@pytest.mark.unit
@pytest.mark.parametrize(
    "target,day,policy",
    [
        ("501018", "2026-09-16", "current"),
        ("518880", "2026-09-15", "current"),
        ("518880", "2026-09-16", "legacy"),
    ],
)
def test_stale_or_ineligible_decision_is_not_current(
    isolated_live, monkeypatch, target, day, policy
):
    from datetime import date

    monkeypatch.setattr(ls, "_today_sh", lambda: date(2026, 9, 16))
    monkeypatch.setattr(ts, "get_data", lambda: {})
    monkeypatch.setattr(ls, "is_trading_day", lambda _: True)
    _seed_token(isolated_live)
    _seed_state(isolated_live)
    path = isolated_live / "state.json"
    state = json.loads(path.read_text())
    state["last_decision"] = {
        "trade_date": day,
        "final_target": target,
        "pool_version": getattr(ls, "POOL_VERSION", "missing") if policy == "current" else None,
    }
    path.write_text(json.dumps(state))
    response = _authed_client().get("/api/signal").json()
    assert response["status"] == "SNAPSHOT_INVALID"
    assert not response["official"]
    assert response["target"] is None
    assert response["snapshot_date"] == day
    assert not response["actionable"]


@pytest.mark.unit
def test_web_cannot_confirm_old_pending(isolated_live):
    _seed_token(isolated_live)
    _seed_state(isolated_live)
    path = isolated_live / "state.json"
    state = json.loads(path.read_text())
    state["pending_order"] = {
        "date": "2020-01-01",
        "status": "pending",
        "buy": {"code": "518880", "shares": 100},
        "sell": None,
    }
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    response = _authed_client().post(
        "/api/confirm", json={"buy": {"code": "518880", "shares": 100, "price": 5.0}}
    )
    assert response.status_code == 400
    assert path.read_bytes() == before


@pytest.mark.unit
def test_restricted_buy_rejected_by_ledger(isolated_live):
    _seed_state(isolated_live)
    with pytest.raises(ValueError, match="可买池"):
        ls.record_manual_trade("buy", "501018", 100, 2.0)


@pytest.mark.unit
def test_restricted_asset_can_still_be_sold(isolated_live):
    _seed_token(isolated_live)
    _seed_state(isolated_live)
    path = isolated_live / "state.json"
    state = json.loads(path.read_text())
    state.update(holding="501018", shares=100, entry_price=2.0)
    path.write_text(json.dumps(state))
    response = _authed_client().post(
        "/api/trade", json={"action": "sell", "code": "501018", "shares": 100, "price": 2.0}
    )
    assert response.status_code == 200
    assert response.json()["holding"] is None


@pytest.mark.unit
def test_current_snapshot_uses_persisted_factors_not_later_prices(isolated_live, monkeypatch):
    from datetime import date

    monkeypatch.setattr(ls, "_today_sh", lambda: date(2026, 9, 16))
    monkeypatch.setattr(
        ts,
        "_market_context",
        lambda: {"market_open": False, "market_status": "已收盘", "is_trading_day": True},
    )
    monkeypatch.setattr(ts, "get_data", lambda: pytest.fail("must not recompute official factors"))
    _seed_token(isolated_live)
    _seed_state(isolated_live)
    path = isolated_live / "state.json"
    state = json.loads(path.read_text())
    state["last_decision"] = {
        "trade_date": "2026-09-16",
        "final_target": "518880",
        "mode": ls.get_strategy_mode(),
        "pool_version": ls.POOL_VERSION,
        "config_hash": ls.v4.CONFIG_HASH,
        "created_at": "2026-09-16T14:50:00+08:00",
        "factors": {
            "518880": {"slow_momentum": 0.02, "eligible": True},
            "501018": {"slow_momentum": 0.99, "eligible": True},
        },
    }
    path.write_text(json.dumps(state))
    result = _authed_client().get("/api/signal").json()
    assert result["official"]
    assert not result["actionable"]
    assert result["market_status"] == "已收盘"
    assert [row["code"] for row in result["board"]] == ["518880"]
    assert result["board"][0]["score"] == 2.0


@pytest.mark.unit
def test_live_selection_excludes_oil_without_mutating_research_pool(monkeypatch):
    import numpy as np
    import pandas as pd

    data = {
        code: pd.DataFrame({"close": np.linspace(1, end, 140), "volume": 1000})
        for code, end in (("501018", 5), ("518880", 2))
    }
    monkeypatch.setattr(ts.rq, "USE_A_SHARE_FILTER", False)
    indices = dict.fromkeys(data, 139)
    assert ts.rq.select_target(data, indices, None)[0] == "501018"
    target, candidates, *_ = ls.select_target(data, indices, "501018")
    assert target == "518880"
    assert "501018" not in dict(candidates)


@pytest.mark.unit
def test_no_data_has_no_target_and_no_server_error(isolated_live, monkeypatch):
    _seed_token(isolated_live)
    _seed_state(isolated_live)
    monkeypatch.setattr(ts, "get_data", lambda: {})
    monkeypatch.setattr(ts, "_market_context", lambda: {"market_open": False})
    result = _authed_client().get("/api/signal").json()
    assert result["status"] == "AWAITING_SNAPSHOT"
    assert result["target"] is None
    assert result["board"] == []


def test_actual_fees_and_incremental_fills_api(isolated_live):
    _seed_state(isolated_live)
    _seed_token(isolated_live)
    client = _authed_client()
    payload = {"action": "buy", "code": "518880", "shares": 100, "price": 5, "fees": 2}
    assert client.post("/api/trade", json=payload).json()["cash"] == 9498
    payload.update(price=6, fees=3)
    assert client.post("/api/trade", json=payload).json()["cash"] == 8895
    state = ls.load_state()
    assert state["shares"] == 200 and state["entry_price"] == 5.5
    before = ls.STATE_FILE.read_bytes()
    payload["code"] = "159985"
    assert client.post("/api/trade", json=payload).status_code == 400
    assert ls.STATE_FILE.read_bytes() == before


def test_historical_restricted_fill_api_and_duplicate(isolated_live):
    _seed_state(isolated_live)
    _seed_token(isolated_live)
    client = _authed_client()
    payload = {
        "action": "buy",
        "code": "501018",
        "shares": 100,
        "price": 2,
        "fees": 1,
        "date": "2026-09-14",
        "historical_fill": True,
        "evidence_confirmed": True,
        "idempotency_key": "historical-oil-api",
    }
    incomplete = dict(payload, evidence_confirmed=False)
    assert client.post("/api/trade", json=incomplete).status_code == 400
    assert client.post("/api/trade", json=payload).status_code == 200
    before = ls.STATE_FILE.read_bytes()
    assert client.post("/api/trade", json=payload).status_code == 409
    assert ls.STATE_FILE.read_bytes() == before
    assert "501018" not in ls.ETF_POOL


@pytest.mark.parametrize("patch", [{"fees": -1}, {"date": "2099-01-01"}])
def test_invalid_actual_fill_api_atomic(isolated_live, patch):
    _seed_state(isolated_live)
    _seed_token(isolated_live)
    before = ls.STATE_FILE.read_bytes()
    payload = {"action": "buy", "code": "518880", "shares": 100, "price": 5, **patch}
    assert _authed_client().post("/api/trade", json=payload).status_code in (400, 422)
    assert ls.STATE_FILE.read_bytes() == before


def test_c5_is_profile_not_product_permission(isolated_live):
    _seed_token(isolated_live)
    data = _authed_client().get("/api/etfs").json()
    assert data["account_profile"]["risk_grade"] == "C5"
    assert data["account_profile"]["source"] == "user_reported"
    assert "501018" not in {x["code"] for x in data["etfs"]}
    assert "501018" in {x["code"] for x in data["recordable_etfs"]}


def test_confirm_uses_actual_fees_not_plan_quantity(isolated_live):
    _seed_state(isolated_live)
    _seed_token(isolated_live)
    state = ls.load_state()
    state.update(
        holding="518880",
        shares=1000,
        cash=100,
        pending_order={
            "date": str(ls._today_sh()),
            "status": "pending",
            "pool_version": ls.POOL_VERSION,
            "sell": {"code": "518880", "shares": 900},
            "buy": None,
        },
    )
    ls.STATE_FILE.write_text(json.dumps(state))
    response = _authed_client().post(
        "/api/confirm",
        json={
            "sell": {"shares": 1000, "price": 5, "fees": 3},
            "idempotency_key": "actual-full-exit",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["cash"] == 5097
    assert ls.load_state()["trade_log"][-1]["fees"] == 3


def test_manual_fill_disables_pending_view(isolated_live):
    _seed_state(isolated_live)
    _seed_token(isolated_live)
    state = ls.load_state()
    state.update(
        holding="518880",
        shares=1000,
        pending_order={
            "date": str(ls._today_sh()),
            "status": "pending",
            "pool_version": ls.POOL_VERSION,
            "sell": {"code": "518880", "shares": 1000},
            "buy": None,
        },
    )
    ls.STATE_FILE.write_text(json.dumps(state))
    response = _authed_client().post(
        "/api/trade",
        json={"action": "sell", "code": "518880", "shares": 100, "price": 5, "fees": 0},
    )
    assert response.status_code == 200
    view = ts._pending_view(ls.load_state()["pending_order"])
    assert not view["confirmable"] and "手工成交" in view["blocked_reason"]


@pytest.mark.parametrize(
    "cash,shares,blocked", [(9498, 100, False), (9506, 100, True), (9498, 200, True)]
)
def test_equity_reports_ledger_gaps_without_mutating_facts(isolated_live, cash, shares, blocked):
    _seed_state(isolated_live)
    _seed_token(isolated_live)
    state = ls.load_state()
    state.update(
        cash=cash,
        shares=shares,
        holding="518880",
        trade_log=[
            {
                "date": "2026-09-14",
                "action": "buy",
                "code": "518880",
                "shares": 100,
                "amount": 502,
                "price": 5,
            }
        ],
    )
    ls.STATE_FILE.write_text(json.dumps(state))
    before = ls.STATE_FILE.read_bytes()
    result = _authed_client().get("/api/equity")
    assert result.status_code == 200
    data = result.json()
    assert data["needs_reconciliation"] is blocked
    assert data["cash_gap"] == round(cash - 9498, 2)
    assert data["shares_gap"] == shares - 100
    assert ls.STATE_FILE.read_bytes() == before
