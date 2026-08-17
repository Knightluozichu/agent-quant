"""七星 V4 实盘记账网页 (FastAPI 后端).

手机端使用流程:
  1. 每个交易日 14:50 服务器自动生成信号并 Bark 推送 (信号挂起为"待确认")
  2. 你在易淘金 App 手动下单
  3. 打开本网页, 填入真实成交价/数量, 点"确认成交" → 持仓状态更新为现实

核心原则: 持仓状态以网页确认的真实成交为准, 系统不自动假设成交.

启动:
  uv run python scripts/trade_server.py --port 8090
首次设置访问密码 (交互式输入, 不进 shell history):
  uv run python scripts/trade_server.py --set-password
  或: WEB_PASSWORD=xxx uv run python scripts/trade_server.py --set-password
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import hmac
import json
import math
import os
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from starlette.types import ASGIApp, Receive, Scope, Send

sys.path.insert(0, str(Path(__file__).parent))

import live_signal as ls
import run_qixing_v3 as rq
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from notify import load_config, save_config
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.responses import PlainTextResponse

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(
    title="七星V4 实盘记账",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# 请求体大小上限: 合法请求体只有几百字节 (登录密码/确认成交表单), 64KB 绰绰有余
MAX_BODY_BYTES = 64 * 1024


class _BodySizeLimitMiddleware:
    """Content-Length 超上限直接 413, 不读 body.

    只检查 Content-Length 头; 无 Content-Length 的 chunked 请求不强行 buffer
    全量 (会把带宽 DoS 变成内存 DoS, 更糟), 由 pydantic 字段级长度限制兜底.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        raw = headers.get(b"content-length")
        if raw is not None:
            try:
                size = int(raw)
            except ValueError:
                resp = PlainTextResponse("非法 Content-Length", status_code=400)
                await resp(scope, receive, send)
                return
            if size > MAX_BODY_BYTES:
                resp = PlainTextResponse("请求体超过 64KB 上限", status_code=413)
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


app.add_middleware(_BodySizeLimitMiddleware)

# 行情数据内存缓存 (带 mtime + 文件数失效: cron 14:50 更新数据后自动重载)
_DATA_CACHE: dict | None = None
_DATA_CACHE_TIME: float = 0.0
_DATA_CACHE_FILES: int = 0

# Token 过期时间 (秒): 24 小时
TOKEN_TTL = 86400

# 登录限流: 每分钟最多 5 次尝试
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_LOGIN_RATE_LIMIT = 5
_LOGIN_WINDOW = 60.0

# 幂等去重: 已处理的 idempotency_key (持久化到文件, 保留 1 小时)
_IDEMPOTENCY_FILE = ls.LIVE_DIR / "idempotency.json"
_IDEMPOTENCY_LOCK_FILE = ls.LIVE_DIR / "idempotency.lock"
_IDEMPOTENCY_TTL = 3600.0


def _load_idempotency() -> dict[str, float]:
    """读取幂等表. 文件损坏时备份为 .corrupt 并重建, 不让写接口 500."""
    if not _IDEMPOTENCY_FILE.exists():
        return {}
    try:
        with open(_IDEMPOTENCY_FILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        print(f"  ⚠️  幂等文件损坏, 备份后重建: {e}")
        with contextlib.suppress(OSError):
            _IDEMPOTENCY_FILE.replace(_IDEMPOTENCY_FILE.with_suffix(".corrupt"))
        return {}
    if not isinstance(data, dict):
        print("  ⚠️  幂等文件内容非 dict, 备份后重建")
        with contextlib.suppress(OSError):
            _IDEMPOTENCY_FILE.replace(_IDEMPOTENCY_FILE.with_suffix(".corrupt"))
        return {}
    now = time.time()
    store: dict[str, float] = {}
    for k, v in data.items():
        # 逐条校验: 非法条目丢弃, 不拖垮整个文件
        if (
            isinstance(k, str)
            and isinstance(v, (int, float))
            and not isinstance(v, bool)
            and now - float(v) <= _IDEMPOTENCY_TTL
        ):
            store[k] = float(v)
    return store


def _save_idempotency(store: dict[str, float]) -> None:
    """原子写幂等表 (临时文件 + fsync + os.replace), 失败时告警不静默."""
    tmp = _IDEMPOTENCY_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(store, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(_IDEMPOTENCY_FILE)
        ls._fsync_dir(_IDEMPOTENCY_FILE.parent)
    except OSError as e:
        print(f"  ⚠️  幂等记录写盘失败 (重启后可能重复放行): {e}")


@contextlib.contextmanager
def _idempotency_guard(key: str | None) -> Iterator[None]:
    """幂等占位: 持锁完成 检查→执行→记录, 消除并发同 key 双成交窗口.

    锁顺序约定 (固定, 防死锁):
      config.lock → idempotency.lock → state lock (quant_state.lock)
    config.lock 只由 require_token/login/logout/set_password 持有且在端点体
    之前释放, 不会与本锁同持; 本锁内只允许再取 state lock, 禁止反向获取.
    """
    _IDEMPOTENCY_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(_IDEMPOTENCY_LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        store = _load_idempotency()
        if key and key in store:
            raise HTTPException(status_code=409, detail="重复提交 (idempotency_key 已处理)")
        yield
        # 执行成功才记录; 失败 (如 400) 不记录, 允许客户端修正后重试
        if key:
            store[key] = time.time()
        _save_idempotency(store)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


# 金额/数量安全上限 (拒绝超大值, 防输入错误/注入)
MAX_TRADE_AMOUNT = 1e9


def get_data() -> dict:
    global _DATA_CACHE, _DATA_CACHE_TIME, _DATA_CACHE_FILES
    files = list(ls.DATA_DIR.glob("*.parquet"))
    if not files:
        _DATA_CACHE = None
        _DATA_CACHE_TIME = 0.0
        _DATA_CACHE_FILES = 0
        return ls.load_data()
    mtime = max(f.stat().st_mtime for f in files)
    # 同时检查文件数量变化 (删除/新增)
    if _DATA_CACHE is None or mtime > _DATA_CACHE_TIME or len(files) != _DATA_CACHE_FILES:
        _DATA_CACHE = ls.load_data()
        _DATA_CACHE_TIME = time.time()
        _DATA_CACHE_FILES = len(files)
    # 返回副本防止 inject_realtime 污染全局缓存
    return copy.deepcopy(_DATA_CACHE)


def refresh_data() -> None:
    """强制重载缓存, 并同步注入当日实时行情 (与 status/signal 惰性注入同源).

    inject_realtime 内部 fail-closed: 行情缺失/过期/冲突时不注入, 缓存保持历史数据.
    """
    global _DATA_CACHE, _DATA_CACHE_TIME, _DATA_CACHE_FILES
    files = list(ls.DATA_DIR.glob("*.parquet"))
    _DATA_CACHE = ls.inject_realtime(ls.load_data())
    _DATA_CACHE_TIME = time.time()
    _DATA_CACHE_FILES = len(files)


# --------------------------------------------------------------------------- #
# 鉴权: 密码哈希 + token (带过期 + 限流)
# --------------------------------------------------------------------------- #
# config.json 读-改-写串行化锁 (跨线程/跨进程, 防并发 login 与 token 清理互相覆盖)
_CONFIG_LOCK_FILE = ls.LIVE_DIR / "config.lock"


@contextlib.contextmanager
def _config_lock() -> Iterator[None]:
    """config.json 读-改-写串行化. 锁文件打不开时降级为无锁 + 告警 (不 500).

    锁顺序约定 (固定, 防死锁):
      config.lock → idempotency.lock → state lock (quant_state.lock)
    require_token 在端点体之前执行且锁随函数返回释放, login/logout/set_password
    只持 config.lock, 因此不存在持 config.lock 再取后两把锁之外的反向路径.
    """
    lock_fd: int | None = None
    try:
        _CONFIG_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(str(_CONFIG_LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    except OSError as e:
        print(f"  ⚠️  config 锁不可用, 降级为无锁: {e}")
        lock_fd = None
    try:
        yield
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


def _hash_password(pwd: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pwd.encode(), salt.encode(), 100_000).hex()


def set_password(pwd: str) -> None:
    with _config_lock():
        cfg = load_config()
        salt = secrets.token_hex(8)
        cfg["web_salt"] = salt
        cfg["web_password"] = _hash_password(pwd, salt)
        # 密码修改后全部 token 失效
        cfg["web_tokens"] = []
        save_config(cfg)


def verify_password(pwd: str) -> bool:
    cfg = load_config()
    salt = cfg.get("web_salt")
    expected = cfg.get("web_password")
    if not salt or not expected:
        return False
    return hmac.compare_digest(_hash_password(pwd, salt), expected)


def _cleanup_expired_tokens(cfg: dict) -> dict:
    """清除过期 token, 返回更新后的 cfg."""
    now = time.time()
    tokens = cfg.get("web_tokens", [])
    # 兼容旧格式 (纯字符串 token) → 迁移为带时间戳的 dict
    cleaned = []
    for t in tokens:
        if isinstance(t, str):
            # 旧格式 token 无过期时间, 保留但标记为即将过期
            cleaned.append({"token": t, "expires": now + 3600})
        elif isinstance(t, dict) and t.get("expires", 0) > now:
            cleaned.append(t)
    cfg["web_tokens"] = cleaned
    return cfg


def _extract_token(request: Request) -> str:
    """从 Bearer header 或 HttpOnly cookie 中提取 token."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.cookies.get("qx_token", "")


def require_token(request: Request) -> None:
    # 读-改-写 (清理过期 token + 保存) 全程持锁, 防与并发 login/logout 互相覆盖
    with _config_lock():
        cfg = load_config()
        cfg = _cleanup_expired_tokens(cfg)
        save_config(cfg)
    tokens = [t["token"] for t in cfg.get("web_tokens", [])]
    token = _extract_token(request)
    if not token or not any(hmac.compare_digest(token, t) for t in tokens):
        raise HTTPException(status_code=401, detail="未授权或 token 已过期")


def _check_login_rate(client_ip: str) -> None:
    """登录限流: 每分钟最多 5 次, 超出返回 429."""
    now = time.time()
    attempts = _LOGIN_ATTEMPTS.get(client_ip, [])
    recent = [t for t in attempts if now - t < _LOGIN_WINDOW]
    if len(recent) >= _LOGIN_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="登录尝试过于频繁, 请稍后再试",
        )
    _LOGIN_ATTEMPTS[client_ip] = recent


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)

    @field_validator("password")
    @classmethod
    def password_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("密码不能为空")
        return v


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


# --------------------------------------------------------------------------- #
# 鉴权接口
# --------------------------------------------------------------------------- #
@app.post("/api/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict:
    client_ip = request.client.host if request.client else "unknown"
    _check_login_rate(client_ip)

    if not load_config().get("web_password"):
        raise HTTPException(
            status_code=503,
            detail="尚未设置访问密码, 请在服务器运行: "
            "python scripts/trade_server.py --set-password",
        )
    if not verify_password(payload.password):
        # 记录失败尝试
        now = time.time()
        attempts = _LOGIN_ATTEMPTS.get(client_ip, [])
        attempts.append(now)
        _LOGIN_ATTEMPTS[client_ip] = attempts
        raise HTTPException(status_code=401, detail="密码错误")

    # 登录成功, 清除该 IP 的失败记录
    _LOGIN_ATTEMPTS.pop(client_ip, None)

    token = secrets.token_hex(16)
    # token 追加是读-改-写, 持锁防与并发 require_token 清理/logout 互相覆盖
    with _config_lock():
        cfg = load_config()
        cfg = _cleanup_expired_tokens(cfg)
        cfg.setdefault("web_tokens", []).append(
            {
                "token": token,
                "expires": time.time() + TOKEN_TTL,
                "created": time.time(),
            }
        )
        save_config(cfg)

    # 设置 HttpOnly cookie (浏览器自动管理, JS 无法读取, 防 XSS 窃取)
    is_https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(
        key="qx_token",
        value=token,
        max_age=TOKEN_TTL,
        httponly=True,
        samesite="lax",
        secure=is_https,
        path="/",
    )
    return {"token": token, "expires_in": TOKEN_TTL}


@app.post("/api/logout")
def logout(request: Request, response: Response) -> dict:
    """注销当前 token (同时清除 cookie 和服务端记录)."""
    token = _extract_token(request)
    if token:
        with _config_lock():
            cfg = load_config()
            cfg["web_tokens"] = [
                t
                for t in cfg.get("web_tokens", [])
                if isinstance(t, dict) and t.get("token") != token
            ]
            save_config(cfg)
    response.delete_cookie(key="qx_token", path="/")
    return {"ok": True}


@app.get("/api/health")
def api_health(_: None = Depends(require_token)) -> dict:
    """健康检查: 只返回安全的运行指标, 不暴露持仓/token/凭证."""
    state = ls.load_state()
    return {
        "status": "ok" if state else "uninitialized",
        "service": "qixing-v4",
        "version": "4.0",
        "strategy_mode": ls.get_strategy_mode(),
        "strategy_id": ls.v4.STRATEGY_ID,
        "config_hash": ls.v4.CONFIG_HASH,
        "has_state": state is not None,
        "data_files": len(list(ls.DATA_DIR.glob("*.parquet"))),
    }


# --------------------------------------------------------------------------- #
# 数据接口
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def api_status(_: None = Depends(require_token)) -> dict:
    state = ls.load_state()
    if state is None:
        return {"initialized": False}
    data = get_data()
    data = ls.inject_realtime(data)  # 注入当日实时行情, 盘中也能显示今天
    td = ls.get_trading_dates(data)[-1]
    holding = state["holding"]
    holding_info = None
    used_realtime = False
    if holding:
        entry_date = state.get("entry_date")
        # 优先用实时行情(当日真实价), 失败回退历史价
        p_realtime = ls.get_realtime_price(holding)
        p_hist = ls.price_on(data, holding, td)
        if p_realtime:
            p = p_realtime
            price_stale = False
            used_realtime = True
        else:
            p = p_hist or 0.0
            # 数据日期 < 买入日 → 数据滑后, 用成本价代替, 盈亏待更新
            price_stale = bool(entry_date and str(td) < str(entry_date))
            if price_stale:
                p = state["entry_price"]
        mv = state["shares"] * p if p else 0.0
        if price_stale:
            pnl_pct = 0.0
        elif p and state["entry_price"]:
            pnl_pct = round((p / state["entry_price"] - 1) * 100, 2)
        else:
            pnl_pct = 0.0
        holding_info = {
            "code": holding,
            "name": ls.name_of(holding),
            "shares": state["shares"],
            "entry_price": state["entry_price"],
            "price": round(p, 3) if p else None,
            "market_value": round(mv, 2),
            "pnl_pct": pnl_pct,
            "price_stale": price_stale,
        }
    total = state["cash"] + (holding_info["market_value"] if holding_info else 0.0)
    ret = (total / state["initial_capital"] - 1) * 100
    return {
        "initialized": True,
        "trade_date": str(td),
        "realtime": used_realtime,
        "cash": round(state["cash"], 2),
        "holding": holding_info,
        "total": round(total, 2),
        "initial_capital": state["initial_capital"],
        "return_pct": round(ret, 2),
        "pending_order": state.get("pending_order"),
        "strategy_mode": ls.get_strategy_mode(),
        "strategy_id": ls.v4.STRATEGY_ID,
        "config_hash": ls.v4.CONFIG_HASH,
        "last_decision": state.get("last_decision"),
    }


@app.get("/api/signal")
def api_signal(_: None = Depends(require_token)) -> dict:
    state = ls.load_state()
    if state and state.get("last_decision"):
        decision = state["last_decision"]
        data = get_data()
        td = decision["trade_date"]
        target = decision["final_target"]
        try:
            board_date = datetime.strptime(td, "%Y-%m-%d").date()
            board = ls.momentum_board_data(data, board_date, state.get("holding"), target)
        except (KeyError, IndexError, ValueError):
            board = []
        return {
            "status": "OK",
            "official": True,
            "trade_date": td,
            "is_trading_day": ls.is_trading_day(ls._today_sh()),
            "strategy_mode": decision.get("mode", ls.get_strategy_mode()),
            "strategy_id": decision.get("strategy_id", ls.v4.STRATEGY_ID),
            "config_hash": decision.get("config_hash", ls.v4.CONFIG_HASH),
            "decision_id": decision.get("decision_id"),
            "target": {"code": target, "name": ls.name_of(target)},
            "holding": state.get("holding"),
            "v3g_target": decision.get("v3g_target"),
            "raw_v4_target": decision.get("raw_v4_target"),
            "confirmation_hits": decision.get("confirmation_hits", 0),
            "confirmation_required": decision.get("confirmation_required", 2),
            "v4_triggered": decision.get("v4_triggered", False),
            "v4_blocked_by": decision.get("v4_blocked_by", ""),
            "scheduled_lock": decision.get("scheduled_lock", False),
            "board": board,
            "pending_order": state.get("pending_order"),
        }
    data = get_data()
    # 注入当日实时行情 (解决parquet没有今天数据的问题)
    data = ls.inject_realtime(data)
    td = ls.get_trading_dates(data)[-1]
    today = ls._today_sh()

    # fail-closed: 非交易日标记但不阻断展示
    is_trading = ls.is_trading_day(today)

    # fail-closed: 实时行情注入失败 (数据停在昨日)
    if td < today:
        return {
            "status": "DATA_UNAVAILABLE",
            "reason": f"实时行情注入失败 (数据停在 {td}), 信号不可用",
            "trade_date": str(td),
            "is_trading_day": is_trading,
            "holding": state["holding"] if state else None,
            "pending_order": state.get("pending_order") if state else None,
        }

    # fail-closed: 数据完整性检查 (所有ETF必须有当日数据)
    ok, missing = ls.check_data_availability(data, td)
    if not ok:
        return {
            "status": "DATA_UNAVAILABLE",
            "reason": f"数据缺失: {', '.join(missing)}",
            "trade_date": str(td),
            "is_trading_day": is_trading,
            "holding": state["holding"] if state else None,
            "pending_order": state.get("pending_order") if state else None,
        }

    holding = state["holding"] if state else None
    idx_map = ls.build_etf_data_at_date(data, td)
    target, _candidates, _best, a_share_weak = ls.select_target(data, idx_map, holding)
    board = ls.momentum_board_data(data, td, holding, target)
    return {
        "status": "OK",
        "official": False,
        "strategy_mode": ls.get_strategy_mode(),
        "strategy_id": ls.v4.STRATEGY_ID,
        "trade_date": str(td),
        "is_trading_day": is_trading,
        "target": {"code": target, "name": ls.name_of(target)},
        "holding": holding,
        "a_share_weak": bool(a_share_weak),
        "board": board,
        "pending_order": state.get("pending_order") if state else None,
    }


@app.get("/api/etfs")
def api_etfs(_: None = Depends(require_token)) -> dict:
    pool = [{"code": c, "name": n} for c, n in ls.ETF_POOL.items()]
    pool.append({"code": ls.DEFENSE, "name": ls.name_of(ls.DEFENSE)})
    return {"etfs": pool}


@app.get("/api/history")
def api_history(_: None = Depends(require_token)) -> dict:
    state = ls.load_state()
    log = state.get("trade_log", []) if state else []
    return {"trades": list(reversed(log))}


@app.get("/api/equity")
def api_equity(_: None = Depends(require_token)) -> dict:
    state = ls.load_state()
    if not state:
        return {"curve": [], "initial_capital": 0}
    curve = ls.build_equity_curve(state, get_data())
    return {"curve": curve, "initial_capital": state["initial_capital"]}


# 回测结果缓存 (数据更新后自动重算)
_BACKTEST_CACHE: dict | None = None
_BACKTEST_TIME: float = 0.0


@app.get("/api/backtest")
def api_backtest(_: None = Depends(require_token)) -> dict:
    """返回 V3-G 基线回测结果 (供网页绘制基准全景图)."""
    global _BACKTEST_CACHE, _BACKTEST_TIME
    mtime = max((f.stat().st_mtime for f in ls.DATA_DIR.glob("*.parquet")), default=0.0)
    if _BACKTEST_CACHE is None or mtime > _BACKTEST_TIME:
        data = rq.load_data()
        result = rq.run_qixing_v3_no_lookahead(data)
        eq = result["equity_curve"]
        cummax = eq["equity"].cummax()
        dd = (eq["equity"] - cummax) / cummax * 100
        span = (eq["trade_date"].iloc[-1] - eq["trade_date"].iloc[0]).days / 365.25
        _BACKTEST_CACHE = {
            "metrics": {
                "total_return": round(result["total_return"] * 100, 1),
                "ann_return": round(result["ann_return"] * 100, 1),
                "sharpe": round(result["sharpe"], 2),
                "max_drawdown": round(result["max_drawdown"] * 100, 1),
                "n_trades": result["n_trades"],
                "n_cancelled": result.get("n_cancelled", 0),
                "start": str(eq["trade_date"].iloc[0])[:7],
                "end": str(eq["trade_date"].iloc[-1])[:7],
                "years": round(span, 1),
                "param_hash": result.get("param_hash", ""),
                "data_hash": result.get("data_hash", ""),
            },
            "dates": eq["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
            "values": [round(float(v), 0) for v in eq["equity"]],
            "drawdown": [round(float(v), 2) for v in dd],
            "holdings": eq["holding"].tolist(),
            "yearly": [
                {
                    "year": y,
                    "return": round(v["return"] * 100, 1),
                    "max_dd": round(v["max_dd"] * 100, 1),
                }
                for y, v in sorted(result["yearly"].items())
            ],
            "etf_names": {**rq.ETF_POOL, rq.DEFENSE: "货币基金"},
        }
        _BACKTEST_TIME = time.time()
    return _BACKTEST_CACHE


# --------------------------------------------------------------------------- #
# 写接口 (记账)
# --------------------------------------------------------------------------- #
class _SellLeg(BaseModel):
    """卖出腿 (确认成交用)."""

    model_config = ConfigDict(extra="forbid")
    shares: int = Field(gt=0)
    price: float = Field(gt=0)

    @field_validator("shares")
    @classmethod
    def _shares_bound(cls, v: int) -> int:
        if v > MAX_TRADE_AMOUNT:
            raise ValueError("卖出数量超限")
        return v

    @field_validator("price")
    @classmethod
    def _price_finite(cls, v: float) -> float:
        if not math.isfinite(v) or v > MAX_TRADE_AMOUNT:
            raise ValueError("价格非法 (NaN/Infinity/超大金额)")
        return v


class _BuyLeg(BaseModel):
    """买入腿 (确认成交用)."""

    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=6, max_length=6)
    shares: int = Field(gt=0)
    price: float = Field(gt=0)

    @field_validator("shares")
    @classmethod
    def _shares_bound(cls, v: int) -> int:
        if v > MAX_TRADE_AMOUNT:
            raise ValueError("买入数量超限")
        return v

    @field_validator("price")
    @classmethod
    def _price_finite(cls, v: float) -> float:
        if not math.isfinite(v) or v > MAX_TRADE_AMOUNT:
            raise ValueError("价格非法 (NaN/Infinity/超大金额)")
        return v


class ConfirmRequest(BaseModel):
    """确认待确认订单 (用真实成交数据)."""

    model_config = ConfigDict(extra="forbid")
    sell: _SellLeg | None = None
    buy: _BuyLeg | None = None
    idempotency_key: str | None = Field(default=None, max_length=128)
    order_id: str | None = Field(default=None, max_length=128)
    expected_state_version: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _at_least_one_leg(self) -> ConfirmRequest:
        if self.sell is None and self.buy is None:
            raise ValueError("sell 和 buy 至少需提供其一")
        return self


class TradeRequest(BaseModel):
    """手动记账 (买入/卖出)."""

    model_config = ConfigDict(extra="forbid")
    action: str
    code: str = Field(min_length=6, max_length=6)
    shares: int = Field(gt=0)
    price: float = Field(gt=0)
    date: str | None = Field(default=None)
    idempotency_key: str | None = Field(default=None, max_length=128)

    @field_validator("action")
    @classmethod
    def _valid_action(cls, v: str) -> str:
        if v not in ("buy", "sell"):
            raise ValueError("action 必须为 buy 或 sell")
        return v

    @field_validator("shares")
    @classmethod
    def _shares_bound(cls, v: int) -> int:
        if v > MAX_TRADE_AMOUNT:
            raise ValueError("数量超限")
        return v

    @field_validator("price")
    @classmethod
    def _price_finite(cls, v: float) -> float:
        if not math.isfinite(v) or v > MAX_TRADE_AMOUNT:
            raise ValueError("价格非法 (NaN/Infinity/超大金额)")
        return v

    @field_validator("date")
    @classmethod
    def _valid_date(cls, v: str | None) -> str | None:
        if not v:
            return None
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError("日期格式非法, 应为 YYYY-MM-DD") from e
        return v


@app.post("/api/confirm")
def api_confirm(payload: ConfirmRequest, _: None = Depends(require_token)) -> dict:
    # 幂等占位持锁覆盖 检查→执行→记录 全程 (锁顺序: config.lock → idempotency.lock → state lock)
    with _idempotency_guard(payload.idempotency_key):
        state = ls.load_state()
        if state is None:
            raise HTTPException(status_code=400, detail="账户未初始化")
        pending = state.get("pending_order")
        if not pending:
            raise HTTPException(status_code=400, detail="无待确认订单")
        # 订单必须处于 pending 状态, 拒绝重复确认
        if pending.get("status") != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"订单已处理 (状态: {pending.get('status')}), 不可重复确认",
            )
        try:
            state = ls.confirm_order(
                payload.sell.model_dump() if payload.sell else None,
                payload.buy.model_dump() if payload.buy else None,
                order_id=payload.order_id,
                expected_state_version=payload.expected_state_version,
                idempotency_key=payload.idempotency_key,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True, "cash": round(state["cash"], 2), "holding": state["holding"]}


@app.post("/api/skip")
def api_skip(_: None = Depends(require_token)) -> dict:
    try:
        ls.skip_pending()
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"ok": True}


@app.post("/api/trade")
def api_trade(payload: TradeRequest, _: None = Depends(require_token)) -> dict:
    # 幂等占位持锁覆盖 检查→执行→记录 全程 (锁顺序: config.lock → idempotency.lock → state lock)
    with _idempotency_guard(payload.idempotency_key):
        state = ls.load_state()
        if state is None:
            raise HTTPException(status_code=400, detail="账户未初始化")

        # 代码必须在交易池内 (ETF_POOL + DEFENSE)
        allowed_codes = set(ls.ETF_POOL.keys()) | {ls.DEFENSE}
        if payload.code not in allowed_codes:
            raise HTTPException(status_code=400, detail=f"非法代码 {payload.code}, 不在交易池")

        if payload.action == "buy":
            # 数量须为 100 的整数倍
            if payload.shares % 100 != 0:
                raise HTTPException(status_code=400, detail="买入数量必须为 100 的整数倍")
            # 现金充足 (含手续费+滑点)
            cost = payload.shares * payload.price * (1 + ls.FEE + ls.SLIPPAGE)
            if cost > state["cash"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"现金不足 (需 {cost:.2f}, 可用 {state['cash']:.2f})",
                )
            # 买入前必须空仓
            if state["holding"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"已有持仓 {state['holding']}, 请先卖出再买入",
                )
        else:  # sell
            # 空仓不可卖出
            if not state["holding"]:
                raise HTTPException(status_code=400, detail="当前空仓, 无法卖出")
            # 卖出代码必须匹配当前持仓
            if payload.code != state["holding"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"卖出代码 {payload.code} 与当前持仓 {state['holding']} 不匹配",
                )
            # 卖出数量不得超过持仓
            if payload.shares > state["shares"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"卖出数量 {payload.shares} 超过持仓 {state['shares']}",
                )

        try:
            state = ls.record_manual_trade(
                payload.action,
                payload.code,
                payload.shares,
                payload.price,
                payload.date,
            )
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True, "cash": round(state["cash"], 2), "holding": state["holding"]}


@app.post("/api/refresh")
def api_refresh(_: None = Depends(require_token)) -> dict:
    refresh_data()  # 重载缓存并注入当日实时行情 (fail-closed 在 inject_realtime 内部)
    # 校验注入结果: 缓存最新交易日是否已推进到今天 (Asia/Shanghai)
    today = ls._today_sh()
    dates = ls.get_trading_dates(_DATA_CACHE) if _DATA_CACHE else []
    ok = bool(dates) and dates[-1] >= today
    reason: str | None = None
    if not ok:
        # 补充诊断信息 (哪只 ETF 缺失/过期)
        spot = ls._fetch_tencent_spot()
        reason = ls.validate_realtime_data(spot)[1] if spot else "无实时数据"
    return {
        "ok": True,
        "realtime_ok": ok,
        "realtime_reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="七星V4 实盘记账网页")
    parser.add_argument("--port", type=int, default=8090, help="监听端口 (默认8090)")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认 127.0.0.1, 仅本地访问)")
    parser.add_argument(
        "--set-password",
        action="store_true",
        help="设置网页访问密码 (从 stdin 或 WEB_PASSWORD 环境变量读取)",
    )
    args = parser.parse_args()

    if args.set_password:
        import os

        pwd = os.environ.get("WEB_PASSWORD", "").strip()
        if not pwd:
            import getpass

            pwd = getpass.getpass("请输入新密码: ").strip()
            if not pwd:
                print("  ❌ 密码不能为空")
                sys.exit(1)
            confirm = getpass.getpass("再次输入确认: ").strip()
            if pwd != confirm:
                print("  ❌ 两次输入不一致")
                sys.exit(1)
        set_password(pwd)
        print("  ✓ 访问密码已设置")
        return

    import uvicorn

    print(f"  🚀 七星V4 记账网页启动: http://{args.host}:{args.port}")
    print(f"     手机浏览器访问 http://<服务器IP>:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
