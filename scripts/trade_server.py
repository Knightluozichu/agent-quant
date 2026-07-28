"""七星V3 实盘记账网页 (FastAPI 后端).

手机端使用流程:
  1. 每个交易日 14:50 服务器自动生成信号并 Bark 推送 (信号挂起为"待确认")
  2. 你在易淘金 App 手动下单
  3. 打开本网页, 填入真实成交价/数量, 点"确认成交" → 持仓状态更新为现实

核心原则: 持仓状态以网页确认的真实成交为准, 系统不自动假设成交.

启动:
  uv run python scripts/trade_server.py --port 8090
首次设置访问密码:
  uv run python scripts/trade_server.py --set-password <你的密码>
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

import live_signal as ls  # noqa: E402
import run_qixing_v3 as rq  # noqa: E402
from notify import load_config, save_config  # noqa: E402

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="七星V3 实盘记账")

# 行情数据内存缓存 (带 mtime 失效: cron 14:50 更新数据后自动重载)
_DATA_CACHE: dict | None = None
_DATA_CACHE_TIME: float = 0.0


def get_data() -> dict:
    global _DATA_CACHE, _DATA_CACHE_TIME
    mtime = max(
        (f.stat().st_mtime for f in ls.DATA_DIR.glob("*.parquet")), default=0.0
    )
    if _DATA_CACHE is None or mtime > _DATA_CACHE_TIME:
        _DATA_CACHE = ls.load_data()
        _DATA_CACHE_TIME = time.time()
    return _DATA_CACHE


def refresh_data() -> None:
    global _DATA_CACHE
    _DATA_CACHE = ls.load_data()


# --------------------------------------------------------------------------- #
# 鉴权: 密码哈希 + token
# --------------------------------------------------------------------------- #
def _hash_password(pwd: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pwd.encode(), salt.encode(), 100_000).hex()


def set_password(pwd: str) -> None:
    cfg = load_config()
    salt = secrets.token_hex(8)
    cfg["web_salt"] = salt
    cfg["web_password"] = _hash_password(pwd, salt)
    cfg.setdefault("web_tokens", [])
    save_config(cfg)


def verify_password(pwd: str) -> bool:
    cfg = load_config()
    salt = cfg.get("web_salt")
    expected = cfg.get("web_password")
    if not salt or not expected:
        return False
    return hmac.compare_digest(_hash_password(pwd, salt), expected)


def require_token(request: Request) -> None:
    cfg = load_config()
    tokens = cfg.get("web_tokens", [])
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = request.query_params.get("token", "")
    if token not in tokens:
        raise HTTPException(status_code=401, detail="未授权, 请先登录")


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
def login(payload: dict) -> dict:
    if not load_config().get("web_password"):
        raise HTTPException(
            status_code=503,
            detail="尚未设置访问密码, 请在服务器运行: "
            "uv run python scripts/trade_server.py --set-password <密码>",
        )
    if not verify_password(payload.get("password", "")):
        raise HTTPException(status_code=401, detail="密码错误")
    token = secrets.token_hex(16)
    cfg = load_config()
    cfg.setdefault("web_tokens", []).append(token)
    save_config(cfg)
    return {"token": token}


# --------------------------------------------------------------------------- #
# 数据接口
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def api_status(_: None = Depends(require_token)) -> dict:
    state = ls.load_state()
    if state is None:
        return {"initialized": False}
    data = get_data()
    td = ls.get_trading_dates(data)[-1]
    holding = state["holding"]
    holding_info = None
    if holding:
        entry_date = state.get("entry_date")
        # 数据日期 < 买入日 → 数据滑后(现价是买入前的价), 用成本价代替, 盈亏待更新
        price_stale = bool(entry_date and str(td) < str(entry_date))
        p_raw = ls.price_on(data, holding, td)
        p = state["entry_price"] if price_stale else p_raw
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
        "cash": round(state["cash"], 2),
        "holding": holding_info,
        "total": round(total, 2),
        "initial_capital": state["initial_capital"],
        "return_pct": round(ret, 2),
        "pending_order": state.get("pending_order"),
    }


@app.get("/api/signal")
def api_signal(_: None = Depends(require_token)) -> dict:
    state = ls.load_state()
    data = get_data()
    td = ls.get_trading_dates(data)[-1]
    holding = state["holding"] if state else None
    idx_map = ls.build_etf_data_at_date(data, td)
    target, candidates, _best, a_share_weak = ls.select_target(data, idx_map, holding)
    board = ls.momentum_board_data(data, td, holding, target)
    return {
        "trade_date": str(td),
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
    """返回七星V3完整池回测结果 (供网页绘制策略全景图)."""
    global _BACKTEST_CACHE, _BACKTEST_TIME
    mtime = max(
        (f.stat().st_mtime for f in ls.DATA_DIR.glob("*.parquet")), default=0.0
    )
    if _BACKTEST_CACHE is None or mtime > _BACKTEST_TIME:
        data = rq.load_data()
        result = rq.run_qixing_v3(data)
        eq = result["equity_curve"]
        cummax = eq["equity"].cummax()
        dd = ((eq["equity"] - cummax) / cummax * 100)
        span = (eq["trade_date"].iloc[-1] - eq["trade_date"].iloc[0]).days / 365.25
        _BACKTEST_CACHE = {
            "metrics": {
                "total_return": round(result["total_return"] * 100, 1),
                "ann_return": round(result["ann_return"] * 100, 1),
                "sharpe": round(result["sharpe"], 2),
                "max_drawdown": round(result["max_drawdown"] * 100, 1),
                "n_trades": result["n_trades"],
                "start": str(eq["trade_date"].iloc[0])[:7],
                "end": str(eq["trade_date"].iloc[-1])[:7],
                "years": round(span, 1),
            },
            "dates": eq["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
            "values": [round(float(v), 0) for v in eq["equity"]],
            "drawdown": [round(float(v), 2) for v in dd],
            "holdings": eq["holding"].tolist(),
            "yearly": [
                {"year": y, "return": round(v["return"] * 100, 1),
                 "max_dd": round(v["max_dd"] * 100, 1)}
                for y, v in sorted(result["yearly"].items())
            ],
            "etf_names": {**rq.ETF_POOL, rq.DEFENSE: "货币基金"},
        }
        _BACKTEST_TIME = time.time()
    return _BACKTEST_CACHE


# --------------------------------------------------------------------------- #
# 写接口 (记账)
# --------------------------------------------------------------------------- #
@app.post("/api/confirm")
def api_confirm(payload: dict, _: None = Depends(require_token)) -> dict:
    try:
        state = ls.confirm_order(payload.get("sell"), payload.get("buy"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, "cash": round(state["cash"], 2), "holding": state["holding"]}


@app.post("/api/skip")
def api_skip(_: None = Depends(require_token)) -> dict:
    ls.skip_pending()
    return {"ok": True}


@app.post("/api/trade")
def api_trade(payload: dict, _: None = Depends(require_token)) -> dict:
    try:
        state = ls.record_manual_trade(
            payload["action"], payload["code"], int(payload["shares"]),
            float(payload["price"]), payload.get("date"),
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, "cash": round(state["cash"], 2), "holding": state["holding"]}


@app.post("/api/refresh")
def api_refresh(_: None = Depends(require_token)) -> dict:
    refresh_data()
    return {"ok": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="七星V3 实盘记账网页")
    parser.add_argument("--port", type=int, default=8090, help="监听端口 (默认8090)")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--set-password", metavar="PWD", help="设置网页访问密码")
    args = parser.parse_args()

    if args.set_password:
        set_password(args.set_password)
        print("  ✓ 访问密码已设置")
        return

    import uvicorn

    print(f"  🚀 七星V3 记账网页启动: http://{args.host}:{args.port}")
    print("     手机浏览器访问 http://<服务器IP>:%d" % args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
