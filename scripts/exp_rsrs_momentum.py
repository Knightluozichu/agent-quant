"""实验: ETF动量轮动+RSRS择时 全量回测 — 预注册研究 (H3), 不触达生产.

预注册假设 (2026-10-03, 改参数即作废重注册, 全文见 tasks/EXPERIMENTS.md):
  H3: 周频 (5 交易日) 调仓、T 收盘信号 T+1 开盘成交、1x 单边成本 0.0015 下,
      "MOM20 动量轮动选标的 + RSRS(N=18, M=600, ±0.7, 滞回) 决定风险开/关",
      在服务器 2026-09-30 快照全历史上扣费后期望为正。
  证伪标准: 1x 成本利润因子 PF ≤ 1 或样本 < 30 笔; 2x/3x 成本下不得路径悬崖。

方法出处 (散户量化圈通行版本):
  - 动量轮动: 全池 20 日涨幅最大者, 动量 ≤0 退防御 (货币 ETF);
  - RSRS 择时 (光大证券 2017 原版): 过去 N=18 日 high~low OLS 斜率 beta,
    再对 beta 序列做 M=600 日滚动 z 标准化; z>+0.7 允许持有风险资产,
    z<-0.7 强制退防御, 中间区滞回沿用前一状态 (初始允许持有);
  - RSRS 原版用于指数择时, 此处应用于各 ETF 自身高低价序列 (自择时), 已披露。

因果保证:
  - 信号日 T 的一切输入 (动量/beta/z) 只用 ≤T 收盘数据;
  - 成交恒在下一公共交易日开盘, 信号日零成交;
  - beta/z 的滚动窗含 T 当日, 截断验证见 tests/unit/test_rsrs_momentum.py。

执行: uv run python scripts/exp_rsrs_momentum.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_signal as live
import run_qixing_v3 as rq
from review_oil_proxy_20260916 import prepare_bars
from review_rotation_20260916 import replay_next_open
from review_since_entry_20260916 import summarize_trades

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "artifacts" / "entry_compare_20261002" / "inputs"
OUT_DIR = ROOT / "data" / "qixing_results"
RESULT = OUT_DIR / "rsrs_momentum_20261003.json"

COST_SIDE = rq.FEE + rq.SLIPPAGE  # 0.0015 单边
WINDOW_START = pd.Timestamp("2026-07-28").date()
WINDOW_END = pd.Timestamp("2026-09-30").date()
INITIAL = 100_000.0

# 预注册固定参数
MOM_N = 20
RSRS_N = 18
RSRS_M = 600
RSRS_THR = 0.7
REBALANCE_DAYS = 5


# --------------------------------------------------------------------------- #
# 指标原语 (纯函数, 可测)
# --------------------------------------------------------------------------- #
def ols_slope(highs: np.ndarray, lows: np.ndarray) -> float:
    """high ~ low 一元 OLS 斜率."""
    x = np.asarray(lows, dtype=float)
    y = np.asarray(highs, dtype=float)
    x_c = x - x.mean()
    denom = float(x_c @ x_c)
    if denom == 0.0:
        return 0.0
    return float((x_c @ (y - y.mean())) / denom)


def rsrs_beta_series(high: np.ndarray, low: np.ndarray, n: int = RSRS_N) -> pd.Series:
    """滚动 N 日 OLS 斜率; 第 i 个值只用 [i-n+1, i] 的数据."""
    betas = [np.nan] * (n - 1)
    for i in range(n - 1, len(high)):
        betas.append(ols_slope(high[i - n + 1 : i + 1], low[i - n + 1 : i + 1]))
    return pd.Series(betas, dtype=float)


def rsrs_z(beta: pd.Series, m: int = RSRS_M) -> pd.Series:
    """滚动 M 日 z 标准化; 第 i 个 z 只用 ≤i 的 beta."""
    mean = beta.rolling(m, min_periods=m).mean()
    std = beta.rolling(m, min_periods=m).std(ddof=1)
    return (beta - mean) / std


def momentum(close: pd.Series, n: int = MOM_N) -> float:
    """T 日 N 日动量; 数据不足返回 NaN."""
    if len(close) <= n:
        return float("nan")
    past = float(close.iloc[-n - 1])
    if past <= 0:
        return float("nan")
    return float(close.iloc[-1]) / past - 1.0


def pick_target(mom: dict[str, float]) -> str | None:
    """动量最大且 >0 者; 全 ≤0 或全缺失 → None (防御)."""
    valid = {c: v for c, v in mom.items() if np.isfinite(v)}
    if not valid:
        return None
    best = max(valid, key=lambda c: valid[c])
    return best if valid[best] > 0 else None


def decide_risk_state(z: float, prev: bool, thr: float = RSRS_THR) -> bool:
    """RSRS 滞回: z>thr → True; z<-thr → False; 中间/缺失 → 沿用 prev."""
    if not np.isfinite(z):
        return prev
    if z > thr:
        return True
    if z < -thr:
        return False
    return prev


# --------------------------------------------------------------------------- #
# 回测引擎
# --------------------------------------------------------------------------- #
def run_backtest(
    data: dict[str, pd.DataFrame],
    *,
    pool: list[str],
    defense: str,
    cost: float = COST_SIDE,
    rebalance_days: int = REBALANCE_DAYS,
    mom_n: int = MOM_N,
    rsrs_n: int = RSRS_N,
    rsrs_m: int = RSRS_M,
    initial: float = INITIAL,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, Any]:
    """周频网格: T 收盘信号 → T+1 开盘成交; 每日收盘估值."""
    codes = [*pool, defense]
    dates = sorted(set.intersection(*(set(d.trade_date) for d in (data[c] for c in codes))))
    if end is not None:
        dates = [d for d in dates if d <= end]
    maps = {c: dict(zip(data[c].trade_date, range(len(data[c])), strict=True)) for c in codes}
    grid = set(dates[::rebalance_days])

    # 预计算全历史 beta/z (只用各自 ≤T 数据, 逐日取用不引入未来)
    zs: dict[str, pd.Series] = {}
    for c in pool:
        beta = rsrs_beta_series(data[c].high.to_numpy(), data[c].low.to_numpy(), n=rsrs_n)
        zs[c] = rsrs_z(beta, m=rsrs_m)

    cash, holding, shares = initial, None, 0.0
    risk_on = True  # 滞回初始: 允许持有
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    pending: tuple[str | None, object] | None = None  # (target_code 或 None=防御, signal_date)

    for td in dates:
        # 1) 开盘成交昨日信号
        if pending is not None and (start is None or td > start):
            target, sig_date = pending
            pending = None
            if holding is not None:
                price = float(data[holding].open.iloc[maps[holding][td]])
                cash += shares * price * (1 - cost)
                trades.append({"date": td, "signal_date": sig_date, "action": "SELL",
                               "code": holding, "price": round(price, 4)})
                holding, shares = None, 0.0
            if target is not None:
                price = float(data[target].open.iloc[maps[target][td]])
                shares = cash * (1 - cost) / price
                cash = 0.0
                holding = target
                trades.append({"date": td, "signal_date": sig_date, "action": "BUY",
                               "code": target, "price": round(price, 4)})
        elif pending is not None:
            pending = None  # start 当日无此前信号, 丢弃

        # 2) 收盘估值
        if holding is not None:
            equity = shares * float(data[holding].close.iloc[maps[holding][td]])
        else:
            equity = cash
        if start is None or td >= start:
            curve.append({"date": td, "equity": round(equity, 2)})

        # 3) 网格日生成信号 (只用 ≤T 收盘)
        if td in grid and (start is None or td >= start):
            mom = {}
            for c in pool:
                idx = maps[c][td]
                mom[c] = momentum(data[c].close.iloc[: idx + 1], n=mom_n)
            candidate = pick_target(mom)
            if candidate is None:
                target = None
            else:
                risk_on = decide_risk_state(float(zs[candidate].iloc[maps[candidate][td]]), risk_on)
                target = candidate if risk_on else None
            if target != holding:
                pending = (target, td)

    metrics = _metrics(curve, trades, initial, cost)
    return {"trades": trades, "curve": curve, **metrics}


def _metrics(curve: list[dict[str, Any]], trades: list[dict[str, Any]],
             initial: float, cost: float = COST_SIDE) -> dict[str, Any]:
    eq = pd.Series([p["equity"] for p in curve], dtype=float)
    if eq.empty:
        return {"ending_equity": initial, "cagr": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "profit_factor": 0.0, "legs": 0}
    days = max((curve[-1]["date"] - curve[0]["date"]).days, 1)
    cagr = (float(eq.iloc[-1]) / initial) ** (365.25 / days) - 1
    ret = eq.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if len(ret) > 1 and ret.std() > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min())
    # 利润因子: 逐腿往返 (SELL 价净 / 对应 BUY 价净)
    wins = losses = 0.0
    open_lot: dict[str, float] = {}
    legs = 0
    for t in trades:
        if t["action"] == "BUY":
            open_lot[t["code"]] = t["price"] * (1 + cost)
        elif t["code"] in open_lot:
            pnl = t["price"] * (1 - cost) - open_lot.pop(t["code"])
            legs += 1
            if pnl > 0:
                wins += pnl
            else:
                losses -= pnl
    pf = wins / losses if losses > 0 else (float("inf") if wins > 0 else 0.0)
    return {"ending_equity": round(float(eq.iloc[-1]), 2), "cagr": round(cagr, 4),
            "sharpe": round(sharpe, 3), "max_drawdown": round(mdd, 4),
            "profit_factor": round(pf, 3), "legs": legs}


# --------------------------------------------------------------------------- #
# 主流程: 全量 1x/2x/3x + 入市窗口 V4 对比
# --------------------------------------------------------------------------- #
def main() -> None:
    codes = [*live.ETF_POOL, live.DEFENSE]
    paths = {c: SNAPSHOT / f"{c}.parquet" for c in codes}
    data = {c: prepare_bars(pd.read_parquet(p), c, WINDOW_END) for c, p in paths.items()}
    pool = [c for c in live.ETF_POOL]
    input_sha = {c: hashlib.sha256(p.read_bytes()).hexdigest() for c, p in paths.items()}

    full: dict[str, dict[str, Any]] = {}
    for mult in (1, 2, 3):
        res = run_backtest(data, pool=pool, defense=live.DEFENSE, cost=COST_SIDE * mult)
        full[f"{mult}x"] = {k: res[k] for k in
                            ("ending_equity", "cagr", "sharpe", "max_drawdown", "profit_factor", "legs")}
        full[f"{mult}x"]["trades"] = len(res["trades"])
        print(f"[{mult}x] {json.dumps(full[f'{mult}x'], ensure_ascii=False)}")

    # 入市窗口同口径: RSRS 策略 + V4 回放 (复用 2026-10-02 对比口径)
    window = run_backtest(data, pool=pool, defense=live.DEFENSE,
                          start=WINDOW_START, end=WINDOW_END)
    dates = sorted(set.intersection(*(set(d.trade_date) for d in data.values())))
    grid = set(dates[130 :: rq.REBALANCE_DAYS])
    replay = replay_next_open(data, start=WINDOW_START, end=WINDOW_END,
                              grid=grid, calendar=dates, cost_multiplier=1)
    last_prices = {c: float(d.close.iloc[-1]) for c, d in data.items()}
    v4_acc = summarize_trades(replay["trades"], last_prices, cost=COST_SIDE)

    report = {
        "experiment": "H3: ETF动量轮动+RSRS择时 (预注册 2026-10-03)",
        "params": {"mom_n": MOM_N, "rsrs_n": RSRS_N, "rsrs_m": RSRS_M,
                   "rsrs_thr": RSRS_THR, "rebalance_days": REBALANCE_DAYS,
                   "cost_side": COST_SIDE},
        "pool": {c: live.ETF_POOL[c] for c in pool},
        "input_sha256": input_sha,
        "full_history": {
            "span": [str(d) for d in (min(dates), max(dates))],
            **full,
        },
        "entry_window": {
            "window": [str(WINDOW_START), str(WINDOW_END)],
            "rsrs_momentum": {k: window[k] for k in
                              ("ending_equity", "cagr", "max_drawdown", "profit_factor", "legs")},
            "v4_reference": {"ending_equity": round(v4_acc["ending_equity"], 2),
                             "trades": len(replay["trades"])},
        },
        "falsification": "1x PF ≤ 1 或 legs < 30 → 证伪; 2x/3x 路径悬崖 → 冻结",
        "caveats": [
            "RSRS 此处为 ETF 自择时, 非原版指数择时",
            "窗口对比仅参照, 不构成方法优劣证据",
            "快照与 V4 对比实验同输入, 哈希见 input_sha256",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"已归档 {RESULT}")


if __name__ == "__main__":
    main()
