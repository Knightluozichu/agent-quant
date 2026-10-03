"""V4 vs 缠论: 用户入市窗口 (2026-07-28 → 2026-09-30) 同口径对比, 10 万本金.

口径声明:
  - 同一输入: 服务器 2026-09-30 行情快照 (artifacts/entry_compare_20261002/inputs);
  - 同一执行: T 收盘信号 → 下一公共交易日开盘, 单边成本 FEE+SLIPPAGE (1x);
  - V4: 生产当前池 (排除 501018), 沿用 review_rotation_20260916.replay_next_open,
    当前规则追溯回放, 不等于当时各日实际部署版本的实绩;
  - 缠论: exp_structure_trade 引擎, 全历史构建几何, 只取 entry >= 窗口起点的交易,
    期末未平按 9/30 收盘强平 (与 V4 期末按收盘估值同口径);
  - 真实账户: 服务器账本 97,684.45 (-2.32%), 为实际成交, 与模拟口径不同, 仅作参照;
  - 窗口仅 ~2 个月, 样本极小, 任何排序都不是方法优劣证据 (AGENTS.md 纪律).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # review_* 以 scripts 包方式互导
import run_qixing_v3 as rq  # noqa: E402
import live_signal as live  # noqa: E402
import exp_structure_trade as est  # noqa: E402
from review_oil_proxy_20260916 import prepare_bars  # noqa: E402
from review_rotation_20260916 import replay_next_open  # noqa: E402
from review_since_entry_20260916 import summarize_trades  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "entry_compare_20261002"
START, END = date(2026, 7, 28), date(2026, 9, 30)
INITIAL = 100_000.0
REAL_ACCOUNT_EQUITY = 97_684.45  # 服务器账本 2026-09-14 对账后现金, 全现金持币


def main() -> None:
    codes = [*live.ETF_POOL, live.DEFENSE]  # 生产当前池 (501018 已排除)
    paths = {c: OUT / "inputs" / f"{c}.parquet" for c in codes}
    data = {c: prepare_bars(pd.read_parquet(p), c, END) for c, p in paths.items()}
    dates = sorted(set.intersection(*(set(d.trade_date) for d in data.values())))
    assert START in dates and END in dates, "窗口端点不在公共日历"
    grid = set(dates[130 :: rq.REBALANCE_DAYS])
    last_prices = {c: float(d.close.iloc[-1]) for c, d in data.items()}
    cost = rq.FEE + rq.SLIPPAGE

    # ---- V4 生产池回放 ----
    replay = replay_next_open(data, start=START, end=END, grid=grid, calendar=dates, cost_multiplier=1)
    v4_acc = summarize_trades(replay["trades"], last_prices, cost=cost)

    # ---- 缠论窗口回放 ----
    chan_map: dict[str, list[dict]] = {}
    for c in codes:
        if c == live.DEFENSE:
            continue
        trades = est.run_chan_on_series(data[c], cost=cost)
        chan_map[c] = [t for t in trades if t["entry_date"] >= str(START)]
    curve, chan_taken = est.simulate_account(chan_map, INITIAL)
    chan_equity = curve[-1]["equity"] if curve else INITIAL

    report = {
        "window": {"start": str(START), "end": str(END), "initial": INITIAL},
        "pool": {c: live.ETF_POOL[c] for c in codes if c != live.DEFENSE},
        "input_sha256": {c: hashlib.sha256(p.read_bytes()).hexdigest() for c, p in paths.items()},
        "v4": {
            "ending_equity": round(v4_acc["ending_equity"], 2),
            "total_pnl": round(v4_acc["total_pnl"], 2),
            "trades": len(replay["trades"]),
            "legs": [
                {k: t[k] for k in ("date", "action", "code", "price")}
                for t in replay["trades"]
            ],
            "final_holding": v4_acc["holding"],
        },
        "chan": {
            "ending_equity": round(chan_equity, 2),
            "total_pnl": round(chan_equity - INITIAL, 2),
            "trades": len(chan_taken),
            "legs": [
                {
                    "entry": t["entry_date"],
                    "exit": t["exit_date"],
                    "code": t["code"],
                    "buy_type": t["buy_type"],
                    "ret": round(t["ret"], 4),
                    "exit_reason": t["exit_reason"],
                }
                for t in chan_taken
            ],
        },
        "real_account_reference": {
            "equity": REAL_ACCOUNT_EQUITY,
            "total_pnl": round(REAL_ACCOUNT_EQUITY - INITIAL, 2),
            "note": "实际成交口径, 与模拟不同, 仅参照",
        },
        "caveats": [
            "窗口约 2 个月, 样本极小, 排序不构成方法优劣证据",
            "V4 为当前规则追溯回放, 不等于当时部署实绩",
            "缠论为日线笔级别代理, 一买精度受区间套限制",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("v4", "chan", "real_account_reference")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
