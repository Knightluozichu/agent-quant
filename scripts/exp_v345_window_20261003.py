"""分析: V3-G / V4 / V5 样本外窗口 (2026-07-01 → 2026-09-30) 对比 — 过拟合检验.

背景: 三策略参数均在 ~2026-08 前的历史上确定 (V3-G/V4 网格搜索, V5 的 RSRS
参数为光大原版未在本数据拟合)。本窗口 7/1-8/11 对 V3-G/V4 部分在样本内,
8/11-9/30 为真样本外; 对 V5 全窗口近似样本外。窗口仅 3 个月, 样本小,
任何排序都不构成方法优劣证据 (AGENTS.md 纪律)。

口径: same-close (14:50 当日成交近似), 生产池 (删 501018), 10 万本金,
2026-07-01 全新现金起步, 调仓网格锚定全序列 (与生产对齐), 单边 0.0015 (1x)。
数据: 服务器 2026-09-30 快照 (国庆休市, 9/30 即最新交易日)。

执行: uv run python scripts/exp_v345_window_20261003.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_signal as live
import qixing_v4 as v4
import run_qixing_v3 as rq
from exp_v3g_full_pool_fast_slow import run_full_pool_strategy
from exp_v5_compare import RsrsTargetFilter
from qixing_v4 import FullPoolParams
from review_oil_proxy_20260916 import prepare_bars

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "artifacts" / "entry_compare_20261002" / "inputs"
RESULT = ROOT / "data" / "qixing_results" / "v345_window_20261003.json"

START = date(2026, 7, 1)
END = date(2026, 9, 30)
INITIAL = 100_000.0

# 全历史 1x 参照 (2026-10-03 V5 对比实验归档值, 2020-06-19 → 2026-09-15)
FULL_HISTORY_REF = {
    "V3-G": {"cagr": 0.8177, "max_drawdown": -0.2253, "sharpe": 2.379},
    "V4": {"cagr": 0.9066, "max_drawdown": -0.2862, "sharpe": 2.571},
    "V5": {"cagr": 0.7061, "max_drawdown": -0.1896, "sharpe": 2.263},
}
# 真实账户参照: 7/28 入市 10 万 → 9/16 对账 97,684.45, 此后全现金
REAL_ACCOUNT = {"entry": "2026-07-28", "equity": 97_684.45, "return": -0.0232}


def window_metrics(curve: pd.DataFrame) -> dict[str, Any]:
    eq = curve["equity"].astype(float)
    days = max((pd.Timestamp(curve["trade_date"].iloc[-1])
                - pd.Timestamp(curve["trade_date"].iloc[0])).days, 1)
    ret = eq.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if len(ret) > 1 and ret.std() > 0 else 0.0
    # 逐月收益
    df = curve.copy()
    df["month"] = pd.to_datetime(df["trade_date"]).dt.to_period("M")
    month_end = df.groupby("month")["equity"].last()
    monthly, prev = {}, INITIAL
    for m, e in month_end.items():
        monthly[str(m)] = round(float(e) / prev - 1, 4)
        prev = float(e)
    return {
        "final_value": round(float(eq.iloc[-1]), 2),
        "total_return": round(float(eq.iloc[-1]) / INITIAL - 1, 4),
        "window_cagr_annualized": round((float(eq.iloc[-1]) / INITIAL) ** (365.25 / days) - 1, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
        "monthly": monthly,
    }


def main() -> None:
    codes = [*live.ETF_POOL, live.DEFENSE]
    data = {c: prepare_bars(pd.read_parquet(SNAPSHOT / f"{c}.parquet"), c, END) for c in codes}
    prod_pool = {c: n for c, n in rq.ETF_POOL.items() if c != "501018"}

    strategies: dict[str, tuple[Any, Any]] = {
        "V3-G": (FullPoolParams.disabled(), None),
        "V4": (v4.V4_PARAMS, None),
        "V5": (v4.V4_PARAMS, RsrsTargetFilter.from_data(data, list(prod_pool), rq.DEFENSE)),
    }
    report: dict[str, Any] = {
        "analysis": "V3-G/V4/V5 样本外窗口对比 (过拟合检验)",
        "window": [str(START), str(END)],
        "convention": "same-close 14:50 当日成交近似, 生产池, 10万, 1x 成本",
        "sample_note": "7/1-8/11 对 V3-G/V4 部分在样本内, 之后真样本外; V5 参数未在本数据拟合",
        "full_history_ref": FULL_HISTORY_REF,
        "real_account_ref": REAL_ACCOUNT,
        "results": {},
    }
    with patch.object(rq, "ETF_POOL", prod_pool):
        for name, (params, tfilter) in strategies.items():
            res = run_full_pool_strategy(data, params, target_filter=tfilter, start=START)
            m = window_metrics(res["equity_curve"])
            m["trade_legs"] = len([t for t in res["trades"] if t["action"] == "sell"])
            m["trades"] = [
                {k: t[k] for k in ("date", "action", "code", "price")} for t in res["trades"]
            ]
            if name == "V5" and tfilter is not None:
                m["rsrs_blocks"] = tfilter.blocks
            report["results"][name] = m
            print(f"[{name}] {json.dumps({k: v for k, v in m.items() if k != 'trades'}, ensure_ascii=False)}")

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"已归档 {RESULT}")


if __name__ == "__main__":
    main()
