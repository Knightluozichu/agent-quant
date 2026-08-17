"""audit_risk.py — 风控误杀审计 (只读, 手动/月度运行).

评估 V32 改进版风控的触发质量: 每次降仓类事件 (H1/H2、高波动衰减) 后
5 个交易日内标的走势, 判定是"误杀"还是"正确避损":

  误杀     = 5日内最高反弹 >5% 且 5日最低 ≥ 触发日收盘×0.98 (躲错了)
  正确避损 = 5日内最低 < 触发日收盘×0.95 (明显继续跌, 躲对了)
  中性     = 其余

汇总误杀率, >30% 触发降级建议 (监控版). 组合级事件 (熔断) 不判定.

用法:
  uv run python scripts/audit_risk.py               # 审计 state.json 的 risk_log
  uv run python scripts/audit_risk.py --bark        # 审计 + Bark 月度摘要
  uv run python scripts/audit_risk.py --state PATH  # 指定状态文件 (测试用)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

STATE_FILE = PROJECT_ROOT / "data" / "live" / "state.json"
DATA_DIR = PROJECT_ROOT / "data" / "cross_asset"

# 误杀判定参数 (通用逻辑)
REBOUND_THR = 0.05  # 5日内最高反弹 >5%
NOT_DROP_THR = -0.02  # 5日最低 ≥ -2% (未显著跌破)
DROP_THR = -0.05  # 5日最低 < -5% (明显继续跌)

# 组合级事件 (不判定误杀)
SKIP_TYPES = {"熔断-30%清仓", "熔断-25%告警", "熔断-12%降仓", "目标不可交易"}


def load_risk_log(state_path: Path) -> list[dict]:
    """读取 state.json 的 risk_log (旧 state 无 → 空)."""
    if not state_path.exists():
        print(f"  ⚠️ 状态文件不存在: {state_path}")
        return []
    with open(state_path) as f:
        state = json.load(f)
    log = state.get("risk_log", [])
    print(f"  状态文件: {state_path} | 风控事件 {len(log)} 条")
    return log


def load_price(code: str) -> pd.DataFrame | None:
    f = DATA_DIR / f"{code}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f).sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def judge(df: pd.DataFrame, trigger_date: str) -> dict | None:
    """触发后 5 个交易日的误杀判定."""
    try:
        idx = df.index[df["trade_date"] == pd.Timestamp(trigger_date).date()][0]
    except (IndexError, KeyError):
        return None
    p0 = float(df.iloc[idx]["close"])
    seg = df.iloc[idx + 1 : idx + 6]["close"]
    if len(seg) == 0 or p0 <= 0:
        return None
    h5 = float(seg.max())
    l5 = float(seg.min())
    rebound = h5 / p0 - 1.0
    drop = l5 / p0 - 1.0
    if rebound > REBOUND_THR and drop >= NOT_DROP_THR:
        verdict = "误杀"
    elif drop < DROP_THR:
        verdict = "正确避损"
    else:
        verdict = "中性"
    return {"rebound5": rebound, "drop5": drop, "verdict": verdict}


def audit(state_path: Path) -> dict:
    log = load_risk_log(state_path)
    if not log:
        print("  无风控事件, 无需审计")
        return {"n_events": 0, "n_audited": 0, "n_kill": 0, "kill_rate": 0.0, "rows": []}

    rows = []
    for ev in log:
        etype = ev.get("type", "")
        if etype in SKIP_TYPES:
            continue  # 组合级不判定
        code = ev.get("asset") or ev.get("from")
        if not code:
            continue
        df = load_price(code)
        if df is None:
            continue
        res = judge(df, ev.get("date", ""))
        if res is None:
            continue
        rows.append(
            {
                "date": ev.get("date"),
                "type": etype,
                "asset": code,
                "rebound5": round(res["rebound5"], 4),
                "drop5": round(res["drop5"], 4),
                "verdict": res["verdict"],
            }
        )

    n_kill = sum(1 for r in rows if r["verdict"] == "误杀")
    n_ok = sum(1 for r in rows if r["verdict"] == "正确避损")
    kill_rate = n_kill / len(rows) if rows else 0.0

    hdr = f"\n  {'日期':<12} {'类型':<12} {'标的':<8} {'5日最高反弹':>10} "
    hdr += f"{'5日最低':>9} {'判定':<8}"
    print(hdr)
    print("  " + "-" * 62)
    for r in rows:
        print(
            f"  {r['date']:<12} {r['type']:<12} {r['asset']:<8} "
            f"{r['rebound5']:>+9.1%} {r['drop5']:>+8.1%} {r['verdict']:<8}"
        )

    print("\n" + "=" * 62)
    print(
        f"  审计汇总: 事件 {len(rows)} | 误杀 {n_kill} | 正确避损 {n_ok} | "
        f"中性 {len(rows) - n_kill - n_ok}"
    )
    print(f"  误杀率: {kill_rate:.1%}")
    if kill_rate > 0.30:
        print("  ⚠️ 误杀率 >30% → 建议降级为监控版 (risk_mode=monitor, 仅告警不干预)")
    else:
        print("  ✓ 误杀率 ≤30% → 风控干预质量可接受")
    print("=" * 62)

    return {
        "n_events": len(log),
        "n_audited": len(rows),
        "n_kill": n_kill,
        "n_ok": n_ok,
        "kill_rate": round(kill_rate, 4),
        "rows": rows,
    }


def push_summary(result: dict) -> None:
    """Bark 月度摘要 (Key 缺失时静默跳过)."""
    from notify import get_bark_key, push_bark

    if not get_bark_key():
        print("  ⚠️ Bark Key 未配置, 跳过推送")
        return
    body = (
        f"风控审计: 事件{result['n_events']} 已判{result['n_audited']} "
        f"误杀{result['n_kill']} 正确{result['n_ok']} "
        f"误杀率{result['kill_rate']:.1%}"
    )
    push_bark("📊 七星V3 风控月度审计", body, level="active")
    print("  ✓ Bark 摘要已推送")


def main() -> None:
    parser = argparse.ArgumentParser(description="风控误杀审计 (只读)")
    parser.add_argument("--state", default=str(STATE_FILE), help="状态文件路径")
    parser.add_argument("--bark", action="store_true", help="推送月度摘要")
    args = parser.parse_args()

    print("=" * 62)
    print("  七星V3 风控误杀审计 (V32)")
    print("=" * 62)
    result = audit(Path(args.state))
    if args.bark and result["n_audited"]:
        push_summary(result)


if __name__ == "__main__":
    main()
