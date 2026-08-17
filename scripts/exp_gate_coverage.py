"""门控方案 · 目标场景覆盖率统计: 震荡上行被甩案例中的命中与落地.

回答: "落地无效"是断言还是事实? 精确统计:
  1. 全周期内, 基线持仓因触发暴跌过滤(DROP_FILTER)被甩出的事件总数
  2. 被甩品种后5/10/20日收益 → 误杀率 (正收益=震荡上行被甩=假摔)
  3. 门控条件(ret60<0 且 动量>0) 在误杀案例中的命中率
  4. 门控命中且该品种成为候选第一 (落地) 的比例 → 量化落地瓶颈
  5. 对照: 真跌被甩案例中门控的命中率 (若门控在真跌也放行=有害)

口径: 基线持仓轨迹取自 run_v3_risk_h3(基线版), 调仓日网格与回测一致;
      候选第一 = 当日生产动量评分最高且>0 (简化, 未含A股走弱等过滤).
用法: uv run python scripts/exp_gate_coverage.py
输出: data/v9_results/gate_coverage.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import exp_drop_gate_h3 as h3
import run_qixing_v3 as rq

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"
WARMUP = 130
HORIZONS = (5, 10, 20)


def mom_score(close: np.ndarray) -> float:
    r10 = (close[-1] - close[-11]) / close[-11] if len(close) > 11 else 0.0
    r20 = (close[-1] - close[-21]) / close[-21] if len(close) > 21 else 0.0
    return 0.5 * r10 + 0.5 * r20


def gate_hit(data: dict, code: str, idx: int) -> tuple[bool, float | None]:
    """门控条件命中? (近5日暴跌 且 ret60<0 且 动量>0). 返回 (命中, ret60)."""
    close = data[code]["close"].values[: idx + 1].astype(float)
    if len(close) <= 61:
        return False, None
    drop = any(
        (close[i] - close[i - 1]) / close[i - 1] < rq.DROP_THRESHOLD
        for i in range(-rq.DROP_LOOKBACK, 0)
    )
    if not drop:
        return False, None
    ret60 = (close[-1] - close[-61]) / close[-61]
    return (ret60 < 0.0 and mom_score(close) > 0.0), float(ret60)


def rank_first(data: dict, code: str, idx: int) -> tuple[bool, int, float]:
    """该品种当日动量评分排名 (含>0过滤). 返回 (是否第一, 排名, 评分)."""
    scores = []
    for c in rq.ETF_POOL:
        if c not in data:
            continue
        close = data[c]["close"].values[: idx + 1].astype(float)
        if len(close) < 121:
            continue
        s = rq.calc_momentum_score(close)
        if s > 0:
            scores.append((c, s))
    scores.sort(key=lambda x: -x[1])
    for rank, (c, s) in enumerate(scores, 1):
        if c == code:
            return rank == 1, rank, s
    return False, len(scores) + 1, 0.0


def main() -> None:
    print("=" * 78)
    print("  门控方案覆盖率统计 | 震荡上行被甩案例中的命中与落地")
    print("=" * 78)

    data = rq.load_data()
    # 基线轨迹: 用决策日志记录每个调仓日的 (date, holding=换仓前, target=换仓后)
    log: list[tuple] = []
    ORIG_SELECT = rq.select_target

    def wrapped(data_, etf_data_at_date, holding):
        t, c, s, a = ORIG_SELECT(data_, etf_data_at_date, holding)
        td = None
        for code in rq.ETF_POOL:
            if code in etf_data_at_date:
                td = data_[code].iloc[etf_data_at_date[code]]["trade_date"]
                break
        log.append((td, holding, t))
        return t, c, s, a

    rq.check_single_day_drop = h3.ORIG_CHECK
    rq.select_target = wrapped
    h3.select_target = wrapped
    h3.H3_ENABLED = False
    h3.run_v3_risk_h3(data)  # 返回值不用, 副作用(wrapped 统计 GATE_STATS)生效
    rq.check_single_day_drop = h3.ORIG_CHECK
    rq.select_target = ORIG_SELECT
    h3.select_target = ORIG_SELECT
    h3.H3_ENABLED = False

    # 公共日历 (用于后向收益)
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    ci = {d: i for i, d in enumerate(dates)}

    events = []
    for td, holding, target in log:
        # 被甩事件: 换仓 (target != holding) 且 旧持仓触发暴跌过滤
        if target == holding or not holding or holding == rq.DEFENSE:
            continue
        mask = data[holding]["trade_date"] <= td
        if mask.sum() < WARMUP:
            continue
        idx = mask.sum() - 1
        close = data[holding]["close"].values[: idx + 1].astype(float)
        drop = any(
            (close[i] - close[i - 1]) / close[i - 1] < rq.DROP_THRESHOLD
            for i in range(-rq.DROP_LOOKBACK, 0)
        )
        if not drop:
            continue
        # 后向收益 (被甩品种若继续持有)
        pos = ci.get(td)
        fwd = {}
        for hz in HORIZONS:
            if pos is not None and pos + hz < len(dates):
                fut = dates[pos + hz]
                frow = data[holding][data[holding]["trade_date"] == fut]
                if not frow.empty:
                    fwd[hz] = float(frow.iloc[0]["close"]) / close[-1] - 1.0
        hit, ret60 = gate_hit(data, holding, idx)
        first, rank, sc = rank_first(data, holding, idx)
        events.append(
            {
                "date": str(td),
                "code": holding,
                "name": rq.ETF_POOL[holding],
                "to": target,
                "ret60": round(ret60, 4) if ret60 is not None else None,
                "gate_hit": hit,
                "rank": rank,
                "score": round(float(sc), 4),
                "fwd5": fwd.get(5),
                "fwd10": fwd.get(10),
                "fwd20": fwd.get(20),
            }
        )

    total = len(events)
    print(f"\n  被甩事件总数 (基线持仓触发暴跌过滤): {total}")

    for hz in HORIZONS:
        v = [e for e in events if e[f"fwd{hz}"] is not None]
        n_fake = sum(1 for e in v if e[f"fwd{hz}"] > 0)
        avg = float(np.mean([e[f"fwd{hz}"] for e in v])) if v else 0
        print(
            f"  [{hz}日] 误杀(反弹) {n_fake}/{len(v)} = {n_fake / len(v) * 100:.0f}% "
            f"| 平均 {avg:+.2%}"
        )

    # 门控命中率 (分误杀/真跌)
    for hz in (5, 10, 20):
        v = [e for e in events if e[f"fwd{hz}"] is not None]
        fake = [e for e in v if e[f"fwd{hz}"] > 0]
        real = [e for e in v if e[f"fwd{hz}"] <= 0]
        fake_hit = sum(1 for e in fake if e["gate_hit"])
        real_hit = sum(1 for e in real if e["gate_hit"])
        print(f"\n  [门控命中率 {hz}日口径]")
        print(
            f"    误杀组(震荡被甩): {fake_hit}/{len(fake)} = "
            f"{fake_hit / len(fake) * 100:.0f}% 被门控覆盖"
        )
        print(
            f"    真跌组(甩对了):  {real_hit}/{len(real)} = "
            f"{real_hit / len(real) * 100:.0f}% 门控也放行(有害)"
        )
        # 落地
        fake_first = sum(1 for e in fake if e["gate_hit"] and e["rank"] == 1)
        print(
            f"    误杀组中 门控命中+候选第一(可落地): {fake_first}/{len(fake)} "
            f"= {fake_first / len(fake) * 100:.0f}%"
        )
        # 命中但未落地明细
        hit_not_first = [e for e in fake if e["gate_hit"] and e["rank"] != 1]
        if hit_not_first:
            ranks = [e["rank"] for e in hit_not_first]
            print(
                f"    命中但未落地 {len(hit_not_first)} 次, 排名分布: "
                f"{dict(sorted({r: ranks.count(r) for r in set(ranks)}.items()))}"
            )

    # 明细
    print("\n  【被甩事件明细 (后5日)】")
    for e in sorted(events, key=lambda x: -(x["fwd5"] or -9)):
        print(
            f"    {e['date']} {e['name']:<6} ret60={e['ret60'] if e['ret60'] is not None else 0:+.1%} "
            f"门控{'✅' if e['gate_hit'] else '❌'} 排名{e['rank']} "
            f"后5日{e['fwd5'] if e['fwd5'] is not None else 0:+.1%} "
            f"后20日{e['fwd20'] if e['fwd20'] is not None else 0:+.1%}"
        )

    out = {
        "n_events": total,
        "events": events,
        "note": "口径: 基线持仓触发暴跌过滤; 门控=ret60<0且动量>0; 候选第一=动量评分最高且>0",
    }
    path = OUTPUT_DIR / "gate_coverage.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {path}")


if __name__ == "__main__":
    main()
