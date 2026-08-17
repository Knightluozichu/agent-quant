"""V3.1 研究成果归档: 汇总全部实验结果到 data/qixing_results/v31_research_results.json.

归档内容:
  1. 实盘镜像基线 (V3) 全周期/IS/OOS 指标与年度明细
  2. V3.1 保守版 (σ0.28+P2) 对比
  3. 激进方案 (A0/A1危机模式/A2趋势门控/A3高σ) 对比
  4. 双目标探索 (C系列 alpha 改进) 结果
  5. 收益放大探索 (D系列) 结果
  6. 最终结论与效率前沿

用法: uv run python scripts/archive_v31_results.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import exp_v31_sharpe as ex
import run_qixing_v3 as rq

OUTPUT = Path(__file__).parent.parent / "data" / "qixing_results" / "v31_research_results.json"


def m(eq, s, e):
    r = ex.metrics(eq, s, e)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}


def snapshot(eq, exits=None):
    d = {
        "full": m(eq, "2020-01-01", "2026-12-31"),
        "is_2020_2023": m(eq, *ex.IS_RANGE),
        "oos_2024_2026": m(eq, *ex.OOS_RANGE),
        "final_value": round(float(eq["equity"].iloc[-1]), 0),
        "yearly": {
            int(y): m(eq, f"{y}-01-01", f"{y}-12-31")
            for y in sorted(eq["trade_date"].dt.year.unique())
        },
    }
    if exits is not None:
        d["n_exits"] = exits
    return d


def main() -> None:
    data = rq.load_data()
    dmap = ex.build_dmap(data)
    P2 = {"exit_intraday": -0.05, "exit_dd": -0.10}

    archive = {
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "data_end": str(data["518880"]["trade_date"].max()),
        "initial_capital": 100_000.0,
        "note": "实盘V3原版维持不动; 本档案为研究归档, 复现: uv run python scripts/exp_v31_sharpe.py [--compare|--aggressive|--alpha|--gain]",
        "configs": {},
        "conclusions": [],
    }

    # 1. 基线
    mir = rq.run_qixing_v3_same_day(data, 100_000.0, live_mirror=True)
    archive["configs"]["V3_mirror_baseline"] = snapshot(mir["equity_curve"])

    # 2. V3.1 保守版 + 激进系列
    for name, P in {
        "V31_conservative_sigma028": {"vol_target": 0.28, "vol_floor": 0.3, **P2},
        "A0_exit_only": dict(P2),
        "A1_crisis_sigma_h050": {"vol_mode": "crisis", "sigma_high": 0.50, "vol_floor": 0.3, **P2},
        "A2_trend_gate_sigma028": {"vol_target": 0.28, "vol_floor": 0.3, "trend_gate": 0.10, **P2},
        "A3_high_sigma040": {"vol_target": 0.40, "vol_floor": 0.3, **P2},
    }.items():
        r = ex.run_v31(data, P, dmap)
        archive["configs"][name] = snapshot(r["equity"], r["n_exits"])
        archive["configs"][name]["params"] = P

    # 3. 双目标 C 系列
    for name, P in {
        "C1_ma_trend": {**P2, "ma_trend": True},
        "C2_entry_th002": {**P2, "entry_th": 0.02},
        "C3_fast_reenter": {**P2, "fast_reenter": True},
    }.items():
        r = ex.run_v31(data, P, dmap)
        archive["configs"][name] = snapshot(r["equity"])

    # 4. 收益放大 D 系列
    r = ex.run_v31(data, {"daily_strong": True, "strong_th": 0.10}, dmap)
    archive["configs"]["D1_daily_strong_th010"] = snapshot(r["equity"])
    orig_p, orig_w = rq.MOM_PERIODS, rq.MOM_WEIGHTS
    rq.MOM_PERIODS, rq.MOM_WEIGHTS = (5, 10), (0.5, 0.5)
    try:
        archive["configs"]["D2_momentum_5_10"] = snapshot(ex.run_v31(data, {}, dmap)["equity"])
    finally:
        rq.MOM_PERIODS, rq.MOM_WEIGHTS = orig_p, orig_w
    phase_finals = {}
    for off in range(5):
        eq = ex.run_v31(data, {"grid_offset": off}, dmap)["equity"]
        phase_finals[f"offset_{off}"] = float(eq["equity"].iloc[-1])
    archive["configs"]["D3_grid_phase_finals"] = phase_finals

    archive["conclusions"] = [
        "P1波动率目标是唯一通过验证的风控改进: 夏普1.67→1.78(σ0.28+P2), 回撤-40.7%→-22.6%, 年化62%→53.1%",
        "P2日频退出冗余(P1已覆盖); P3波动调整动量有害(夏普1.18)已否决; P4拥挤惩罚无增量",
        "激进档A1危机模式(σh=0.50): 终值154.1万/夏普1.73/回撤-25.2%, 为攻守平衡推荐点",
        "双目标探索8组alpha改进全部恶化(C系列), 收益放大3类(D系列)全部为负贡献",
        "V3当前配置(10/20动量+5日网格+自适应阈值)是该策略族帕累托顶点, 实盘维持不动",
        "网格相位敏感性±261%(当前相位0最优190.9万, 五相位平均约94万), 长期收益中枢预期+800%~+1800%",
    ]

    OUTPUT.write_text(json.dumps(archive, indent=2, ensure_ascii=False, default=str))
    print(f"  ✓ 研究档案已保存: {OUTPUT}")
    print(f"    含 {len(archive['configs'])} 组配置结果 + {len(archive['conclusions'])} 条结论")


if __name__ == "__main__":
    main()
