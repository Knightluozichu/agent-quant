"""反过拟合审计 ③ 门控+豁免+H3 Walk-forward (训练/测试分离) + PBO.

设计 (严格样本外):
  - 4 段 expanding window (复用 walk_forward.py 边界 + purge/embargo):
      WF-1: train 2020-01~2021-06 → test 2021-07~2022-06
      WF-2: train 2020-01~2022-06 → test 2022-07~2023-06
      WF-3: train 2020-01~2023-06 → test 2023-07~2024-06
      WF-4: train 2020-01~2024-06 → test 2024-07~2025-06
  - 每段仅在 train 段扫描门控阈值 ret60_thr ∈ {-0.05, 0.00, 0.05}
    (δ=2% expo=0.3 固定), 选 train 期末最高者; test 段用该参数验证
  - 对比同段基线 (原版过滤), 检验 test 段是否有改善
  - PBO (CSCV): 参数×段落 Sharpe 矩阵, train 最优参数在 test 段
    排名低于中位数的比例

注: 审计①已证 2026-07 前新方案与基线零差异, 本测试检验前 4 段
    测试窗口的样本外表现 (预期多为零差异, 验证机制稀疏性).
用法: uv run python scripts/exp_walk_forward_gate.py
输出: data/v9_results/walk_forward_gate.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
import exp_drop_gate_h3 as h3  # noqa: E402
from exp_drop_gate_exempt import select_target_exempt  # noqa: E402

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"
WARMUP = 130
RET60_GRID = (-0.05, 0.00, 0.05)

SEGMENTS = [
    {
        "name": "WF-1",
        "tr0": "2020-01-02",
        "tr1": "2021-06-30",
        "te0": "2021-07-05",
        "te1": "2022-06-30",
    },
    {
        "name": "WF-2",
        "tr0": "2020-01-02",
        "tr1": "2022-06-30",
        "te0": "2022-07-05",
        "te1": "2023-06-30",
    },
    {
        "name": "WF-3",
        "tr0": "2020-01-02",
        "tr1": "2023-06-30",
        "te0": "2023-07-05",
        "te1": "2024-06-30",
    },
    {
        "name": "WF-4",
        "tr0": "2020-01-02",
        "tr1": "2024-06-30",
        "te0": "2024-07-05",
        "te1": "2025-06-30",
    },
]


def filter_data(data: dict, s0: str, s1: str) -> dict:
    start = pd.to_datetime(s0)
    end = pd.to_datetime(s1)
    out = {}
    for code, df in data.items():
        dates = pd.to_datetime(df["trade_date"])
        out[code] = df[(dates >= start) & (dates <= end)].reset_index(drop=True)
    return out


def run_variant(data: dict, thr: float | None, exempt: bool, h3on: bool) -> dict:
    """thr=None → 基线(原版); 否则门控(thr)+豁免+H3."""
    rq.check_single_day_drop = h3.make_gated(thr, True) if thr is not None else h3.ORIG_CHECK
    h3.select_target = select_target_exempt if exempt else rq.select_target
    h3.H3_ENABLED = h3on
    h3.H3_DELTA, h3.H3_ACTION, h3.H3_EXPO = 0.02, "reduce", 0.3
    rep = h3.run_v3_risk_h3(data)
    rq.check_single_day_drop = h3.ORIG_CHECK
    h3.select_target = rq.select_target
    h3.H3_ENABLED = False
    return rep


def main() -> None:
    print("=" * 78)
    print("  反过拟合审计③ Walk-forward | 门控+豁免+H3 vs 基线")
    print("=" * 78)

    full = rq.load_data()
    results = []
    sharpe_matrix = []  # rows=参数集, cols=段落 (test段 Sharpe)

    for seg in SEGMENTS:
        print(
            f"\n  [{seg['name']}] train {seg['tr0']}~{seg['tr1']} → test {seg['te0']}~{seg['te1']}"
        )
        train_data = filter_data(full, seg["tr0"], seg["tr1"])
        test_data = filter_data(full, seg["te0"], seg["te1"])

        # --- train 段扫描 ret60_thr ---
        train_best = None
        train_scores = {}
        for thr in RET60_GRID:
            rep = run_variant(train_data, thr, exempt=True, h3on=True)
            train_scores[thr] = rep["final_value"]
            print(
                f"    train thr={thr:+.2f}: 期末 {rep['final_value']:>10,.0f} "
                f"夏普 {rep['sharpe']:.2f}"
            )
        train_best = max(train_scores, key=lambda k: train_scores[k])
        print(f"    → train 最优 thr={train_best:+.2f}")

        # --- test 段用最优参数验证 + 基线对照 ---
        rep_test = run_variant(test_data, train_best, exempt=True, h3on=True)
        rep_base = run_variant(test_data, None, exempt=False, h3on=False)
        diff = rep_test["final_value"] / rep_base["final_value"] - 1
        print(
            f"    test 新方案: 期末 {rep_test['final_value']:>10,.0f} 夏普 {rep_test['sharpe']:.2f}"
        )
        print(
            f"    test 基线  : 期末 {rep_base['final_value']:>10,.0f} "
            f"夏普 {rep_base['sharpe']:.2f} | 差异 {diff:+.1%}"
        )

        # test 段全部参数 (供 PBO)
        row_sharpe = []
        for thr in RET60_GRID:
            rep_t = run_variant(test_data, thr, exempt=True, h3on=True)
            row_sharpe.append(rep_t["sharpe"])
        sharpe_matrix.append(row_sharpe)

        results.append(
            {
                "segment": seg["name"],
                "train_best_thr": float(train_best),
                "train_scores": {str(k): round(v, 0) for k, v in train_scores.items()},
                "test_new_final": rep_test["final_value"],
                "test_base_final": rep_base["final_value"],
                "test_diff": round(float(diff), 4),
                "test_new_sharpe": rep_test["sharpe"],
                "test_base_sharpe": rep_base["sharpe"],
                "test_sharpes_all": [round(s, 3) for s in row_sharpe],
            }
        )

    # === PBO (CSCV) ===
    m = np.array(sharpe_matrix).T  # (n_params, n_segments)
    print("\n" + "=" * 78)
    print("  PBO (CSCV, 参数×段落 Sharpe 矩阵):")
    print(f"    ret60_thr 网格: {[f'{t:+.2f}' for t in RET60_GRID]}")
    for i, t in enumerate(RET60_GRID):
        print(f"    thr={t:+.2f}: 各段夏普 {[f'{s:.2f}' for s in m[i]]}")
    n_params, n_segs = m.shape
    n_half = n_segs // 2
    from itertools import combinations

    n_overfit, n_total = 0, 0
    for tr_idx in combinations(range(n_segs), n_half):
        te_idx = [i for i in range(n_segs) if i not in tr_idx]
        tr_sh = m[:, list(tr_idx)].mean(axis=1)
        te_sh = m[:, te_idx].mean(axis=1)
        best_tr = int(np.argmax(tr_sh))
        te_rank = np.argsort(np.argsort(te_sh))
        if te_rank[best_tr] < n_params // 2:
            n_overfit += 1
        n_total += 1
    pbo = n_overfit / n_total if n_total else 0.0
    print(
        f"    PBO = {pbo:.1%} ({n_overfit}/{n_total}) "
        f"→ {'过拟合风险低' if pbo < 0.5 else '过拟合风险高'}"
    )

    out = {
        "segments": results,
        "pbo": round(float(pbo), 3),
        "ret60_grid": [float(t) for t in RET60_GRID],
        "note": "审计①已证 2026-07 前新方案与基线零差异; 本测试覆盖 2021H2~2025H1 测试窗口",
    }
    path = OUTPUT_DIR / "walk_forward_gate.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {path}")


if __name__ == "__main__":
    main()
