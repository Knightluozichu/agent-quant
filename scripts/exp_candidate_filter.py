"""C队阶段1: 候选规律快速筛选 (效应量 + 事件聚类重抽样 + 多重比较校正).

对 6 个候选规律 (源自 short_window_patterns.json 统计实验) 逐一裁决:
  1. 效应量排序: Cohen's d = |mean_diff| / pooled_std, <0.5 直接淘汰
  2. 事件聚类重抽样: 对事件型规律 (暴涨后持续/暴跌后反弹), 将相距≤5日的
     连续事件合并为簇, 以簇为独立样本重算效应量 — 符号不变且保留>50% 才通过
  3. 多重比较校正: Welch t 检验 p 值 vs Bonferroni 阈值 α'=0.05/N_TESTS
     (N_TESTS≈503 为实验全部统计检验数的估计)

候选规律:
  R1 3日动量反转      R3最强组(Q5) vs 最弱组(Q1) 未来5日收益差
  R2 10日动量符号     动量>0 vs ≤0 未来5日收益差
  R3 类别轮动         最强类别下周 vs 最弱类别下周收益差
  R4 暴涨后持续       单日>3% 事件后5日 vs 全样本5日
  R5 暴跌后反弹       近5日含单日<-3% 后5日 vs 干净池 (聚类后: 暴跌簇结束后5日)
  R6 高波动收益       20日年化波动>30% vs <15% 后5日收益差

输出: data/v9_results/candidate_filter.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 复用实验脚本的数据口径
from exp_short_window_patterns import (  # noqa: E402
    BIG_MOVE,
    CATEGORIES,
    close_matrix,
    common_dates,
    daily_rets,
    load_data,
    window_ret,
)

# === 筛选参数 (C队阶段1) ===
N_TESTS = 503            # 实验全部统计检验数的估计
ALPHA_BONF = 0.05 / N_TESTS  # ≈ 0.0001
MIN_EFFECT = 0.5         # Cohen's d 阈值
CLUSTER_WINDOW = 5       # 事件聚类窗口 (交易日)
KEEP_RATIO = 0.5         # 聚类后须保留的效应量比例


# --------------------------------------------------------------------------- #
# 统计工具
# --------------------------------------------------------------------------- #
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """效应量: 组间均值差 / 合并标准差."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if sp < 1e-12:
        return 0.0
    return float(abs(a.mean() - b.mean()) / sp)


def welch_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch t 检验 → (t, p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(t), float(p)


def cluster_events(dr: np.ndarray, thr: float, window: int = CLUSTER_WINDOW,
                   direction: str = "up") -> list[list[int]]:
    """将相距≤window日的连续事件聚类为簇.

    direction='up' 取 dr>=thr; 'down' 取 dr<=thr.
    Returns: [[idx1, idx2, ...], ...] 时间正序簇列表.
    """
    mask = dr >= thr if direction == "up" else dr <= thr
    idxs = np.where(mask)[0]
    clusters: list[list[int]] = []
    for idx in idxs:
        if clusters and idx - clusters[-1][-1] <= window:
            clusters[-1].append(int(idx))
        else:
            clusters.append([int(idx)])
    return clusters


def summarize(a: np.ndarray) -> dict:
    a = np.asarray(a, float)
    return {
        "n": len(a),
        "mean": round(float(a.mean()), 6) if len(a) else None,
        "median": round(float(np.median(a)), 6) if len(a) else None,
        "std": round(float(a.std(ddof=1)), 6) if len(a) > 1 else None,
        "pct_pos": round(float((a > 0).mean()), 4) if len(a) else None,
    }


# --------------------------------------------------------------------------- #
# 候选规律定义 (每项返回评估所需统计量)
# --------------------------------------------------------------------------- #
def r1_three_day_reversal(mat: dict) -> dict:
    """R1: 池级合并 R3 最强组(Q5) vs 最弱组(Q1) 未来5日收益."""
    q5_all, q1_all = [], []
    for close in mat.values():
        mom = window_ret(close, 3)
        fwd = close[3 + 5:] / close[3:-5] - 1.0  # 动量结束后持有5日 (与实验同口径)
        m = mom[:len(fwd)]
        qs = pd.qcut(m, 5, labels=False, duplicates="drop")
        q5_all.append(fwd[qs == 4])
        q1_all.append(fwd[qs == 0])
    q5 = np.concatenate(q5_all)
    q1 = np.concatenate(q1_all)
    return {
        "id": "R1", "name": "3日动量反转(Q5 vs Q1, 5日)",
        "type": "cross-sectional", "a": q5, "b": q1,
        "a_label": "R3最强组Q5", "b_label": "R3最弱组Q1",
        "note": "5/7资产为反转方向; 白银为唯一持续例外",
    }


def r2_ten_day_sign(mat: dict) -> dict:
    """R2: 池级合并 10日动量>0 vs ≤0 未来5日收益."""
    pos, neg = [], []
    for close in mat.values():
        m10 = window_ret(close, 10)
        fwd = close[10 + 5:] / close[10:-5] - 1.0
        m = m10[:len(fwd)]
        pos.append(fwd[m > 0])
        neg.append(fwd[m <= 0])
    return {
        "id": "R2", "name": "10日动量符号(>0 vs ≤0, 5日)",
        "type": "cross-sectional", "a": np.concatenate(pos), "b": np.concatenate(neg),
        "a_label": "动量>0", "b_label": "动量≤0",
        "note": "实验 mom_sign 差 +3.09pp, 判别力最强",
    }


def r3_best_category(mat: dict, dates: list) -> dict:
    """R3: 最强类别下周 vs 最弱类别下周 收益差 (周频)."""
    r5: dict[str, np.ndarray] = {}
    for code, close in mat.items():
        r5[code] = window_ret(close, 5)
    n = len(dates) - 5
    cat_r5 = {cat: np.mean([r5[c] for c in codes if c in r5], axis=0)
              for cat, codes in CATEGORIES.items()}
    idxs = list(range(0, n, 5))
    best_next, worst_next = [], []
    for i in range(len(idxs) - 1):
        t, t2 = idxs[i], idxs[i + 1]
        if t2 >= n:
            break
        ranks = sorted(cat_r5, key=lambda k: -cat_r5[k][t])
        best_next.append(cat_r5[ranks[0]][t2])
        worst_next.append(cat_r5[ranks[-1]][t2])
    return {
        "id": "R3", "name": "最强类别下周 vs 最弱类别下周",
        "type": "cross-sectional",
        "a": np.array(best_next), "b": np.array(worst_next),
        "a_label": "最强类别下周", "b_label": "最弱类别下周",
        "note": "实验: 差+0.26pp, 胜率53.1% (弱信号)",
    }


def _after_events(mat: dict, thr: float, direction: str, clustered: bool) -> np.ndarray:
    """事件后5日收益. clustered=True 时用事件簇代表日."""
    out = []
    for close in mat.values():
        dr = daily_rets(close)
        if clustered:
            clusters = cluster_events(dr, thr, direction=direction)
            ev = [c[0] for c in clusters]  # 每簇代表日 (第一个事件日)
        else:
            ev = np.where(dr >= thr if direction == "up" else dr <= thr)[0]
        for t in ev:
            if t + 5 < len(dr):
                out.append(close[t + 5] / close[t] - 1.0)
    return np.array(out)


def _all_5d(mat: dict) -> np.ndarray:
    """全样本5日收益 (基准)."""
    out = []
    for close in mat.values():
        out.extend(window_ret(close, 5))
    return np.array(out)


def r4_surge_persistence(mat: dict) -> dict:
    """R4: 暴涨>3% 事件后5日 vs 全样本5日 (含事件聚类重抽样)."""
    raw = _after_events(mat, BIG_MOVE, "up", clustered=False)
    clu = _after_events(mat, BIG_MOVE, "up", clustered=True)
    base = _all_5d(mat)
    return {
        "id": "R4", "name": "暴涨>3%后5日持续",
        "type": "event",
        "raw": {"after": raw, "base": base},
        "clustered": {"after": clu, "base": base},
        "a_label": "暴涨后5日", "b_label": "全样本5日",
        "note": "事件重叠严重 (原油80次/白银76次), 须聚类复核",
    }


def _drop_rebound(mat: dict, clustered: bool) -> tuple[np.ndarray, np.ndarray]:
    """暴跌后反弹: 近5日含<-3% (raw) 或 暴跌簇结束后 (clustered) 的5日收益.

    raw:      近5日有单日<-3% → 未来5日  vs  近5日无 (干净池)
    clustered: 暴跌簇结束后第1天起5日 (V型反弹起点)  vs  全样本5日
    """
    if not clustered:
        dropped, clean = [], []
        for close in mat.values():
            dr = daily_rets(close)
            for t in range(5, len(dr) - 5):
                fwd = close[t + 5] / close[t] - 1.0
                (dropped if (dr[t - 5:t] < -BIG_MOVE).any() else clean).append(fwd)
        return np.array(dropped), np.array(clean)
    rebound, base = [], []
    for close in mat.values():
        dr = daily_rets(close)
        clusters = cluster_events(dr, -BIG_MOVE, direction="down")
        for cl in clusters:
            t = cl[-1] + 1  # 簇结束后第1天 (反弹起点)
            if t + 5 <= len(dr):
                rebound.append(close[t + 5] / close[t] - 1.0)
        base.extend(window_ret(close, 5))
    return np.array(rebound), np.array(base)


def r5_drop_rebound(mat: dict) -> dict:
    """R5: 暴跌后反弹."""
    raw_a, raw_b = _drop_rebound(mat, clustered=False)
    clu_a, clu_b = _drop_rebound(mat, clustered=True)
    return {
        "id": "R5", "name": "暴跌后短期反弹",
        "type": "event",
        "raw": {"after": raw_a, "base": raw_b},
        "clustered": {"after": clu_a, "base": clu_b},
        "a_label": "暴跌后5日", "b_label": "基准5日",
        "note": "raw口径: 暴跌池+0.74% vs 干净池+0.24%; 需聚类复核均值回归偏差",
    }


def r6_vol_regime(mat: dict) -> dict:
    """R6: 高波动(>30%) vs 低波动(<15%) 未来5日收益."""
    high, low = [], []
    for close in mat.values():
        dr = daily_rets(close)
        for t in range(20, len(dr) - 5):
            v = np.std(dr[t - 20:t]) * np.sqrt(252)
            fwd = close[t + 5] / close[t] - 1.0
            if v > 0.30:
                high.append(fwd)
            elif v < 0.15:
                low.append(fwd)
    return {
        "id": "R6", "name": "高波动>30% vs 低波动<15% (5日)",
        "type": "cross-sectional",
        "a": np.array(high), "b": np.array(low),
        "a_label": "高波动>30%", "b_label": "低波动<15%",
        "note": "实验: +0.71% vs +0.18%, 高波动胜率54.0%",
    }


# --------------------------------------------------------------------------- #
# 裁决流程
# --------------------------------------------------------------------------- #
def evaluate(cand: dict) -> dict:
    """对单个候选规律执行三关筛选, 返回裁决结果."""
    out = {"id": cand["id"], "name": cand["name"], "type": cand["type"], "note": cand["note"]}

    # 第1关: 效应量 (对 cross-sectional: a vs b; 对 event: raw after vs base)
    if cand["type"] == "cross-sectional":
        a, b = cand["a"], cand["b"]
        d = cohens_d(a, b)
        t_stat, p = welch_test(a, b)
        out["raw"] = {
            "a": summarize(a), "b": summarize(b),
            "mean_diff": round(float(a.mean() - b.mean()), 6),
            "cohens_d": round(d, 4),
            "t_stat": round(t_stat, 3), "p_value": round(p, 8),
            "effect_pass": bool(d >= MIN_EFFECT),
            "bonf_pass": bool(p < ALPHA_BONF),
        }
    else:
        ra, rb = cand["raw"]["after"], cand["raw"]["base"]
        d_raw = cohens_d(ra, rb)
        t_raw, p_raw = welch_test(ra, rb)
        out["raw"] = {
            "a": summarize(ra), "b": summarize(rb),
            "mean_diff": round(float(ra.mean() - rb.mean()), 6),
            "cohens_d": round(d_raw, 4),
            "t_stat": round(t_raw, 3), "p_value": round(p_raw, 8),
            "effect_pass": bool(d_raw >= MIN_EFFECT),
            "bonf_pass": bool(p_raw < ALPHA_BONF),
        }
        # 第2关: 事件聚类重抽样
        ca, cb = cand["clustered"]["after"], cand["clustered"]["base"]
        d_clu = cohens_d(ca, cb)
        t_clu, p_clu = welch_test(ca, cb)
        keep_ratio = (d_clu / d_raw) if d_raw > 1e-12 else 0.0
        same_sign = bool(np.sign(d_clu) == np.sign(d_raw) or abs(d_clu) < 1e-12)
        out["clustered"] = {
            "a": summarize(ca), "b": summarize(cb),
            "mean_diff": round(float(ca.mean() - cb.mean()), 6),
            "cohens_d": round(d_clu, 4),
            "t_stat": round(t_clu, 3), "p_value": round(p_clu, 8),
            "n_reduction": f"{len(ra)}→{len(ca)}",
            "keep_ratio": round(keep_ratio, 3),
            "same_sign": same_sign,
            "cluster_pass": bool(same_sign and keep_ratio >= KEEP_RATIO),
            "bonf_pass": bool(p_clu < ALPHA_BONF),
        }

    # 裁决
    raw = out["raw"]
    checks = [raw["effect_pass"], raw["bonf_pass"]]
    reasons = []
    if not raw["effect_pass"]:
        reasons.append(f"效应量d={raw['cohens_d']:.2f}<{MIN_EFFECT}")
    if not raw["bonf_pass"]:
        reasons.append(f"p={raw['p_value']:.4g}≥α'={ALPHA_BONF:.2g}")
    if cand["type"] == "event":
        clu = out["clustered"]
        checks.append(clu["cluster_pass"])
        if not clu["cluster_pass"]:
            reasons.append(
                f"聚类后效应量保留{clu['keep_ratio']:.0%}或符号翻转 (d={clu['cohens_d']:.2f})"
            )
        if not clu["bonf_pass"]:
            reasons.append(f"聚类后p={clu['p_value']:.4g}不显著")

    all_pass = all(checks)
    if all_pass:
        out["verdict"] = "待定(进入阶段2)"
    else:
        out["verdict"] = "淘汰"
    out["reasons"] = reasons
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("  C队阶段1: 候选规律快速筛选")
    print(f"  效应量阈值 d≥{MIN_EFFECT} | Bonferroni α'={ALPHA_BONF:.2g} (N={N_TESTS}) | "
          f"聚类保留≥{KEEP_RATIO:.0%}")
    print("=" * 74)

    data = load_data()
    dates = common_dates(data)
    mat = close_matrix(data, dates)
    print(f"\n  数据: {len(dates)} 个交易日 ({dates[0]} ~ {dates[-1]})")

    candidates = [
        r1_three_day_reversal(mat),
        r2_ten_day_sign(mat),
        r3_best_category(mat, dates),
        r4_surge_persistence(mat),
        r5_drop_rebound(mat),
        r6_vol_regime(mat),
    ]

    results = []
    print(f"\n  {'规律':<8} {'效应量d':>8} {'均值差':>9} {'p值':>10} {'裁决':<14} 备注")
    print("  " + "-" * 72)
    for cand in candidates:
        res = evaluate(cand)
        results.append(res)
        raw = res["raw"]
        mean_diff = raw["mean_diff"]
        p = raw["p_value"]
        # 方向: mean_diff 为 a - b
        direction = "+" if mean_diff > 0 else ""
        line = (f"  {res['id']:<8} {raw['cohens_d']:>8.2f} {direction}{mean_diff:>8.2%} "
                f"{p:>10.3g} {res['verdict']:<14}")
        print(line)
        if res["type"] == "event":
            clu = res["clustered"]
            print(f"      └ 聚类重抽样: n={clu['a']['n']:>5}  d={clu['cohens_d']:.2f}  "
                  f"保留{clu['keep_ratio']:.0%}  p={clu['p_value']:.3g}")
        if res["reasons"]:
            print(f"      └ 淘汰原因: {'; '.join(res['reasons'])}")

    # 汇总
    passed = [r for r in results if r["verdict"].startswith("待定")]
    print("\n" + "=" * 74)
    if passed:
        print(f"  通过筛选: {', '.join(r['id'] for r in passed)} — 进入阶段2 walk-forward 验证")
    else:
        print("  全部候选规律被淘汰 — 6个候选规律均未通过阶段1")
    print("=" * 74)

    out_path = OUTPUT_DIR / "candidate_filter.json"
    with open(out_path, "w") as f:
        json.dump({
            "meta": {"n_tests": N_TESTS, "bonf_alpha": ALPHA_BONF,
                     "min_effect": MIN_EFFECT, "cluster_window": CLUSTER_WINDOW,
                     "period": f"{dates[0]} ~ {dates[-1]}"},
            "candidates": results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
