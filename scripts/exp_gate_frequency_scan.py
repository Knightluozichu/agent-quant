"""反过拟合审计 ④ 触发频率-表现扫描: 放宽参数扩大门控触发面.

用户质疑: "n=1 不能证明" 是机制触发面过窄的结果, 不是机制无效的证据.
          V3 周频Top1 + 暴跌+ret60<0+动量>0 三重条件叠加后天然稀疏.
本脚本: 放宽三个参数维度扩大触发面, 检验"触发更多次后是否仍有效":
  - ret60_thr ∈ {0.00, 0.10, 0.20, 0.30}  (放宽放行: 前期上涨也放行)
  - mom_thr   ∈ {0.00, -0.05}              (放宽动量守卫)
  - exempt    ∈ {on, off}                  (豁免开关)
每变体记录: 全周期期末/夏普/回撤, IS/OOS, 放行次数, 实质决策分歧次数
  (与基线决策不同的调仓日数 = 有效样本量).
判定: 若存在"分歧次数≥5 且 全周期≥基线"的参数区 → 机制有信号;
      若分歧增加后表现断崖 → 原+9.3% 是单次运气.
用法: uv run python scripts/exp_gate_frequency_scan.py
输出: data/v9_results/gate_frequency_scan.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
import exp_drop_gate_h3 as h3  # noqa: E402
from exp_drop_gate_exempt import select_target_exempt  # noqa: E402

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"
ORIG_SELECT = rq.select_target
DECISIONS: list[tuple] = []


def make_gated_par(ret60_thr: float, mom_thr: float):
    """参数化门控: ret60>=thr 排除; 动量>mom_thr 才放行."""
    def gated(close: np.ndarray) -> bool:
        if len(close) < rq.DROP_LOOKBACK + 1:
            return True
        triggered = False
        for i in range(-rq.DROP_LOOKBACK, 0):
            dr = (close[i] - close[i - 1]) / close[i - 1]
            if dr < rq.DROP_THRESHOLD:
                triggered = True
                break
        if not triggered:
            return True
        if len(close) <= 61:
            return True
        ret60 = (close[-1] - close[-61]) / close[-61]
        if ret60 >= ret60_thr:
            return False
        r10 = (close[-1] - close[-11]) / close[-11] if len(close) > 11 else 0.0
        r20 = (close[-1] - close[-21]) / close[-21] if len(close) > 21 else 0.0
        if 0.5 * r10 + 0.5 * r20 <= mom_thr:
            return False
        return True
    return gated


def make_exempt_par(ret60_thr: float):
    """参数化豁免: 放行类型持仓 (ret60<thr 且近5日暴跌) 不享缓冲."""
    def exempt_select(data, etf_data_at_date, holding):
        t, c, s, a = ORIG_SELECT(data, etf_data_at_date, holding)
        if holding and holding in etf_data_at_date and holding != rq.DEFENSE:
            hclose = data[holding]["close"].values[:etf_data_at_date[holding] + 1].astype(float)
            if len(hclose) > rq.DROP_LOOKBACK + 1 and len(hclose) > 61:
                drop = any(
                    (hclose[i] - hclose[i - 1]) / hclose[i - 1] < rq.DROP_THRESHOLD
                    for i in range(-rq.DROP_LOOKBACK, 0))
                ret60 = (hclose[-1] - hclose[-61]) / hclose[-61]
                if drop and ret60 < ret60_thr:
                    # 重新评估: 强制换最强 (threshold=0)
                    if c:
                        best_target = c[0][0]
                        best_score = c[0][1]
                        cur_score = dict(c).get(holding, -999)
                        if cur_score > 0 and best_score <= cur_score:
                            t = holding
                        else:
                            t = best_target
        return t, c, s, a
    return exempt_select


def wrapped_select(data, etf_data_at_date, holding):
    """记录每次调仓 (date, holding, target)."""
    global DECISIONS
    t, c, s, a = rq.select_target(data, etf_data_at_date, holding)
    td = None
    for code in rq.ETF_POOL:
        if code in etf_data_at_date:
            td = data[code].iloc[etf_data_at_date[code]]["trade_date"]
            break
    DECISIONS.append((str(td), holding, t))
    return t, c, s, a


def run_variant(data, gate_fn, exempt_fn, h3on: bool, log: bool) -> dict:
    global DECISIONS
    DECISIONS = []
    rq.check_single_day_drop = gate_fn
    base_select = exempt_fn if exempt_fn is not None else ORIG_SELECT

    def wrapped(data_, etf_data_at_date, holding):
        global DECISIONS
        t, c, s, a = base_select(data_, etf_data_at_date, holding)
        td = None
        for code in rq.ETF_POOL:
            if code in etf_data_at_date:
                td = data_[code].iloc[etf_data_at_date[code]]["trade_date"]
                break
        DECISIONS.append((str(td), holding, t))
        return t, c, s, a

    rq.select_target = wrapped
    h3.select_target = wrapped  # run_v3_risk_h3 读取 h3 模块绑定
    h3.H3_ENABLED = h3on
    h3.H3_DELTA, h3.H3_ACTION, h3.H3_EXPO = 0.02, "reduce", 0.3
    rep = h3.run_v3_risk_h3(data)
    log_snapshot = list(DECISIONS)
    rq.check_single_day_drop = h3.ORIG_CHECK
    rq.select_target = ORIG_SELECT
    h3.select_target = ORIG_SELECT
    h3.H3_ENABLED = False
    if log:
        return rep, log_snapshot
    return rep, None


def count_divergence(log_a: list, log_b: list) -> int:
    """统计 log_b 与基线 log_a 的实质决策分歧次数 (同date下 target 不同).

    注: 只比较 target (决策目标), 不比较 holding — holding 分叉是决策
    分歧的后果, 比较它会产生大量伪分歧.
    """
    amap = {d: t for d, h, t in log_a}
    div = 0
    for d, h, t in log_b:
        a = amap.get(d)
        if a is not None and a != t:
            div += 1
    return div


def main() -> None:
    print("=" * 80)
    print("  反过拟合审计④ 触发频率-表现扫描 | 放宽参数扩大样本")
    print("=" * 80)

    data = rq.load_data()

    # 基线 (决策日志)
    rep_base, log_base = run_variant(data, h3.ORIG_CHECK, None, False, log=True)
    base_final = rep_base["final_value"]
    print(f"\n  基线: 期末 {base_final:,.0f} 夏普 {rep_base['sharpe']:.2f} "
          f"回撤 {rep_base['max_drawdown']:.1%}")

    rows = []
    for ret60_thr in (0.00, 0.10, 0.20, 0.30):
        for mom_thr in (0.00, -0.05):
            for exempt in (True, False):
                gate_fn = make_gated_par(ret60_thr, mom_thr)
                ex_fn = make_exempt_par(ret60_thr) if exempt else None
                rep, log = run_variant(data, gate_fn, ex_fn, True, log=True)
                div = count_divergence(log_base, log)
                diff = rep["final_value"] / base_final - 1
                rows.append({
                    "ret60_thr": ret60_thr, "mom_thr": mom_thr,
                    "exempt": exempt, "final": rep["final_value"],
                    "diff_vs_base": diff, "sharpe": rep["sharpe"],
                    "max_dd": rep["max_drawdown"], "divergences": div,
                    "n_trades": rep["n_trades"],
                })
                print(f"  thr={ret60_thr:+.2f} mom={mom_thr:+.2f} "
                      f"豁免={'on' if exempt else 'off'}: "
                      f"期末{rep['final_value']:>12,.0f} ({diff:+.1%}) "
                      f"夏普{rep['sharpe']:.2f} 分歧{div:>3}次")

    # === 分析: 分歧次数与表现的关系 ===
    print("\n" + "=" * 80)
    print("  分歧次数 vs 表现 (有效样本量与收益的关系):")
    active = [r for r in rows if r["divergences"] >= 3]
    for r in sorted(active, key=lambda x: -x["diff_vs_base"]):
        print(f"    分歧{r['divergences']:>3}次 thr={r['ret60_thr']:+.2f} "
              f"mom={r['mom_thr']:+.2f} 豁免={'on' if r['exempt'] else 'off'}: "
              f"{r['diff_vs_base']:+.1%}")
    if not active:
        print("    (无分歧≥3次的变体)")

    # 判定: 是否存在 分歧≥5 且 全周期≥基线 的参数区
    robust = [r for r in rows if r["divergences"] >= 5 and r["diff_vs_base"] >= 0]
    print(f"\n  判定: 分歧≥5 且 全周期≥基线 的参数区: "
          f"{len(robust)} 个 {'✅ 机制有信号' if robust else '❌ 无稳健参数区'}")
    for r in robust[:5]:
        print(f"    thr={r['ret60_thr']:+.2f} mom={r['mom_thr']:+.2f} "
              f"豁免={'on' if r['exempt'] else 'off'} 分歧{r['divergences']}次 "
              f"{r['diff_vs_base']:+.1%}")

    out = {"base_final": base_final, "rows": rows,
           "robust_zone": [r for r in robust]}
    path = OUTPUT_DIR / "gate_frequency_scan.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {path}")


if __name__ == "__main__":
    main()
