"""V3基线 vs V3+RF跌概率减仓层 A/B 验证 (C队阶段3).

背景: ML 研究 (exp_ml_up_down.py) 显示 RandomForest 跌模型是唯一提供稳定增量的模型
(跌AUC 0.535, +0.020 vs 基线). 本实验验证其作为 V3 减仓确认层的交易价值:

  - RF 跌模型: 预测未来5日收益<0 的概率 (22个动态特征, 与 ML 研究同特征工程)
  - 减仓层: 每个调仓日, 若当前持仓为风险资产且 RF跌概率 > 阈值 → 目标改为货币基金
    (防御优先; 下一调仓日恢复正常 V3 决策)
  - 时序正确性: 4个OOS段 (260日训练→20日缓冲→260日测试), 每段用段前全部数据
    训练 RF (无未来函数), 段内冻结运行
  - 对比: V3基线 / V3+RF(0.50) / V3+RF(0.55) / V3+RF(0.60)

评估: 年化/夏普/回撤/交易数 (复刻 run_qixing_v3 指标算法, 生产脚本零改动).
输出: data/v9_results/v3_rf_guard_ab.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from exp_ml_up_down import build_features
from exp_short_window_patterns import close_matrix, common_dates
from run_qixing_v3 import (
    DEFENSE,
    ETF_POOL,
    FEE,
    REBALANCE_DAYS,
    SLIPPAGE,
    load_data,
    select_target,
)

WARMUP = 130  # V3 预热期 (与生产一致)
INITIAL_CAPITAL = 100_000.0
GUARD_THRESHOLDS = (0.50, 0.55, 0.60)

# === OOS 段划分 (与 walk-forward 一致) ===
TRAIN_DAYS = 260
TEST_DAYS = 260
BUFFER_DAYS = 20
STEP = (TRAIN_DAYS + TEST_DAYS) // 2


# --------------------------------------------------------------------------- #
# V3 回测引擎 (复刻 run_qixing_v3 逻辑, 支持时间区间 + RF减仓层)
# --------------------------------------------------------------------------- #
def run_v3_window(
    data: dict,
    dates: list,
    start_idx: int,
    end_idx: int,
    rf_proba: dict | None = None,
    guard_thr: float | None = None,
) -> dict:
    """在日期索引区间 [start_idx, end_idx) 运行 V3 周频动量轮动.

    Args:
        rf_proba: {(date_str, code): 跌概率} 或 None (无减仓层)
        guard_thr: 减仓阈值; rf_proba 提供且持仓概率>阈值 → 切防御
    """
    # 绝对调仓网格 (与生产一致: 全局公共日历 warmup 后每5个交易日)
    trading_dates = dates[WARMUP:]
    global_rebalance = set(trading_dates[::REBALANCE_DAYS])
    window_dates = dates[start_idx:end_idx]

    cash = float(INITIAL_CAPITAL)
    holding: str | None = None
    holding_shares: int = 0
    equity_history = []
    n_trades = 0
    guard_hits = 0

    for td in window_dates:
        if td not in global_rebalance:
            continue  # 非调仓日 (V3 仅在调仓日采样 equity)

        # 各ETF在td的索引 (与生产一致)
        etf_data_at_date = {}
        for code in [*list(ETF_POOL.keys()), DEFENSE]:
            if code not in data:
                continue
            df = data[code]
            mask = df["trade_date"] <= td
            if mask.sum() < WARMUP:
                continue
            etf_data_at_date[code] = mask.sum() - 1

        # 核心决策 (与生产同一函数)
        target, _candidates, _score, _a_weak = select_target(data, etf_data_at_date, holding)

        # === RF 减仓层: 持仓风险资产且跌概率>阈值 → 切防御 ===
        if guard_thr is not None and rf_proba and holding and holding != DEFENSE:
            p = rf_proba.get((str(td), holding))
            if p is not None and p > guard_thr:
                target = DEFENSE
                guard_hits += 1

        # 交易执行 (与生产一致)
        if target != holding:
            # 卖出
            if holding and holding in data:
                row = data[holding][data[holding]["trade_date"] == td]
                if not row.empty:
                    price = row.iloc[0]["close"]
                    cash += holding_shares * price * (1 - FEE - SLIPPAGE)
                    n_trades += 1
                    holding = None
                    holding_shares = 0
            # 买入
            if target in data:
                row = data[target][data[target]["trade_date"] == td]
                if not row.empty:
                    price = row.iloc[0]["close"]
                    shares = int(cash * 0.99 / price / 100) * 100
                    if shares > 0:
                        cost = shares * price * (1 + FEE + SLIPPAGE)
                        cash -= cost
                        holding = target
                        holding_shares = shares
                        n_trades += 1

        # 记录 equity (调仓日采样, 与生产一致)
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

    if not equity_history:
        return {"error": "no data"}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    total_return = eq_df["equity"].iloc[-1] / INITIAL_CAPITAL - 1
    rets = eq_df["equity"].pct_change().dropna()
    periods_per_year = 252 / REBALANCE_DAYS
    ann_vol = rets.std() * np.sqrt(periods_per_year) if len(rets) > 1 else 0.0
    span_days = (eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days
    span_years = max(span_days / 365.25, 1e-9)
    ann_ret = (1 + total_return) ** (1 / span_years) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = float(((eq_df["equity"] - cummax) / cummax).min())

    return {
        "total_return": round(float(total_return), 4),
        "ann_return": round(float(ann_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(max_dd, 4),
        "n_trades": n_trades,
        "guard_hits": guard_hits,
        "n_rebalance": len(equity_history),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("  V3基线 vs V3+RF跌概率减仓层 A/B 验证")
    print(f"  减仓阈值: {GUARD_THRESHOLDS} | 时序: 每OOS段用段前数据训练RF (无未来函数)")
    print("=" * 74)

    # 1. 数据与特征面板
    data = load_data()
    dates = common_dates(data)
    mat = close_matrix(data, dates)
    vol_mat = {c: data[c].set_index("trade_date")["volume"].reindex(dates).values for c in ETF_POOL}
    print("\n  构建特征面板...")
    x, meta, feat_names = build_features(mat, vol_mat, dates)
    order = sorted(range(len(meta)), key=lambda i: meta[i]["date"])
    x = x[order]
    meta = [meta[i] for i in order]
    y_dn = np.array([m["y_dn"] for m in meta])
    dates_s = [m["date"] for m in meta]
    print(f"  样本: {len(meta)} | 特征: {len(feat_names)}")

    # 2. OOS 段
    max_start = len(dates) - TRAIN_DAYS - BUFFER_DAYS - TEST_DAYS
    starts = list(range(0, max_start, STEP))[:4]
    if len(starts) < 4:
        print(f"  ⚠️ 数据不足, 实际段数: {len(starts)}")

    # 3. 逐段 A/B
    configs = ["V3基线", "V3+RF(0.50)", "V3+RF(0.55)", "V3+RF(0.60)"]
    agg = {c: {"ann": [], "sharpe": [], "dd": [], "trades": []} for c in configs}
    windows_out = []

    from sklearn.ensemble import RandomForestClassifier

    for wi, s0 in enumerate(starts, 1):
        test_start = s0 + TRAIN_DAYS + BUFFER_DAYS
        test_end = test_start + TEST_DAYS
        print(f"\n  ── 段{wi}: 测试 {dates[test_start]}~{dates[test_end - 1]}")

        # 训练 RF (段前全部样本, 无未来函数)
        test_start_date = str(dates[test_start])
        tr_mask = np.array([d < test_start_date for d in dates_s])
        # 测试段日期集合
        te_dates_set = {str(dates[i]) for i in range(test_start, test_end)}
        te_mask = np.array([d in te_dates_set for d in dates_s])

        if tr_mask.sum() < 500:
            print("    ⚠️ 训练样本不足, 跳过")
            continue
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=50, random_state=42, n_jobs=-1
        )
        rf.fit(x[tr_mask], y_dn[tr_mask])
        proba = rf.predict_proba(x)[:, 1]  # 全样本概率 (段内取用)

        # 构建 (date, code) → 跌概率 映射 (仅测试段)
        proba_map = {}
        for i in np.where(te_mask)[0]:
            proba_map[(meta[i]["date"], meta[i]["code"])] = float(proba[i])

        # 各配置回测
        print(f"    {'配置':<14} {'年化':>8} {'夏普':>6} {'回撤':>8} {'交易':>4} {'减仓':>4}")
        row = {
            "window": f"W{wi}",
            "test_span": [str(dates[test_start]), str(dates[test_end - 1])],
            "n_train": int(tr_mask.sum()),
            "configs": {},
        }
        for cfg, thr in zip(configs, [None, *GUARD_THRESHOLDS], strict=False):
            res = run_v3_window(
                data,
                dates,
                test_start,
                test_end,
                rf_proba=proba_map if thr is not None else None,
                guard_thr=thr,
            )
            if "error" in res:
                print(f"    {cfg:<14} ERROR: {res['error']}")
                continue
            agg[cfg]["ann"].append(res["ann_return"])
            agg[cfg]["sharpe"].append(res["sharpe"])
            agg[cfg]["dd"].append(res["max_drawdown"])
            agg[cfg]["trades"].append(res["n_trades"])
            row["configs"][cfg] = res
            print(
                f"    {cfg:<14} {res['ann_return']:>+8.1%} {res['sharpe']:>6.2f} "
                f"{res['max_drawdown']:>8.1%} {res['n_trades']:>4} {res['guard_hits']:>4}"
            )
        windows_out.append(row)

    # 4. 汇总
    print("\n" + "=" * 74)
    print("  汇总 (OOS段平均)")
    print("=" * 74)
    print(f"  {'配置':<14} {'年化':>8} {'夏普':>6} {'回撤':>8} {'交易/段':>8}")
    summary = {}
    for cfg in configs:
        a = agg[cfg]
        if not a["ann"]:
            continue
        ann = float(np.mean(a["ann"]))
        shp = float(np.mean(a["sharpe"]))
        dd = float(np.mean(a["dd"]))
        tr = float(np.mean(a["trades"]))
        summary[cfg] = {
            "ann": round(ann, 4),
            "sharpe": round(shp, 3),
            "max_dd": round(dd, 4),
            "trades": round(tr, 1),
        }
        print(f"  {cfg:<14} {ann:>+8.1%} {shp:>6.2f} {dd:>8.1%} {tr:>8.0f}")

    # 判定: 减仓层 vs 基线
    base = summary.get("V3基线")
    if base:
        print("\n  减仓层收益/风险权衡:")
        for cfg in configs[1:]:
            s = summary.get(cfg)
            if not s:
                continue
            d_ann = (s["ann"] - base["ann"]) * 100
            d_dd = (s["max_dd"] - base["max_dd"]) * 100
            d_shp = s["sharpe"] - base["sharpe"]
            print(
                f"    {cfg:<14} 年化 {d_ann:+.1f}pp | 夏普 {d_shp:+.2f} | "
                f"回撤 {d_dd:+.1f}pp {'(改善)' if d_dd > 0 else '(恶化)'}"
            )

    out = {
        "meta": {
            "thresholds": list(GUARD_THRESHOLDS),
            "rf": "RandomForest(n=300, depth=6, leaf=50), 22动态特征, 段前训练",
            "windows": starts,
            "pass_note": "A/B对比: 减仓层是否在保住年化的前提下改善夏普/回撤",
        },
        "windows": windows_out,
        "summary": summary,
    }
    out_path = OUTPUT_DIR / "v3_rf_guard_ab.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
