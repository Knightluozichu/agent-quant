"""6年短期窗口价格变动模式统计实验.

为三队专家研究(跌征兆/涨征兆/防过拟合)提供统一口径的结构化统计:
  1. 窗口收益分布: R1/R2/R3/R5/R10 的均值/中位数/分位数/正收益概率/偏度
  2. 涨跌持续性: 连涨/连跌天数分布 + 次日续涨/续跌条件概率
  3. 大涨/大跌事件: |r|>3% 与 |r|>5% 事件后 1/3/5 日条件收益
  4. 短期动量分位组: R3 五分位组别的未来 3/5/10 日收益 (持续性 vs 反转)
  5. 类别轮动: 商品/海外/A股/债券 5日窗口相对强弱 + 强弱持续性 + spread
  6. 暴涨/暴跌前兆: 事件前5日特征 (前期涨跌/连涨天数/波动率/距高点距离)
  7. 波动率: 20日年化波动分布、聚类特征、高/低波动后的收益
  8. 已有规则验证: 单日暴跌过滤 / A股MA过滤 / 动量符号 的有效性

数据: data/cross_asset/*.parquet (与回测同源前复权)
区间: 最近6年 (公共交易日, 2020-08-03 ~ 2026-08-03)
输出: data/v9_results/short_window_patterns.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cross_asset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ETF_POOL = {
    "518880": "黄金ETF",
    "159985": "豆粕ETF",
    "501018": "南方原油",
    "161226": "白银LOF",
    "513100": "纳指ETF",
    "159915": "创业板ETF",
    "511220": "城投债ETF",
}
DEFENSE = "511880"
CATEGORIES = {
    "商品": ["518880", "159985", "501018", "161226"],
    "海外": ["513100"],
    "A股": ["159915"],
    "债券": ["511220"],
}
WINDOWS = (1, 2, 3, 5, 10)
PERIOD_START = "2020-08-03"  # 6年窗口起点
BIG_MOVE = 0.03
EXTREME_MOVE = 0.05


def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for code in [*list(ETF_POOL.keys()), DEFENSE]:
        f = DATA_DIR / f"{code}.parquet"
        if f.exists():
            df = pd.read_parquet(f).sort_values("trade_date").reset_index(drop=True)
            data[code] = df
    return data


def common_dates(data: dict) -> list:
    """公共交易日 (含货币基金)."""
    common: set = set()
    for code in data:
        dates = set(data[code]["trade_date"].tolist())
        common = dates if not common else common & dates
    return sorted(d for d in common if d >= pd.Timestamp(PERIOD_START).date())


def close_matrix(data: dict, dates: list) -> dict[str, np.ndarray]:
    """每资产对齐公共日历后的收盘序列."""
    mat = {}
    for code, df in data.items():
        idx = df.set_index("trade_date")["close"]
        mat[code] = idx.reindex(dates).astype(float).values
    return mat


def daily_rets(close: np.ndarray) -> np.ndarray:
    return close[1:] / close[:-1] - 1.0


def window_ret(close: np.ndarray, n: int) -> np.ndarray:
    """滚动 n 日收益序列 R_n(t), 长度 = len(close) - n."""
    return close[n:] / close[:-n] - 1.0


def stats_of(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    return {
        "n": len(x),
        "mean": round(float(np.mean(x)), 6),
        "std": round(float(np.std(x)), 6),
        "skew": round(float(pd.Series(x).skew()), 4),
        "q05": round(float(np.quantile(x, 0.05)), 6),
        "q25": round(float(np.quantile(x, 0.25)), 6),
        "median": round(float(np.median(x)), 6),
        "q75": round(float(np.quantile(x, 0.75)), 6),
        "q95": round(float(np.quantile(x, 0.95)), 6),
        "pct_pos": round(float((x > 0).mean()), 4),
        "min": round(float(x.min()), 6),
        "max": round(float(x.max()), 6),
    }


# --------------------------------------------------------------------------- #
# 1. 窗口收益分布
# --------------------------------------------------------------------------- #
def window_stats(close: np.ndarray) -> dict:
    return {f"r{n}": stats_of(window_ret(close, n)) for n in WINDOWS}


# --------------------------------------------------------------------------- #
# 2. 涨跌持续性 (连涨/连跌)
# --------------------------------------------------------------------------- #
def streak_analysis_v2(dr: np.ndarray) -> dict:
    """连涨/连跌: 长度分布 + 次日续涨概率 (简洁实现)."""
    out = {"up": {}, "down": {}}
    for direction, key in ((1, "up"), (-1, "down")):
        # 统计连续同向段长度分布
        segs = []
        cur = 1
        for i in range(1, len(dr)):
            same = ((dr[i] > 0) == (direction == 1)) and ((dr[i - 1] > 0) == (direction == 1))
            if same:
                cur += 1
            else:
                if (dr[i - 1] > 0) == (direction == 1):
                    segs.append(cur)
                cur = 1
        if (dr[-1] > 0) == (direction == 1):
            segs.append(cur)
        # 长度分布 + 次日续涨条件概率
        segs_arr = np.array(segs)
        for seg_len in range(1, 6):
            cnt = int((segs_arr == seg_len).sum())
            total = len(segs_arr)
            # 条件概率: 连涨/连跌>=L天后, 次日继续同向
            cont = 0
            n_cont = 0
            for i in range(seg_len, len(dr)):
                ok = True
                for j in range(1, seg_len + 1):
                    if (dr[i - j] > 0) != (direction == 1):
                        ok = False
                        break
                if ok:
                    n_cont += 1
                    if (dr[i] > 0) == (direction == 1):
                        cont += 1
            out[key][f"len{seg_len}"] = {
                "seg_count": cnt,
                "seg_pct": round(cnt / max(total, 1), 4),
                "next_n": n_cont,
                "continue_pct": round(cont / n_cont, 4) if n_cont else None,
            }
    return out


# --------------------------------------------------------------------------- #
# 3. 大涨/大跌事件后条件收益
# --------------------------------------------------------------------------- #
def big_move_analysis(close: np.ndarray, dr: np.ndarray) -> dict:
    out = {}
    for label, thr in (("up3", BIG_MOVE), ("down3", -BIG_MOVE),
                       ("up5", EXTREME_MOVE), ("down5", -EXTREME_MOVE)):
        ev = {}
        for h in (1, 3, 5):
            fwd = []
            for t in range(len(dr)):
                hit = (dr[t] >= thr) if thr > 0 else (dr[t] <= thr)
                if hit and t + h < len(dr):
                    r = close[t + h] / close[t] - 1.0
                    fwd.append(r)
            if fwd:
                fwd = np.array(fwd)
                ev[f"after{h}d"] = {
                    "n": len(fwd),
                    "mean": round(float(fwd.mean()), 6),
                    "median": round(float(np.median(fwd)), 6),
                    "win_rate": round(float((fwd > 0).mean()), 4),
                    "q25": round(float(np.quantile(fwd, 0.25)), 6),
                    "q75": round(float(np.quantile(fwd, 0.75)), 6),
                }
        out[label] = ev
    return out


# --------------------------------------------------------------------------- #
# 4. 短期动量分位组 (R3/R5 → 未来收益)
# --------------------------------------------------------------------------- #
def quintile_analysis(close: np.ndarray) -> dict:
    """R3/R5 五分位 → 未来 3/5/10 日收益 (检验短期动量持续还是反转).

    严格无重叠: 动量形成期 [t, t+mom_w) 结束后, 持有期 [t+mom_w, t+mom_w+h].
    """
    out = {}
    for mom_w in (3, 5):
        mom = window_ret(close, mom_w)  # mom[t] = close[t+mom_w]/close[t]-1
        entry = {}
        for h in (3, 5, 10):
            # 未来 h 日收益 (动量结束后才开始): fwd[t] = close[t+mom_w+h]/close[t+mom_w]-1
            fwd = close[mom_w + h:] / close[mom_w:-h] - 1.0
            m = mom[:len(fwd)]
            qs = pd.qcut(m, 5, labels=False, duplicates="drop")
            groups = {}
            for g in range(5):
                sel = fwd[qs == g]
                if len(sel):
                    groups[f"q{g + 1}"] = {
                        "n": len(sel),
                        "fwd_mean": round(float(sel.mean()), 6),
                        "win_rate": round(float((sel > 0).mean()), 4),
                    }
            entry[f"h{h}"] = groups
        out[f"mom{mom_w}"] = entry
    return out


# --------------------------------------------------------------------------- #
# 5. 类别轮动
# --------------------------------------------------------------------------- #
def category_rotation(close_mat: dict, dates: list) -> dict:
    """4类资产 5日窗口强弱: 周频轮动持续性与 spread 均值回归."""
    # 每资产 5日收益序列 (按公共日历)
    r5: dict[str, np.ndarray] = {}
    for code, close in close_mat.items():
        r5[code] = window_ret(close, 5)
    n = len(dates) - 5
    # 类别 5日收益 (等权)
    cat_r5: dict[str, np.ndarray] = {}
    for cat, codes in CATEGORIES.items():
        arr = np.mean([r5[c] for c in codes if c in r5], axis=0)
        cat_r5[cat] = arr

    # 周频样本 (每5个交易日取1)
    idxs = list(range(0, n, 5))
    # 强弱持续性: 本周最强类别 → 下周类别收益
    persist = []
    for i in range(len(idxs) - 1):
        t = idxs[i]
        t2 = idxs[i + 1]
        if t2 >= n:
            break
        ranks = sorted(cat_r5, key=lambda k: -cat_r5[k][t])
        best_cat = ranks[0]
        worst_cat = ranks[-1]
        persist.append({
            "best_cat": best_cat, "best_next": float(cat_r5[best_cat][t2]),
            "worst_cat": worst_cat, "worst_next": float(cat_r5[worst_cat][t2]),
        })
    dfp = pd.DataFrame(persist)
    best_next = dfp["best_next"].values
    worst_next = dfp["worst_next"].values
    cat_names = list(cat_r5.keys())

    # spread: 最强-最弱类别 5日窗口
    spread = np.array([np.ptp([cat_r5[k][t] for k in cat_names]) for t in range(n)])
    spread_now = spread[idxs[:-1]]
    spread_fwd = []
    for i in range(len(idxs) - 1):
        t2 = idxs[i + 1]
        if t2 < n:
            spread_fwd.append(spread[t2])
    spread_fwd = np.array(spread_fwd)

    # 高spread(>4%)后收敛检验
    high_sp = spread_now > 0.04
    low_sp = spread_now < 0.02
    high_fwd = spread_fwd[high_sp]
    low_fwd = spread_fwd[low_sp]

    return {
        "category_5d_stats": {k: stats_of(cat_r5[k]) for k in cat_names},
        "best_cat_next_5d": stats_of(best_next),
        "worst_cat_next_5d": stats_of(worst_next),
        "best_minus_worst_next_5d": {
            "mean": round(float((best_next - worst_next).mean()), 6),
            "win_rate": round(float(((best_next - worst_next) > 0).mean()), 4),
        },
        "spread_5d": stats_of(spread),
        "spread_high_persistence": {
            "n": len(high_fwd),
            "mean_fwd_spread": round(float(high_fwd.mean()), 6),
            "mean_fwd_diff_from_all": round(float(high_fwd.mean() - spread.mean()), 6),
        },
        "spread_low_persistence": {
            "n": len(low_fwd),
            "mean_fwd_spread": round(float(low_fwd.mean()), 6),
            "mean_fwd_diff_from_all": round(float(low_fwd.mean() - spread.mean()), 6),
        },
        "per_cat_best_freq": {
            k: round(float((dfp["best_cat"] == k).mean()), 4) for k in cat_names
        },
    }


# --------------------------------------------------------------------------- #
# 6. 暴涨/暴跌前兆特征 (池级合并样本)
# --------------------------------------------------------------------------- #
def precursor_analysis(close_mat: dict) -> dict:
    """事件前5日特征 vs 全样本: 前期涨跌/连涨天数/20日波动/距20日高点."""
    feats = {
        "prev5_ret": [],      # 前5日累计
        "up_streak": [],      # 事件前连涨天数
        "vol20": [],          # 20日年化波动
        "dist_high20": [],    # 距20日高点 (0=最高)
        "dist_low20": [],     # 距20日低点
        "label": [],          # 1=暴涨>3%, -1=暴跌<-3%, 0=普通日
    }
    for close in close_mat.values():
        dr = daily_rets(close)
        for t in range(21, len(dr)):
            r5 = close[t] / close[t - 5] - 1.0
            vol = np.std(dr[t - 20:t]) * np.sqrt(252)
            win20 = close[t - 20:t + 1]
            dist_high = close[t] / win20.max() - 1.0
            dist_low = close[t] / win20.min() - 1.0
            up_streak = 0
            for j in range(t, 0, -1):
                if dr[j - 1] > 0:
                    up_streak += 1
                else:
                    break
            label = 1 if dr[t] > BIG_MOVE else (-1 if dr[t] < -BIG_MOVE else 0)
            feats["prev5_ret"].append(r5)
            feats["up_streak"].append(min(up_streak, 5))
            feats["vol20"].append(vol)
            feats["dist_high20"].append(dist_high)
            feats["dist_low20"].append(dist_low)
            feats["label"].append(label)

    out = {}
    arr = {k: np.array(v) for k, v in feats.items()}
    for label, name in ((1, "up3"), (-1, "down3"), (0, "normal")):
        sel = arr["label"] == label
        out[name] = {
            "n": int(sel.sum()),
            "prev5_ret_mean": round(float(arr["prev5_ret"][sel].mean()), 6),
            "prev5_ret_median": round(float(np.median(arr["prev5_ret"][sel])), 6),
            "up_streak_mean": round(float(arr["up_streak"][sel].mean()), 4),
            "vol20_mean": round(float(arr["vol20"][sel].mean()), 6),
            "dist_high20_mean": round(float(arr["dist_high20"][sel].mean()), 6),
            "dist_low20_mean": round(float(arr["dist_low20"][sel].mean()), 6),
        }
    # 事件后表现 (暴涨后/暴跌后 5日)
    after = {"up3": [], "down3": []}
    for close in close_mat.values():
        dr = daily_rets(close)
        for t in range(21, len(dr) - 5):
            if dr[t] > BIG_MOVE:
                after["up3"].append(close[t + 5] / close[t] - 1.0)
            elif dr[t] < -BIG_MOVE:
                after["down3"].append(close[t + 5] / close[t] - 1.0)
    for k, v in after.items():
        if v:
            v = np.array(v)
            out[k]["after5d_mean"] = round(float(v.mean()), 6)
            out[k]["after5d_win"] = round(float((v > 0).mean()), 4)
    return out


# --------------------------------------------------------------------------- #
# 7. 波动率特征
# --------------------------------------------------------------------------- #
def vol_analysis(close_mat: dict) -> dict:
    out = {"per_asset": {}, "regime": {}}
    all_vol = []
    all_fwd = []
    all_labels = []
    for code, close in close_mat.items():
        dr = daily_rets(close)
        vols = []
        for t in range(20, len(dr)):
            vols.append(np.std(dr[t - 20:t]) * np.sqrt(252))
        vols = np.array(vols)
        out["per_asset"][code] = stats_of(vols)
        for t in range(20, len(dr) - 5):
            v = np.std(dr[t - 20:t]) * np.sqrt(252)
            fwd = close[t + 5] / close[t] - 1.0
            all_vol.append(v)
            all_fwd.append(fwd)
            all_labels.append(1 if v > 0.30 else (0 if v < 0.15 else 2))  # 高/低/中
    av = np.array(all_vol)
    af = np.array(all_fwd)
    al = np.array(all_labels)
    out["regime"] = {
        "high_vol_gt30": {
            "n": int((al == 1).sum()),
            "fwd5_mean": round(float(af[al == 1].mean()), 6),
            "fwd5_win": round(float((af[al == 1] > 0).mean()), 4),
        },
        "low_vol_lt15": {
            "n": int((al == 0).sum()),
            "fwd5_mean": round(float(af[al == 0].mean()), 6),
            "fwd5_win": round(float((af[al == 0] > 0).mean()), 4),
        },
        "vol_autocorr_5d": round(float(pd.Series(av).autocorr(5)), 4),
    }
    return out


# --------------------------------------------------------------------------- #
# 8. 已有规则验证
# --------------------------------------------------------------------------- #
def rule_validation(close_mat: dict) -> dict:
    """验证V3策略的三个规则是否有效 (未来5日收益对比)."""
    out = {}
    # 8.1 单日暴跌过滤: 近5日有单日<-3% 的资产, 未来5日收益
    f_drop, f_clean = [], []
    for close in close_mat.values():
        dr = daily_rets(close)
        for t in range(5, len(dr) - 5):
            recent = dr[t - 5:t]
            fwd = close[t + 5] / close[t] - 1.0
            if (recent < -BIG_MOVE).any():
                f_drop.append(fwd)
            else:
                f_clean.append(fwd)
    out["drop_filter"] = {
        "dropped_fwd5": stats_of(np.array(f_drop)) if f_drop else {},
        "clean_fwd5": stats_of(np.array(f_clean)) if f_clean else {},
        "diff_mean": round(float(np.mean(f_clean) - np.mean(f_drop)), 6),
    }
    # 8.2 动量符号: 10日动量>0 vs <0 的未来5日收益 (严格无重叠)
    # 注: 原实现 fwd=window_ret(close,5)[5:] 使持有终点与动量终点重合 (终点偏差),
    #     高估了 mom_sign 判别力; 现改为动量形成期结束后才开始持有.
    f_pos, f_neg = [], []
    for close in close_mat.values():
        m10 = window_ret(close, 10)          # m10[t] = close[t+10]/close[t]-1
        fwd = close[10 + 5:] / close[10:-5] - 1.0  # 持有 [t+10, t+15], 与动量期无重叠
        m = m10[:len(fwd)]
        f_pos.append(fwd[m > 0])
        f_neg.append(fwd[m <= 0])
    f_pos = np.concatenate(f_pos) if f_pos else np.array([])
    f_neg = np.concatenate(f_neg) if f_neg else np.array([])
    out["mom_sign"] = {
        "mom_pos_fwd5": stats_of(f_pos) if len(f_pos) else {},
        "mom_neg_fwd5": stats_of(f_neg) if len(f_neg) else {},
        "diff_mean": round(float(f_pos.mean() - f_neg.mean()), 6),
        "note": "严格无重叠口径 (动量期结束后持有5日), 原重叠口径差+3.09pp为终点偏差"
    }
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 70)
    print("  6年短期窗口价格变动模式统计实验")
    print(f"  区间: {PERIOD_START} ~ (数据末日) | 池: {len(ETF_POOL)}只风险资产 + 货币基金")
    print("=" * 70)

    data = load_data()
    dates = common_dates(data)
    print(f"\n  公共交易日: {len(dates)} 天 ({dates[0]} ~ {dates[-1]})")

    mat = close_matrix(data, dates)
    # 移除防御资产(不参与动量统计), 但保留在矩阵中供对比
    risk = {c: mat[c] for c in ETF_POOL}

    result: dict = {"meta": {
        "period": f"{dates[0]} ~ {dates[-1]}",
        "n_days": len(dates),
        "pool": ETF_POOL,
        "categories": CATEGORIES,
    }}

    # 1. 窗口收益分布
    print("\n[1/8] 窗口收益分布...")
    result["window_stats"] = {c: window_stats(close) for c, close in risk.items()}

    # 2. 涨跌持续性
    print("[2/8] 涨跌持续性 (连涨/连跌)...")
    result["streak"] = {
        c: streak_analysis_v2(daily_rets(close)) for c, close in risk.items()
    }

    # 3. 大涨/大跌事件
    print("[3/8] 大涨/大跌事件后表现...")
    result["big_move"] = {
        c: big_move_analysis(close, daily_rets(close)) for c, close in risk.items()
    }

    # 4. 短期动量分位组
    print("[4/8] 短期动量分位组 (R3/R5 → 未来收益)...")
    result["quintile"] = {c: quintile_analysis(close) for c, close in risk.items()}

    # 5. 类别轮动
    print("[5/8] 类别轮动...")
    result["category"] = category_rotation(risk, dates)

    # 6. 暴涨/暴跌前兆
    print("[6/8] 暴涨/暴跌前兆特征...")
    result["precursor"] = precursor_analysis(risk)

    # 7. 波动率
    print("[7/8] 波动率特征...")
    result["vol"] = vol_analysis(risk)

    # 8. 规则验证
    print("[8/8] 已有规则验证...")
    result["rule_validation"] = rule_validation(risk)

    out_path = OUTPUT_DIR / "short_window_patterns.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")

    # === 控制台摘要 ===
    print("\n" + "=" * 70)
    print("  关键发现预览")
    print("=" * 70)
    print("\n  [窗口收益 5日正收益概率]")
    for c, st in result["window_stats"].items():
        r5 = st.get("r5", {})
        print(f"    {ETF_POOL[c]:<8} pct_pos={r5.get('pct_pos', 0):.1%}  "
              f"mean={r5.get('mean', 0):+.2%}  median={r5.get('median', 0):+.2%}")

    print("\n  [涨跌持续性] 连涨3天后次日续涨概率 vs 连跌3天后次日续跌概率")
    for c, st in result["streak"].items():
        up3 = st["up"].get("len3", {}).get("continue_pct")
        dn3 = st["down"].get("len3", {}).get("continue_pct")
        print(f"    {ETF_POOL[c]:<8} 连涨3续 {up3 if up3 is None else f'{up3:.1%}':>6}  "
              f"连跌3续 {dn3 if dn3 is None else f'{dn3:.1%}':>6}")

    print("\n  [动量分位] R3最强组(Q5) vs 最弱组(Q1) 未来5日收益 (持续性检验)")
    for c, qt in result["quintile"].items():
        g = qt["mom3"]["h5"]
        q1, q5 = g.get("q1", {}), g.get("q5", {})
        if q1 and q5:
            diff = q5.get("fwd_mean", 0) - q1.get("fwd_mean", 0)
            print(f"    {ETF_POOL[c]:<8} Q5={q5.get('fwd_mean', 0):+.2%}  "
                  f"Q1={q1.get('fwd_mean', 0):+.2%}  差值={diff:+.2%}")

    print("\n  [类别轮动] 最强类别下周收益 vs 最弱类别")
    cat = result["category"]
    print(f"    最强类别下周: mean={cat['best_cat_next_5d'].get('mean', 0):+.2%} "
          f"win={cat['best_cat_next_5d'].get('win_rate', 0):.1%}")
    print(f"    最弱类别下周: mean={cat['worst_cat_next_5d'].get('mean', 0):+.2%}")
    print(f"    强弱差: mean={cat['best_minus_worst_next_5d'].get('mean', 0):+.2%} "
          f"win={cat['best_minus_worst_next_5d'].get('win_rate', 0):.1%}")

    print("\n  [暴涨/暴跌前兆] (池级合并)")
    pre = result["precursor"]
    for k, name in (("up3", "暴涨>3%"), ("down3", "暴跌<-3%"), ("normal", "普通日")):
        p = pre[k]
        if "after5d_mean" in p:
            print(f"    {name}: n={p['n']:>6}  前5日={p['prev5_ret_mean']:+.2%}  "
                  f"连涨={p['up_streak_mean']:.2f}天  vol20={p['vol20_mean']:.0%}  "
                  f"距高={p['dist_high20_mean']:+.1%}  后5日={p['after5d_mean']:+.2%}")
        else:
            print(f"    {name}: n={p['n']:>6}  前5日={p['prev5_ret_mean']:+.2%}")

    print("\n  [规则验证]")
    rv = result["rule_validation"]
    df_ = rv["drop_filter"]
    print(f"    暴跌过滤: 干净池后5日 {df_['clean_fwd5'].get('mean', 0):+.2%} vs "
          f"暴跌池 {df_['dropped_fwd5'].get('mean', 0):+.2%} (差 {df_['diff_mean']:+.2%})")
    ms = rv["mom_sign"]
    print(f"    动量符号: 动量>0后5日 {ms['mom_pos_fwd5'].get('mean', 0):+.2%} vs "
          f"动量<0 {ms['mom_neg_fwd5'].get('mean', 0):+.2%} (差 {ms['diff_mean']:+.2%})")

    print("\n  完成.")


if __name__ == "__main__":
    main()
