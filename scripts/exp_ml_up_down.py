"""涨/跌征兆机器学习研究 — 动态特征 + 多模型 + 时序验证.

回应研究方法的动态化要求:
  - 特征全部为滚动计算的动态信号 (多尺度动量/波动/量能/连涨连跌/距高低点/
    市场环境/类别相对强弱), 不使用固定窗口决策, 无未来函数
  - 涨模型: 预测未来5日收益>0; 跌模型: 预测未来5日收益<0 (分开训练, 分别找特征)
  - 模型: RandomForest / sklearn GBDT(HistGB) / XGBoost / LightGBM
  - 验证: TimeSeriesSplit 时序 walk-forward (训练过去→预测未来, 非随机KFold),
    与多数类基线/动量规则基线对比增量价值
  - 特征重要性: 涨/跌模型分别输出 top10

数据: data/cross_asset/*.parquet 公共日历 (2020-08-03 ~ 2026-08-03)
输出: data/v9_results/ml_up_down.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from exp_short_window_patterns import (  # noqa: E402
    CATEGORIES,
    ETF_POOL,
    close_matrix,
    common_dates,
    load_data,
)

WARMUP = 60  # 特征需 mom60
FWD = 5  # 预测未来5日
N_SPLITS = 4  # 时序折数
RANDOM_STATE = 42

# === 类别映射 (商品/海外/A股/债券) ===
CAT_OF: dict[str, str] = {}
for cat, codes in CATEGORIES.items():
    for c in codes:
        CAT_OF[c] = cat


# --------------------------------------------------------------------------- #
# 特征工程 (滚动动态特征, 截至 t 日, 无未来函数)
# --------------------------------------------------------------------------- #
def _streak(arr: np.ndarray, direction: int) -> int:
    """截至最后一天的连续同向天数 (direction: 1涨/-1跌)."""
    n = 0
    for i in range(len(arr) - 1, 0, -1):
        if (arr[i] > 0) == (direction == 1):
            n += 1
        else:
            break
    return n


def build_features(mat: dict, vol_mat: dict, dates: list) -> tuple[np.ndarray, list]:
    """构建 {样本: [date, code, 特征...]} 面板.

    Returns:
        X: [n, n_feat]; meta: [n] 每行 {date, code, y5}
    """
    codes = [c for c in ETF_POOL if c in mat]  # 仅风险资产 (排除货币基金)
    feat_names = [
        "mom3",
        "mom5",
        "mom10",
        "mom20",
        "mom60",
        "vol5",
        "vol20",
        "vol60",
        "mom_short_dev",  # mom3 - mom20 (短期背离)
        "up_streak",
        "dn_streak",  # 动态连涨/连跌天数
        "dist_high20",
        "dist_low20",
        "dist_high60",
        "vol_ratio",  # 5日均量/20日均量
        "ret_vol5",  # 近5日收益×近5日波动 (量价风格)
        "cyb_ma_state",  # 创业板>MA20 (A股环境)
        "pool_mom5",  # 全池平均5日动量 (市场)
        "a_share_ret5",  # 创业板5日动量 (风险偏好)
        "cat_rel_mom5",  # 资产5日动量 - 同类别均值
        "pool_rel_mom5",  # 资产5日动量 - 全池均值
    ]
    n = len(dates)
    # 跨资产市场特征 (逐日)
    cyb = mat["159915"]
    cyb_ma20 = np.array([np.mean(cyb[max(0, t - 19) : t + 1]) for t in range(n)])
    cyb_ma_state = (cyb > cyb_ma20).astype(float)
    a_share_ret5 = np.array([cyb[t] / cyb[t - 5] - 1 if t >= 5 else 0.0 for t in range(n)])
    pool_mom5_all = np.mean([mat[c][5:] / mat[c][:-5] - 1 for c in codes], axis=0)
    pool_mom5 = np.concatenate([[0.0] * 5, pool_mom5_all])

    x_rows, meta = [], []
    for code in codes:
        close = mat[code].astype(float)
        volume = vol_mat[code].astype(float)
        cat = CAT_OF.get(code, "A股")
        cat_codes = CATEGORIES[cat]
        for t in range(WARMUP, n - FWD):
            y5 = close[t + FWD] / close[t] - 1.0
            feats = {}
            for w, k in ((3, "mom3"), (5, "mom5"), (10, "mom10"), (20, "mom20"), (60, "mom60")):
                feats[k] = close[t] / close[t - w] - 1.0
            dr = np.diff(close[t - 60 : t + 1]) / close[t - 60 : t]
            for w, k in ((5, "vol5"), (20, "vol20"), (60, "vol60")):
                feats[k] = np.std(dr[-w:]) * np.sqrt(252)
            feats["mom_short_dev"] = feats["mom3"] - feats["mom20"]
            feats["up_streak"] = float(_streak(np.diff(close[t - 5 : t + 1]) / close[t - 5 : t], 1))
            feats["dn_streak"] = float(
                _streak(np.diff(close[t - 5 : t + 1]) / close[t - 5 : t], -1)
            )
            win20 = close[t - 19 : t + 1]
            win60 = close[t - 59 : t + 1]
            feats["dist_high20"] = close[t] / win20.max() - 1.0
            feats["dist_low20"] = close[t] / win20.min() - 1.0
            feats["dist_high60"] = close[t] / win60.max() - 1.0
            v5 = volume[t - 4 : t + 1].mean()
            v20 = volume[t - 19 : t + 1].mean()
            feats["vol_ratio"] = v5 / v20 if v20 > 0 else 1.0
            feats["ret_vol5"] = feats["mom5"] * feats["vol5"]
            feats["cyb_ma_state"] = cyb_ma_state[t]
            feats["pool_mom5"] = pool_mom5[t]
            feats["a_share_ret5"] = a_share_ret5[t]
            cat_mom = np.mean([mat[c][t] / mat[c][t - 5] - 1 for c in cat_codes])
            feats["cat_rel_mom5"] = feats["mom5"] - cat_mom
            feats["pool_rel_mom5"] = feats["mom5"] - pool_mom5[t]
            x_rows.append([feats[k] for k in feat_names])
            meta.append(
                {
                    "date": str(dates[t]),
                    "code": code,
                    "y5": float(y5),
                    "y_up": float(y5 > 0),
                    "y_dn": float(y5 < 0),
                }
            )
    return np.array(x_rows, dtype=float), meta, feat_names


# --------------------------------------------------------------------------- #
# 模型工厂 (失败自动跳过)
# --------------------------------------------------------------------------- #
def make_models() -> dict[str, object]:
    models = {}
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE, n_jobs=-1
    )
    models["GBDT(HistGB)"] = HistGradientBoostingClassifier(
        max_depth=4,
        l2_regularization=1.0,
        max_iter=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        print("  ⚠️ xgboost 未安装, 跳过")
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
    except ImportError:
        print("  ⚠️ lightgbm 未安装, 跳过")
    return models


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score

    y_pred = (y_prob > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "auc": round(float(auc), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "pos_rate": round(float(y_true.mean()), 4),
    }


def importance_report(model, feat_names: list, top_k: int = 10) -> list[dict]:
    try:
        imp = model.feature_importances_
    except AttributeError:
        return []
    order = np.argsort(-imp)
    return [
        {"feature": feat_names[i], "importance": round(float(imp[i]), 5)} for i in order[:top_k]
    ]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("  涨/跌征兆 ML 研究 — 动态特征 + 4模型 + 时序验证")
    print("  预测目标: 未来5日收益>0 (涨) / <0 (跌) | 特征: 滚动动态信号")
    print("=" * 74)

    data = load_data()
    dates = common_dates(data)
    mat = close_matrix(data, dates)
    vol_mat = {c: data[c].set_index("trade_date")["volume"].reindex(dates).values for c in ETF_POOL}
    print(f"\n  构建特征面板: {len(dates)} 交易日 × {len(ETF_POOL)} 资产...")
    x, meta, feat_names = build_features(mat, vol_mat, dates)
    # 按日期重排 (面板按资产分组构建, 时序切分必须时间有序)
    order = sorted(range(len(meta)), key=lambda i: meta[i]["date"])
    x = x[order]
    meta = [meta[i] for i in order]
    y_up = np.array([m["y_up"] for m in meta])
    y_dn = np.array([m["y_dn"] for m in meta])
    print(f"  样本: {len(meta)} | 涨占比 {y_up.mean():.1%} | 跌占比 {y_dn.mean():.1%}")

    models = make_models()
    print(f"  模型: {list(models.keys())}")

    # === 时序 walk-forward 验证 (TimeSeriesSplit) ===
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = {
        "meta": {
            "period": f"{dates[0]} ~ {dates[-1]}",
            "n_samples": len(meta),
            "features": feat_names,
            "fwd_days": FWD,
            "up_rate": round(float(y_up.mean()), 4),
            "dn_rate": round(float(y_dn.mean()), 4),
            "models": list(models.keys()),
        },
        "splits": [],
        "models": {},
    }

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(x), 1):
        print(
            f"\n  ── 折{fold}: 训练至 {meta[tr_idx[-1]]['date']} "
            f"| 测试 {meta[te_idx[0]]['date']}~{meta[te_idx[-1]]['date']}"
        )
        split = {
            "fold": fold,
            "train_span": [meta[tr_idx[0]]["date"], meta[tr_idx[-1]]["date"]],
            "test_span": [meta[te_idx[0]]["date"], meta[te_idx[-1]]["date"]],
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "models": {},
        }
        for name, model in models.items():
            for task, y in (("up", y_up), ("dn", y_dn)):
                params = model.get_params() if hasattr(model, "get_params") else {}
                clf = model.__class__(**params)
                clf.fit(x[tr_idx], y[tr_idx])
                proba = clf.predict_proba(x[te_idx])[:, 1]
                split["models"].setdefault(name, {})[task] = evaluate(y[te_idx], proba)
                results["models"].setdefault(name, {}).setdefault(task, []).append(
                    split["models"][name][task]
                )
        # 基线: 动量规则 (10日动量>0 → 预测涨)
        mom10_te = x[te_idx, feat_names.index("mom10")]
        base_up = evaluate(y_up[te_idx], (mom10_te > 0).astype(float))
        base_dn = evaluate(y_dn[te_idx], (mom10_te < 0).astype(float))
        split["baselines"] = {"mom_rule_up": base_up, "mom_rule_dn": base_dn}
        print(f"    基线(动量规则): 涨AUC {base_up['auc']:.3f} | 跌AUC {base_dn['auc']:.3f}")
        for name in models:
            r = split["models"][name]
            print(
                f"    {name:<15} 涨AUC {r['up']['auc']:.3f} (acc {r['up']['accuracy']:.1%})  "
                f"跌AUC {r['dn']['auc']:.3f} (acc {r['dn']['accuracy']:.1%})"
            )
        results["splits"].append(split)

    # === 汇总 ===
    print("\n" + "=" * 74)
    print("  汇总 (4折平均 AUC)")
    print("=" * 74)
    print(f"  {'模型':<15} {'涨AUC':>8} {'跌AUC':>8} {'涨-基线':>8} {'跌-基线':>8}")
    base_up_avg = np.mean([s["baselines"]["mom_rule_up"]["auc"] for s in results["splits"]])
    base_dn_avg = np.mean([s["baselines"]["mom_rule_dn"]["auc"] for s in results["splits"]])
    summary = {
        "baseline_mom_rule": {
            "up_auc": round(float(base_up_avg), 4),
            "dn_auc": round(float(base_dn_avg), 4),
        }
    }
    for name in models:
        up_auc = np.mean([r["auc"] for r in results["models"][name]["up"]])
        dn_auc = np.mean([r["auc"] for r in results["models"][name]["dn"]])
        print(
            f"  {name:<15} {up_auc:>8.3f} {dn_auc:>8.3f} "
            f"{up_auc - base_up_avg:>+8.3f} {dn_auc - base_dn_avg:>+8.3f}"
        )
        summary[name] = {"up_auc": round(float(up_auc), 4), "dn_auc": round(float(dn_auc), 4)}
    summary["baseline_mom_rule"]["up_auc"] = round(float(base_up_avg), 4)
    summary["baseline_mom_rule"]["dn_auc"] = round(float(base_dn_avg), 4)
    results["summary"] = summary

    # === 特征重要性 (最后一折最大训练集, 涨/跌分别) ===
    print("\n" + "=" * 74)
    print("  特征重要性 Top10 (最后一折训练, 涨/跌模型分别)")
    print("=" * 74)
    tr_idx, te_idx = list(tscv.split(x))[-1]
    imp_summary = {}
    for task, y in (("涨", y_up), ("跌", y_dn)):
        print(f"\n  [{task}模型]")
        feats_rank = {}
        for _name, model in models.items():
            params = model.get_params() if hasattr(model, "get_params") else {}
            clf = model.__class__(**params)
            clf.fit(x[tr_idx], y[tr_idx])
            for item in importance_report(clf, feat_names):
                feats_rank.setdefault(item["feature"], []).append(item["importance"])
        if feats_rank:
            avg_imp = {k: float(np.mean(v)) for k, v in feats_rank.items()}
            order = sorted(avg_imp, key=lambda k: -avg_imp[k])[:10]
            imp_summary[task] = [{"feature": k, "importance": round(avg_imp[k], 5)} for k in order]
            for i, it in enumerate(imp_summary[task], 1):
                print(f"    {i:>2}. {it['feature']:<18} {it['importance']:.4f}")
    results["feature_importance"] = imp_summary

    # 涨/跌模型差异最大特征
    print("\n  涨/跌模型特征分歧 (重要性排名差):")
    if "涨" in imp_summary and "跌" in imp_summary:
        up_rank = {it["feature"]: i for i, it in enumerate(imp_summary["涨"])}
        dn_rank = {it["feature"]: i for i, it in enumerate(imp_summary["跌"])}
        all_f = set(up_rank) | set(dn_rank)
        diff = sorted(
            all_f, key=lambda f: abs(up_rank.get(f, 20) - dn_rank.get(f, 20)), reverse=True
        )[:6]
        for f in diff:
            up_r = up_rank.get(f)
            dn_r = dn_rank.get(f)
            up_s = f"第{up_r + 1}名" if up_r is not None else "-名"
            dn_s = f"第{dn_r + 1}名" if dn_r is not None else "-名"
            print(f"    {f:<18} 涨{up_s} vs 跌{dn_s}")

    out_path = OUTPUT_DIR / "ml_up_down.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
