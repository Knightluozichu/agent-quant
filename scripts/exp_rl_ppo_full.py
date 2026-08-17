"""V3基线 vs RL+CNN — 全量滚动回测 (2020至今, 14:50口径, 无未来函数).

RL 无法从数据起点直接预测 (需训练数据), 采用滚动重训拼接覆盖全区间:
  1. CNN 特征器: 训练 ≤2022-06-30 (2.5年) → 全样本特征表 (CNN 对之后数据为外推, 无泄漏)
  2. PPO 段1: 训练用 2021-01~2022-06 特征交互 → 测试 2022-07-01~2023-12-31 → 期末 F1
  3. PPO 段2: 训练用 2021-01~2023-12 特征交互 → 测试 2024-01-01~2026-08-03 (初始资金=F1)
  4. 拼接两段净值 → 全量曲线 (覆盖 2022-07~2026-08)
  5. 对比 V3: (a) 全量 2019-12~2026-08 (b) 同覆盖区间 2022-07~2026-08

输出: data/v9_results/rl_ppo_full.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from exp_cnn_feats import Featurizer, build_dataset, train_cnn
from exp_rl_ppo import ETFEnv, ppo_eval, ppo_train
from exp_short_window_patterns import close_matrix
from exp_v3_r4_sameday import run_v3_r4_sameday
from run_qixing_v3 import ETF_POOL, load_data

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LR, LAYERS, FILTERS, DROPOUT = 1e-3, 3, 32, 0.3
N_EPISODES = 500

CNN_CUTOFF = "2022-06-30"  # CNN 训练截止 (之后特征=外推, 无泄漏)
PPO1_TRAIN = ("2021-01-04", "2022-06-30")
PPO1_TEST = ("2022-07-01", "2023-12-31")
PPO2_TRAIN = ("2021-01-04", "2023-12-31")
PPO2_TEST = ("2024-01-02", "2026-08-03")


def main() -> None:
    print("=" * 74)
    print("  V3 vs RL+CNN — 全量滚动回测 (14:50口径, 无未来函数)")
    print(f"  device={DEVICE} | CNN训练≤{CNN_CUTOFF} | PPO {N_EPISODES}轮×2段")
    print("=" * 74)

    data = load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    n = len(dates)
    codes = [c for c in ETF_POOL if c in data]
    mat = close_matrix(data, dates)
    ohlcv = {}
    for c in codes:
        df = data[c].set_index("trade_date")
        ohlcv[c] = {
            k: df[k].reindex(dates).astype(float).values
            for k in ("open", "high", "low", "close", "volume")
        }

    print(f"\n  数据: {n} 交易日 ({dates[0]} ~ {dates[-1]})")

    # ---- 1. 渲染 + CNN 训练 (≤2022-06) + 特征表 ----
    print("\n  渲染图像数据集...")
    imgs, ys, rows, _codes = build_dataset(data, dates, ohlcv)
    print(f"  样本: {len(ys)}")

    cnn_cut = next(i for i, d in enumerate(dates) if str(d) >= CNN_CUTOFF)
    ts_arr = np.array([r[0] for r in rows])
    tr_mask = ts_arr < cnn_cut
    tr_idx = np.where(tr_mask)[0]
    va_idx = tr_idx[int(len(tr_idx) * 0.85) :]
    tr_idx = tr_idx[: int(len(tr_idx) * 0.85)]
    print(f"\n  CNN 训练 (≤{CNN_CUTOFF}): {len(tr_idx)} 样本...")
    model, _ = train_cnn(
        imgs[tr_idx], ys[tr_idx], imgs[va_idx], ys[va_idx], LR, LAYERS, FILTERS, DROPOUT
    )
    featurizer = Featurizer(model).to(DEVICE)
    featurizer.eval()
    with torch.no_grad():
        feats_all = featurizer(torch.tensor(imgs, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    feats_map = {}
    for j, (t, code) in enumerate(rows):
        feats_map[(str(dates[t]), code)] = feats_all[j]
    print(f"  特征表: {feats_all.shape} (CNN 外推, 对 {CNN_CUTOFF} 后数据无泄漏)")

    # ---- 2. 价格特征 ----
    price_map = {}
    for c in codes:
        df = data[c].set_index("trade_date")
        close = df["close"].reindex(dates)
        for i in range(20, n):
            mom5 = (
                close.iloc[i] / close.iloc[i - 5] - 1.0
                if np.isfinite(close.iloc[i]) and np.isfinite(close.iloc[i - 5])
                else 0.0
            )
            seg = close.iloc[i - 19 : i + 1].astype(float)
            if seg.isna().any():
                vol20 = 0.3
            else:
                dr = seg.diff().dropna() / seg.shift(1).dropna()
                vol20 = float(dr.std() * np.sqrt(252)) if len(dr) > 1 else 0.3
            price_map[(str(dates[i]), c)] = [float(mom5), vol20]

    def seg_idx(s0: str, s1: str) -> tuple[int, int]:
        a = next(i for i, d in enumerate(dates) if str(d) >= s0)
        b = next(i for i, d in enumerate(dates) if str(d) >= s1)
        return a, min(b, n)

    # ---- 3. PPO 段1 ----
    tr_s, tr_e = seg_idx(*PPO1_TRAIN)
    te_s, te_e = seg_idx(*PPO1_TEST)
    print(
        f"\n  段1: PPO训练 {dates[tr_s]}~{dates[tr_e - 1]} | 测试 {dates[te_s]}~{dates[te_e - 1]}"
    )
    env1_tr = ETFEnv(data, dates, mat, feats_map, price_map, codes, tr_s, tr_e)
    ac1 = ppo_train(env1_tr, N_EPISODES, DEVICE)
    env1_te = ETFEnv(data, dates, mat, feats_map, price_map, codes, te_s, te_e)
    r1 = ppo_eval(env1_te, ac1, DEVICE)
    f1 = r1["final_value"]
    print(f"  段1 期末: {f1:,.0f} ({r1['total_return']:+.1%}) 夏普{r1['sharpe']:.2f}")

    # ---- 4. PPO 段2 (初始资金=F1) ----
    tr_s2, tr_e2 = seg_idx(*PPO2_TRAIN)
    te_s2, te_e2 = seg_idx(*PPO2_TEST)
    print(
        f"\n  段2: PPO训练 {dates[tr_s2]}~{dates[tr_e2 - 1]} | "
        f"测试 {dates[te_s2]}~{dates[te_e2 - 1]} (初始资金 {f1:,.0f})"
    )
    env2_tr = ETFEnv(
        data, dates, mat, feats_map, price_map, codes, tr_s2, tr_e2, initial_capital=f1
    )
    ac2 = ppo_train(env2_tr, N_EPISODES, DEVICE)
    env2_te = ETFEnv(
        data, dates, mat, feats_map, price_map, codes, te_s2, te_e2, initial_capital=f1
    )
    r2 = ppo_eval(env2_te, ac2, DEVICE)
    f2 = r2["final_value"]
    print(f"  段2 期末: {f2:,.0f} ({r2['total_return']:+.1%}) 夏普{r2['sharpe']:.2f}")

    # ---- 5. V3 对比 ----
    r_v3_full = run_v3_r4_sameday(data, mat, thr=1.0)  # 全量 2019-12 起
    v3_full = {
        k: r_v3_full[k]
        for k in ("final_value", "total_return", "ann_return", "sharpe", "max_drawdown", "n_trades")
    }
    te_s0, _te_e0 = seg_idx(*PPO1_TEST)
    tb = max(te_s0 - 130, 0)
    r_v3_seg = run_v3_r4_sameday(data, mat, thr=1.0, start_idx=tb)  # 2022-07 起
    v3_seg = {
        k: r_v3_seg[k]
        for k in ("final_value", "total_return", "ann_return", "sharpe", "max_drawdown", "n_trades")
    }

    # RL 拼接 (覆盖区间 2022-07~2026-08)
    rl_total = (f2 / 100_000.0) - 1.0

    print("\n" + "=" * 74)
    print("  A. RL 覆盖区间对比 (2022-07-01 ~ 2026-08-03, 起点 10万)")
    print("=" * 74)
    print(f"  {'配置':<14} {'期末金额':>10} {'总收益':>9} {'夏普':>6} {'回撤':>8}")
    for name, r in (
        ("V3(同区间)", v3_seg),
        (
            "RL+CNN(拼接)",
            {
                "final_value": f2,
                "total_return": rl_total,
                "sharpe": round((r1["sharpe"] + r2["sharpe"]) / 2, 3),
                "max_drawdown": max(r1["max_drawdown"], r2["max_drawdown"]),
            },
        ),
    ):
        print(
            f"  {name:<14} {r['final_value']:>10,.0f} {r['total_return']:>+9.1%} "
            f"{r['sharpe']:>6.2f} {r['max_drawdown']:>8.1%}"
        )
    beat = f2 > v3_seg["final_value"]

    print("\n" + "=" * 74)
    print("  B. V3 全量参考 (2019-12-05 ~ 2026-08-03, 起点 10万)")
    print("=" * 74)
    print(
        f"  V3全量: 期末 {v3_full['final_value']:>10,.0f} ({v3_full['total_return']:+.1%}) "
        f"夏普{v3_full['sharpe']:.2f} 回撤{v3_full['max_drawdown']:.1%}"
    )
    print(f"  → RL+CNN {'跑赢' if beat else '跑输'} V3 (同覆盖区间)")

    out = {
        "meta": {
            "note": "滚动重训拼接: CNN≤2022-06, PPO段1(2022-07~2023-12)+段2(2024-01~2026-08), 资金拼接",
            "device": DEVICE,
            "ppo_episodes": N_EPISODES,
            "rl_span": "2022-07-01~2026-08-03",
            "v3_full_span": "2019-12-05~2026-08-03",
        },
        "rl_seg1": r1,
        "rl_seg2": r2,
        "rl_combined": {"final_value": f2, "total_return": rl_total},
        "v3_full": v3_full,
        "v3_same_span": v3_seg,
        "beat": beat,
    }
    out_path = OUT_DIR / "rl_ppo_full.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
