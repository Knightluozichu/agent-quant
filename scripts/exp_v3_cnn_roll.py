"""CNN 滚动 OOS 验证 — 多段未参与训练的年份数据测试.

主实验 (exp_v3_cnn.py) 显示 CNN 在 2023-07~2026-08 显著跑赢 V3 (期末 800K vs 661K,
夏普 2.41 vs 1.96). 本脚本验证泛化能力: 3 个滚动 OOS 段, 每段用段前数据训练
(固定超参 lr1e-3/L3/F32/drop0.3 = 主实验网格最优), 段内 14:50 口径回测对比 V3.

  段A: 训练 ≤2021-12-31 → 测试 2022-01-01~2023-06-30 (1.5年)
  段B: 训练 ≤2023-06-30 → 测试 2023-07-01~2024-12-31 (1.5年)
  段C: 训练 ≤2024-12-31 → 测试 2025-01-01~2026-08-03 (1.6年)

通过标准: 至少 2/3 段 CNN 期末金额跑赢同段 V3 (项目三滚动窗口规范).

输出: data/v9_results/v3_cnn_roll.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from exp_short_window_patterns import close_matrix
from exp_v3_cnn import (
    FWD,
    SEQ_N,
    predict_cnn,
    render_tech_image,
    train_cnn,
)
from exp_v3_dl import run_dl_backtest
from exp_v3_r4_sameday import run_v3_r4_sameday
from run_qixing_v3 import ETF_POOL, load_data

WARMUP = 130
INITIAL_CAPITAL = 100_000.0
BUF = 0.005  # CNN 换手缓冲 (主实验最优)
LR, LAYERS, FILTERS, DROPOUT = 1e-3, 3, 32, 0.3

SEGMENTS = [
    ("A", "2022-01-01", "2023-06-30"),
    ("B", "2023-07-01", "2024-12-31"),
    ("C", "2025-01-01", "2026-08-03"),
]


def main() -> None:
    print("=" * 74)
    print("  CNN 滚动 OOS 验证 (3段, 每段段前训练, 14:50口径)")
    print(f"  超参固定: lr{LR:.0e} L{LAYERS} F{FILTERS} drop{DROPOUT} buffer{BUF:.1%}")
    print("=" * 74)

    data = load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    n = len(dates)
    codes = [c for c in ETF_POOL if c in data]
    ohlcv = {}
    for c in codes:
        df = data[c].set_index("trade_date")
        ohlcv[c] = {
            k: df[k].reindex(dates).astype(float).values
            for k in ("open", "high", "low", "close", "volume")
        }

    # 渲染数据集 (一次, 共享)
    print("\n  渲染 K线图数据集...")
    imgs, ys, rows = [], [], []
    for code in codes:
        o = ohlcv[code]
        for t in range(SEQ_N + 1, n - FWD):
            if not np.all(np.isfinite(o["close"][t - SEQ_N : t + 1])):
                continue
            imgs.append(
                render_tech_image(
                    o["close"][: t + 1],
                    o["high"][: t + 1],
                    o["low"][: t + 1],
                    o["open"][: t + 1],
                    o["volume"][: t + 1],
                )
            )
            ys.append(o["close"][t + FWD] / o["close"][t] - 1.0)
            rows.append((t, code))
    imgs = np.array(imgs, np.float32)
    ys = np.array(ys, float)
    ts_arr = np.array([r[0] for r in rows])
    print(f"  样本: {len(ys)} | 张量 {imgs.shape}")

    seg_out = []
    for seg_name, test_start_str, test_end_str in SEGMENTS:
        te_start = next(i for i, d in enumerate(dates) if str(d) >= test_start_str)
        te_end = next(i for i, d in enumerate(dates) if str(d) >= test_end_str) + 1
        te_end = min(te_end, n)
        tr_mask = ts_arr < te_start
        tr_idx = np.where(tr_mask)[0]
        # 验证集: 训练末 15%
        va_idx = tr_idx[int(len(tr_idx) * 0.85) :]
        tr_idx = tr_idx[: int(len(tr_idx) * 0.85)]
        tb_start = max(te_start - WARMUP, 0)

        print(
            f"\n  ── 段{seg_name}: 训练 ≤{test_start_str} | 测试 "
            f"{dates[te_start]}~{dates[te_end - 1]}"
        )
        # CNN 训练 + 预测
        model, _ = train_cnn(
            imgs[tr_idx], ys[tr_idx], imgs[va_idx], ys[va_idx], LR, LAYERS, FILTERS, DROPOUT
        )
        preds = predict_cnn(model, imgs)
        pmap = {}
        for j, (t, code) in enumerate(rows):
            if te_start <= t < te_end:
                pmap[(str(dates[t]), code)] = float(preds[j])

        mat = close_matrix(data, dates)
        r_cnn = run_dl_backtest(data, mat, tb_start, None, pmap, buffer=BUF)
        r_v3 = run_v3_r4_sameday(data, mat, thr=1.0, start_idx=tb_start)
        beat = r_cnn["final_value"] > r_v3["final_value"]
        print(
            f"    CNN  期末 {r_cnn['final_value']:>10,.0f} ({r_cnn['total_return']:+.1%}) "
            f"夏普{r_cnn['sharpe']:.2f} 回撤{r_cnn['max_drawdown']:.1%} 换手{r_cnn['n_trades']}"
        )
        print(
            f"    V3   期末 {r_v3['final_value']:>10,.0f} ({r_v3['total_return']:+.1%}) "
            f"夏普{r_v3['sharpe']:.2f} 回撤{r_v3['max_drawdown']:.1%}"
        )
        print(f"    → CNN {'跑赢' if beat else '跑输'} V3")
        seg_out.append(
            {
                "segment": seg_name,
                "test_span": [str(dates[te_start]), str(dates[te_end - 1])],
                "cnn": {
                    k: r_cnn[k]
                    for k in (
                        "final_value",
                        "total_return",
                        "ann_return",
                        "sharpe",
                        "max_drawdown",
                        "n_trades",
                    )
                },
                "v3": {
                    k: r_v3[k]
                    for k in (
                        "final_value",
                        "total_return",
                        "ann_return",
                        "sharpe",
                        "max_drawdown",
                        "n_trades",
                    )
                },
                "beat": beat,
            }
        )

    wins = sum(1 for s in seg_out if s["beat"])
    passed = wins >= 2
    print("\n" + "=" * 74)
    print(
        f"  滚动 OOS 判定: CNN 跑赢 {wins}/3 段 (标准: ≥2/3) → "
        f"{'✅ 通过' if passed else '❌ 未通过'}"
    )
    print("=" * 74)

    out = {
        "meta": {
            "model": f"CNN L{LAYERS} F{FILTERS} lr{LR:.0e} drop{DROPOUT} buffer{BUF:.1%}",
            "note": "滚动OOS: 每段用段前数据训练, 测试段未参与训练(无未来函数)",
        },
        "segments": seg_out,
        "wins": wins,
        "passed": passed,
    }
    out_path = OUTPUT_DIR / "v3_cnn_roll.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
