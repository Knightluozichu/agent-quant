"""CNN K线图视觉特征学习 — 动态调仓回测 (14:50 同日口径).

背景: DL (GRU+特征融合) 已证短期可预测性弱; 本实验验证 CNN 能否从
K线图+技术图表的视觉形态中提取信号 (蜡烛形态/量能/MACD/RSI):

1. 图像渲染 (numpy 向量化, 无外部绘图依赖):
   - ch0: 蜡烛图 (high-low 竖线 + open-close 实体, 涨亮跌暗) + MA5/MA20 线
   - ch1: 成交量柱 (归一化)
   - ch2: MACD 曲线 + RSI 曲线
   - 尺寸 32×32, 最近 30 个交易日
2. CNN: 卷积层×L (3×3, BN, ReLU, MaxPool) → FC → 未来5日收益回归
3. 网格调优: lr {1e-3,3e-4} × 层数 {2,3} × 滤波器 {16,32} (验证集MSE选优)
4. OOS 验证: 训练 2019-12~2023-06 → 测试 2023-07~2026-08 (与 DL/V3 同区间)
5. 对比: V3基线 / R4规则 / CNN (buffer 0.5%, 1%) — 14:50口径, 10万本金
6. 分析: 通道消融 (各视觉通道对预测的贡献) + 年度环境表现 + 样例形态统计

输出: data/v9_results/v3_cnn_ab.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from run_qixing_v3 import ETF_POOL, load_data

WARMUP = 130
INITIAL_CAPITAL = 100_000.0
SEQ_N = 30  # 图像窗口 (交易日)
IMG_SIZE = 32  # 图像尺寸
FWD = 5  # 预测未来5日
TEST_START = "2023-07-03"

# === 网格 ===
GRID_LR = (1e-3, 3e-4)
GRID_LAYERS = (2, 3)
GRID_FILTERS = (16, 32)
FIX_DROPOUT = 0.3
FIX_BATCH = 128


# --------------------------------------------------------------------------- #
# 技术指标
# --------------------------------------------------------------------------- #
def ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def rsi14(close: np.ndarray, period: int = 14) -> np.ndarray:
    diff = np.diff(close)
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    if len(up) < period:
        return np.full(len(close), 50.0)
    ru, rd = np.empty(len(up)), np.empty(len(dn))
    ru[0] = up[:period].mean()
    rd[0] = dn[:period].mean()
    for i in range(1, len(up)):
        ru[i] = (ru[i - 1] * (period - 1) + up[i]) / period
        rd[i] = (rd[i - 1] * (period - 1) + dn[i]) / period
    rs = ru / (rd + 1e-10)
    return np.concatenate([[50.0], 100.0 - 100.0 / (1.0 + rs)])


# --------------------------------------------------------------------------- #
# 图像渲染 (numpy 向量化 → 3通道像素张量)
# --------------------------------------------------------------------------- #
def render_tech_image(close, high, low, open_, vol, n: int = SEQ_N, size: int = IMG_SIZE):
    """渲染最近 n 日 → (3, size, size) float32 张量.

    ch0: 蜡烛图 (high-low 竖线 0.8 + open-close 实体 涨0.9/跌0.3) + MA5(1.0)/MA20(0.6) 线
    ch1: 成交量柱 (归一化 0~1)
    ch2: MACD 曲线(1.0) + RSI 曲线(0.6)
    """
    c = np.asarray(close[-n:], float)
    h = np.asarray(high[-n:], float)
    lo_ = np.asarray(low[-n:], float)
    o = np.asarray(open_[-n:], float)
    v = np.asarray(vol[-n:], float)
    lo, hi = float(np.min(lo_)), float(np.max(h))
    rng = (hi - lo) or 1.0
    img = np.zeros((3, size, size), np.float32)

    def yp(x):
        return np.clip(((x - lo) / rng * (size - 1)).round().astype(int), 0, size - 1)

    x = np.linspace(0, size - 1, n).round().astype(int)
    yh, yl, yo, yc = yp(h), yp(lo_), yp(o), yp(c)
    for i in range(n):
        img[0, min(yh[i], yl[i]) : max(yh[i], yl[i]) + 1, x[i]] = 0.8
        a, b = sorted((yo[i], yc[i]))
        img[0, a : max(b, a + 1) + 1, x[i]] = 0.9 if yc[i] >= yo[i] else 0.3
    for w, val in ((5, 1.0), (20, 0.6)):
        if n >= w:
            ma = np.convolve(c, np.ones(w) / w, mode="valid")
            for j in range(len(ma)):
                img[0, yp(ma[j]), x[w - 1 + j]] = val

    vn = v / (float(v.max()) or 1.0)
    for i in range(n):
        img[1, : int(vn[i] * (size - 1)) + 1, x[i]] = vn[i]

    def norm_curve(series):
        s = np.asarray(series[-n:], float)
        lo2, hi2 = float(s.min()), float(s.max())
        r2 = (hi2 - lo2) or 1.0
        return (s - lo2) / r2

    macd = ema(c, 12) - ema(c, 26)
    rsi = rsi14(c)
    for series, val in ((norm_curve(macd), 1.0), (norm_curve(rsi), 0.6)):
        ys = (series * (size - 1)).round().astype(int)
        for j in range(n):
            img[2, ys[j], x[j]] = val
    return img


# --------------------------------------------------------------------------- #
# CNN 模型
# --------------------------------------------------------------------------- #
class KlineCNN(nn.Module):
    """K线图 CNN: 卷积×L → 全局池化 → FC (无BN, 金融小数据BN易致ReLU死亡)."""

    def __init__(self, n_layers: int = 2, filters: int = 16, dropout: float = 0.3):
        super().__init__()
        convs = []
        in_ch = 3
        f = filters
        for _ in range(n_layers):
            convs += [nn.Conv2d(in_ch, f, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = f
            f *= 2
        self.conv = nn.Sequential(*convs)
        flat = in_ch * (IMG_SIZE // (2**n_layers)) ** 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.conv(x)).squeeze(-1)


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train_cnn(imgs_tr, y_tr, imgs_va, y_va, lr, n_layers, filters, dropout, epochs=30, patience=8):
    torch.manual_seed(42)  # 可复现性
    model = KlineCNN(n_layers, filters, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt = torch.tensor(imgs_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    xv = torch.tensor(imgs_va, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.float32)
    best_va, best_state, bad = float("inf"), None, 0
    n = len(yt)
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, FIX_BATCH):
            idx = perm[i : i + FIX_BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(xv), yv).item()
        if va_loss < best_va:
            best_va = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_va


def predict_cnn(model: KlineCNN, imgs: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(imgs, dtype=torch.float32)).numpy()


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("  CNN K线图视觉特征 — 动态调仓全量回测 (14:50口径)")
    print(f"  图像: {SEQ_N}日→3通道{IMG_SIZE}×{IMG_SIZE} (蜡烛+MA/量/MACD+RSI)")
    print(f"  网格: lr{GRID_LR} × 层数{GRID_LAYERS} × 滤波器{GRID_FILTERS}")
    print("=" * 74)

    data = load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    n = len(dates)
    codes = [c for c in ETF_POOL if c in data]
    # OHLCV 矩阵 (对齐公共日历)
    ohlcv = {}
    for c in codes:
        df = data[c].set_index("trade_date")
        ohlcv[c] = {
            k: df[k].reindex(dates).astype(float).values
            for k in ("open", "high", "low", "close", "volume")
        }
    print(f"\n  数据: {n} 交易日 ({dates[0]} ~ {dates[-1]}) | 资产 {len(codes)}")

    # === 1. 渲染数据集 ===
    print("\n  渲染 K线图数据集...")
    imgs, ys, rows = [], [], []
    for code in codes:
        o = ohlcv[code]
        for t in range(SEQ_N + 1, n - FWD):
            if not np.all(np.isfinite(o["close"][t - SEQ_N : t + 1])):
                continue
            img = render_tech_image(
                o["close"][: t + 1],
                o["high"][: t + 1],
                o["low"][: t + 1],
                o["open"][: t + 1],
                o["volume"][: t + 1],
            )
            imgs.append(img)
            ys.append(o["close"][t + FWD] / o["close"][t] - 1.0)
            rows.append((t, code))
    imgs = np.array(imgs, np.float32)
    ys = np.array(ys, float)
    print(f"  图像样本: {len(ys)} | 张量 {imgs.shape} | 内存 {imgs.nbytes / 1e6:.0f}MB")

    # === 2. 训练/验证切分 (训练末15%为验证) ===
    te_start = next(i for i, d in enumerate(dates) if str(d) >= TEST_START)
    ts_arr = np.array([r[0] for r in rows])
    tr_mask = ts_arr < te_start
    tr_idx = np.where(tr_mask)[0]
    va_idx = tr_idx[int(len(tr_idx) * 0.85) :]
    tr_idx = tr_idx[: int(len(tr_idx) * 0.85)]
    print(f"  训练 {len(tr_idx)} | 验证 {len(va_idx)} | 测试 {int((~tr_mask).sum())}")

    # === 3. 网格搜索 (验证集 MSE 选优) ===
    print("\n  网格搜索:")
    grid = {}
    for lr in GRID_LR:
        for nl in GRID_LAYERS:
            for nf in GRID_FILTERS:
                model, va = train_cnn(
                    imgs[tr_idx], ys[tr_idx], imgs[va_idx], ys[va_idx], lr, nl, nf, FIX_DROPOUT
                )
                key = f"lr{lr:.0e}_L{nl}_F{nf}"
                grid[key] = {"val_mse": round(float(va), 6)}
                print(f"    {key:<16} 验证MSE {va:.6f}")
    best_key = min(grid, key=lambda k: grid[k]["val_mse"])
    best_grid = grid[best_key]
    print(f"  → 最优: {best_key} (验证MSE {best_grid['val_mse']})")
    lr_b, nl_b, nf_b = (
        float(best_key.split("_")[0][2:]),
        int(best_key.split("_")[1][1:]),
        int(best_key.split("_")[2][1:]),
    )

    # === 4. 最优模型: 训练(含验证) + 测试预测 ===
    model, _ = train_cnn(
        imgs[tr_idx], ys[tr_idx], imgs[va_idx], ys[va_idx], lr_b, nl_b, nf_b, FIX_DROPOUT
    )
    # 用全量训练(训练+验证)重新训练
    tr_all = np.where(tr_mask)[0]
    model, _ = train_cnn(
        imgs[tr_all], ys[tr_all], imgs[va_idx], ys[va_idx], lr_b, nl_b, nf_b, FIX_DROPOUT
    )
    preds = predict_cnn(model, imgs)
    print(f"  预测分布: std={np.std(preds):.2e} min={preds.min():.4f} max={preds.max():.4f}")
    pmap = {}
    for j, (t, code) in enumerate(rows):
        if t >= te_start:
            pmap[(str(dates[t]), code)] = float(preds[j])

    # 测试集预测 IC (预测力检验)
    te_idx = np.where(~tr_mask)[0]
    pred_te = preds[te_idx]
    y_te = ys[te_idx]
    from scipy.stats import spearmanr

    if np.std(pred_te) < 1e-12 or np.std(y_te) < 1e-12:
        ic, ic_p = 0.0, 1.0
    else:
        ic, ic_p = spearmanr(pred_te, y_te)
    print(f"\n  测试集 IC (Spearman): {ic:.4f} (p={ic_p:.3f})")

    # === 5. 回测对比 (14:50口径, 复用 DL 引擎) ===
    from exp_v3_dl import run_dl_backtest
    from exp_v3_r4_sameday import run_v3_r4_sameday

    tb_start = max(te_start - WARMUP, 0)
    hdr = f"  {'配置':<20} {'期末金额':>10} {'总收益':>9} {'年化':>8} "
    hdr += f"{'夏普':>6} {'回撤':>8} {'换手':>4}"
    print(hdr)
    results = {}
    r_v3 = run_v3_r4_sameday(data, mat_placeholder(data, dates), thr=1.0, start_idx=tb_start)
    results["V3基线"] = {
        k: r_v3[k]
        for k in (
            "final_value",
            "total_return",
            "ann_return",
            "sharpe",
            "max_drawdown",
            "n_trades",
            "n_events",
        )
    }
    print(
        f"  {'V3基线':<20} {r_v3['final_value']:>10,.0f} {r_v3['total_return']:>+9.1%} "
        f"{r_v3['ann_return']:>+8.1%} {r_v3['sharpe']:>6.2f} "
        f"{r_v3['max_drawdown']:>8.1%} {r_v3['n_trades']:>4}"
    )
    r_r4 = run_v3_r4_sameday(
        data, mat_placeholder(data, dates), thr=0.02, buffer=0.02, start_idx=tb_start
    )
    results["V3+R4(2.0%_b2%)"] = {
        k: r_r4[k]
        for k in (
            "final_value",
            "total_return",
            "ann_return",
            "sharpe",
            "max_drawdown",
            "n_trades",
            "n_events",
        )
    }
    print(
        f"  {'V3+R4(2.0%_b2%)':<20} {r_r4['final_value']:>10,.0f} "
        f"{r_r4['total_return']:>+9.1%} {r_r4['ann_return']:>+8.1%} "
        f"{r_r4['sharpe']:>6.2f} {r_r4['max_drawdown']:>8.1%} "
        f"{r_r4['n_trades']:>4}"
    )
    for buf in (0.005, 0.01):
        r_cnn = run_dl_backtest(
            data, mat_placeholder(data, dates), tb_start, None, pmap, buffer=buf
        )
        name = f"CNN({buf:.1%})"
        results[name] = {
            k: r_cnn[k]
            for k in (
                "final_value",
                "total_return",
                "ann_return",
                "sharpe",
                "max_drawdown",
                "n_trades",
                "n_events",
            )
        }
        print(
            f"  {name:<20} {r_cnn['final_value']:>10,.0f} "
            f"{r_cnn['total_return']:>+9.1%} {r_cnn['ann_return']:>+8.1%} "
            f"{r_cnn['sharpe']:>6.2f} {r_cnn['max_drawdown']:>8.1%} "
            f"{r_cnn['n_trades']:>4}"
        )

    # === 6. 通道消融 (最优模型, 测试样本 IC) ===
    print("\n  通道消融 (测试集 IC, 各视觉通道对预测的贡献):")
    from scipy.stats import pearsonr

    abl = {}
    for ch_name, ch_idx in (
        ("全通道", None),
        ("无成交量ch1", 1),
        ("无MACD/RSI ch2", 2),
        ("无蜡烛ch0", 0),
    ):
        imgs_ab = imgs.copy()
        if ch_idx is not None:
            imgs_ab[:, ch_idx, :, :] = 0.0
        p_ab = predict_cnn(model, imgs_ab)[te_idx]
        if np.std(p_ab) < 1e-12 or np.std(y_te) < 1e-12:
            ic_ab = 0.0
        else:
            ic_ab, _ = pearsonr(p_ab, y_te)
        abl[ch_name] = round(float(ic_ab), 4)
        print(f"    {ch_name:<16} IC {ic_ab:+.4f}")

    # === 7. 年度表现 (CNN 净值曲线分年) ===
    r_cnn_best = run_dl_backtest(
        data, mat_placeholder(data, dates), tb_start, None, pmap, buffer=0.005
    )
    eq_cnn = r_cnn_best.get("equity_curve")
    yearly = {}
    if eq_cnn is not None:
        eq_cnn = eq_cnn.copy()
        eq_cnn["year"] = pd.to_datetime(eq_cnn["trade_date"]).dt.year
        prev = INITIAL_CAPITAL
        for y in sorted(eq_cnn["year"].unique()):
            ydf = eq_cnn[eq_cnn["year"] == y]
            if ydf.empty:
                continue
            yearly[int(y)] = round(float(ydf["equity"].iloc[-1] / prev - 1), 4)
            prev = ydf["equity"].iloc[-1]
        print("\n  CNN 年度收益:", {k: f"{v:+.1%}" for k, v in yearly.items()})

    out = {
        "meta": {
            "model": f"CNN L{nl_b} F{nf_b} lr{lr_b:.0e} drop{FIX_DROPOUT}",
            "image": f"{SEQ_N}日→3ch×{IMG_SIZE}×{IMG_SIZE} (蜡烛+MA/成交量/MACD+RSI)",
            "test_ic": round(float(ic), 4),
            "train": "2019-12~2023-06",
            "test": "2023-07~2026-08",
            "decision": "每日预测选优, buffer换仓, T日收盘成交(14:50口径)",
        },
        "grid": grid,
        "best": best_key,
        "results": results,
        "ablation": abl,
        "yearly_cnn": yearly,
    }
    out_path = OUTPUT_DIR / "v3_cnn_ab.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


def mat_placeholder(data: dict, dates: list):
    """构造 close 矩阵 (供 run_v3_r4_sameday / run_dl_backtest 使用)."""
    from exp_short_window_patterns import close_matrix

    return close_matrix(data, dates)


if __name__ == "__main__":
    main()
