"""深度学习 vs R4规则 — 动态调仓全量回测对比 (14:50 同日成交口径).

背景: R4 提前换手规则 (thr2.0%_b2% / thr1.5%_b0%) 在 14:50 口径下大幅跑输 V3
(期末 97.9万/95.1万 vs 190.6万). 本实验验证深度学习是否优于简单阈值规则:

模型 (特征融合):
  - 序列隐特征: GRU(hidden=24) 编码该资产最近 SEQ=10 日收益率序列
    (学习短期窗口价格变动模式)
  - 手工特征: 22维 (多尺度动量/波动/量能/市场环境, 复用 build_features)
  - R4 规则特征: 昨日单日涨幅 / 10日动量 / V3动量评分 (与规则特征结合)
  - concat → Linear(32) → Linear(1) 回归输出 未来5日收益

决策 (动态调仓, 最快 T+1 与实盘对齐):
  - 每日收盘: 模型对 7 资产预测未来5日收益, 选最高者
  - 预测最高者 ≠ 当前持仓 且 差>buffer → 当日收盘换仓 (14:50 口径同日成交)
  - buffer 网格 {0, 0.5%, 1%}; DL+R4 模式: R4 事件(昨日涨幅>2%)时 buffer 减半

验证: 训练 2019-12~2023-06 → 全量测试 2023-07~2026-08 (10万本金连续回测)
     同区间对比 V3 / V3+R4规则两配置 / V3+DL / DL+R4特征组合
     追加滚动验证: 训练至2024-12 → 测 2025-01~2026-08

输出: data/v9_results/v3_dl_ab.json
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
from exp_ml_up_down import build_features  # noqa: E402
from exp_short_window_patterns import close_matrix  # noqa: E402
from run_qixing_v3 import (  # noqa: E402
    DEFENSE,
    ETF_POOL,
    FEE,
    SLIPPAGE,
    calc_momentum_score,
    load_data,
)

WARMUP = 130
INITIAL_CAPITAL = 100_000.0
SEQ = 10          # 序列窗口
FWD = 5           # 预测未来5日
DEVICE = "cpu"

# === 区间划分 ===
TRAIN_END = "2023-06-30"    # 训练截止
TEST_START = "2023-07-03"   # 全量测试起点
ROLL_TRAIN_END = "2024-12-31"  # 滚动验证训练截止
ROLL_TEST_START = "2025-01-06"

# === 手工特征列 (与 exp_ml_up_down 一致) ===
FEAT_COLS = [
    "mom3", "mom5", "mom10", "mom20", "mom60",
    "vol5", "vol20", "vol60", "mom_short_dev",
    "up_streak", "dn_streak",
    "dist_high20", "dist_low20", "dist_high60",
    "vol_ratio", "ret_vol5",
    "cyb_ma_state", "pool_mom5", "a_share_ret5",
    "cat_rel_mom5", "pool_rel_mom5",
]


# --------------------------------------------------------------------------- #
# 数据构建: 每样本 = (t, asset) → 序列 + 手工特征 + R4特征 → y(未来5日)
# --------------------------------------------------------------------------- #
def build_dl_dataset(mat: dict, dates: list, x_feats: np.ndarray, meta: list,
                     feat_names: list) -> dict:
    """构建 DL 样本集 (全部资产×交易日)."""
    n = len(dates)
    codes = [c for c in ETF_POOL if c in mat]
    # 手工特征 (date, code) → 特征向量
    feat_map = {}
    for i, m in enumerate(meta):
        if m["code"] in ETF_POOL:
            feat_map[(m["date"], m["code"])] = x_feats[i]
    mom_feat_idx = {f: feat_names.index(f) for f in FEAT_COLS}

    seqs, feats, y, rows = [], [], [], []
    for code in codes:
        close = mat[code].astype(float)
        for t in range(SEQ + 1, n - FWD):
            # 序列: 最近SEQ日收益率
            s = close[t - SEQ + 1:t + 1] / close[t - SEQ:t] - 1.0
            if not np.all(np.isfinite(s)):
                continue
            # 手工特征
            key = (str(dates[t]), code)
            if key not in feat_map:
                continue
            f = np.array([feat_map[key][mom_feat_idx[c]] for c in FEAT_COLS], float)
            # R4 规则特征: 昨日涨幅 / 10日动量 / V3动量评分
            prev_ret = close[t] / close[t - 1] - 1.0 if close[t - 1] > 0 else 0.0
            mom10 = close[t] / close[t - 10] - 1.0 if close[t - 10] > 0 else 0.0
            score = float(calc_momentum_score(close[:t + 1])) if len(close) >= 121 else 0.0
            r4f = np.array([prev_ret, mom10, score], float)
            seqs.append(s)
            feats.append(np.concatenate((f, r4f)))
            y.append(close[t + FWD] / close[t] - 1.0)
            rows.append((t, code))
    return {
        "seq": np.array(seqs, float),       # [N, SEQ]
        "feat": np.array(feats, float),     # [N, 25] (22手工+3规则)
        "y": np.array(y, float),            # [N]
        "feat_names": [*FEAT_COLS, "r4_prev_ret", "r4_mom10", "r4_score"],
        "rows": rows,                       # [(t, code)] 与样本一一对应
    }


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class SeqFeatNet(nn.Module):
    """GRU 序列隐特征 + 手工特征融合 → 未来5日收益."""

    def __init__(self, seq_len: int, n_feat: int, hidden: int = 24):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_feat, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, seq, feat):
        # seq: [B, SEQ, 1], feat: [B, n_feat]
        _, h = self.gru(seq)              # h: [1, B, hidden]
        h = h[-1]                          # [B, hidden]
        x = torch.cat([h, feat], dim=1)   # [B, hidden+n_feat]
        return self.head(x).squeeze(-1)


def train_model(seq_tr, feat_tr, y_tr, seq_va, feat_va, y_va,
                epochs: int = 40, lr: float = 1e-3, patience: int = 8) -> tuple[SeqFeatNet, dict]:
    """训练 + 早停 (验证集 MSE). 特征用训练集统计量标准化."""
    feat_mean = feat_tr.mean(0)
    feat_std = feat_tr.std(0) + 1e-8
    seq_mean = seq_tr.mean()
    seq_std = seq_tr.std() + 1e-8

    def norm_seq(s):
        return (s - seq_mean) / seq_std

    def norm_feat(f):
        return (f - feat_mean) / feat_std

    model = SeqFeatNet(SEQ, feat_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    xs = torch.tensor(norm_seq(seq_tr)[:, :, None], dtype=torch.float32)
    xf = torch.tensor(norm_feat(feat_tr), dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    xvs = torch.tensor(norm_seq(seq_va)[:, :, None], dtype=torch.float32)
    xvf = torch.tensor(norm_feat(feat_va), dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.float32)
    n = len(yt)
    best_va, best_state, bad = float("inf"), None, 0
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(model(xs[idx], xf[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(xvs, xvf), yv).item()
        if va_loss < best_va:
            best_va = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"feat_mean": feat_mean, "feat_std": feat_std,
                   "seq_mean": seq_mean, "seq_std": seq_std, "best_va": best_va}


def predict_all(model: SeqFeatNet, ds: dict, scaler: dict) -> np.ndarray:
    """对全部样本预测未来5日收益."""
    model.eval()
    seq = torch.tensor(((ds["seq"] - scaler["seq_mean"]) / scaler["seq_std"])[:, :, None],
                       dtype=torch.float32)
    feat = torch.tensor((ds["feat"] - scaler["feat_mean"]) / scaler["feat_std"],
                        dtype=torch.float32)
    with torch.no_grad():
        return model(seq, feat).numpy()


# --------------------------------------------------------------------------- #
# 回测: 14:50 同日口径 + DL 动态调仓 (最快 T+1)
# --------------------------------------------------------------------------- #
def run_dl_backtest(
    data: dict, mat: dict,
    start_idx: int, end_idx: int,
    pred_map: dict,                 # {(date_str, code): 预测收益}
    buffer: float = 0.005,
    r4_boost: bool = False,         # R4 事件时 buffer 减半
) -> dict:
    """每日预测选最优 → 换仓 (T日收盘成交, 涨跌停检查/卖出失败卡仓)."""
    common: set = set()
    for code in ETF_POOL:
        if code not in data:
            continue
        common = set(data[code]["trade_date"].tolist()) if not common \
            else common & set(data[code]["trade_date"].tolist())
    common &= set(data[DEFENSE]["trade_date"].tolist())
    all_dates = sorted(common)
    trading_dates = all_dates[WARMUP:]

    cash = float(INITIAL_CAPITAL)
    holding: str | None = None
    holding_shares = 0
    equity_history: list[dict] = []
    events: list[dict] = []
    n_trades = 0

    def _tradable(code: str, td) -> bool:
        row = data[code][data[code]["trade_date"] == td]
        if row.empty:
            return False
        price = float(row.iloc[0]["close"])
        if price <= 0:
            return False
        hist = data[code][data[code]["trade_date"] < td]
        if not hist.empty:
            prev = float(hist.iloc[-1]["close"])
            if prev > 0 and abs(price / prev - 1) >= 0.099:
                return False
        return True

    def _sell(code: str, td) -> bool:
        nonlocal cash, holding, holding_shares
        if not _tradable(code, td):
            return False
        price = float(data[code][data[code]["trade_date"] == td].iloc[0]["close"])
        cash += holding_shares * price * (1 - FEE - SLIPPAGE)
        holding, holding_shares = None, 0
        return True

    def _buy(code: str, td) -> bool:
        nonlocal cash, holding, holding_shares
        if not _tradable(code, td):
            return False
        price = float(data[code][data[code]["trade_date"] == td].iloc[0]["close"])
        shares = int(cash * 0.99 / price / 100) * 100
        if shares <= 0:
            return False
        cash -= shares * price * (1 + FEE + SLIPPAGE)
        holding, holding_shares = code, shares
        return True

    idx_of = {d: i for i, d in enumerate(all_dates)}
    for td in trading_dates[start_idx:end_idx]:
        td_s = str(td)
        # 预测选优
        best_code, best_p = None, -np.inf
        for code in ETF_POOL:
            p = pred_map.get((td_s, code))
            if p is not None and p > best_p:
                best_p, best_code = p, code
        if best_code:
            eff_buffer = buffer
            if r4_boost:
                # R4 事件 (昨日涨幅>2%) → 更积极换手
                i = idx_of[td]
                if i >= 1 and mat[best_code][i - 1] > 0 and np.isfinite(mat[best_code][i - 1]):
                    p2 = mat[best_code][i - 2] if i >= 2 else 0.0
                    prev_ret = mat[best_code][i - 1] / p2 - 1.0 if p2 > 0 else 0.0
                    if prev_ret > 0.02:
                        eff_buffer = buffer / 2
            if holding is None or holding == DEFENSE:
                if _buy(best_code, td):
                    n_trades += 1
                    events.append({"date": td_s, "type": "enter", "asset": best_code,
                                   "pred": round(float(best_p), 4)})
            elif holding in ETF_POOL:
                cur_p = pred_map.get((td_s, holding))
                if (cur_p is not None and best_p > cur_p + eff_buffer
                        and _sell(holding, td) and _buy(best_code, td)):
                    n_trades += 1
                    events.append({"date": td_s, "type": "switch", "asset": best_code,
                                   "from": holding, "pred": round(float(best_p), 4),
                                   "cur": round(float(cur_p), 4)})
        # 每日净值
        equity = cash
        if holding and holding in data:
            row = data[holding][data[holding]["trade_date"] == td]
            if not row.empty:
                equity += holding_shares * float(row.iloc[0]["close"])
        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    total = eq_df["equity"].iloc[-1] / INITIAL_CAPITAL - 1
    rets = eq_df["equity"].pct_change().dropna()
    ann_vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
    span_days = max((eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days, 1)
    ann_ret = (1 + total) ** (365.25 / span_days) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = float(((eq_df["equity"] - cummax) / cummax).min())
    return {
        "final_value": round(float(eq_df["equity"].iloc[-1]), 0),
        "total_return": round(float(total), 4),
        "ann_return": round(float(ann_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(max_dd, 4),
        "n_trades": n_trades,
        "n_events": len(events),
        "events": events,
        "equity_curve": eq_df,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("  深度学习 vs R4规则 — 动态调仓全量回测 (14:50同日口径)")
    print(f"  模型: GRU(序列{SEQ}日) + {len(FEAT_COLS)}手工特征 + R4规则特征 → 未来{FWD}日收益")
    print("=" * 74)

    data = load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    mat = close_matrix(data, dates)
    vol_mat = {c: data[c].set_index("trade_date")["volume"].reindex(dates).values
               for c in ETF_POOL}
    n = len(dates)
    print(f"\n  数据: {n} 交易日 ({dates[0]} ~ {dates[-1]})")

    # 手工特征面板 (复用 ML 研究)
    x_feats, meta, feat_names = build_features(mat, vol_mat, dates)
    order = sorted(range(len(meta)), key=lambda i: meta[i]["date"])
    x_feats = x_feats[order]
    meta = [meta[i] for i in order]

    # DL 数据集 (全样本, rows 与样本一一对应)
    ds = build_dl_dataset(mat, dates, x_feats, meta, feat_names)
    print(f"  DL样本: {len(ds['y'])} | 特征维度: {ds['feat'].shape[1]}")

    sample_rows = ds["rows"]  # [(t, code)]
    ts_arr = np.array([r[0] for r in sample_rows])

    def train_and_predict(test_start_str: str) -> dict:
        """训练 (用 te_start 前数据), 返回测试期 (date,code)→预测 映射."""
        te_start = next(i for i, d in enumerate(dates) if str(d) >= test_start_str)
        train_mask_local = ts_arr < te_start
        tr_idx = np.where(train_mask_local)[0]
        va_idx = tr_idx[int(len(tr_idx) * 0.85):]
        tr_idx = tr_idx[:int(len(tr_idx) * 0.85)]
        model, scaler = train_model(
            ds["seq"][tr_idx], ds["feat"][tr_idx], ds["y"][tr_idx],
            ds["seq"][va_idx], ds["feat"][va_idx], ds["y"][va_idx])
        preds = predict_all(model, ds, scaler)
        pmap = {}
        for j, (t, code) in enumerate(sample_rows):
            if t >= te_start:
                pmap[(str(dates[t]), code)] = float(preds[j])
        return pmap, te_start

    # === 全量测试区间 (2023-07 ~ 2026-08) ===
    te_start = next(i for i, d in enumerate(dates) if str(d) >= TEST_START)
    te_end = n
    tb_start = max(te_start - WARMUP, 0)  # trading_dates (warmup后) 相对索引
    print(f"\n  全量测试区间: {dates[te_start]} ~ {dates[te_end-1]} ({te_end-te_start}天)")
    print("  训练: 2019-12-05 ~ 2023-06-30 | 所有配置同一测试区间对比")

    print("\n  训练 DL 模型 (早停)...")
    pmap, _te_start = train_and_predict(TEST_START)

    from exp_v3_r4_sameday import run_v3_r4_sameday

    configs = [
        ("V3基线", "v3"),
        ("V3+R4(2.0%_b2%)", "r4_20"),
        ("V3+R4(1.5%_b0%)", "r4_15"),
        ("DL(0.5%)", "dl_0.5"),
        ("DL(1.0%)", "dl_1.0"),
        ("DL+R4(0.5%)", "dlr4_0.5"),
    ]
    hdr = f"  {'配置':<18} {'期末金额':>10} {'总收益':>9} {'年化':>8} "
    hdr += f"{'夏普':>6} {'回撤':>8} {'交易':>4} {'换手':>4}"
    print(hdr)
    results = {}
    for name, kind in configs:
        if kind == "v3":
            r = run_v3_r4_sameday(data, mat, thr=1.0, start_idx=tb_start)
        elif kind == "r4_20":
            r = run_v3_r4_sameday(data, mat, thr=0.02, buffer=0.02, start_idx=tb_start)
        elif kind == "r4_15":
            r = run_v3_r4_sameday(data, mat, thr=0.015, buffer=0.0, start_idx=tb_start)
        else:
            buf = 0.005 if "0.5" in kind else 0.01
            r4b = "r4" in kind
            r = run_dl_backtest(data, mat, tb_start, None, pmap,
                                buffer=buf, r4_boost=r4b)
        r = {k: r[k] for k in ("final_value", "total_return", "ann_return",
                               "sharpe", "max_drawdown", "n_trades", "n_events")}
        results[name] = r
        print(f"  {name:<18} {r['final_value']:>10,.0f} {r['total_return']:>+9.1%} "
              f"{r['ann_return']:>+8.1%} {r['sharpe']:>6.2f} {r['max_drawdown']:>8.1%} "
              f"{r['n_trades']:>4} {r['n_events']:>4}")

    out = {"meta": {
        "model": "GRU(24) + 22手工特征 + 3规则特征 → 未来5日收益回归",
        "train": "2019-12-05~2023-06-30", "test": "2023-07-03~2026-08-03",
        "decision": "每日预测选优, 差>buffer 换仓, T日收盘成交(14:50口径), 最快T+1",
        "configs": ["V3基线", "V3+R4(2.0%_b2%)", "V3+R4(1.5%_b0%)", "DL", "DL+R4"],
    }, "results": {k: {kk: vv for kk, vv in v.items() if kk != "events"}
                   for k, v in results.items()}}
    out_path = OUTPUT_DIR / "v3_dl_ab.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
