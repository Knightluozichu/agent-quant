"""DL 学 V3 门控: 暴跌事件二分类 (真跌=排除 vs 假摔=放行).

任务: 规则门控 (ret60<0.01 且 动量>0) 是基于2个手工因子的线性规则;
      本实验用 DNN(MLP)/CNN 学习全部可用特征, 判断能否超越规则门控.

数据集: 全部"单日跌>3%"事件 (8资产, 无未来函数, 特征用当日及之前)
标签: 后5日收益>0 → 假摔(1, 应放行) / ≤0 → 真跌(0, 应排除)
切分: 时间切分 训练 2020-2023 → 测试 2024-2026 (禁止随机切分)
评估: AUC / 准确率 / 假摔召回率 vs 规则门控基准 (ret60<0.01 且 动量>0)

用法: uv run python scripts/exp_v3_dl_gate.py
输出: data/v9_results/v3_dl_gate.json
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

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"
DROP_THR = -0.03
RET60_THR = 0.01

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def collect_events(data: dict) -> list[dict]:
    """收集全部暴跌事件 (特征+标签, 无未来函数)."""
    events = []
    for code in rq.ETF_POOL:
        df = data[code].sort_values("trade_date").reset_index(drop=True)
        c = df["close"].astype(float).values
        o = df["open"].astype(float).values
        v = df["volume"].astype(float).values
        for i in range(61, len(c) - 5):
            cr = (c[i] - c[i - 1]) / c[i - 1]
            if cr >= DROP_THR:
                continue
            ret60 = (c[i] - c[i - 61]) / c[i - 61]
            r5 = (c[i] - c[i - 6]) / c[i - 6]
            r10 = (c[i] - c[i - 11]) / c[i - 11]
            r20 = (c[i] - c[i - 21]) / c[i - 21]
            mom = 0.5 * r10 + 0.5 * r20
            dr = np.diff(c[i - 21 : i]) / c[i - 21 : i - 1]
            vol20 = float(np.std(dr) * np.sqrt(252))
            w60 = c[i - 60 : i]
            pos60 = (c[i] - w60.min()) / (w60.max() - w60.min()) if w60.max() > w60.min() else 0.5
            peak60 = float(np.max(c[i - 61 : i - 1]))
            dd_peak = c[i] / peak60 - 1.0 if peak60 > 0 else 0.0
            avg_vol = float(np.mean(v[i - 21 : i - 1])) if i >= 22 else 1.0
            vol_ratio = float(v[i] / avg_vol) if avg_vol > 0 else 1.0
            open_gap = (o[i] - c[i - 1]) / c[i - 1]
            intraday = (c[i] - o[i]) / o[i] if o[i] > 0 else 0.0
            fwd5 = (c[i + 5] - c[i]) / c[i]
            events.append(
                {
                    "date": str(df.iloc[i]["trade_date"]),
                    "code": code,
                    "name": rq.ETF_POOL[code],
                    "f": [
                        ret60,
                        r5,
                        r10,
                        r20,
                        mom,
                        vol20,
                        pos60,
                        dd_peak,
                        vol_ratio,
                        open_gap,
                        intraday,
                        cr,
                    ],
                    "kline": c[i - 30 : i + 1] / c[i] - 1.0,  # 31日归一化K线
                    "label": 1.0 if fwd5 > 0 else 0.0,  # 假摔=1
                    "rule_gate": 1.0 if ret60 < RET60_THR and mom > 0 else 0.0,
                    "year": str(df.iloc[i]["trade_date"])[:4],
                }
            )
    return events


def split_data(events: list[dict]):
    """时间切分: 训练 2020-2023 → 测试 2024-2026."""
    tr = [e for e in events if e["year"] <= "2023"]
    te = [e for e in events if e["year"] >= "2024"]
    x_tr = np.array([e["f"] for e in tr], dtype=np.float32)
    ytr = np.array([e["label"] for e in tr], dtype=np.float32)
    x_te = np.array([e["f"] for e in te], dtype=np.float32)
    yte = np.array([e["label"] for e in te], dtype=np.float32)
    k_tr = np.array([e["kline"] for e in tr], dtype=np.float32)
    k_te = np.array([e["kline"] for e in te], dtype=np.float32)
    return x_tr, ytr, x_te, yte, k_tr, k_te, tr, te


def auc_score(y, p):
    """AUC (无sklearn依赖实现)."""
    order = np.argsort(-p)
    y_s = y[order]
    n_pos = int(y_s.sum())
    n_neg = len(y_s) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = sum(i + 1 for i, v in enumerate(y_s) if v == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, in_len=31):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * (in_len // 4), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(-1)


def train_eval(model, x, y, epochs=60, lr=1e-3, batch=32):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    x_t = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n = len(x)
    for _epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = lossf(model(x_t[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x_t)).numpy()


def main() -> None:
    print("=" * 76)
    print("  DL 学 V3 门控 | 暴跌事件二分类 (真跌vs假摔)")
    print("=" * 76)

    data = rq.load_data()
    events = collect_events(data)
    x_tr, y_tr, x_te, y_te, k_tr, k_te, tr, te = split_data(events)
    print(f"\n  事件总数: {len(events)} | 训练 {len(tr)} (2020-2023) | 测试 {len(te)} (2024-2026)")
    print(
        f"  假摔率(标签=1): 训练 {(y_tr == 1).mean() * 100:.1f}% | "
        f"测试 {(y_te == 1).mean() * 100:.1f}%"
    )

    # 基准1: 规则门控 (ret60<0.01 且 动量>0)
    rule_p = np.array([e["rule_gate"] for e in te], dtype=np.float32)
    rule_acc = float((np.round(rule_p) == y_te).mean())
    rule_auc = auc_score(y_te, rule_p)
    rule_recall = float(((rule_p == 1) & (y_te == 1)).sum() / max((y_te == 1).sum(), 1))
    print(
        f"\n  【规则门控基准】 准确率 {rule_acc * 100:.1f}% | "
        f"AUC {rule_auc:.3f} | 假摔召回 {rule_recall * 100:.1f}%"
    )

    # DNN (MLP) — 12维手工特征
    torch.manual_seed(42)
    mlp = MLP(x_tr.shape[1])
    train_eval(mlp, x_tr, y_tr)
    with torch.no_grad():
        p_mlp_te = torch.sigmoid(mlp(torch.tensor(x_te, dtype=torch.float32))).numpy()
    acc_mlp = float(((p_mlp_te > 0.5) == y_te).mean())
    auc_mlp = auc_score(y_te, p_mlp_te)
    recall_mlp = float(((p_mlp_te > 0.5) & (y_te == 1)).sum() / max((y_te == 1).sum(), 1))
    print(
        f"  【DNN(MLP)】     准确率 {acc_mlp * 100:.1f}% | "
        f"AUC {auc_mlp:.3f} | 假摔召回 {recall_mlp * 100:.1f}%"
    )

    # CNN (1D) — 31日K线
    torch.manual_seed(42)
    cnn = CNN1D(k_tr.shape[1])
    train_eval(cnn, k_tr, y_tr)
    with torch.no_grad():
        p_cnn_te = torch.sigmoid(cnn(torch.tensor(k_te, dtype=torch.float32))).numpy()
    acc_cnn = float(((p_cnn_te > 0.5) == y_te).mean())
    auc_cnn = auc_score(y_te, p_cnn_te)
    recall_cnn = float(((p_cnn_te > 0.5) & (y_te == 1)).sum() / max((y_te == 1).sum(), 1))
    print(
        f"  【CNN(1D K线)】  准确率 {acc_cnn * 100:.1f}% | "
        f"AUC {auc_cnn:.3f} | 假摔召回 {recall_cnn * 100:.1f}%"
    )

    # 融合: DNN特征 + CNN K线
    torch.manual_seed(42)

    class Fusion(nn.Module):
        def __init__(self, in_dim, klen):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 16, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Flatten(),
                nn.Linear(32 * (klen // 4), 16),
                nn.ReLU(),
            )
            self.head = nn.Sequential(nn.Linear(in_dim + 16, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x, k):
            kf = self.cnn(k.unsqueeze(1))
            return self.head(torch.cat([x, kf], dim=1)).squeeze(-1)

    fus = Fusion(x_tr.shape[1], k_tr.shape[1])
    opt = torch.optim.Adam(fus.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    x_t, y_t, k_t = (
        torch.tensor(x_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
        torch.tensor(k_tr, dtype=torch.float32),
    )
    for _epoch in range(60):
        perm = torch.randperm(len(x_t))
        for i in range(0, len(x_t), 32):
            idx = perm[i : i + 32]
            opt.zero_grad()
            loss = lossf(fus(x_t[idx], k_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
    fus.eval()
    with torch.no_grad():
        p_fus_te = torch.sigmoid(
            fus(torch.tensor(x_te, dtype=torch.float32), torch.tensor(k_te, dtype=torch.float32))
        ).numpy()
    acc_fus = float(((p_fus_te > 0.5) == y_te).mean())
    auc_fus = auc_score(y_te, p_fus_te)
    recall_fus = float(((p_fus_te > 0.5) & (y_te == 1)).sum() / max((y_te == 1).sum(), 1))
    print(
        f"  【DNN+CNN融合】  准确率 {acc_fus * 100:.1f}% | "
        f"AUC {auc_fus:.3f} | 假摔召回 {recall_fus * 100:.1f}%"
    )

    # 判定: DL 是否显著优于规则门控 (AUC 提升 > 0.05)
    best_auc = max(auc_mlp, auc_cnn, auc_fus)
    verdict = f"AUC 最高 {best_auc:.3f} (规则 {rule_auc:.3f}), " + (
        "✅ DL 显著优于规则"
        if best_auc - rule_auc > 0.05
        else "❌ DL 未显著优于规则 (信号太弱, 与历史ML层结论一致)"
    )

    print("\n" + "=" * 76)
    print(f"  判定: {verdict}")
    print("=" * 76)

    out = {
        "n_train": len(tr),
        "n_test": len(te),
        "rule": {"acc": rule_acc, "auc": rule_auc, "recall": rule_recall},
        "dnn": {"acc": acc_mlp, "auc": auc_mlp, "recall": recall_mlp},
        "cnn": {"acc": acc_cnn, "auc": auc_cnn, "recall": recall_cnn},
        "fusion": {"acc": acc_fus, "auc": auc_fus, "recall": recall_fus},
        "verdict": verdict,
    }
    path = OUTPUT_DIR / "v3_dl_gate.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {path}")


if __name__ == "__main__":
    main()
