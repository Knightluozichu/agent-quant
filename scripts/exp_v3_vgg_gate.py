"""VGG-8 学 V3 门控: 暴跌事件K线图像 → 自适应放行/排除判据.

任务: 规则门控(ret60<0.01且动量>0)是2因子线性规则; 用户要求用 VGG-8
      从K线图像自适应学习"最优门控判据" (隐式拟合所有门控参数).

数据: 全部"单日跌>3%"事件 → render_tech_image 生成 3ch×32×32 K线图
      (蜡烛+MA / 成交量 / MACD+RSI, 复用 exp_v3_cnn 方案)
标签: 后5日收益>0 → 假摔(1,放行) / ≤0 → 真跌(0,排除)
切分: 时间切分 训练 2020-2023 → 测试 2024-2026 (禁随机)
模型: VGG-8 (64×2→128×2→256×2→FC, 对32×32小图适配)
评估: AUC/准确率/假摔召回 vs 规则门控; AUC 显著(>0.55)则回测落地

用法: uv run python scripts/exp_v3_vgg_gate.py
输出: data/v9_results/v3_vgg_gate.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq
from exp_v3_cnn import IMG_SIZE, render_tech_image

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"
DROP_THR, RET60_THR = -0.03, 0.01

import torch
import torch.nn as nn


class VGG8(nn.Module):
    """VGG-8 (轻量适配 32×32): conv64×2→pool→conv128×2→pool→conv256×2→pool→FC."""

    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat = 256 * (IMG_SIZE // 8) ** 2
        self.classifier = nn.Sequential(
            nn.Linear(flat, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)  # flatten (N, 256, 4, 4) → (N, 4096)
        return self.classifier(f).squeeze(-1)


def auc_score(y, p):
    order = np.argsort(-p)
    ys = y[order]
    n_pos = int(ys.sum())
    n_neg = len(ys) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = sum(i + 1 for i, v in enumerate(ys) if v == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    print("=" * 76)
    print("  VGG-8 学 V3 门控 | K线图像 → 自适应放行/排除")
    print("=" * 76)

    data = rq.load_data()
    imgs, labels, rules, years, metas = [], [], [], [], []
    for code in rq.ETF_POOL:
        df = data[code].sort_values("trade_date").reset_index(drop=True)
        c = df["close"].astype(float).values
        h = df["high"].astype(float).values
        low = df["low"].astype(float).values
        o = df["open"].astype(float).values
        v = df["volume"].astype(float).values
        for i in range(61, len(c) - 5):
            cr = (c[i] - c[i - 1]) / c[i - 1]
            if cr >= DROP_THR:
                continue
            img = render_tech_image(
                c[: i + 1],
                h[: i + 1],
                low[: i + 1],
                o[: i + 1],
                v[: i + 1],
                n=30,
                size=IMG_SIZE,
            )
            ret60 = (c[i] - c[i - 61]) / c[i - 61]
            r10 = (c[i] - c[i - 11]) / c[i - 11]
            r20 = (c[i] - c[i - 21]) / c[i - 21]
            mom = 0.5 * r10 + 0.5 * r20
            fwd5 = (c[i + 5] - c[i]) / c[i]
            imgs.append(img)
            labels.append(1.0 if fwd5 > 0 else 0.0)
            rules.append(1.0 if ret60 < RET60_THR and mom > 0 else 0.0)
            years.append(str(df.iloc[i]["trade_date"])[:4])
            metas.append(
                {"date": str(df.iloc[i]["trade_date"]), "code": code, "name": rq.ETF_POOL[code]}
            )

    imgs = np.stack(imgs).astype(np.float32)
    labels = np.array(labels, dtype=np.float32)
    rules = np.array(rules, dtype=np.float32)
    years = np.array(years)
    tr_mask = years <= "2023"
    te_mask = years >= "2024"
    print(f"\n  事件总数: {len(imgs)} | 训练 {tr_mask.sum()} | 测试 {te_mask.sum()}")
    print(
        f"  假摔率: 训练 {labels[tr_mask].mean() * 100:.1f}% | "
        f"测试 {labels[te_mask].mean() * 100:.1f}%"
    )

    # 规则门控基准
    rule_auc = auc_score(labels[te_mask], rules[te_mask])
    rule_acc = float((np.round(rules[te_mask]) == labels[te_mask]).mean())
    rule_recall = float(
        ((rules[te_mask] == 1) & (labels[te_mask] == 1)).sum()
        / max((labels[te_mask] == 1).sum(), 1)
    )
    print(
        f"\n  【规则门控】 准确率 {rule_acc * 100:.1f}% | "
        f"AUC {rule_auc:.3f} | 假摔召回 {rule_recall * 100:.1f}%"
    )

    # VGG-8 训练
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG8().to(device)
    x_tr = torch.tensor(imgs[tr_mask]).to(device)
    ytr = torch.tensor(labels[tr_mask]).to(device)
    x_te = torch.tensor(imgs[te_mask]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    n = len(x_tr)
    for ep in range(50):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i : i + 32]
            opt.zero_grad()
            loss = lossf(model(x_tr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep + 1}: loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(x_te)).cpu().numpy()
    acc = float(((p_te > 0.5) == labels[te_mask]).mean())
    auc = auc_score(labels[te_mask], p_te)
    recall = float(
        ((p_te > 0.5) & (labels[te_mask] == 1)).sum() / max((labels[te_mask] == 1).sum(), 1)
    )
    print(
        f"\n  【VGG-8】      准确率 {acc * 100:.1f}% | AUC {auc:.3f} | 假摔召回 {recall * 100:.1f}%"
    )

    verdict = f"VGG-8 AUC {auc:.3f} vs 规则 {rule_auc:.3f}, " + (
        "✅ VGG-8 显著优于规则门控"
        if auc - rule_auc > 0.05
        else "❌ VGG-8 未显著优于规则 (暴跌后走向接近随机, 与历史4次ML实验结论一致)"
    )
    print("\n" + "=" * 76)
    print(f"  判定: {verdict}")
    print("=" * 76)

    out = {
        "n_train": int(tr_mask.sum()),
        "n_test": int(te_mask.sum()),
        "rule": {"acc": rule_acc, "auc": rule_auc, "recall": rule_recall},
        "vgg8": {"acc": acc, "auc": auc, "recall": recall},
        "verdict": verdict,
    }
    path = OUTPUT_DIR / "v3_vgg_gate.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {path}")


if __name__ == "__main__":
    main()
