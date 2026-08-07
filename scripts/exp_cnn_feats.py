"""CNN 视觉特征提取器 + 小模型对比 + Grad-CAM 特征分析.

模块1 (RL 全链路第一步):
  1. 自研 CNN (L3/F32/无BN, 已验证) → 特征提取器 (FC64 前激活 = 64维视觉特征)
  2. 对比 MobileNetV3-Small (torchvision 最轻量视觉模型, 随机初始化适配32×32)
  3. 验证集 MSE 选优 → 保存特征器
  4. Grad-CAM 特征分析: 涨/跌样本平均热力图 + 通道激活差异 → "关键视觉模式"
  5. 预计算特征表: 全样本 (t, asset) → 64维视觉特征 (RL 训练查表用)

输出: data/v9_results/cnn_feats/{featurizer.pt, feats.npz, grad_cam.json}
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "v9_results" / "cnn_feats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from exp_v3_cnn import (  # noqa: E402
    FWD,
    SEQ_N,
    KlineCNN,
    render_tech_image,
    train_cnn,
)
from run_qixing_v3 import ETF_POOL, load_data  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TEST_START = "2023-07-03"
LR, LAYERS, FILTERS, DROPOUT = 1e-3, 3, 32, 0.3  # 主实验最优超参


class Featurizer(nn.Module):
    """特征提取器: CNN 卷积 → FC64 激活 (64维视觉特征)."""

    def __init__(self, cnn: KlineCNN):
        super().__init__()
        self.conv = cnn.conv
        self.fc1 = cnn.head[1]       # Linear(flat → 64)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.conv(x)
        h = self.fc1(h.flatten(1))
        return self.act(h)           # [B, 64]


def build_dataset(data: dict, dates: list, ohlcv: dict):
    """渲染全样本图像 + 标签 (与 exp_v3_cnn 一致)."""
    n = len(dates)
    codes = [c for c in ETF_POOL if c in data]
    imgs, ys, rows = [], [], []
    for code in codes:
        o = ohlcv[code]
        for t in range(SEQ_N + 1, n - FWD):
            if not np.all(np.isfinite(o["close"][t - SEQ_N:t + 1])):
                continue
            imgs.append(render_tech_image(o["close"][:t + 1], o["high"][:t + 1],
                                          o["low"][:t + 1], o["open"][:t + 1],
                                          o["volume"][:t + 1]))
            ys.append(o["close"][t + FWD] / o["close"][t] - 1.0)
            rows.append((t, code))
    return np.array(imgs, np.float32), np.array(ys, float), rows, codes


def train_mobilenet(imgs_tr, y_tr, imgs_va, y_va, epochs=30, patience=8):
    """MobileNetV3-Small (随机初始化, 适配32×32) 早停训练."""
    from torchvision.models import mobilenet_v3_small
    torch.manual_seed(42)
    net = mobilenet_v3_small(weights=None, num_classes=1)
    # 适配 32×32: 首层 stride 1
    net.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                             nn.Linear(576, 64), nn.ReLU())  # 64维视觉特征
    head = nn.Linear(64, 1)  # 回归头 (仅训练用)
    model = nn.Sequential(backbone, head)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    xt = torch.tensor(imgs_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    xv = torch.tensor(imgs_va, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_va, dtype=torch.float32, device=DEVICE)
    best_va, best_state, bad = float("inf"), None, 0
    n = len(yt)
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(xv).squeeze(-1), yv).item()
        if va_loss < best_va:
            best_va = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return backbone, best_va


def grad_cam_analysis(model: KlineCNN, imgs: np.ndarray, ys: np.ndarray, top_k: int = 64):
    """Grad-CAM: 涨/跌样本平均热力图 (按卷积层梯度加权激活).

    Returns: {up_heat, dn_heat} shape (last_conv_channels,) 每通道平均权重贡献.
    """
    model.to(DEVICE)
    model.eval()
    conv = model.conv
    # 取最后一个卷积层
    children = list(conv.children())
    target_idx = None
    for i, m in enumerate(children):
        if isinstance(m, nn.Conv2d):
            target_idx = i
    imgs_t = torch.tensor(imgs, dtype=torch.float32, device=DEVICE)
    activations, gradients = {}, {}

    def hook_a(_module, _input, output):
        activations["a"] = output.detach()

    def hook_g(_module, _grad_input, grad_output):
        gradients["g"] = grad_output[0].detach()

    h_a = children[target_idx].register_forward_hook(hook_a)
    h_g = children[target_idx].register_full_backward_hook(hook_g)

    def cam_for(select_idx: np.ndarray) -> np.ndarray:
        x = imgs_t[select_idx]
        y = torch.tensor(ys[select_idx], dtype=torch.float32, device=DEVICE)
        out = model(x).squeeze(-1)
        loss = (out * y).mean()  # 加权: 预测与真实同向贡献
        model.zero_grad()
        loss.backward()
        _act = activations["a"]   # [B, C, H, W] (仅触发hook用)
        g = gradients["g"]        # [B, C, H, W]
        weights = g.mean(dim=(2, 3))   # [B, C]
        return weights.mean(dim=0).cpu().numpy()  # [C]

    order = np.argsort(ys)
    dn_idx = order[:top_k]
    up_idx = order[-top_k:]
    up_w = cam_for(up_idx)
    dn_w = cam_for(dn_idx)
    h_a.remove()
    h_g.remove()
    return up_w, dn_w


def main() -> None:
    print("=" * 74)
    print("  CNN 视觉特征提取器 + 小模型对比 + Grad-CAM 分析")
    print(f"  Device: {DEVICE}")
    print("=" * 74)

    data = load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    _n = len(dates)
    codes = [c for c in ETF_POOL if c in data]
    ohlcv = {}
    for c in codes:
        df = data[c].set_index("trade_date")
        ohlcv[c] = {k: df[k].reindex(dates).astype(float).values
                    for k in ("open", "high", "low", "close", "volume")}

    print("\n  渲染图像数据集...")
    imgs, ys, rows, codes = build_dataset(data, dates, ohlcv)
    print(f"  样本: {len(ys)} | 张量 {imgs.shape}")

    te_start = next(i for i, d in enumerate(dates) if str(d) >= TEST_START)
    ts_arr = np.array([r[0] for r in rows])
    tr_mask = ts_arr < te_start
    tr_idx = np.where(tr_mask)[0]
    va_idx = tr_idx[int(len(tr_idx) * 0.85):]
    tr_idx = tr_idx[:int(len(tr_idx) * 0.85)]
    print(f"  训练 {len(tr_idx)} | 验证 {len(va_idx)} | 测试 {int((~tr_mask).sum())}")

    # ---- 1. 小模型对比: 自研CNN vs MobileNetV3-Small ----
    print("\n  小模型对比 (验证集 MSE):")
    cnn, mse_cnn = train_cnn(imgs[tr_idx], ys[tr_idx], imgs[va_idx], ys[va_idx],
                             LR, LAYERS, FILTERS, DROPOUT)
    print(f"    自研CNN (L3/F32)          MSE {mse_cnn:.6f}")
    mob, mse_mob = train_mobilenet(imgs[tr_idx], ys[tr_idx], imgs[va_idx], ys[va_idx])
    print(f"    MobileNetV3-Small         MSE {mse_mob:.6f}")

    # ---- 2. 选优特征器 + 全量训练 ----
    best = cnn if mse_cnn <= mse_mob else mob
    featurizer = Featurizer(cnn) if isinstance(best, KlineCNN) else None
    print(f"  → 特征器: {'自研CNN' if isinstance(best, KlineCNN) else 'MobileNetV3-Small'}")

    # 全量训练集 (训练+验证) 重新训练特征器
    tr_all = np.where(tr_mask)[0]
    if isinstance(best, KlineCNN):
        model, _ = train_cnn(imgs[tr_all], ys[tr_all], imgs[va_idx], ys[va_idx],
                             LR, LAYERS, FILTERS, DROPOUT)
        featurizer = Featurizer(model)
    else:
        featurizer, _ = train_mobilenet(imgs[tr_all], ys[tr_all], imgs[va_idx], ys[va_idx])

    # ---- 3. 预计算特征表 (全样本) ----
    featurizer.eval()
    featurizer.to(DEVICE)
    with torch.no_grad():
        feats = featurizer(torch.tensor(imgs, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    print(f"  特征表: {feats.shape} (全样本 64维视觉特征)")
    t_arr = np.array([r[0] for r in rows], dtype=np.int64)
    code_arr = np.array([r[1] for r in rows], dtype="U6")
    dates_str = np.array([str(d) for d in dates], dtype="U10")
    np.savez_compressed(OUT_DIR / "feats.npz", feats=feats, ys=ys,
                        t_arr=t_arr, code_arr=code_arr, dates_str=dates_str)

    # ---- 4. Grad-CAM 特征分析 ----
    print("\n  Grad-CAM 特征分析 (涨/跌 top64 样本, 卷积通道权重)...")
    if isinstance(best, KlineCNN):
        model.to(DEVICE)
        up_w, dn_w = grad_cam_analysis(model, imgs, ys)
        diff = up_w - dn_w
        top_ch = np.argsort(-np.abs(diff))[:8]
        gc = {"up_top_channels": [int(i) for i in top_ch if diff[i] > 0],
              "dn_top_channels": [int(i) for i in top_ch if diff[i] < 0],
              "channel_diff": {int(i): round(float(diff[i]), 5) for i in top_ch}}
        print(f"    区分度最高通道: {gc}")
        with open(OUT_DIR / "grad_cam.json", "w") as f:
            json.dump(gc, f, indent=2, ensure_ascii=False)
    else:
        print("    (MobileNet 特征器, 跳过 Grad-CAM 通道分析)")

    torch.save(featurizer.state_dict(), OUT_DIR / "featurizer.pt")
    print(f"\n  已保存: {OUT_DIR}/featurizer.pt, feats.npz, grad_cam.json")


if __name__ == "__main__":
    main()
