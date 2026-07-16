"""专家3: 深度时序模型 (PatchTST 风格, 纯 PyTorch 自实现)。

角色: 用注意力机制捕捉多变量、长依赖的时序模式, 提供与 GBDT 互补的视角。
输入: 每个事件回看 lookback 根 bar 的特征窗口; 输出: 该下注/盈利概率。
若未安装 torch, 会抛出清晰的提示(不影响其他专家运行)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseExpert


def _require_torch():
    try:
        import torch  # noqa
        return True
    except Exception as e:
        raise ImportError(
            "DeepTSExpert 需要 PyTorch。请先安装: pip install torch (按你的 CUDA 版本)。"
        ) from e


class DeepTSExpert(BaseExpert):
    name = "deep_ts"
    needs_panel = True

    def _build_model(self, n_feats: int):
        import torch.nn as nn

        p = self.cfg
        d_model = int(p.get("d_model", 64))
        patch_len = int(p.get("patch_len", 8))
        lookback = int(p.get("lookback", 64))
        n_patches = lookback // patch_len

        class PatchTST(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_len = patch_len
                self.n_patches = n_patches
                self.embed = nn.Linear(patch_len * n_feats, d_model)
                self.pos = nn.Parameter(0.02 * __import__("torch").randn(1, n_patches, d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=int(p.get("n_heads", 4)),
                    dim_feedforward=d_model * 4, dropout=float(p.get("dropout", 0.1)),
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=int(p.get("n_layers", 2)))
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model), nn.Linear(d_model, 1)
                )

            def forward(self, x):  # x: (B, lookback, n_feats)
                B = x.shape[0]
                x = x[:, : self.n_patches * self.patch_len, :]
                x = x.reshape(B, self.n_patches, -1)  # 拼接 patch 内的时间×特征
                h = self.embed(x) + self.pos
                h = self.encoder(h)
                h = h.mean(dim=1)
                return self.head(h).squeeze(-1)

        return PatchTST()

    def _windows(self, index: pd.Index) -> np.ndarray:
        """从面板切出每个事件的历史窗口, 左侧不足零填充。返回 (n, lookback, n_feats)。"""
        assert self._panel is not None, "DeepTSExpert 需要先 set_panel"
        lookback = int(self.cfg.get("lookback", 64))
        panel = self._panel[self.feature_cols].astype(float).fillna(0.0)
        pv = panel.values
        locs = panel.index.get_indexer(index)
        out = np.zeros((len(index), lookback, pv.shape[1]), dtype=np.float32)
        for i, loc in enumerate(locs):
            if loc < 0:
                continue
            start = max(0, loc - lookback + 1)
            seg = pv[start : loc + 1]
            out[i, lookback - len(seg) :] = seg
        return out

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None):
        _require_torch()
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        p = self.cfg
        dev = p.get("device", "auto")
        device = torch.device("cuda" if (dev == "auto" and torch.cuda.is_available()) or dev == "cuda" else "cpu")

        Xw = self._windows(X.index)
        # 标准化(按训练集统计), 保存供推理使用
        self.mu_ = Xw.reshape(-1, Xw.shape[-1]).mean(0)
        self.sd_ = Xw.reshape(-1, Xw.shape[-1]).std(0) + 1e-8
        Xw = (Xw - self.mu_) / self.sd_

        n_feats = Xw.shape[-1]
        self.model = self._build_model(n_feats).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(p.get("lr", 1e-3)))
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        ds = TensorDataset(
            torch.tensor(Xw), torch.tensor(y.astype(np.float32)), torch.tensor(w.astype(np.float32))
        )
        dl = DataLoader(ds, batch_size=int(p.get("batch_size", 256)), shuffle=True)

        self.model.train()
        for _ in range(int(p.get("epochs", 15))):
            for xb, yb, wb in dl:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                opt.zero_grad()
                logit = self.model(xb)
                loss = (loss_fn(logit, yb) * wb).mean()
                loss.backward()
                opt.step()
        self.device = device
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        Xw = self._windows(X.index)
        Xw = (Xw - self.mu_) / self.sd_
        self.model.eval()
        with torch.no_grad():
            logit = self.model(torch.tensor(Xw).to(self.device))
            prob = torch.sigmoid(logit).cpu().numpy()
        return prob
