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

        # 可复现: 固定 torch 种子 + DataLoader 采样器 generator(否则跨折/多次运行结果漂移)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        g = torch.Generator()
        g.manual_seed(self.seed)

        Xw = self._windows(X.index)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        # 时间切分验证集(非随机), 用于 early stopping; 标准化仅用训练段统计
        n = len(y)
        val_frac = float(p.get("val_frac", 0.15))
        patience = int(p.get("early_stop_patience", 3))
        n_val = max(int(n * val_frac), 1) if n >= 40 and patience > 0 else 0
        n_tr = n - n_val if n_val else n

        tr_flat = Xw[:n_tr].reshape(-1, Xw.shape[-1])
        self.mu_ = tr_flat.mean(0)
        self.sd_ = tr_flat.std(0) + 1e-8
        Xw = (Xw - self.mu_) / self.sd_

        n_feats = Xw.shape[-1]
        self.model = self._build_model(n_feats).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(p.get("lr", 1e-3)))
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        def _run_epoch(loader, train: bool) -> float:
            self.model.train(train)
            total, wsum = 0.0, 0.0
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                if train:
                    opt.zero_grad()
                logit = self.model(xb)
                loss_vec = loss_fn(logit, yb) * wb
                # 与日志一致: 加权均值 sum(loss*w)/sum(w), 避免 batch 大小稀释权重
                w_batch = wb.sum().clamp_min(1e-8)
                loss = loss_vec.sum() / w_batch
                if train:
                    loss.backward()
                    opt.step()
                total += float(loss_vec.sum().detach().cpu())
                wsum += float(wb.sum().detach().cpu())
            return total / max(wsum, 1e-8)

        tr_ds = TensorDataset(
            torch.tensor(Xw[:n_tr]), torch.tensor(y[:n_tr].astype(np.float32)),
            torch.tensor(w[:n_tr].astype(np.float32)),
        )
        tr_dl = DataLoader(tr_ds, batch_size=int(p.get("batch_size", 256)), shuffle=True, generator=g)
        va_dl = None
        if n_val:
            va_ds = TensorDataset(
                torch.tensor(Xw[n_tr:]), torch.tensor(y[n_tr:].astype(np.float32)),
                torch.tensor(w[n_tr:].astype(np.float32)),
            )
            va_dl = DataLoader(va_ds, batch_size=int(p.get("batch_size", 256)), shuffle=False)

        best_state, best_val, wait = None, float("inf"), 0
        for _ in range(int(p.get("epochs", 15))):
            _run_epoch(tr_dl, train=True)
            if va_dl is None:
                continue
            vloss = _run_epoch(va_dl, train=False)
            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
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
