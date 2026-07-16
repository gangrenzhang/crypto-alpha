"""专家2: 时序基础模型 (Chronos / TimesFM), 支持新闻协变量。

角色: 大规模预训练的数值时序表征。对每个事件取历史收盘序列做基线预测,
再把该预测分与**新闻协变量**(情绪/互证等)一起送入概率头, 输出 P(盈利)。

关于协变量的专业说明:
- **Chronos/Chronos-Bolt 是单变量模型, 原生不支持外生协变量。** 因此新闻协变量通过
  "**协变量融合头**"注入: TSFM 预测分 ⊕ 新闻协变量 -> 逻辑回归/GBDT 头 -> 概率。
  该机制对任意基线预测器都通用, 且严格无泄漏(协变量为已 as-of 对齐的 news_* 特征)。
- **TimesFM 2.0+ 提供原生协变量接口**(forecast_with_covariates); backend=timesfm 时
  基线预测本身可纳入动态协变量(此处给出结构, 具体调用随所装版本适配)。

后端(backend):
    chronos  -> amazon/chronos-bolt-*   (需 chronos-forecasting; 单变量)
    timesfm  -> google/timesfm          (需 timesfm; 支持原生协变量)
    naive    -> 内置动量基线            (零依赖, 离线可跑, 用于打通链路/测试)
无 chronos/timesfm 依赖时自动回退到 naive。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseExpert
from ..features.news_features import NEWS_FEATURE_COLS


class TSFMExpert(BaseExpert):
    name = "tsfm"
    needs_panel = True

    # ---------------- 后端加载 ----------------
    def _load_backend(self):
        backend = self.cfg.get("backend", "chronos")
        self._kind = "naive"
        if backend == "naive":
            return
        if backend == "chronos":
            try:
                from chronos import ChronosBoltPipeline  # type: ignore
                import torch

                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self.pipe = ChronosBoltPipeline.from_pretrained(
                    self.cfg.get("model_name", "amazon/chronos-bolt-base"), device_map=dev
                )
                self._kind = "chronos"
            except Exception as e:
                print(f"[warn] Chronos 不可用({e}); TSFM 回退 naive 基线。")
        elif backend == "timesfm":
            try:
                import timesfm  # type: ignore

                self._tfm = timesfm
                self._kind = "timesfm"
            except Exception as e:
                print(f"[warn] TimesFM 不可用({e}); TSFM 回退 naive 基线。")
        else:
            print(f"[warn] 未知 TSFM backend={backend}; 使用 naive 基线。")

    # ---------------- 上下文与协变量 ----------------
    def _context_series(self, index: pd.Index) -> list[np.ndarray]:
        assert self._panel is not None, "TSFMExpert 需要先 set_panel"
        ctx_len = int(self.cfg.get("context_length", 512))
        close = self._panel["close"].astype(float)
        locs = close.index.get_indexer(index)
        return [close.values[max(0, loc - ctx_len + 1): loc + 1] for loc in locs]

    def _covariate_cols(self) -> list[str]:
        """确定协变量列: 配置 auto => 面板中存在的新闻数值特征。"""
        cfg_cols = self.cfg.get("covariate_cols", "auto")
        if cfg_cols == "auto" or cfg_cols is None:
            cand = list(NEWS_FEATURE_COLS)
        else:
            cand = list(cfg_cols)
        panel_cols = set(self._panel.columns) if self._panel is not None else set()
        return [c for c in cand if c in panel_cols]

    def _covariates(self, index: pd.Index) -> np.ndarray:
        cols = self._covariate_cols()
        if not cols:
            return np.zeros((len(index), 0), dtype=float)
        return self._panel.loc[index, cols].astype(float).fillna(0.0).values

    # ---------------- 基线预测分 ----------------
    def _raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        """side 调整后的预测收益/波动比(未校准原始分数), 由所选后端产生。"""
        horizon = int(self.cfg.get("horizon", 24))
        series = self._context_series(X.index)
        sides = X["side"].values if "side" in X.columns else np.ones(len(X))
        scores = np.zeros(len(series))
        for i, s in enumerate(series):
            if len(s) < 8:
                continue
            pred_ret = self._forecast_return(s, horizon)
            vol = np.std(np.diff(np.log(s[-64:] + 1e-9))) + 1e-6
            scores[i] = sides[i] * pred_ret / (vol * np.sqrt(horizon))
        return scores

    def _forecast_return(self, s: np.ndarray, horizon: int) -> float:
        if self._kind == "chronos":
            import torch

            fc = self.pipe.predict(torch.tensor(s, dtype=torch.float32), prediction_length=horizon)
            med = np.median(fc[0].numpy(), axis=0)
            return float(np.log(med[-1] + 1e-9) - np.log(s[-1] + 1e-9))
        if self._kind == "timesfm":
            pred = self._timesfm_forecast(s, horizon)
            return float(np.log(pred[-1] + 1e-9) - np.log(s[-1] + 1e-9))
        # naive: 近端对数收益动量外推
        logret = np.diff(np.log(s[-min(len(s) - 1, 24):] + 1e-9))
        return float(np.mean(logret) * horizon)

    def _timesfm_forecast(self, s, horizon):  # pragma: no cover - 需 timesfm 运行时
        # 原生协变量路径示意: 组织 dynamic covariates 后调用 forecast_with_covariates。
        raise NotImplementedError("请按所装 timesfm 版本实现 forecast/forecast_with_covariates")

    # ---------------- 概率头(协变量融合) ----------------
    def _new_head(self):
        if self.cfg.get("head", "logistic") == "gbdt":
            import lightgbm as lgb

            return lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05, verbose=-1)
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=300))])

    def _design_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """特征 = [TSFM 预测分, 新闻协变量...]。"""
        raw = self._raw_scores(X).reshape(-1, 1)
        cov = self._covariates(X.index)
        return np.hstack([raw, cov]) if cov.shape[1] else raw

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self._load_backend()
        Z = self._design_matrix(X)
        self._head = self._new_head()
        if self.cfg.get("head", "logistic") == "gbdt" and sample_weight is not None:
            self._head.fit(Z, y, sample_weight=sample_weight)
        else:
            self._head.fit(Z, y)
        self._cov_cols = self._covariate_cols()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._design_matrix(X)
        return self._head.predict_proba(Z)[:, 1]
