"""专家1: 梯度提升树 (LightGBM)。

角色: 捕捉表格特征间的非线性交互, 稳健、抗过拟合, 是集成的压舱石。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseExpert


class GBDTExpert(BaseExpert):
    name = "gbdt"
    needs_panel = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None):
        import lightgbm as lgb

        p = self.cfg
        self.model = lgb.LGBMClassifier(
            n_estimators=int(p.get("n_estimators", 400)),
            learning_rate=float(p.get("learning_rate", 0.03)),
            num_leaves=int(p.get("num_leaves", 31)),
            max_depth=int(p.get("max_depth", -1)),
            subsample=float(p.get("subsample", 0.8)),
            subsample_freq=1,
            colsample_bytree=float(p.get("colsample_bytree", 0.8)),
            min_child_samples=int(p.get("min_child_samples", 50)),
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        Xv = X[self.feature_cols].astype(float).fillna(0.0)
        self.model.fit(Xv, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xv = X[self.feature_cols].astype(float).fillna(0.0)
        return self.model.predict_proba(Xv)[:, 1]
