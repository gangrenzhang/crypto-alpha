"""Stacking(堆叠泛化)集成。

思想: 用 Purged K-Fold 产生每个专家的 out-of-fold(OOF) 概率作为二层特征,
再训练元学习器融合。因误差不相关, 融合后方差下降、稳健性与准度提升。
OOF 严格无泄漏, 保证元学习器不会看到自己训练样本的预测。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..experts.base import BaseExpert
from ..validation.purged_kfold import PurgedKFold


class StackingEnsemble:
    def __init__(self, experts: list[BaseExpert], cfg: dict, seed: int = 42):
        self.experts = experts
        self.cfg = cfg
        self.seed = seed

    def _new_meta(self):
        kind = self.cfg.get("meta_learner", "logistic")
        if kind == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(C=float(self.cfg.get("C", 1.0)), max_iter=500)
        else:
            import lightgbm as lgb

            return lgb.LGBMClassifier(n_estimators=100, num_leaves=7, learning_rate=0.05, verbose=-1)

    def build_oof(
        self, X: pd.DataFrame, y: np.ndarray, t1: pd.Series, sample_weight: np.ndarray | None,
        n_splits: int, embargo_pct: float,
    ) -> pd.DataFrame:
        """用 Purged K-Fold 生成各专家的 OOF 概率矩阵。"""
        pkf = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)
        oof = pd.DataFrame(index=X.index, columns=[e.name for e in self.experts], dtype=float)

        for tr, te in pkf.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr = y[tr]
            wtr = None if sample_weight is None else sample_weight[tr]
            for e in self.experts:
                clone = e.clone()
                clone.fit(Xtr, ytr, sample_weight=wtr)
                oof.iloc[te, oof.columns.get_loc(e.name)] = clone.predict_proba(Xte)
        return oof.astype(float)

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, t1: pd.Series,
        sample_weight: np.ndarray | None = None, n_splits: int = 6, embargo_pct: float = 0.01,
    ):
        # 1) OOF 二层特征 -> 训练元学习器
        oof = self.build_oof(X, y, t1, sample_weight, n_splits, embargo_pct)
        mask = oof.notna().all(axis=1).values
        self.meta_ = self._new_meta()
        self.meta_.fit(oof.values[mask], y[mask])
        self.oof_ = oof
        # 2) 各专家在全量数据上重训, 供部署推理
        for e in self.experts:
            w = None if sample_weight is None else sample_weight
            e.fit(X, y, sample_weight=w)
        return self

    def base_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {e.name: e.predict_proba(X) for e in self.experts}, index=X.index
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base = self.base_proba(X)
        return self.meta_.predict_proba(base.values)[:, 1]

    def oof_proba(self) -> np.ndarray:
        """返回元学习器在 OOF 上的融合概率(用于校准与回测, 无泄漏)。"""
        oof = self.oof_
        mask = oof.notna().all(axis=1)
        pred = pd.Series(np.nan, index=oof.index)
        pred[mask] = self.meta_.predict_proba(oof.values[mask.values])[:, 1]
        return pred.values
