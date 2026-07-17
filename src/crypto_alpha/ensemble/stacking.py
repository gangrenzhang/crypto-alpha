"""Stacking(堆叠泛化)集成。

思想: 用 Purged K-Fold 产生每个专家的 out-of-fold(OOF) 概率作为二层特征,
再训练元学习器融合。因误差不相关, 融合后方差下降、稳健性与准度提升。

无泄漏纪律(两层都要):
- 一层: 每个专家在 Purged K-Fold 上产出干净 OOF。
- 二层: 元学习器**也**走 Purged K-Fold 交叉拟合(nested OOF), 从不在自己训练过的
  行上做预测。因此 oof_proba() 返回的融合概率对二层同样是样本外, 主面板指标与
  CPCV 路径口径一致(此前二层"自训自评"会让 AUC/Brier/回测系统性虚高)。
- 弱专家剪枝: OOF AUC 明显低于 0.5(随机)的专家会被自动剔除, 避免拖累集成。
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
        self.dropped_experts: list[tuple[str, float]] = []

    def _new_meta(self):
        kind = self.cfg.get("meta_learner", "logistic")
        if kind == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(C=float(self.cfg.get("C", 1.0)), max_iter=500)
        else:
            import lightgbm as lgb

            return lgb.LGBMClassifier(n_estimators=100, num_leaves=7, learning_rate=0.05, verbose=-1)

    @staticmethod
    def _fit_meta(meta, Xm: np.ndarray, ym: np.ndarray, w: np.ndarray | None):
        """元学习器拟合, 支持样本权重(与一层口径一致); 不支持时优雅回退。"""
        if w is None:
            meta.fit(Xm, ym)
            return
        try:
            meta.fit(Xm, ym, sample_weight=w)
        except TypeError:
            meta.fit(Xm, ym)

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

    def _prune_weak_experts(self, oof: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """剔除 OOF AUC < min_expert_auc 的弱专家(至少保留 1 个)。"""
        min_auc = float(self.cfg.get("min_expert_auc", 0.5))
        if min_auc <= 0 or len(self.experts) <= 1:
            return oof
        from sklearn.metrics import roc_auc_score

        mask = oof.notna().all(axis=1).values
        yv = np.asarray(y)[mask]
        aucs: dict[str, float] = {}
        for e in self.experts:
            col = oof[e.name].values[mask]
            try:
                aucs[e.name] = float(roc_auc_score(yv, col))
            except Exception:
                aucs[e.name] = float("nan")

        keep = [e for e in self.experts if not (aucs[e.name] < min_auc)]  # NaN 视为保留
        if not keep:  # 全员低于阈值时, 保留 AUC 最高者, 避免空集成
            best = max(self.experts, key=lambda e: (aucs[e.name] if np.isfinite(aucs[e.name]) else -1))
            keep = [best]
        self.dropped_experts = [(e.name, aucs[e.name]) for e in self.experts if e not in keep]
        if self.dropped_experts:
            info = ", ".join(f"{n}(auc={a:.3f})" for n, a in self.dropped_experts)
            print(f"[ensemble] 剔除弱专家: {info}")
        self.experts = keep
        return oof[[e.name for e in keep]]

    def _meta_cross_fit(
        self, oof: pd.DataFrame, y: np.ndarray, t1: pd.Series,
        sample_weight: np.ndarray | None, n_splits: int, embargo_pct: float,
    ) -> np.ndarray:
        """二层 nested OOF: 元学习器在 Purged K-Fold 上交叉拟合, 得到无泄漏融合概率。"""
        mask = oof.notna().all(axis=1)
        idx = oof.index[mask.values]
        pred = pd.Series(np.nan, index=oof.index)
        if len(idx) < n_splits * 2:  # 样本过少无法交叉拟合 -> 退回单折(会有轻微乐观, 但已剪枝)
            m = self._new_meta()
            w = None if sample_weight is None else sample_weight[mask.values]
            self._fit_meta(m, oof.loc[idx].values, y[mask.values], w)
            pred.loc[idx] = m.predict_proba(oof.loc[idx].values)[:, 1]
            return pred.values

        Xm = oof.loc[idx]
        ym = np.asarray(y)[mask.values]
        wm = None if sample_weight is None else np.asarray(sample_weight)[mask.values]
        t1m = t1.loc[idx]
        pkf = PurgedKFold(n_splits=n_splits, t1=t1m, embargo_pct=embargo_pct)
        Xv = Xm.values
        for tr, te in pkf.split(Xm):
            m = self._new_meta()
            self._fit_meta(m, Xv[tr], ym[tr], None if wm is None else wm[tr])
            pred.loc[idx[te]] = m.predict_proba(Xv[te])[:, 1]
        return pred.values

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, t1: pd.Series,
        sample_weight: np.ndarray | None = None, n_splits: int = 6, embargo_pct: float = 0.01,
    ):
        # 1) 一层 OOF 特征
        oof = self.build_oof(X, y, t1, sample_weight, n_splits, embargo_pct)
        # 2) 弱专家剪枝(基于 OOF AUC)
        oof = self._prune_weak_experts(oof, y)
        self.oof_ = oof
        # 3) 二层 nested OOF: 无泄漏融合概率(用于校准/回测/评估)
        self.meta_oof_ = self._meta_cross_fit(oof, y, t1, sample_weight, n_splits, embargo_pct)
        # 4) 部署用元学习器: 在全部干净 OOF 行上拟合一次
        mask = oof.notna().all(axis=1).values
        self.meta_ = self._new_meta()
        w = None if sample_weight is None else np.asarray(sample_weight)[mask]
        self._fit_meta(self.meta_, oof.values[mask], np.asarray(y)[mask], w)
        # 5) 各(保留的)专家在全量数据上重训, 供部署推理
        for e in self.experts:
            e.fit(X, y, sample_weight=(None if sample_weight is None else sample_weight))
        return self

    def base_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {e.name: e.predict_proba(X) for e in self.experts}, index=X.index
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base = self.base_proba(X)
        return self.meta_.predict_proba(base.values)[:, 1]

    def oof_proba(self) -> np.ndarray:
        """返回二层 nested OOF 融合概率(对元学习器亦为样本外, 无泄漏)。"""
        return self.meta_oof_
