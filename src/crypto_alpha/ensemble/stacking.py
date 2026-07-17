"""Stacking(堆叠泛化)集成。

思想: 用 Purged K-Fold 产生每个专家的 out-of-fold(OOF) 概率作为二层特征,
再训练元学习器融合。因误差不相关, 融合后方差下降、稳健性与准度提升。

无泄漏纪律(两层都要):
- 一层: 每个专家在 Purged K-Fold 上产出干净 OOF。
- 二层: 元学习器**也**走 Purged K-Fold 交叉拟合(nested OOF), 从不在自己训练过的
  行上做预测。因此 oof_proba() 返回的融合概率对二层同样是样本外, 主面板指标与
  CPCV 路径口径一致(此前二层"自训自评"会让 AUC/Brier/回测系统性虚高)。
- 弱专家剪枝: OOF AUC 明显低于 0.5(随机)的专家会被自动剔除, 避免拖累集成。
- 伪 OOF 专家(pseudo_oof=True, 如 LLM fit 只加载 adapter): 默认**不进入元学习器**,
  分数仅保留供诊断; 避免污染 nested OOF / 回测 / 校准。
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
        self.degradations: list[str] = []
        #: 被排除出元学习器的伪 OOF 分数(列=专家名), 仅诊断用
        self.pseudo_oof_: pd.DataFrame = pd.DataFrame()
        #: 与 X/y 对齐: 剪枝后用于报告/回测的后半窗(选型半窗不进评估)
        self.prune_eval_mask_: np.ndarray | None = None

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
        """用 Purged K-Fold 生成各专家的 OOF 概率矩阵。

        伪 OOF 专家(pseudo_oof=True, fit 为 no-op)只 fit+predict 一次, 广播到所有折,
        避免重复加载模型且诚实标记非真正 CV。
        """
        pkf = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)
        oof = pd.DataFrame(index=X.index, columns=[e.name for e in self.experts], dtype=float)

        # 分离伪 OOF 专家: 非真正 CV, 只跑一次
        pseudo_experts = [e for e in self.experts if getattr(e, "pseudo_oof", False)]
        regular_experts = [e for e in self.experts if not getattr(e, "pseudo_oof", False)]
        for e in pseudo_experts:
            tag = f"{e.name}:pseudo_oof_not_cross_validated"
            if tag not in self.degradations:
                self.degradations.append(tag)
            clone = e.clone()
            clone.fit(X, y, sample_weight=sample_weight)
            self._sync_degraded(e, clone)
            prob = clone.predict_proba(X)
            # 广播到所有折(标注为伪 OOF, 不做真正交叉验证)
            col_idx = oof.columns.get_loc(e.name)
            for _tr, te in pkf.split(X):
                oof.iloc[te, col_idx] = prob[te]

        for tr, te in pkf.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr = y[tr]
            wtr = None if sample_weight is None else sample_weight[tr]
            # DeepTS 早停 val 不得用测试折之后的样本: 传入测试折最早时刻作 cutoff
            es_cutoff = X.index[te].min()
            for e in regular_experts:
                clone = e.clone()
                clone.fit(Xtr, ytr, sample_weight=wtr, es_cutoff_time=es_cutoff)
                self._sync_degraded(e, clone)
                oof.iloc[te, oof.columns.get_loc(e.name)] = clone.predict_proba(Xte)
        return oof.astype(float)

    def _sync_degraded(self, expert: BaseExpert, clone: BaseExpert) -> None:
        """折内 clone 的降级状态回写到原专家并记入 degradations(剪枝后也不丢失)。"""
        if not getattr(clone, "degraded", False):
            return
        expert.degraded = True
        expert.degraded_reason = getattr(clone, "degraded_reason", "degraded")
        tag = f"{expert.name}:{expert.degraded_reason}"
        if tag not in self.degradations:
            self.degradations.append(tag)

    def _prune_weak_experts(self, oof: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """剔除弱专家: 前半窗 AUC 选型; 后半窗留给报告/回测(防 selection-on-evaluation)。

        元学习器仍在保留专家的完整 OOF 上 nested 交叉拟合(部署用); ``prune_eval_mask_``
        标记后半窗, 供主路径集成报告、专家 base_report 与回测使用(同窗可比)。
        """
        n_rows = len(oof)
        mask = oof.notna().all(axis=1).values
        pos = np.where(mask)[0]
        min_auc = float(self.cfg.get("min_expert_auc", 0.5))
        if min_auc <= 0 or len(self.experts) <= 1:
            self.prune_eval_mask_ = mask.copy()
            return oof
        from sklearn.metrics import roc_auc_score

        # 按行序(=事件时间序): 前半选型, 后半评估
        if len(pos) < 20:
            select = mask
            eval_m = mask.copy()
            tag = f"expert_prune_full_window_selection(n={int(mask.sum())})"
            if tag not in self.degradations:
                self.degradations.append(tag)
            print(f"[ensemble] WARN: {tag}; 选型与评估同窗, 指标可能偏乐观")
        else:
            n_sel = max(len(pos) // 2, 10)
            select = np.zeros(n_rows, dtype=bool)
            select[pos[:n_sel]] = True
            eval_m = np.zeros(n_rows, dtype=bool)
            eval_m[pos[n_sel:]] = True
        self.prune_eval_mask_ = eval_m

        yv = np.asarray(y)[select]
        aucs: dict[str, float] = {}
        for e in self.experts:
            col = oof[e.name].values[select]
            try:
                aucs[e.name] = float(roc_auc_score(yv, col))
            except Exception:
                aucs[e.name] = float("nan")

        keep = [e for e in self.experts if not (aucs[e.name] < min_auc)]  # NaN 视为保留
        if not keep:  # 全员低于阈值时, 保留 AUC 最高者, 避免空集成
            best = max(self.experts, key=lambda e: (aucs[e.name] if np.isfinite(aucs[e.name]) else -1))
            keep = [best]
        dropped = [e for e in self.experts if e not in keep]
        self.dropped_experts = [(e.name, aucs[e.name]) for e in dropped]
        # 被剪枝专家的 degraded 已在 build_oof 写入 degradations; 再记 AUC 原因
        for e in dropped:
            tag = f"{e.name}:dropped_low_auc({aucs[e.name]:.3f})"
            if tag not in self.degradations:
                self.degradations.append(tag)
        if self.dropped_experts:
            info = ", ".join(f"{n}(auc={a:.3f})" for n, a in self.dropped_experts)
            print(f"[ensemble] 剔除弱专家(选型半窗): {info}")
        self.experts = keep
        return oof[[e.name for e in keep]]

    def _exclude_pseudo_oof_from_meta(self, oof: pd.DataFrame) -> pd.DataFrame:
        """默认把伪 OOF 专家移出元学习器, 防止冻结 adapter 污染评估/回测。

        分数写入 ``pseudo_oof_`` 供 base_report 诊断; 配置
        ``exclude_pseudo_oof_from_meta: false`` 可关闭(不推荐)。
        """
        if not bool(self.cfg.get("exclude_pseudo_oof_from_meta", True)):
            return oof
        pseudo = [e for e in self.experts if getattr(e, "pseudo_oof", False)]
        regular = [e for e in self.experts if not getattr(e, "pseudo_oof", False)]
        if not pseudo:
            return oof
        names = [e.name for e in pseudo]
        self.pseudo_oof_ = oof[names].copy()
        for n in names:
            tag = f"{n}:excluded_from_meta_pseudo_oof"
            if tag not in self.degradations:
                self.degradations.append(tag)
        print(
            f"[ensemble] WARN: 伪OOF专家不进入元学习器(避免污染 nested OOF): {names}; "
            f"分数保留于 pseudo_oof_ 供诊断"
        )
        if not regular:
            raise ValueError(
                "enabled 专家全部为伪OOF(如仅 llm), 无法构建 stacking 元学习器。"
                "请至少启用一个可折内重训的专家(gbdt / deep_ts / tsfm)。"
            )
        self.experts = regular
        return oof[[e.name for e in regular]]

    def _meta_cross_fit(
        self, oof: pd.DataFrame, y: np.ndarray, t1: pd.Series,
        sample_weight: np.ndarray | None, n_splits: int, embargo_pct: float,
    ) -> np.ndarray:
        """二层 nested OOF: 元学习器在 Purged K-Fold 上交叉拟合, 得到无泄漏融合概率。"""
        mask = oof.notna().all(axis=1)
        idx = oof.index[mask.values]
        pred = pd.Series(np.nan, index=oof.index)
        if len(idx) < n_splits * 2:
            # 样本过少无法 nested CV → 同批 fit+predict(评估偏乐观); 显式降级可追踪
            tag = f"meta_nested_oof_fallback_insample(n={len(idx)},n_splits={n_splits})"
            if tag not in self.degradations:
                self.degradations.append(tag)
            print(
                f"[ensemble] WARN: {tag}; 二层 OOF 退回自训自评, 指标可能偏乐观"
            )
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
        self.pseudo_oof_ = pd.DataFrame(index=X.index)
        self.prune_eval_mask_ = None
        # 1) 一层 OOF 特征(伪 OOF 专家仅冻结推理并记 degradations)
        oof = self.build_oof(X, y, t1, sample_weight, n_splits, embargo_pct)
        # 2) 伪 OOF 默认不进元学习器(护栏)
        oof = self._exclude_pseudo_oof_from_meta(oof)
        # 3) 弱专家剪枝(前半选型 / 后半评估; degraded 已在 build_oof 收集)
        oof = self._prune_weak_experts(oof, y)
        self.oof_ = oof
        # 4) 二层 nested OOF: 无泄漏融合概率(用于校准/回测/评估)
        self.meta_oof_ = self._meta_cross_fit(oof, y, t1, sample_weight, n_splits, embargo_pct)
        # 5) 部署用元学习器: 在全部干净 OOF 行上拟合一次
        mask = oof.notna().all(axis=1).values
        self.meta_ = self._new_meta()
        w = None if sample_weight is None else np.asarray(sample_weight)[mask]
        self._fit_meta(self.meta_, oof.values[mask], np.asarray(y)[mask], w)
        # 6) 各(进入 meta 的)专家在全量数据上重训, 供部署推理
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
