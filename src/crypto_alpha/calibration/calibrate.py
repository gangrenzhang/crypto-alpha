"""概率校准 + 保形预测。

- 校准(Isotonic/Platt): 让 "模型说 70%" 真的对应约 70% 的经验频率, 否则仓位管理失真。
- 保形预测(Split Conformal): 给出有覆盖率保证的预测集; 当预测集不确定时可 "弃权观望",
  从而只在高置信时下注, 显著提升下单子集的实际胜率。
"""
from __future__ import annotations

import numpy as np


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method

    def fit(self, prob: np.ndarray, y: np.ndarray):
        prob = np.asarray(prob, dtype=float)
        y = np.asarray(y, dtype=int)
        m = ~np.isnan(prob)
        prob, y = prob[m], y[m]
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self._m = IsotonicRegression(out_of_bounds="clip")
            self._m.fit(prob, y)
        else:  # sigmoid / Platt
            from sklearn.linear_model import LogisticRegression

            self._m = LogisticRegression()
            self._m.fit(prob.reshape(-1, 1), y)
        return self

    def transform(self, prob: np.ndarray) -> np.ndarray:
        prob = np.asarray(prob, dtype=float)
        if self.method == "isotonic":
            return self._m.predict(prob)
        return self._m.predict_proba(prob.reshape(-1, 1))[:, 1]


class ConformalBinary:
    """分裂式保形预测(二分类)。nonconformity = 1 - p(真实类)。"""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, prob: np.ndarray, y: np.ndarray):
        prob = np.asarray(prob, dtype=float)
        y = np.asarray(y, dtype=int)
        p_true = np.where(y == 1, prob, 1 - prob)
        scores = 1 - p_true
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat_ = float(np.quantile(scores, min(q, 1.0)))
        return self

    def predict_set(self, prob: np.ndarray) -> dict:
        """返回每个样本的预测集包含情况与是否可下注(仅含单一类=可下注)。"""
        prob = np.asarray(prob, dtype=float)
        in_1 = (1 - prob) <= self.qhat_
        in_0 = prob <= self.qhat_
        confident = in_1 ^ in_0  # 恰好一个类 => 高置信
        return {"in_class1": in_1, "in_class0": in_0, "confident": confident}


def cross_fitted_calibrated(
    prob: np.ndarray, y: np.ndarray, t1, method: str = "isotonic",
    n_splits: int = 5, embargo_pct: float = 0.01,
) -> np.ndarray:
    """交叉拟合校准: 用 Purged K-Fold 在训练折拟合校准器、在测试折 transform,
    得到**无泄漏**的校准概率, 供报告/回测评估(避免"拟合即评估"的乐观偏差)。

    prob 与 y、t1 一一对齐(prob 中的 NaN 表示该行无 OOF)。返回同长度数组,
    不可交叉拟合的位置为 NaN。部署用的最终校准器仍在全量上单独拟合。
    """
    from ..validation.purged_kfold import PurgedKFold
    import pandas as pd

    prob = np.asarray(prob, dtype=float)
    yy = np.asarray(y, dtype=int)
    out = np.full(len(prob), np.nan)
    pos = np.where(~np.isnan(prob))[0]
    if len(pos) < n_splits * 2:
        return out  # 样本过少, 不足以交叉拟合(调用方会回退)

    t1 = pd.Series(t1)
    idx = t1.index[pos]
    t1m = t1.loc[idx]
    Xdummy = pd.DataFrame(index=idx)
    pkf = PurgedKFold(n_splits=n_splits, t1=t1m, embargo_pct=embargo_pct)
    for tr, te in pkf.split(Xdummy):
        cal = ProbabilityCalibrator(method).fit(prob[pos[tr]], yy[pos[tr]])
        out[pos[te]] = cal.transform(prob[pos[te]])
    return out


def cross_fitted_conformal_flags(
    prob: np.ndarray, y: np.ndarray, t1, alpha: float = 0.1,
    n_splits: int = 5, embargo_pct: float = 0.01,
) -> np.ndarray:
    """交叉拟合保形弃权旗标: 训练折拟合 ConformalBinary, 测试折出 confident。

    与回测/评估对齐, 避免用部署保形器在同一批 OOF 上自评。无法交叉拟合时默认 True
    (不额外弃权, 由概率阈值单独把关)。
    """
    from ..validation.purged_kfold import PurgedKFold
    import pandas as pd

    prob = np.asarray(prob, dtype=float)
    yy = np.asarray(y, dtype=int)
    out = np.ones(len(prob), dtype=bool)
    pos = np.where(~np.isnan(prob))[0]
    if len(pos) < n_splits * 2:
        return out

    t1 = pd.Series(t1)
    idx = t1.index[pos]
    t1m = t1.loc[idx]
    Xdummy = pd.DataFrame(index=idx)
    pkf = PurgedKFold(n_splits=n_splits, t1=t1m, embargo_pct=embargo_pct)
    for tr, te in pkf.split(Xdummy):
        if len(np.unique(yy[pos[tr]])) < 2 or len(tr) < 10:
            continue
        conf = ConformalBinary(alpha=alpha).fit(prob[pos[tr]], yy[pos[tr]])
        out[pos[te]] = conf.predict_set(prob[pos[te]])["confident"]
    return out


def fit_deploy_calibrator_and_conformal(
    prob: np.ndarray, y: np.ndarray, method: str = "isotonic",
    alpha: float = 0.1, conformal_frac: float = 0.3,
) -> tuple[ProbabilityCalibrator, ConformalBinary]:
    """部署用校准器 + 保形器: **时间切分**独立保形集, 保证 split conformal 覆盖率语义。

    - 较早 (1-conformal_frac) 的有效 OOF 拟合校准器;
    - 较晚 conformal_frac 在**该校准器变换后**的概率上拟合保形器;
    - 返回的校准器即部署所用(与保形同一基), 不再在全量上重拟合以免破坏分割。
    """
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    m = ~np.isnan(prob)
    p, yy = prob[m], y[m]
    n = len(p)
    if n < 40:
        cal = ProbabilityCalibrator(method=method).fit(p, yy)
        conf = ConformalBinary(alpha=alpha).fit(cal.transform(p), yy)
        return cal, conf

    frac = float(np.clip(conformal_frac, 0.15, 0.5))
    n_conf = max(int(n * frac), 20)
    n_cal = n - n_conf
    # 假定 prob 与事件时间顺序一致(OOF 按 X.index 排列)
    cal = ProbabilityCalibrator(method=method).fit(p[:n_cal], yy[:n_cal])
    conf_p = cal.transform(p[n_cal:])
    conf = ConformalBinary(alpha=alpha).fit(conf_p, yy[n_cal:])
    return cal, conf


def classification_report_probs(prob: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    m = ~np.isnan(prob)
    prob, y = prob[m], y[m]
    out = {"n": int(m.sum())}
    try:
        out["auc"] = float(roc_auc_score(y, prob))
    except Exception:
        out["auc"] = float("nan")
    out["brier"] = float(brier_score_loss(y, prob))
    eps = 1e-6
    out["logloss"] = float(log_loss(y, np.clip(prob, eps, 1 - eps), labels=[0, 1]))
    out["accuracy"] = float(((prob > 0.5).astype(int) == y).mean())
    out["base_rate"] = float(y.mean())
    return out
