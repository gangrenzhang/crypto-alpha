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
        if str(self.method).lower() == "auto":
            raise ValueError(
                "ProbabilityCalibrator 不接受 method='auto'; "
                "请先 resolve_calibrator_method 或走 fit_deploy_calibrator_and_conformal"
            )
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


def count_unique_prob_levels(prob: np.ndarray, decimals: int = 6) -> int:
    """校准后概率的唯一台阶数(忽略 NaN)。"""
    p = np.asarray(prob, dtype=float)
    p = p[~np.isnan(p)]
    if len(p) == 0:
        return 0
    return int(len(np.unique(np.round(p, int(decimals)))))


def resolve_calibrator_method(
    method: str,
    prob: np.ndarray,
    y: np.ndarray,
    min_unique_levels: int = 20,
) -> tuple[str, list[str]]:
    """解析校准方法。``auto``: 先试 isotonic, 唯一台阶过少则回退 sigmoid。

    单类标签无法拟合 Platt 时打 ``auto_fallback_blocked_single_class`` 并保留 isotonic。
    显式 ``isotonic``/``sigmoid`` 不自动切换(即使台阶塌缩)。
    """
    tags: list[str] = []
    m = str(method or "isotonic").strip().lower()
    if m in ("", "isotonic", "sigmoid", "platt"):
        resolved = "sigmoid" if m in ("sigmoid", "platt") else "isotonic"
        tags.append(f"deploy_cal_method={resolved}")
        return resolved, tags
    if m != "auto":
        raise ValueError(
            f"未知 calibration.method={method!r}; 请用 auto / isotonic / sigmoid"
        )

    p = np.asarray(prob, dtype=float)
    yy = np.asarray(y, dtype=int)
    mask = ~np.isnan(p)
    p, yy = p[mask], yy[mask]
    n_uniq = 0
    if len(p) > 0:
        try:
            cal_try = ProbabilityCalibrator("isotonic").fit(p, yy)
            n_uniq = count_unique_prob_levels(cal_try.transform(p))
        except Exception:
            n_uniq = 0
    thr = int(min_unique_levels)
    if n_uniq >= thr:
        tags.append("deploy_cal_method=isotonic")
        tags.append(f"auto_kept_isotonic(n_unique={n_uniq})")
        return "isotonic", tags
    if len(np.unique(yy)) < 2:
        tags.append("deploy_cal_method=isotonic")
        tags.append("auto_fallback_blocked_single_class")
        return "isotonic", tags
    tags.append("deploy_cal_method=sigmoid")
    tags.append(f"auto_fallback_sigmoid(n_unique={n_uniq}<{thr})")
    return "sigmoid", tags


class ConformalBinary:
    """分裂式保形预测(二分类)。nonconformity = 1 - p(真实类)。

    ``min_margin``: 在「预测集恰含一类」之外, 要求 ``|p-0.5| >= min_margin``。
    默认 0 保持旧行为; >0 时可避免弱模型下略过 0.55 仍全部 confident。
    """

    def __init__(self, alpha: float = 0.1, min_margin: float = 0.0):
        self.alpha = float(alpha)
        self.min_margin = float(max(min_margin, 0.0))

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
        if self.min_margin > 0.0:
            confident = np.asarray(confident, dtype=bool) & (
                np.abs(prob - 0.5) >= self.min_margin
            )
        return {"in_class1": in_1, "in_class0": in_0, "confident": confident}


def cross_fitted_calibrated(
    prob: np.ndarray, y: np.ndarray, t1, method: str = "isotonic",
    n_splits: int = 5, embargo_pct: float = 0.01,
    min_unique_levels: int = 20,
) -> np.ndarray:
    """交叉拟合校准: 用 Purged K-Fold 在训练折拟合校准器、在测试折 transform,
    得到**无泄漏**的校准概率, 供报告/回测评估(避免"拟合即评估"的乐观偏差)。

    prob 与 y、t1 一一对齐(prob 中的 NaN 表示该行无 OOF)。返回同长度数组,
    不可交叉拟合的位置为 NaN。部署用的最终校准器仍在全量上单独拟合。
    ``method=auto`` 时按全量有效 OOF 探台阶后固定为 isotonic 或 sigmoid。
    """
    from ..validation.purged_kfold import PurgedKFold
    import pandas as pd

    prob = np.asarray(prob, dtype=float)
    yy = np.asarray(y, dtype=int)
    method, _ = resolve_calibrator_method(
        method, prob, yy, min_unique_levels=min_unique_levels,
    )
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
    min_margin: float = 0.0,
) -> np.ndarray:
    """交叉拟合保形弃权旗标: 训练折拟合 ConformalBinary, 测试折出 confident。

    与回测/评估对齐, 避免用部署保形器在同一批 OOF 上自评。无法交叉拟合时默认 True
    (不额外弃权, 由概率阈值单独把关)。

    .. note::
        若 ``prob`` 已是另一轮交叉拟合的校准输出, 会与保形 CF 形成二阶依赖。
        主路径请改用 :func:`cross_fitted_calibrated_and_conformal`。
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
        conf = ConformalBinary(alpha=alpha, min_margin=min_margin).fit(
            prob[pos[tr]], yy[pos[tr]],
        )
        out[pos[te]] = conf.predict_set(prob[pos[te]])["confident"]
    return out


def cross_fitted_calibrated_and_conformal(
    prob: np.ndarray,
    y: np.ndarray,
    t1,
    method: str = "isotonic",
    alpha: float = 0.1,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    min_margin: float = 0.0,
    min_unique_levels: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """同一 Purged 折内联合产出校准概率与保形旗标(主路径评估/回测用)。

    消除「先 ``cross_fitted_calibrated`` 再对结果做 ``cross_fitted_conformal_flags``」
    的二阶依赖: 后者训练集中的校准分可能来自「曾见过测试点」的校准器。

    每折严格顺序(与部署「先校准再保形」同形, 但用交叉拟合代替时间切分)::

        fit(cal | train raw) → cal(train), cal(test)
        fit(conf | cal(train)) → confident(cal(test))

    返回 ``(oof_cal, conf_flags, tags)``:

    - ``oof_cal``: 与 :func:`cross_fitted_calibrated` 同语义; 不可 CF 时全 NaN
    - ``conf_flags``: 默认 False; 仅在该折成功拟合保形器后写入测试折
    - ``tags``: 如某折因单类/过少跳过保形 → ``conformal_cf_fold_skipped``;
      样本不足以 CF → ``conformal_cf_insufficient_samples``
    """
    from ..validation.purged_kfold import PurgedKFold
    import pandas as pd

    tags: list[str] = []
    prob = np.asarray(prob, dtype=float)
    yy = np.asarray(y, dtype=int)
    method, method_tags = resolve_calibrator_method(
        method, prob, yy, min_unique_levels=min_unique_levels,
    )
    tags.extend(method_tags)
    oof_cal = np.full(len(prob), np.nan)
    # 默认不自信: 仅在该折成功拟合保形器后写入 True。
    # 跳过折若保持 True 会让研究回测在无覆盖折上偏多开仓。
    conf_flags = np.zeros(len(prob), dtype=bool)
    pos = np.where(~np.isnan(prob))[0]
    if len(pos) < n_splits * 2:
        tags.append(f"conformal_cf_insufficient_samples(n={len(pos)})")
        return oof_cal, conf_flags, tags

    t1 = pd.Series(t1)
    idx = t1.index[pos]
    t1m = t1.loc[idx]
    Xdummy = pd.DataFrame(index=idx)
    pkf = PurgedKFold(n_splits=n_splits, t1=t1m, embargo_pct=embargo_pct)
    n_conf_skip = 0
    for tr, te in pkf.split(Xdummy):
        cal = ProbabilityCalibrator(method).fit(prob[pos[tr]], yy[pos[tr]])
        p_tr = cal.transform(prob[pos[tr]])
        p_te = cal.transform(prob[pos[te]])
        oof_cal[pos[te]] = p_te
        if len(np.unique(yy[pos[tr]])) < 2 or len(tr) < 10:
            n_conf_skip += 1
            # 跳过折: 保持 False(不确定则弃权)
            continue
        conf = ConformalBinary(alpha=alpha, min_margin=min_margin).fit(p_tr, yy[pos[tr]])
        conf_flags[pos[te]] = conf.predict_set(p_te)["confident"]
    if n_conf_skip > 0:
        tags.append(f"conformal_cf_fold_skipped(n_folds={n_conf_skip})")
    return oof_cal, conf_flags, tags


def fit_deploy_calibrator_and_conformal(
    prob: np.ndarray, y: np.ndarray, method: str = "isotonic",
    alpha: float = 0.1, conformal_frac: float = 0.3,
    min_margin: float = 0.0,
    min_unique_levels: int = 20,
) -> tuple[ProbabilityCalibrator, ConformalBinary, list[str]]:
    """部署用校准器 + 保形器: **时间切分**独立保形集, 保证 split conformal 覆盖率语义。

    - 较早 (1-conformal_frac) 的有效 OOF 拟合校准器;
    - 较晚 conformal_frac 在**该校准器变换后**的概率上拟合保形器;
    - 返回的校准器即部署所用(与保形同一基), 不再在全量上重拟合以免破坏分割。
    - 第三返回值: degradations 标签(如 ``n<40`` 同批回退、``auto_fallback_sigmoid``)。
    - ``method=auto``: 见 ``resolve_calibrator_method``。
    """
    tags: list[str] = []
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    m = ~np.isnan(prob)
    p, yy = prob[m], y[m]
    n = len(p)

    resolved, method_tags = resolve_calibrator_method(
        method, p, yy, min_unique_levels=min_unique_levels,
    )
    tags.extend(method_tags)

    if n < 40:
        tags.append(f"deploy_cal_conformal_fallback_insample(n={n})")
        cal = ProbabilityCalibrator(method=resolved).fit(p, yy)
        conf = ConformalBinary(alpha=alpha, min_margin=min_margin).fit(
            cal.transform(p), yy,
        )
        return cal, conf, tags

    frac = float(np.clip(conformal_frac, 0.15, 0.5))
    n_conf = max(int(n * frac), 20)
    n_cal = n - n_conf
    # 假定 prob 与事件时间顺序一致(OOF 按 X.index 排列)
    # auto 解析用全量有效 OOF 探台阶; 真正 fit 仍只用较早 n_cal 段
    if str(method).strip().lower() == "auto":
        resolved2, tags2 = resolve_calibrator_method(
            "auto", p[:n_cal], yy[:n_cal], min_unique_levels=min_unique_levels,
        )
        # 用校准段重解析更贴合部署 fit; 替换首轮 tags 中的 method 相关
        tags = [t for t in tags if not (
            t.startswith("deploy_cal_method=")
            or t.startswith("auto_kept_")
            or t.startswith("auto_fallback_")
        )]
        tags.extend(tags2)
        resolved = resolved2
    cal = ProbabilityCalibrator(method=resolved).fit(p[:n_cal], yy[:n_cal])
    conf_p = cal.transform(p[n_cal:])
    conf = ConformalBinary(alpha=alpha, min_margin=min_margin).fit(
        conf_p, yy[n_cal:],
    )
    return cal, conf, tags


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
