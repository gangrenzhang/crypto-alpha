"""CPCV 严谨评估: 生成多条回测路径的夏普分布 + 去偏夏普(DSR) + 过拟合概率(PBO)。

与主训练路径一致: 测试折概率先经**训练折 OOF 拟合的校准器**再回测;
单专家与集成分支的保形均用**训练折 OOF**(禁用 in-sample 概率拟合保形)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..validation.cpcv import CombinatorialPurgedCV
from ..validation.purged_kfold import PurgedKFold
from ..calibration.calibrate import ProbabilityCalibrator
from ..backtest.engine import (
    backtest_events,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)


def _calibrator_from_oof(
    oof: np.ndarray, y: np.ndarray, method: str,
) -> ProbabilityCalibrator | None:
    m = ~np.isnan(oof)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return None
    return ProbabilityCalibrator(method=method).fit(oof[m], y[m])


def _expert_oof_calibrator(
    expert, X: pd.DataFrame, y: np.ndarray, t1: pd.Series,
    sample_weight: np.ndarray | None, method: str,
    n_splits: int, embargo_pct: float,
) -> tuple[object, ProbabilityCalibrator | None, np.ndarray]:
    """在训练集上产出专家 OOF → 拟合校准器, 再全量重训专家供测试折推理。

    返回 (fitted_full, calibrator|None, oof_probs)。oof 供保形拟合, 避免用
    训练集内概率破坏 split-conformal 覆盖语义。

    伪 OOF 专家(pseudo_oof): fit 一次后按折填充分数(模型未折内重训), 仍返回
    折结构数组供相对比较; 调用方应记入 caveats。
    """
    pkf = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)
    oof = np.full(len(y), np.nan)

    if getattr(expert, "pseudo_oof", False):
        full = expert.clone()
        full.fit(X, y, sample_weight=sample_weight)
        prob_all = np.asarray(full.predict_proba(X), dtype=float)
        for _tr, te in pkf.split(X):
            oof[te] = prob_all[te]
        cal = _calibrator_from_oof(oof, y, method)
        return full, cal, oof

    for tr, te in pkf.split(X):
        clone = expert.clone()
        w = None if sample_weight is None else sample_weight[tr]
        clone.fit(X.iloc[tr], y[tr], sample_weight=w)
        oof[te] = clone.predict_proba(X.iloc[te])
    cal = _calibrator_from_oof(oof, y, method)
    full = expert.clone()
    full.fit(X, y, sample_weight=sample_weight)
    return full, cal, oof


def cpcv_report(cfg, ds, build_experts_fn) -> dict:
    """对每个 CPCV 划分, 在训练折训练集成、在测试折回测, 汇总路径级指标。

    同时构建 (n_configs, n_splits) 绩效矩阵用于 PBO: 配置 = 各专家 + Stacking 集成。
    """
    from ..ensemble import StackingEnsemble

    vcfg = cfg["validation"]
    ccfg = cfg["calibration"]
    method = ccfg.get("method", "isotonic")
    cv = CombinatorialPurgedCV(
        n_splits=int(vcfg["n_splits"]),
        n_test_groups=int(vcfg["n_test_groups"]),
        t1=ds.t1,
        embargo_pct=float(vcfg["embargo_pct"]),
    )

    from ..calibration.calibrate import ConformalBinary

    payoff = float(cfg["labeling"]["pt_sl"][0]) / float(cfg["labeling"]["pt_sl"][1])
    prices = ds.panel["close"] if "close" in ds.panel.columns else None
    combo_sharpes: list[float] = []
    combo_trades: list[int] = []
    combo_pnls: list[np.ndarray] = []  # 各组合成交 pnl, 供 DSR 估计经验偏度/峰度
    config_names = None
    perf_rows: list[list[float]] = []
    inner_splits = max(3, int(vcfg["n_splits"]) - 1)
    embargo = float(vcfg["embargo_pct"])
    conf_alpha = float(cfg["calibration"].get("conformal_alpha", 0.1))
    pseudo_expert_names: list[str] = []

    for split_id, (tr, te, combo) in enumerate(cv.split(ds.X)):
        Xtr, Xte = ds.X.iloc[tr], ds.X.iloc[te]
        ytr = ds.y[tr]
        wtr = ds.sample_weight[tr]
        t1tr = ds.t1.iloc[tr]

        experts = build_experts_fn(cfg, ds)
        if not pseudo_expert_names:
            pseudo_expert_names = [
                e.name for e in experts if getattr(e, "pseudo_oof", False)
            ]
        col_perf = {}
        for e in experts:
            fitted, cal, oof_tr = _expert_oof_calibrator(
                e, Xtr, ytr, t1tr, wtr, method, inner_splits, embargo,
            )
            p = fitted.predict_proba(Xte)
            if cal is not None:
                p = cal.transform(p)
            # 保形用训练折 OOF(与集成分支 / 主路径交叉拟合口径一致), 禁用 in-sample
            conf_mask = np.ones(len(p), dtype=bool)
            m_oof = ~np.isnan(oof_tr)
            if (
                cal is not None
                and m_oof.sum() >= 30
                and len(np.unique(ytr[m_oof])) >= 2
            ):
                try:
                    cfit = ConformalBinary(alpha=conf_alpha).fit(
                        cal.transform(oof_tr[m_oof]), ytr[m_oof],
                    )
                    conf_mask = cfit.predict_set(p)["confident"]
                except Exception:
                    pass
            bt = backtest_events(
                ds.events.iloc[te], p, cfg["backtest"], cfg["risk"], payoff, prices,
                confident=conf_mask,
            )
            col_perf[e.name] = bt["metrics"]["sharpe"]

        ens = StackingEnsemble([e.clone() for e in experts], cfg["ensemble"], seed=cfg.seed)
        ens.fit(Xtr, ytr, t1tr, sample_weight=wtr, n_splits=inner_splits, embargo_pct=embargo)
        pe = ens.predict_proba(Xte)
        oof_e = ens.oof_proba()
        cal_e = _calibrator_from_oof(oof_e, ytr, method)
        if cal_e is not None:
            pe = cal_e.transform(pe)
        conf_e = np.ones(len(pe), dtype=bool)
        m_oof = ~np.isnan(oof_e)
        if cal_e is not None and m_oof.sum() >= 30 and len(np.unique(ytr[m_oof])) >= 2:
            try:
                cfit = ConformalBinary(alpha=conf_alpha).fit(
                    cal_e.transform(oof_e[m_oof]), ytr[m_oof],
                )
                conf_e = cfit.predict_set(pe)["confident"]
            except Exception:
                pass
        bte = backtest_events(
            ds.events.iloc[te], pe, cfg["backtest"], cfg["risk"], payoff, prices,
            confident=conf_e,
        )
        col_perf["ensemble"] = bte["metrics"]["sharpe"]
        combo_sharpes.append(bte["metrics"]["sharpe"])
        combo_trades.append(int(bte["metrics"].get("n_trades", 0)))
        det = bte.get("detail")
        if det is not None and "size" in det.columns and "pnl" in det.columns and len(det):
            traded_pnl = det.loc[det["size"] > 0, "pnl"].to_numpy(dtype=float)
            if len(traded_pnl):
                combo_pnls.append(traded_pnl)

        if config_names is None:
            config_names = list(col_perf.keys())
        perf_rows.append([col_perf[c] for c in config_names])

    perf_matrix = np.array(perf_rows).T  # (n_configs, n_combos)
    sr = float(np.mean(combo_sharpes))
    n_obs = int(np.mean(combo_trades)) if combo_trades else len(ds.y)
    n_obs = max(n_obs, 2)
    n_trials = max(int(vcfg.get("dsr_n_trials", 50)), perf_matrix.shape[0])
    skew, kurt = 0.0, 3.0
    if combo_pnls:
        pooled = np.concatenate(combo_pnls)
        if len(pooled) >= 8 and float(np.std(pooled)) > 0:
            from scipy.stats import kurtosis as _kurt, skew as _skew

            skew = float(_skew(pooled, bias=False))
            kurt = float(_kurt(pooled, fisher=False, bias=False))
    dsr = deflated_sharpe_ratio(sr, n_trials=n_trials, n_obs=n_obs, skew=skew, kurt=kurt)
    pbo = probability_of_backtest_overfitting(perf_matrix)

    n_configs = int(perf_matrix.shape[0])
    n_combos = int(len(combo_sharpes))
    pbo_warning = bool(n_configs < 8)
    caveats: list[str] = []
    caveats.append(
        f"评估单元为 CPCV **组合**(n_combos={n_combos}), 不是拼接后的完整回测路径; "
        f"理论路径数 n_paths_theoretical={cv.n_paths} 仅供参考。DSR/夏普分布基于相关组合, 偏乐观。"
    )
    if pbo_warning:
        caveats.append(
            f"PBO 仅基于 {n_configs} 个配置(<8), 统计力不足, 数值仅供参考——"
            "需扫更多超参配置才可信。"
        )
    caveats.append(
        "DSR 的 observed_SR 为各 CPCV 组合(共享数据)per-trade 夏普的均值, 方差被低估, "
        "偏乐观; dsr_n_trials 须按你真实试过的策略/超参规模如实填写, 否则去偏失效。"
    )
    if n_trials <= n_configs:
        caveats.append(
            f"dsr_n_trials({n_trials}) ≤ 配置数({n_configs}), 几乎未去偏——请上调为真实研究规模。"
        )
    if pseudo_expert_names:
        caveats.append(
            f"配置含伪OOF专家 {pseudo_expert_names}: 单专家 CPCV 列非折内重训;"
            " stacking 默认已将其排除出元学习器(exclude_pseudo_oof_from_meta)。"
        )

    return {
        "evaluation_unit": "combo",
        "n_combos": n_combos,
        "n_paths_theoretical": cv.n_paths,
        "n_paths": n_combos,  # 兼容旧字段: 现等于组合数, 不再假装为 φ
        "path_sharpes": combo_sharpes,  # 兼容旧字段名
        "combo_sharpes": combo_sharpes,
        "mean_sharpe": sr,
        "std_sharpe": float(np.std(combo_sharpes)),
        "deflated_sharpe": dsr,
        "dsr_n_trials": n_trials,
        "dsr_n_obs": n_obs,
        "dsr_skew": skew,
        "dsr_kurt": kurt,
        "pbo": pbo,
        "pbo_warning": pbo_warning,
        "n_configs": n_configs,
        "caveats": caveats,
        "config_names": config_names,
        "perf_matrix": perf_matrix,
        "calibrated": True,
        "conformal": True,
        "data_source": getattr(ds, "data_source", None),
    }
