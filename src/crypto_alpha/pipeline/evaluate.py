"""CPCV 严谨评估: 生成多条回测路径的夏普分布 + 去偏夏普(DSR) + 过拟合概率(PBO)。

与主训练路径一致: 测试折概率先经**训练折 OOF 拟合的校准器**再回测, 避免
"主路径有校准、CPCV 无校准"的口径分裂。
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
) -> tuple[object, ProbabilityCalibrator | None]:
    """在训练集上产出专家 OOF → 拟合校准器, 再全量重训专家供测试折推理。"""
    pkf = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)
    oof = np.full(len(y), np.nan)
    for tr, te in pkf.split(X):
        clone = expert.clone()
        w = None if sample_weight is None else sample_weight[tr]
        clone.fit(X.iloc[tr], y[tr], sample_weight=w)
        oof[te] = clone.predict_proba(X.iloc[te])
    cal = _calibrator_from_oof(oof, y, method)
    full = expert.clone()
    full.fit(X, y, sample_weight=sample_weight)
    return full, cal


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

    payoff = float(cfg["labeling"]["pt_sl"][0]) / float(cfg["labeling"]["pt_sl"][1])
    path_sharpes: list[float] = []
    path_trades: list[int] = []
    config_names = None
    perf_rows: list[list[float]] = []
    inner_splits = max(3, int(vcfg["n_splits"]) - 1)
    embargo = float(vcfg["embargo_pct"])

    for split_id, (tr, te, combo) in enumerate(cv.split(ds.X)):
        Xtr, Xte = ds.X.iloc[tr], ds.X.iloc[te]
        ytr = ds.y[tr]
        wtr = ds.sample_weight[tr]
        t1tr = ds.t1.iloc[tr]

        experts = build_experts_fn(cfg, ds)
        col_perf = {}
        for e in experts:
            fitted, cal = _expert_oof_calibrator(
                e, Xtr, ytr, t1tr, wtr, method, inner_splits, embargo,
            )
            p = fitted.predict_proba(Xte)
            if cal is not None:
                p = cal.transform(p)
            bt = backtest_events(ds.events.iloc[te], p, cfg["backtest"], cfg["risk"], payoff)
            col_perf[e.name] = bt["metrics"]["sharpe"]

        ens = StackingEnsemble([e.clone() for e in experts], cfg["ensemble"], seed=cfg.seed)
        ens.fit(Xtr, ytr, t1tr, sample_weight=wtr, n_splits=inner_splits, embargo_pct=embargo)
        pe = ens.predict_proba(Xte)
        cal_e = _calibrator_from_oof(ens.oof_proba(), ytr, method)
        if cal_e is not None:
            pe = cal_e.transform(pe)
        bte = backtest_events(ds.events.iloc[te], pe, cfg["backtest"], cfg["risk"], payoff)
        col_perf["ensemble"] = bte["metrics"]["sharpe"]
        path_sharpes.append(bte["metrics"]["sharpe"])
        path_trades.append(int(bte["metrics"].get("n_trades", 0)))

        if config_names is None:
            config_names = list(col_perf.keys())
        perf_rows.append([col_perf[c] for c in config_names])

    perf_matrix = np.array(perf_rows).T  # (n_configs, n_splits)
    sr = float(np.mean(path_sharpes))
    n_obs = int(np.mean(path_trades)) if path_trades else len(ds.y)
    n_obs = max(n_obs, 2)
    n_trials = max(int(vcfg.get("dsr_n_trials", 50)), perf_matrix.shape[0])
    dsr = deflated_sharpe_ratio(sr, n_trials=n_trials, n_obs=n_obs)
    pbo = probability_of_backtest_overfitting(perf_matrix)

    return {
        "n_paths": cv.n_paths,
        "path_sharpes": path_sharpes,
        "mean_sharpe": sr,
        "std_sharpe": float(np.std(path_sharpes)),
        "deflated_sharpe": dsr,
        "dsr_n_trials": n_trials,
        "dsr_n_obs": n_obs,
        "pbo": pbo,
        "pbo_warning": bool(perf_matrix.shape[0] < 8),
        "config_names": config_names,
        "perf_matrix": perf_matrix,
        "calibrated": True,
    }
