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
    prices = ds.panel["close"] if "close" in ds.panel.columns else None
    path_sharpes: list[float] = []
    path_trades: list[int] = []
    path_pnls: list[np.ndarray] = []  # 各路径成交 pnl, 供 DSR 估计经验偏度/峰度
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
            bt = backtest_events(ds.events.iloc[te], p, cfg["backtest"], cfg["risk"], payoff, prices)
            col_perf[e.name] = bt["metrics"]["sharpe"]

        ens = StackingEnsemble([e.clone() for e in experts], cfg["ensemble"], seed=cfg.seed)
        ens.fit(Xtr, ytr, t1tr, sample_weight=wtr, n_splits=inner_splits, embargo_pct=embargo)
        pe = ens.predict_proba(Xte)
        cal_e = _calibrator_from_oof(ens.oof_proba(), ytr, method)
        if cal_e is not None:
            pe = cal_e.transform(pe)
        bte = backtest_events(ds.events.iloc[te], pe, cfg["backtest"], cfg["risk"], payoff, prices)
        col_perf["ensemble"] = bte["metrics"]["sharpe"]
        path_sharpes.append(bte["metrics"]["sharpe"])
        path_trades.append(int(bte["metrics"].get("n_trades", 0)))
        det = bte.get("detail")
        if det is not None and "size" in det.columns and "pnl" in det.columns and len(det):
            traded_pnl = det.loc[det["size"] > 0, "pnl"].to_numpy(dtype=float)
            if len(traded_pnl):
                path_pnls.append(traded_pnl)

        if config_names is None:
            config_names = list(col_perf.keys())
        perf_rows.append([col_perf[c] for c in config_names])

    perf_matrix = np.array(perf_rows).T  # (n_configs, n_splits)
    sr = float(np.mean(path_sharpes))
    n_obs = int(np.mean(path_trades)) if path_trades else len(ds.y)
    n_obs = max(n_obs, 2)
    n_trials = max(int(vcfg.get("dsr_n_trials", 50)), perf_matrix.shape[0])
    # DSR 用**经验**偏度/峰度(加密逐笔 pnl 尖峰厚尾), 否则默认 skew=0/kurt=3 会低估 SR 方差、
    # 系统性高估 DSR。汇总各路径成交 pnl 后估计。
    skew, kurt = 0.0, 3.0
    if path_pnls:
        pooled = np.concatenate(path_pnls)
        if len(pooled) >= 8 and float(np.std(pooled)) > 0:
            from scipy.stats import kurtosis as _kurt, skew as _skew

            skew = float(_skew(pooled, bias=False))
            kurt = float(_kurt(pooled, fisher=False, bias=False))
    dsr = deflated_sharpe_ratio(sr, n_trials=n_trials, n_obs=n_obs, skew=skew, kurt=kurt)
    pbo = probability_of_backtest_overfitting(perf_matrix)

    n_configs = int(perf_matrix.shape[0])
    pbo_warning = bool(n_configs < 8)
    caveats: list[str] = []
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

    return {
        "n_paths": cv.n_paths,
        "path_sharpes": path_sharpes,
        "mean_sharpe": sr,
        "std_sharpe": float(np.std(path_sharpes)),
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
    }
