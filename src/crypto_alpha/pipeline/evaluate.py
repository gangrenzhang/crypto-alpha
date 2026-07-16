"""CPCV 严谨评估: 生成多条回测路径的夏普分布 + 去偏夏普(DSR) + 过拟合概率(PBO)。"""
from __future__ import annotations

import numpy as np

from ..validation.cpcv import CombinatorialPurgedCV
from ..backtest.engine import (
    backtest_events,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)


def cpcv_report(cfg, ds, build_experts_fn) -> dict:
    """对每个 CPCV 划分, 在训练折训练集成、在测试折回测, 汇总路径级指标。

    同时构建 (n_configs, n_splits) 绩效矩阵用于 PBO: 配置 = 各专家 + 简单等权集成。
    """
    from ..ensemble import StackingEnsemble

    vcfg = cfg["validation"]
    cv = CombinatorialPurgedCV(
        n_splits=int(vcfg["n_splits"]),
        n_test_groups=int(vcfg["n_test_groups"]),
        t1=ds.t1,
        embargo_pct=float(vcfg["embargo_pct"]),
    )

    payoff = float(cfg["labeling"]["pt_sl"][0]) / float(cfg["labeling"]["pt_sl"][1])
    path_sharpes: list[float] = []
    config_names = None
    perf_rows: list[list[float]] = []

    for split_id, (tr, te, combo) in enumerate(cv.split(ds.X)):
        Xtr, Xte = ds.X.iloc[tr], ds.X.iloc[te]
        ytr = ds.y[tr]
        wtr = ds.sample_weight[tr]

        experts = build_experts_fn(cfg, ds)
        # 各专家单独绩效(用于 PBO 配置维度)
        col_perf = {}
        for e in experts:
            clone = e.clone()
            clone.fit(Xtr, ytr, sample_weight=wtr)
            p = clone.predict_proba(Xte)
            bt = backtest_events(ds.events.iloc[te], p, cfg["backtest"], cfg["risk"], payoff)
            col_perf[e.name] = bt["metrics"]["sharpe"]

        # 等权集成绩效 = 路径夏普
        ens = StackingEnsemble([e.clone() for e in experts], cfg["ensemble"], seed=cfg.seed)
        ens.fit(Xtr, ytr, ds.t1.iloc[tr], sample_weight=wtr,
                n_splits=max(3, int(vcfg["n_splits"]) - 1), embargo_pct=float(vcfg["embargo_pct"]))
        pe = ens.predict_proba(Xte)
        bte = backtest_events(ds.events.iloc[te], pe, cfg["backtest"], cfg["risk"], payoff)
        col_perf["ensemble"] = bte["metrics"]["sharpe"]
        path_sharpes.append(bte["metrics"]["sharpe"])

        if config_names is None:
            config_names = list(col_perf.keys())
        perf_rows.append([col_perf[c] for c in config_names])

    perf_matrix = np.array(perf_rows).T  # (n_configs, n_splits)
    sr = float(np.mean(path_sharpes))
    dsr = deflated_sharpe_ratio(sr, n_trials=perf_matrix.shape[0], n_obs=len(ds.y))
    pbo = probability_of_backtest_overfitting(perf_matrix)

    return {
        "n_paths": cv.n_paths,
        "path_sharpes": path_sharpes,
        "mean_sharpe": sr,
        "std_sharpe": float(np.std(path_sharpes)),
        "deflated_sharpe": dsr,
        "pbo": pbo,
        "config_names": config_names,
        "perf_matrix": perf_matrix,
    }
