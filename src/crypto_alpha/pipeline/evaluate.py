"""CPCV 严谨评估: 生成多条回测路径的夏普分布 + 去偏夏普(DSR) + 过拟合概率(PBO)。

与主训练路径 / 部署口径对齐:
- 测试折概率经**训练折 OOF**拟合的校准器再回测;
- 校准器与保形器在训练折 OOF 上**时间切分**(复用 ``fit_deploy_calibrator_and_conformal``),
  禁止「同一批 OOF 既 fit 校准又 fit 保形」破坏 split-conformal 独立性;
- 禁用训练集内概率(in-sample)拟合保形。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..validation.cpcv import CombinatorialPurgedCV
from ..validation.purged_kfold import PurgedKFold
from ..calibration.calibrate import fit_deploy_calibrator_and_conformal
from ..backtest.engine import (
    backtest_events,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)


def _expert_oof_probs(
    expert, X: pd.DataFrame, y: np.ndarray, t1: pd.Series,
    sample_weight: np.ndarray | None,
    n_splits: int, embargo_pct: float,
) -> tuple[object, np.ndarray]:
    """在训练集上产出专家 OOF, 再全量重训专家供测试折推理。

    返回 (fitted_full, oof_probs)。校准/保形由调用方用时间切分拟合
    (``fit_deploy_calibrator_and_conformal``), 本函数不再附带校准器。

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
        return full, oof

    for tr, te in pkf.split(X):
        clone = expert.clone()
        w = None if sample_weight is None else sample_weight[tr]
        # 与 stacking.build_oof 一致: DeepTS 折内早停 cutoff = 测试折最早时刻
        clone.fit(
            X.iloc[tr], y[tr], sample_weight=w,
            es_cutoff_time=X.index[te].min(),
        )
        oof[te] = clone.predict_proba(X.iloc[te])
    full = expert.clone()
    # 全量部署 fit: 不传 es_cutoff_time → DeepTS 用最近 val_frac 做早停
    full.fit(X, y, sample_weight=sample_weight)
    return full, oof


# 兼容旧测试/外部引用名
def _expert_oof_calibrator(
    expert, X: pd.DataFrame, y: np.ndarray, t1: pd.Series,
    sample_weight: np.ndarray | None, method: str,
    n_splits: int, embargo_pct: float,
) -> tuple[object, object | None, np.ndarray]:
    """兼容包装: 返回 (fitted, cal|None, oof); cal 来自时间切分部署口径。

    新代码请优先用 ``_expert_oof_probs`` + ``_apply_deploy_cal_conformal``。
    """
    fitted, oof = _expert_oof_probs(
        expert, X, y, t1, sample_weight, n_splits, embargo_pct,
    )
    m = ~np.isnan(oof)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return fitted, None, oof
    cal, _conf, _tags = fit_deploy_calibrator_and_conformal(
        oof, y, method=method, alpha=0.1, conformal_frac=0.3,
    )
    return fitted, cal, oof


def _apply_deploy_cal_conformal(
    oof: np.ndarray,
    y: np.ndarray,
    p_test: np.ndarray,
    *,
    method: str,
    alpha: float,
    conformal_frac: float,
    min_oof: int = 20,
    min_margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[str], object | None]:
    """用训练折 OOF 按部署口径时间切分拟合校准+保形, 变换测试折概率。

    返回 (校准后测试概率, confident 掩码, degradations 标签, 校准器或 None)。
    OOF 不足或单类时: 概率原样返回, confident 全 True, calibrator=None, 并写入 skipped 标签。
    调用方须用 ``cal.transform(train_oof)`` 估计 thr(与测试折同尺度), 禁止用原始 OOF。
    """
    tags: list[str] = []
    p_test = np.asarray(p_test, dtype=float)
    conf_mask = np.ones(len(p_test), dtype=bool)
    m = ~np.isnan(np.asarray(oof, dtype=float))
    yy = np.asarray(y)
    if m.sum() < min_oof or len(np.unique(yy[m])) < 2:
        tags.append(f"cpcv_cal_conformal_skipped(n_oof={int(m.sum())})")
        return p_test, conf_mask, tags, None
    try:
        cal, conf, dep_tags = fit_deploy_calibrator_and_conformal(
            oof, yy, method=method, alpha=alpha, conformal_frac=conformal_frac,
            min_margin=min_margin,
        )
        tags.extend(dep_tags)
        p_cal = cal.transform(p_test)
        conf_mask = np.asarray(conf.predict_set(p_cal)["confident"], dtype=bool)
        return p_cal, conf_mask, tags, cal
    except Exception as ex:
        tags.append(f"cpcv_cal_conformal_error:{type(ex).__name__}")
        return p_test, conf_mask, tags, None


def cpcv_report(cfg, ds, build_experts_fn) -> dict:
    """对每个 CPCV 划分, 在训练折训练集成、在测试折回测, 汇总路径级指标。

    同时构建 (n_configs, n_splits) 绩效矩阵用于 PBO: 配置 = 各专家 + Stacking 集成。
    """
    from ..ensemble import StackingEnsemble
    from ..diagnostics.gates import freeze_threshold_on_reference

    vcfg = cfg["validation"]
    ccfg = cfg["calibration"]
    method = ccfg.get("method", "isotonic")
    conformal_frac = float(ccfg.get("conformal_frac", 0.3))
    conf_margin = float(ccfg.get("conformal_min_margin", 0.0) or 0.0)
    inflate_max = float(ccfg.get("pass_rate_inflate_max", 1.5) or 0.0)

    def _thr_from_train_oof(oof_tr, cal, tags_out: list[str]) -> float:
        """部署同形阈值: cal.transform(train OOF); 无校准器时用原始 OOF。"""
        oof_arr = np.asarray(oof_tr, dtype=float)
        m = np.isfinite(oof_arr)
        raw_ref = oof_arr[m]
        if cal is not None and len(raw_ref):
            ref = np.asarray(cal.transform(raw_ref), dtype=float)
            prefix = "deploy_"
        else:
            ref = raw_ref
            prefix = "deploy_"
            if cal is None and "cpcv_thr_reference_raw_oof(no_calibrator)" not in tags_out:
                tags_out.append("cpcv_thr_reference_raw_oof(no_calibrator)")
        thr, thr_tags = freeze_threshold_on_reference(
            cfg["backtest"], raw_ref, ref,
            pass_rate_inflate_max=inflate_max, tag_prefix=prefix,
        )
        for t in thr_tags:
            if t not in tags_out:
                tags_out.append(t)
        return float(thr)
    cv = CombinatorialPurgedCV(
        n_splits=int(vcfg["n_splits"]),
        n_test_groups=int(vcfg["n_test_groups"]),
        t1=ds.t1,
        embargo_pct=float(vcfg["embargo_pct"]),
    )

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
    cal_degradations: list[str] = []

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
            fitted, oof_tr = _expert_oof_probs(
                e, Xtr, ytr, t1tr, wtr, inner_splits, embargo,
            )
            p_raw = fitted.predict_proba(Xte)
            p, conf_mask, cal_tags, cal_e = _apply_deploy_cal_conformal(
                oof_tr, ytr, p_raw,
                method=method, alpha=conf_alpha, conformal_frac=conformal_frac,
                min_margin=conf_margin,
            )
            for t in cal_tags:
                if t not in cal_degradations:
                    cal_degradations.append(t)
            thr_e = _thr_from_train_oof(oof_tr, cal_e, cal_degradations)
            bt_cfg_e = dict(cfg["backtest"])
            bt_cfg_e["prob_threshold"] = float(thr_e)
            bt = backtest_events(
                ds.events.iloc[te], p, bt_cfg_e, cfg["risk"], payoff, prices,
                confident=conf_mask,
            )
            col_perf[e.name] = bt["metrics"]["sharpe"]

        ens = StackingEnsemble([e.clone() for e in experts], cfg["ensemble"], seed=cfg.seed)
        ens.fit(Xtr, ytr, t1tr, sample_weight=wtr, n_splits=inner_splits, embargo_pct=embargo)
        pe_raw = ens.predict_proba(Xte)
        oof_e = ens.oof_proba()
        pe, conf_e, cal_tags_e, cal_ens = _apply_deploy_cal_conformal(
            oof_e, ytr, pe_raw,
            method=method, alpha=conf_alpha, conformal_frac=conformal_frac,
            min_margin=conf_margin,
        )
        for t in cal_tags_e:
            if t not in cal_degradations:
                cal_degradations.append(t)
        thr_ens = _thr_from_train_oof(oof_e, cal_ens, cal_degradations)
        bt_cfg_ens = dict(cfg["backtest"])
        bt_cfg_ens["prob_threshold"] = float(thr_ens)
        bte = backtest_events(
            ds.events.iloc[te], pe, bt_cfg_ens, cfg["risk"], payoff, prices,
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
    caveats.append(
        f"组合内校准+保形与部署一致: 训练折 OOF 时间切分"
        f"(conformal_frac={conformal_frac}); 有效 OOF<40 时同批回退"
        f"(与 fit_deploy_calibrator_and_conformal 相同)。"
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
    if cal_degradations:
        caveats.append(
            "校准/保形降级: " + "; ".join(cal_degradations[:8])
            + ("…" if len(cal_degradations) > 8 else "")
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
        "degradations": cal_degradations,
        "config_names": config_names,
        "perf_matrix": perf_matrix,
        "calibrated": True,
        "conformal": True,
        "conformal_time_split": True,
        "data_source": getattr(ds, "data_source", None),
    }
