"""端到端编排: 数据 -> 特征 -> 标注 -> 数据集 -> 集成训练/验证 -> 回测 -> 决策。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config, set_global_seed
from ..data import load_symbol_data, refresh_market_data
from ..features.build import build_feature_matrix, feature_columns
from ..features.news_features import add_news_features
from ..labeling.meta_labeling import (
    build_meta_labels,
    primary_signal,
    resolve_event_times,
    _barrier_target,
)
from ..labeling.sample_weights import sample_weights_by_return, time_decay_weights
from ..experts import EXPERT_REGISTRY
from ..ensemble import StackingEnsemble
from ..calibration import (
    ProbabilityCalibrator,
    ConformalBinary,
    classification_report_probs,
    cross_fitted_calibrated_and_conformal,
    fit_deploy_calibrator_and_conformal,
)
from ..backtest import backtest_events
from ..diagnostics.gates import (
    assess_calibration_pass_health,
    build_threshold_reference_mask,
    freeze_threshold_on_reference,
    gate_diagnostics,
)


_DATA_MODE_ZH = {
    "synthetic": "合成",
    "real": "真实",
    "cache": "真实(缓存)",
    "synthetic_fallback": "合成(降级)",
}


@dataclass
class Dataset:
    symbol: str
    panel: pd.DataFrame        # 全量特征面板(按 bar)
    feature_cols: list[str]
    X: pd.DataFrame            # 事件行特征(feature_cols, 含 side)
    y: np.ndarray              # 元标签 {0,1}
    events: pd.DataFrame       # ret / t1 / side / trgt
    t1: pd.Series
    sample_weight: np.ndarray
    data_source: str = "real"  # synthetic / real / cache / synthetic_fallback
    cusum_full_sampling: bool = False
    degradations: list[str] = field(default_factory=list)


def prepare_dataset(cfg: Config, symbol: str, *, for_decide: bool = False) -> Dataset:
    """组装建模数据集。

    仅当 ``for_decide=True`` **且** ``data.refresh_before_decide`` 时, 先
    ``refresh_market_data`` 刷到当下已收盘 tip; 训练/回测勿传 ``for_decide``,
    避免误触 REST。
    """
    dcfg = cfg["data"]
    # 仅决策路径(for_decide)且配置允许时强制刷 tip; 04 训练不传 for_decide, 避免误触 REST。
    if (
        for_decide
        and bool(dcfg.get("refresh_before_decide", True))
        and not dcfg.get("use_synthetic", False)
    ):
        raw = refresh_market_data(cfg, symbol)
    else:
        raw = load_symbol_data(cfg, symbol)
    data_source = str(getattr(raw, "attrs", {}).get("data_source", "real"))
    degradations: list[str] = []
    if data_source == "synthetic_fallback":
        degradations.append("ohlcv_synthetic_fallback")
        print(
            f"[warn] {symbol}: 本次使用合成行情降级(data_source=synthetic_fallback); "
            "勿将回测结果当作真实市场证据。"
        )
    if getattr(raw, "attrs", {}).get("tip_exchange_mismatch"):
        tag = "ohlcv_tip_exchange_fallback"
        if tag not in degradations:
            degradations.append(tag)
            exu = getattr(raw, "attrs", {}).get("exchange_used")
            print(
                f"[warn] {symbol}: tip 来自备用所 {exu} "
                f"(主所 {dcfg.get('exchange')}); 历史缓存与 tip 可能存在微观价差接缝。"
            )

    feat = build_feature_matrix(raw, cfg, symbol=symbol)
    feat["close"] = raw["close"]  # 回测/TSFM 需要收盘价
    feat["high"] = raw["high"] if "high" in raw.columns else raw["close"]
    feat["low"] = raw["low"] if "low" in raw.columns else raw["close"]
    feat = add_news_features(feat, cfg, symbol)  # 新闻数值特征并入(供所有专家共享)
    # 新闻覆盖率告警等写入 feat.attrs.degradations, 汇入 Dataset(不改特征数值)
    for d in list(getattr(feat, "attrs", {}).get("degradations") or []):
        if d not in degradations:
            degradations.append(d)

    labels = build_meta_labels(feat, cfg)
    full_sampling = bool(getattr(labels, "attrs", {}).get("cusum_full_sampling", False))

    # 主信号方向并入面板+建模特征: GBDT/DeepTS 与 TSFM/LLM 共享(元标签需要知道 side)
    lc = cfg["labeling"]
    feat["side"] = primary_signal(
        feat["close"], kind=lc["primary_signal"], lookback=int(lc["primary_lookback"]),
    ).astype(float)
    # 事件行与标签 side 严格对齐(同一定义, 防任何漂移)
    feat.loc[labels.index, "side"] = labels["side"].astype(float).values
    fcols = feature_columns(feat)

    # 对齐: 只保留特征无缺失的事件
    common = labels.index.intersection(feat.index)
    labels = labels.loc[common]
    valid = feat.loc[common, fcols].notna().all(axis=1)
    labels = labels.loc[valid.values]

    X = feat.loc[labels.index, fcols].copy()

    w_uniq = sample_weights_by_return(labels, feat.index).reindex(labels.index).fillna(1.0)
    w_decay = time_decay_weights(labels).reindex(labels.index).fillna(1.0)
    sw = (w_uniq * w_decay).values
    sw = sw / (sw.mean() + 1e-12)

    return Dataset(
        symbol=symbol, panel=feat, feature_cols=fcols, X=X,
        y=labels["bin"].astype(int).values, events=labels,
        t1=labels["t1"], sample_weight=sw,
        data_source=data_source, cusum_full_sampling=full_sampling,
        degradations=degradations,
    )


def _attach_news_to_llm(cfg: Config, symbol: str, experts: list) -> None:
    """为 LLM 专家注入/刷新新闻面板(决策时刻与数值新闻特征对齐)。"""
    for e in experts:
        if not hasattr(e, "set_news"):
            continue
        try:
            from ..data.news import ensure_news_panel
            from ..data.fetch import timeframe_delta

            news_df = ensure_news_panel(cfg, symbol)
            if news_df is not None:
                e.set_news(
                    news_df,
                    int(cfg["news"].get("buffer_minutes", 5)),
                    float(cfg["news"].get("feature_ttl_hours", 24)),
                    decision_delta=timeframe_delta(cfg["data"]["timeframe"]),
                )
        except Exception as ex:
            print(f"[warn] 新闻加载失败(LLM 专家将不使用新闻): {ex}")


def build_experts(cfg: Config, ds: Dataset) -> list:
    experts = []
    for name in cfg["experts"]["enabled"]:
        cls = EXPERT_REGISTRY[name]
        e = cls(cfg["experts"][name], ds.feature_cols, seed=cfg.seed)
        if e.needs_panel:
            e.set_panel(ds.panel)
        experts.append(e)
    _attach_news_to_llm(cfg, ds.symbol, experts)
    return experts


def train_and_validate(cfg: Config, ds: Dataset) -> dict:
    set_global_seed(cfg.seed)
    experts = build_experts(cfg, ds)
    ens = StackingEnsemble(experts, cfg["ensemble"], seed=cfg.seed)
    vcfg = cfg["validation"]
    ens.fit(
        ds.X, ds.y, ds.t1, sample_weight=ds.sample_weight,
        n_splits=int(vcfg["n_splits"]), embargo_pct=float(vcfg["embargo_pct"]),
    )

    oof = ens.oof_proba()  # 无泄漏的二层融合概率(nested OOF)
    mask = ~np.isnan(oof)
    degradations = list(ds.degradations)

    ccfg = cfg["calibration"]
    vcfg = cfg["validation"]
    conf_alpha = float(ccfg["conformal_alpha"])
    conf_margin = float(ccfg.get("conformal_min_margin", 0.0) or 0.0)
    n_cal_splits = int(ccfg.get("calib_splits", 5))
    embargo = float(vcfg["embargo_pct"])
    # 评估/回测: 同一 Purged 折内联合校准+保形, 避免「先 CF 校准再 CF 保形」二阶依赖。
    # 部署仍用下方时间切分 fit_deploy(与 CPCV 组合内同口径); 此处只服务 OOF 报告/回测。
    oof_cal, conf_flags, joint_tags = cross_fitted_calibrated_and_conformal(
        oof, ds.y, ds.t1, method=ccfg["method"], alpha=conf_alpha,
        n_splits=n_cal_splits, embargo_pct=embargo, min_margin=conf_margin,
    )
    for t in joint_tags:
        if t not in degradations:
            degradations.append(t)
            print(f"[calibration] WARN: {t}")
    eval_mask = ~np.isnan(oof_cal)
    if not eval_mask.any():  # 样本过少时退回同批(乐观); 必须写入 degradations
        tag = "calib_cross_fit_fallback_insample"
        if tag not in degradations:
            degradations.append(tag)
        print(
            f"[calibration] WARN: {tag}; 交叉拟合校准/保形不可用, "
            "退回同批 OOF fit+transform, 评估可能偏乐观"
        )
        cal_tmp = ProbabilityCalibrator(method=ccfg["method"]).fit(oof[mask], ds.y[mask])
        oof_cal = np.full_like(oof, np.nan)
        oof_cal[mask] = cal_tmp.transform(oof[mask])
        eval_mask = ~np.isnan(oof_cal)
        # 与部署 n<40 同形: 同批校准后再拟合保形(已标记乐观)
        conf_tmp = ConformalBinary(alpha=conf_alpha, min_margin=conf_margin).fit(
            oof_cal[mask], ds.y[mask],
        )
        conf_flags = np.asarray(
            conf_tmp.predict_set(oof_cal)["confident"], dtype=bool,
        )

    # 部署用: 时间切分独立保形集(校准器与保形器同基且分割) — 不改评估路径出分
    cal, conf, deploy_tags = fit_deploy_calibrator_and_conformal(
        oof, ds.y, method=ccfg["method"],
        alpha=conf_alpha,
        conformal_frac=float(ccfg.get("conformal_frac", 0.3)),
        min_margin=conf_margin,
    )
    for t in deploy_tags:
        if t not in degradations:
            degradations.append(t)
            print(f"[calibration] WARN: {t}")

    # 弱专家半窗选型后: 报告/回测只用后半窗, 避免 selection-on-evaluation
    prune_eval = getattr(ens, "prune_eval_mask_", None)
    if prune_eval is not None:
        report_mask = eval_mask & np.asarray(prune_eval, dtype=bool)
        if not report_mask.any():
            report_mask = eval_mask
    else:
        report_mask = eval_mask

    # 双阈值(方案B): 研究/部署各用与出分同尺度的参考窗冻结 thr, 禁止混用。
    # - thr_research: CF oof_cal[ref] → 仅研究回测
    # - thr_eff (prob_threshold_effective): deploy_cal.transform(raw oof[ref])
    #   → backtest_deploy / decide / serve / 与 walk-forward 同形
    # 禁止 cal.transform(oof_cal)(二次校准); 禁止用评估窗刷分位。
    ref_mask, ref_tags = build_threshold_reference_mask(eval_mask, report_mask, min_ref=20)
    for t in ref_tags:
        if t not in degradations:
            degradations.append(t)
            print(f"[threshold] {t}")
    inflate_max = float(ccfg.get("pass_rate_inflate_max", 1.5) or 0.0)
    raw_ref = np.asarray(oof[ref_mask], dtype=float)
    # 研究: 交叉拟合校准尺度
    thr_research, research_thr_tags = freeze_threshold_on_reference(
        cfg["backtest"], raw_ref, np.asarray(oof_cal[ref_mask], dtype=float),
        pass_rate_inflate_max=inflate_max, tag_prefix="research_",
    )
    for t in research_thr_tags:
        if t not in degradations:
            degradations.append(t)
    # 部署: 同一参考窗上的原始 OOF → deploy 校准器(与 decide 同 cal)
    ref_finite = np.isfinite(raw_ref)
    if int(ref_finite.sum()) >= 1:
        deploy_cal_ref = np.asarray(cal.transform(raw_ref[ref_finite]), dtype=float)
        raw_ref_dep = raw_ref[ref_finite]
    else:
        deploy_cal_ref = np.asarray([], dtype=float)
        raw_ref_dep = raw_ref
    thr_eff, deploy_thr_tags = freeze_threshold_on_reference(
        cfg["backtest"], raw_ref_dep, deploy_cal_ref,
        pass_rate_inflate_max=inflate_max, tag_prefix="deploy_",
    )
    for t in deploy_thr_tags:
        if t not in degradations:
            degradations.append(t)
            print(f"[threshold] {t}")
    print(
        f"[threshold] research={thr_research:.4f} deploy/effective={thr_eff:.4f}",
        flush=True,
    )

    bt_cfg_research = dict(cfg["backtest"])
    bt_cfg_research["prob_threshold"] = float(thr_research)
    bt_cfg_deploy = dict(cfg["backtest"])
    bt_cfg_deploy["prob_threshold"] = float(thr_eff)

    # 研究口径: 报告窗校准健康(只 degradations, 不改 thr)
    for t in assess_calibration_pass_health(
        oof[report_mask], oof_cal[report_mask], thr_research,
        pass_rate_inflate_max=inflate_max,
        min_unique_levels=int(ccfg.get("min_unique_levels", 20) or 0),
    ):
        if t not in degradations:
            degradations.append(t)
            print(f"[calibration] WARN: {t}")

    report = classification_report_probs(oof_cal[report_mask], ds.y[report_mask])

    payoff = float(cfg["labeling"]["pt_sl"][0]) / float(cfg["labeling"]["pt_sl"][1])
    prices = ds.panel["close"] if "close" in ds.panel.columns else None
    events_rep = ds.events.loc[report_mask]
    bt = backtest_events(
        events_rep, oof_cal[report_mask], bt_cfg_research, cfg["risk"],
        payoff=payoff, prices=prices, confident=conf_flags[report_mask],
    )
    gate_research = gate_diagnostics(
        events_rep.index, oof[report_mask], oof_cal[report_mask],
        conf_flags[report_mask], bt.get("detail"), thr_research, conf_obj=None,
    )
    gate_research["path"] = "research_oof_cross_fit"
    gate_research["prob_threshold_source"] = "cross_fitted_oof_cal_ref"

    # 部署口径回测(与 latest_decision / serve 同出分+同 thr_eff; 模型见过全样本 → 标注乐观,
    # 真外推请用 walk-forward 脚本)
    X_rep = ds.X.loc[events_rep.index]
    raw_dep = np.asarray(ens.predict_proba(X_rep), dtype=float)
    prob_dep = np.asarray(cal.transform(raw_dep), dtype=float)
    conf_dep = np.asarray(conf.predict_set(prob_dep)["confident"], dtype=bool)
    for t in assess_calibration_pass_health(
        raw_dep, prob_dep, thr_eff,
        pass_rate_inflate_max=inflate_max,
        min_unique_levels=int(ccfg.get("min_unique_levels", 20) or 0),
    ):
        tag = f"deploy_{t}" if not t.startswith("deploy_") else t
        if tag not in degradations:
            degradations.append(tag)
            print(f"[calibration] WARN: {tag}")
    bt_deploy = backtest_events(
        events_rep, prob_dep, bt_cfg_deploy, cfg["risk"],
        payoff=payoff, prices=prices, confident=conf_dep,
    )
    gate_deploy = gate_diagnostics(
        events_rep.index, raw_dep, prob_dep, conf_dep,
        bt_deploy.get("detail"), thr_eff, conf_obj=conf,
    )
    gate_deploy["path"] = "deploy_predict_fit_deploy"
    gate_deploy["prob_threshold_source"] = "deploy_cal_transform_raw_oof_ref"
    gate_deploy["note_optimism"] = (
        "deploy 路径在 train_and_validate 内对报告窗用全量拟合模型推理, "
        "成交数/胜率偏乐观; 实盘预期以 walk-forward / 滚动再训练为准。"
        "门控阈值=prob_threshold_effective(与 decide 同形)。"
    )

    # 收集降级信息(含被剪枝专家; build_oof 已写入 ens.degradations)
    for e in ens.experts:
        if getattr(e, "degraded", False):
            tag = f"{e.name}:{getattr(e, 'degraded_reason', 'degraded')}"
            if tag not in degradations:
                degradations.append(tag)
    for d in getattr(ens, "degradations", []) or []:
        if d not in degradations:
            degradations.append(d)

    # 专家表与集成 report / 回测共用 report_mask(半窗选型后的后半窗),
    # 避免看板上「专家 AUC vs 集成 AUC」全窗/半窗混比。不改 OOF 本身、部署与 decide。
    y_rep = ds.y[report_mask]
    base_report = {
        e.name: classification_report_probs(ens.oof_[e.name].values[report_mask], y_rep)
        for e in ens.experts
    }
    # 伪 OOF 分数仅诊断(未进 meta); 标注 contaminated 避免误读为真 OOF AUC
    pseudo = getattr(ens, "pseudo_oof_", None)
    if pseudo is not None and len(getattr(pseudo, "columns", [])):
        for name in pseudo.columns:
            br = classification_report_probs(pseudo[name].values[report_mask], y_rep)
            br["pseudo_oof"] = True
            br["note"] = "frozen_adapter_not_cross_validated_excluded_from_meta"
            base_report[name] = br

    # 波动滑点参考: 事件相对 ATR 中位数(与 backtest resolve_event_slippage 同口径)
    slip_ref_trgt = float("nan")
    if "trgt" in ds.events.columns and len(ds.events):
        _tv = pd.to_numeric(ds.events["trgt"], errors="coerce").to_numpy(dtype=float)
        _tv = _tv[np.isfinite(_tv) & (_tv > 0)]
        if len(_tv):
            slip_ref_trgt = float(np.median(_tv))

    # 正式研究路径可写实验日志, 抬高后续 DSR n_trials 下限(诊断/冒烟应关)
    if bool(cfg["validation"].get("log_experiments", True)):
        try:
            from ..diagnostics.experiments import append_experiment

            append_experiment(cfg.artifacts_dir, cfg, source="train_and_validate")
        except Exception as e:
            print(f"[experiments] WARN: 无法写入实验日志: {e}")

    return {
        "ensemble": ens, "calibrator": cal, "conformal": conf,
        "oof_calibrated": oof_cal, "report": report, "backtest": bt,
        "backtest_deploy": bt_deploy,
        "gate_diagnostics_research": gate_research,
        "gate_diagnostics_deploy": gate_deploy,
        # 部署/decide/serve 冻结阈值(deploy 校准尺度); 研究回测用 research
        "prob_threshold_effective": float(thr_eff),
        "prob_threshold_research": float(thr_research),
        "slip_ref_trgt": slip_ref_trgt,
        "base_report": base_report,
        "dropped_experts": ens.dropped_experts,
        "data_source": ds.data_source,
        "data_mode_zh": _DATA_MODE_ZH.get(ds.data_source, ds.data_source),
        "degradations": degradations,
        "cusum_full_sampling": ds.cusum_full_sampling,
    }


def _is_tradable_event(cfg: Config, panel: pd.DataFrame, ts, full_sampling: bool) -> bool:
    """实盘/最新决策是否落在与训练一致的事件集上。"""
    if full_sampling:
        return True
    if not bool(cfg["labeling"].get("serve_require_cusum", True)):
        return True
    lc = cfg["labeling"]
    try:
        vol_window = int(cfg["features"]["vol_window"])
    except Exception:
        vol_window = 50
    trgt = _barrier_target(panel, panel["close"], lc, vol_window)
    events, _ = resolve_event_times(panel["close"], trgt, lc)
    return ts in events


def align_feature_schema(
    feat: pd.DataFrame, feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """对齐训练期特征列 schema: 缺失列以 0.0 补齐, 返回 (面板, 缺失列名)。

    仅做**列存在性**对齐, 不改已有列数值。调用方若发现 missing 非空, 应强制 HOLD
    (勿在分布偏移的特征上继续推理开仓)——与 MTF 冷启动填 0 的训练语义一致, 但实盘
    「整列缺失」意味着辅周期/新闻装配失败, 与训练完整面板不同分布。
    """
    missing = [c for c in feature_cols if c not in feat.columns]
    if not missing:
        return feat, []
    out = feat.copy()
    for c in missing:
        out[c] = 0.0
    return out, missing


def hold_for_schema_mismatch(
    *,
    symbol: str,
    missing_cols: list[str],
    risk_cfg: dict,
    timestamp=None,
    close: float | None = None,
    data_source: str | None = None,
) -> dict:
    """特征 schema 与训练不一致时的安全 HOLD(不推理、不吐 SL/TP)。"""
    from ..risk.sizing import resolve_execution_assumption

    tag = f"feature_schema_mismatch({','.join(missing_cols[:12])}"
    if len(missing_cols) > 12:
        tag += f",+{len(missing_cols) - 12}more"
    tag += ")"
    out = {
        "signal": "HOLD",
        "symbol": symbol,
        "reason": "feature_schema_mismatch",
        "win_probability": None,
        "suggested_position_pct": 0.0,
        "stop_loss": None,
        "take_profit": None,
        "confident": False,
        "execution_assumption": resolve_execution_assumption(risk_cfg),
        "degradations": [tag],
        "missing_feature_cols": list(missing_cols),
    }
    if timestamp is not None:
        out["timestamp"] = str(timestamp)
    if close is not None:
        out["close"] = float(close)
    if data_source is not None:
        out["data_source"] = data_source
        out["data_mode_zh"] = _DATA_MODE_ZH.get(data_source, data_source)
    from ..serve.notifier import attach_decision_description

    return attach_decision_description(out)


def latest_decision(cfg: Config, ds: Dataset, trained: dict) -> dict:
    """对**最新一根特征完整的 bar** 输出结构化交易决策。

    与实盘服务(serve.decide_live)口径一致: 用最新 bar(而非最后一个已标注事件,
    后者至少滞后一个持有期)重算方向与特征后推理, 并接入保形弃权。
    若训练为 CUSUM 事件采样, 则非事件 bar 强制 HOLD(serve_require_cusum)。
    """
    from ..risk.sizing import decide

    ens = trained["ensemble"]
    cal = trained["calibrator"]
    conf = trained.get("conformal")
    fcols = ds.feature_cols
    panel = ds.panel

    lc = cfg["labeling"]
    side_ser = primary_signal(panel["close"], kind=lc["primary_signal"],
                              lookback=int(lc["primary_lookback"]))
    # 与训练一致: side 在建模特征列中时写入面板
    if "side" in fcols:
        panel = panel.copy()
        panel["side"] = side_ser.astype(float)
    # 防御: 训练面板理论上已含全部 feature_cols; 若列缺失则 HOLD(不推理)
    panel, missing = align_feature_schema(panel, fcols)
    if missing:
        print(
            f"[warn] {ds.symbol}: 特征列与训练 schema 不一致, 强制 HOLD; "
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
        _ts = panel.index[-1] if len(panel) else None
        _close = float(panel["close"].iloc[-1]) if len(panel) and "close" in panel.columns else None
        return hold_for_schema_mismatch(
            symbol=ds.symbol, missing_cols=missing, risk_cfg=cfg["risk"],
            timestamp=_ts, close=_close,
            data_source=ds.data_source,
        )
    from ..serve.notifier import attach_decision_description
    from ..diagnostics.decision_audit import attach_decision_audit, build_decision_audit
    from ..diagnostics.env_guard import hold_for_environment, should_hold_for_environment
    from ..backtest.engine import resolve_event_slippage

    # 展示用: 数据面 + 训练期质量标签; 环境 HOLD 只看数据面(见 env_guard)
    deg_data = list(ds.degradations)
    deg_train = list(trained.get("degradations") or [])
    deg_all: list[str] = []
    for t in deg_data + deg_train:
        if t not in deg_all:
            deg_all.append(t)

    def _with_audit(d: dict) -> dict:
        audit = build_decision_audit(
            cfg, panel=panel, feature_cols=fcols, trained=trained,
            degradations=list(d.get("degradations") or deg_all),
        )
        return attach_decision_audit(attach_decision_description(d), audit)

    valid = panel[fcols].notna().all(axis=1)
    if not valid.any():
        _ts = panel.index[-1] if len(panel) else None
        _close = float(panel["close"].iloc[-1]) if len(panel) and "close" in panel.columns else None
        return _with_audit({
            "signal": "HOLD", "symbol": ds.symbol, "reason": "no_valid_feature_bar",
            "timestamp": str(_ts) if _ts is not None else None,
            "close": _close,
            "data_source": ds.data_source,
            "data_mode_zh": _DATA_MODE_ZH.get(ds.data_source, ds.data_source),
            "degradations": deg_all,
        })
    ts = panel.index[valid][-1]
    bar_close = float(panel["close"].loc[ts])

    # 多降级叠加: 只对本次数据集/装配环境计分(不计训练期校准/剪枝标签)
    env_thr = cfg["risk"].get("env_degradation_hold_score", 50)
    hold_env, env_score, env_tag = should_hold_for_environment(deg_data, threshold=env_thr)
    if hold_env and env_tag:
        d_env = hold_for_environment(
            symbol=ds.symbol, score=env_score, reason_tag=env_tag,
            risk_cfg=cfg["risk"], timestamp=ts, close=bar_close,
            data_source=ds.data_source, degradations=deg_all,
        )
        return _with_audit(d_env)

    full_sampling = bool(
        trained.get("cusum_full_sampling", ds.cusum_full_sampling)
    )
    if not _is_tradable_event(cfg, panel, ts, full_sampling):
        from ..risk.sizing import resolve_execution_assumption

        return _with_audit({
            "signal": "HOLD",
            "symbol": ds.symbol,
            "timestamp": str(ts),
            "close": bar_close,
            "reason": "not_cusum_event",
            "win_probability": None,
            "suggested_position_pct": 0.0,
            "stop_loss": None,
            "take_profit": None,
            "confident": False,
            "execution_assumption": resolve_execution_assumption(cfg["risk"]),
            "data_source": ds.data_source,
            "data_mode_zh": _DATA_MODE_ZH.get(ds.data_source, ds.data_source),
            "degradations": deg_all,
        })

    for e in ens.experts:  # 刷新时序面板专家的历史窗口
        if getattr(e, "needs_panel", False):
            e.set_panel(panel)
    _attach_news_to_llm(cfg, ds.symbol, ens.experts)

    x_last = panel.loc[[ts], fcols].copy()
    if "side" not in x_last.columns:
        x_last["side"] = side_ser.loc[ts]
    prob = float(cal.transform(ens.predict_proba(x_last))[0])

    confident = True
    if conf is not None:
        confident = bool(conf.predict_set(np.array([prob]))["confident"][0])

    side = int(side_ser.loc[ts])
    entry = bar_close
    atr = float(panel["atr_14"].loc[ts]) if "atr_14" in panel.columns else entry * 0.01
    pt_sl = (float(lc["pt_sl"][0]), float(lc["pt_sl"][1]))
    payoff = pt_sl[0] / pt_sl[1]
    # fee/slip 传入 decide: null 成本与回测统一回退 2*(fee+slip)
    fee = float(cfg["backtest"].get("fee_bps", 5.0)) / 1e4
    slip = float(cfg["backtest"].get("slippage_bps", 2.0)) / 1e4
    # 与回测同形: 可选按相对 ATR 放大滑点(从不低于 base)
    trgt_now = atr / entry if entry > 0 else None
    ref_trgt = trained.get("slip_ref_trgt")
    if ref_trgt is None or not np.isfinite(float(ref_trgt or np.nan)):
        ref_trgt = float("nan")
    else:
        ref_trgt = float(ref_trgt)
    slip = resolve_event_slippage(slip, trgt_now, ref_trgt, cfg["backtest"])
    risk_cfg = dict(cfg["risk"])
    # 与 train_and_validate 冻结的部署阈值对齐; 缺省禁止用 CF oof_cal 估 thr
    # (尺度错配)。无冻结值时回退配置地板 fixed。
    thr_eff = trained.get("prob_threshold_effective")
    if thr_eff is None:
        thr_eff = float(cfg["backtest"].get("prob_threshold", 0.55))
    d = decide(
        prob, side, entry, atr, risk_cfg,
        prob_threshold=float(thr_eff), payoff=payoff,
        confident=confident, pt_sl=pt_sl, fee=fee, slip=slip,
    )
    d["prob_threshold_effective"] = float(thr_eff)
    d["symbol"] = ds.symbol
    d["timestamp"] = str(ts)
    d["close"] = bar_close
    d["data_source"] = ds.data_source
    d["data_mode_zh"] = _DATA_MODE_ZH.get(ds.data_source, ds.data_source)
    d["degradations"] = deg_all
    d["env_degradation_score"] = env_score
    return _with_audit(d)
