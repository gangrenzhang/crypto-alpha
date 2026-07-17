"""端到端编排: 数据 -> 特征 -> 标注 -> 数据集 -> 集成训练/验证 -> 回测 -> 决策。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config, set_global_seed
from ..data import load_symbol_data
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
    classification_report_probs,
    cross_fitted_calibrated,
    cross_fitted_conformal_flags,
    fit_deploy_calibrator_and_conformal,
)
from ..backtest import backtest_events


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


def prepare_dataset(cfg: Config, symbol: str) -> Dataset:
    raw = load_symbol_data(cfg, symbol)
    data_source = str(getattr(raw, "attrs", {}).get("data_source", "real"))
    degradations: list[str] = []
    if data_source == "synthetic_fallback":
        degradations.append("ohlcv_synthetic_fallback")
        print(
            f"[warn] {symbol}: 本次使用合成行情降级(data_source=synthetic_fallback); "
            "勿将回测结果当作真实市场证据。"
        )

    feat = build_feature_matrix(raw, cfg, symbol=symbol)
    feat["close"] = raw["close"]  # 回测/TSFM 需要收盘价
    feat["high"] = raw["high"] if "high" in raw.columns else raw["close"]
    feat["low"] = raw["low"] if "low" in raw.columns else raw["close"]
    feat = add_news_features(feat, cfg, symbol)  # 新闻数值特征并入(供所有专家共享)

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
    """为 LLM 专家注入/刷新新闻面板。"""
    for e in experts:
        if not hasattr(e, "set_news"):
            continue
        try:
            from ..data.news import load_news_panel

            news_df = load_news_panel(cfg, symbol)
            if news_df is not None:
                e.set_news(
                    news_df,
                    int(cfg["news"].get("buffer_minutes", 5)),
                    float(cfg["news"].get("feature_ttl_hours", 24)),
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
    # 评估用: 交叉拟合校准概率(避免"拟合即评估"乐观偏差)
    oof_cal = cross_fitted_calibrated(
        oof, ds.y, ds.t1, method=ccfg["method"],
        n_splits=int(ccfg.get("calib_splits", 5)), embargo_pct=float(vcfg["embargo_pct"]),
    )
    eval_mask = ~np.isnan(oof_cal)
    if not eval_mask.any():  # 样本过少时退回单折(乐观); 必须写入 degradations
        tag = "calib_cross_fit_fallback_insample"
        if tag not in degradations:
            degradations.append(tag)
        print(
            f"[calibration] WARN: {tag}; 交叉拟合校准不可用, "
            "退回同批 OOF fit+transform, 评估可能偏乐观"
        )
        cal_tmp = ProbabilityCalibrator(method=ccfg["method"]).fit(oof[mask], ds.y[mask])
        oof_cal = np.full_like(oof, np.nan)
        oof_cal[mask] = cal_tmp.transform(oof[mask])
        eval_mask = ~np.isnan(oof_cal)

    # 部署用: 时间切分独立保形集(校准器与保形器同基且分割)
    cal, conf = fit_deploy_calibrator_and_conformal(
        oof, ds.y, method=ccfg["method"],
        alpha=float(ccfg["conformal_alpha"]),
        conformal_frac=float(ccfg.get("conformal_frac", 0.3)),
    )

    # 回测用交叉拟合保形旗标(与实盘 HOLD 口径对齐, 且无部署器自评)
    conf_flags = cross_fitted_conformal_flags(
        oof_cal, ds.y, ds.t1, alpha=float(ccfg["conformal_alpha"]),
        n_splits=int(ccfg.get("calib_splits", 5)), embargo_pct=float(vcfg["embargo_pct"]),
    )

    report = classification_report_probs(oof_cal, ds.y)

    payoff = float(cfg["labeling"]["pt_sl"][0]) / float(cfg["labeling"]["pt_sl"][1])
    prices = ds.panel["close"] if "close" in ds.panel.columns else None
    bt = backtest_events(
        ds.events.loc[eval_mask], oof_cal[eval_mask], cfg["backtest"], cfg["risk"],
        payoff=payoff, prices=prices, confident=conf_flags[eval_mask],
    )

    # 收集降级信息(TSFM→naive / 伪OOF排除 / 二层OOF或校准小样本回退等)
    for e in ens.experts:
        if getattr(e, "degraded", False):
            degradations.append(
                f"{e.name}:{getattr(e, 'degraded_reason', 'degraded')}"
            )
    for d in getattr(ens, "degradations", []) or []:
        if d not in degradations:
            degradations.append(d)

    base_report = {
        e.name: classification_report_probs(ens.oof_[e.name].values, ds.y)
        for e in ens.experts
    }
    # 伪 OOF 分数仅诊断(未进 meta); 标注 contaminated 避免误读为真 OOF AUC
    pseudo = getattr(ens, "pseudo_oof_", None)
    if pseudo is not None and len(getattr(pseudo, "columns", [])):
        for name in pseudo.columns:
            br = classification_report_probs(pseudo[name].values, ds.y)
            br["pseudo_oof"] = True
            br["note"] = "frozen_adapter_not_cross_validated_excluded_from_meta"
            base_report[name] = br

    return {
        "ensemble": ens, "calibrator": cal, "conformal": conf,
        "oof_calibrated": oof_cal, "report": report, "backtest": bt,
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
    valid = panel[fcols].notna().all(axis=1)
    if not valid.any():
        return {"signal": "HOLD", "symbol": ds.symbol, "reason": "no_valid_feature_bar"}
    ts = panel.index[valid][-1]

    full_sampling = bool(
        trained.get("cusum_full_sampling", ds.cusum_full_sampling)
    )
    if not _is_tradable_event(cfg, panel, ts, full_sampling):
        return {
            "signal": "HOLD",
            "symbol": ds.symbol,
            "timestamp": str(ts),
            "reason": "not_cusum_event",
            "win_probability": None,
            "suggested_position_pct": 0.0,
            "stop_loss": None,
            "take_profit": None,
            "confident": False,
            "execution_assumption": str(cfg["risk"].get("execution_assumption", "close_fill")),
            "data_source": ds.data_source,
        }

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
    entry = float(panel["close"].loc[ts])
    atr = float(panel["atr_14"].loc[ts]) if "atr_14" in panel.columns else entry * 0.01
    pt_sl = (float(lc["pt_sl"][0]), float(lc["pt_sl"][1]))
    payoff = pt_sl[0] / pt_sl[1]
    # fee/slip 传入 decide: null 成本与回测统一回退 2*(fee+slip)
    fee = float(cfg["backtest"].get("fee_bps", 5.0)) / 1e4
    slip = float(cfg["backtest"].get("slippage_bps", 2.0)) / 1e4
    risk_cfg = dict(cfg["risk"])
    d = decide(
        prob, side, entry, atr, risk_cfg,
        prob_threshold=float(cfg["backtest"]["prob_threshold"]), payoff=payoff,
        confident=confident, pt_sl=pt_sl, fee=fee, slip=slip,
    )
    d["symbol"] = ds.symbol
    d["timestamp"] = str(ts)
    d["data_source"] = ds.data_source
    d["data_mode_zh"] = _DATA_MODE_ZH.get(ds.data_source, ds.data_source)
    return d
