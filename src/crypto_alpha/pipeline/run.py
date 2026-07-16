"""端到端编排: 数据 -> 特征 -> 标注 -> 数据集 -> 集成训练/验证 -> 回测 -> 决策。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config, set_global_seed
from ..data import load_symbol_data
from ..features.build import build_feature_matrix, feature_columns
from ..features.news_features import add_news_features
from ..labeling.meta_labeling import build_meta_labels
from ..labeling.sample_weights import sample_weights_by_return, time_decay_weights
from ..experts import EXPERT_REGISTRY
from ..ensemble import StackingEnsemble
from ..calibration import ProbabilityCalibrator, ConformalBinary, classification_report_probs
from ..backtest import backtest_events


@dataclass
class Dataset:
    symbol: str
    panel: pd.DataFrame        # 全量特征面板(按 bar)
    feature_cols: list[str]
    X: pd.DataFrame            # 事件行特征(含 side 列)
    y: np.ndarray              # 元标签 {0,1}
    events: pd.DataFrame       # ret / t1 / side / trgt
    t1: pd.Series
    sample_weight: np.ndarray


def prepare_dataset(cfg: Config, symbol: str) -> Dataset:
    raw = load_symbol_data(cfg, symbol)
    feat = build_feature_matrix(raw, cfg)
    feat["close"] = raw["close"]  # 回测/TSFM 需要收盘价
    feat = add_news_features(feat, cfg, symbol)  # 新闻数值特征并入(供所有专家共享)
    fcols = feature_columns(feat)

    labels = build_meta_labels(feat, cfg)

    # 对齐: 只保留特征无缺失的事件
    common = labels.index.intersection(feat.index)
    labels = labels.loc[common]
    valid = feat.loc[common, fcols].notna().all(axis=1)
    labels = labels.loc[valid.values]

    X = feat.loc[labels.index, fcols].copy()
    X["side"] = labels["side"].values  # 供 TSFM/LLM 专家使用

    w_uniq = sample_weights_by_return(labels, feat.index).reindex(labels.index).fillna(1.0)
    w_decay = time_decay_weights(labels).reindex(labels.index).fillna(1.0)
    sw = (w_uniq * w_decay).values
    sw = sw / (sw.mean() + 1e-12)

    return Dataset(
        symbol=symbol, panel=feat, feature_cols=fcols, X=X,
        y=labels["bin"].astype(int).values, events=labels,
        t1=labels["t1"], sample_weight=sw,
    )


def build_experts(cfg: Config, ds: Dataset) -> list:
    experts = []
    for name in cfg["experts"]["enabled"]:
        cls = EXPERT_REGISTRY[name]
        e = cls(cfg["experts"][name], ds.feature_cols, seed=cfg.seed)
        if e.needs_panel:
            e.set_panel(ds.panel)
        if name == "llm":  # 为 LLM 专家注入新闻(无则静默)
            try:
                from ..data.news import load_news_panel

                news_df = load_news_panel(cfg, ds.symbol)
                if news_df is not None:
                    e.set_news(news_df, int(cfg["news"].get("buffer_minutes", 5)))
            except Exception as ex:
                print(f"[warn] 新闻加载失败(LLM 专家将不使用新闻): {ex}")
        experts.append(e)
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

    oof = ens.oof_proba()  # 无泄漏的融合概率
    mask = ~np.isnan(oof)

    # 概率校准(在 OOF 上拟合)
    ccfg = cfg["calibration"]
    cal = ProbabilityCalibrator(method=ccfg["method"]).fit(oof[mask], ds.y[mask])
    oof_cal = np.full_like(oof, np.nan)
    oof_cal[mask] = cal.transform(oof[mask])

    conf = ConformalBinary(alpha=float(ccfg["conformal_alpha"])).fit(oof_cal[mask], ds.y[mask])

    report = classification_report_probs(oof_cal, ds.y)

    # 回测(用无泄漏 OOF 校准概率)
    payoff = float(cfg["labeling"]["pt_sl"][0]) / float(cfg["labeling"]["pt_sl"][1])
    bt = backtest_events(
        ds.events.loc[mask], oof_cal[mask], cfg["backtest"], cfg["risk"], payoff=payoff
    )

    return {
        "ensemble": ens, "calibrator": cal, "conformal": conf,
        "oof_calibrated": oof_cal, "report": report, "backtest": bt,
        "base_report": {
            e.name: classification_report_probs(ens.oof_[e.name].values, ds.y)
            for e in experts
        },
    }


def latest_decision(cfg: Config, ds: Dataset, trained: dict) -> dict:
    """对最新一个事件输出结构化交易决策。"""
    from ..risk.sizing import decide

    ens, cal = trained["ensemble"], trained["calibrator"]
    x_last = ds.X.iloc[[-1]]
    prob = cal.transform(ens.predict_proba(x_last))[0]
    side = int(ds.X["side"].iloc[-1])
    entry = float(ds.panel["close"].iloc[-1])
    atr = float(ds.panel["atr_14"].iloc[-1]) if "atr_14" in ds.panel else entry * 0.01
    payoff = float(cfg["labeling"]["pt_sl"][0]) / float(cfg["labeling"]["pt_sl"][1])
    d = decide(
        prob, side, entry, atr, cfg["risk"],
        prob_threshold=float(cfg["backtest"]["prob_threshold"]), payoff=payoff,
    )
    d["symbol"] = ds.symbol
    d["timestamp"] = str(ds.X.index[-1])
    return d
