"""把新闻数值特征(情绪/互证数/条数/权威度)以无泄漏方式并入特征面板。

供所有专家(GBDT/深度时序)共享, 而不止喂给 LLM。核心纪律:
- as-of 对齐: 用 merge_asof(backward) 把每根 bar 对齐到"可用时刻 <= **决策时刻**"的最新
  新闻桶。决策时刻 = 主 bar 开盘 + 主周期长度(与 MTF 口径一致); 新闻桶已用桶末标记 + buffer。
- 时间衰减: 新闻的影响随时间指数衰减(halflife); 超过 TTL 视为过期 => **全部数值列归零**
  (含 news_sentiment_raw)。
- 缺新闻: 一律填 0/中性, 保证不引入 NaN、不丢样本。
- 覆盖率告警: as_feature 开启但 has_recent_news 均值过低时 warn + attrs.degradations
  (不改数值, 避免长回测把「空新闻」误当成「新闻无 alpha」)。
产出列: news_sentiment(衰减后) / news_sentiment_raw / news_corroboration /
        news_n_items / news_max_authority / news_age_hours / has_recent_news / news_sent_ema
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEWS_FEATURE_COLS = [
    "news_sentiment", "news_sentiment_raw", "news_corroboration", "news_n_items",
    "news_max_authority", "news_age_hours", "has_recent_news", "news_sent_ema",
]


def _empty_news_features(feat: pd.DataFrame, ttl_hours: float) -> pd.DataFrame:
    for c in NEWS_FEATURE_COLS:
        feat[c] = 0.0
    feat["news_age_hours"] = float(ttl_hours)  # 无新闻 => 视为最大陈旧度
    return feat


def _maybe_warn_news_coverage(feat: pd.DataFrame, ncfg: dict, symbol: str) -> None:
    """覆盖率过低时 warn + 写入 ``feat.attrs['degradations']``(不改特征数值)。

    ``news.min_coverage_warn``: 覆盖率阈值(默认 0.05); ≤0 关闭告警。
    覆盖率 = ``has_recent_news`` 均值(TTL 内有新闻的 bar 占比)。
    """
    thr = float(ncfg.get("min_coverage_warn", 0.05))
    if thr <= 0:
        return
    if "has_recent_news" in feat.columns:
        cov = float(pd.Series(feat["has_recent_news"]).fillna(0.0).astype(float).mean())
    else:
        cov = 0.0
    feat.attrs["news_feature_coverage"] = cov
    if cov >= thr:
        return
    tag = f"news_features_sparse(coverage={cov:.4f},threshold={thr:.4f})"
    deg = list(feat.attrs.get("degradations") or [])
    if tag not in deg:
        deg.append(tag)
    feat.attrs["degradations"] = deg
    print(
        f"[news] WARN: {symbol} 新闻数值特征覆盖率过低 coverage={cov:.2%} "
        f"(阈值 {thr:.0%})。长回测请设 news.use_history=true 并回填语料, "
        f"或关闭 news.as_feature, 以免误判「新闻无 alpha」。"
    )


def add_news_features(feat: pd.DataFrame, cfg, symbol: str) -> pd.DataFrame:
    """在特征面板上追加新闻数值特征(无泄漏 as-of + 时间衰减)。"""
    ncfg = cfg["news"]
    ttl = float(ncfg.get("feature_ttl_hours", 24))
    if not ncfg.get("as_feature", True):
        return feat

    from ..data.news import load_news_panel
    from ..data.fetch import timeframe_delta

    news = load_news_panel(cfg, symbol)
    if news is None or len(news) == 0:
        feat = _empty_news_features(feat, ttl)
        _maybe_warn_news_coverage(feat, ncfg, symbol)
        return feat

    buffer_min = int(ncfg.get("buffer_minutes", 5))
    halflife = float(ncfg.get("feature_halflife_hours", 6))
    ema_span = int(ncfg.get("feature_ema_span", 12))
    main_tf = cfg["data"]["timeframe"]
    main_delta = timeframe_delta(main_tf)

    # 右表: 新闻桶(桶末标记) + 传播缓冲 => 可用时刻
    right = news.reset_index().rename(columns={news.index.name or "timestamp": "news_ts"})
    right["news_ts"] = pd.to_datetime(right["news_ts"], utc=True) + pd.Timedelta(minutes=buffer_min)
    keep = ["news_ts", "sentiment", "corroboration", "n_items", "max_authority"]
    right = right[[c for c in keep if c in right.columns]].sort_values("news_ts").reset_index(drop=True)

    # 左表: 与 MTF 一致 — 决策时刻 = 主 bar 开盘 + 主周期(该 bar 收盘才决策)
    main_idx = pd.DatetimeIndex(pd.to_datetime(feat.index, utc=True))
    left = pd.DataFrame({
        "decision_at": (main_idx + main_delta).astype("datetime64[ns, UTC]"),
        "main_ts": main_idx,
    }).sort_values("decision_at")
    right["news_ts"] = right["news_ts"].astype("datetime64[ns, UTC]")
    merged = pd.merge_asof(
        left, right, left_on="decision_at", right_on="news_ts", direction="backward",
    )
    merged = merged.set_index("main_ts")

    age_h = (merged["decision_at"] - merged["news_ts"]).dt.total_seconds() / 3600.0
    fresh = (age_h.notna()) & (age_h <= ttl)
    decay = np.exp(-age_h.fillna(np.inf) / max(halflife, 1e-6))
    decay = decay.where(fresh, 0.0)

    sent_raw = merged["sentiment"].fillna(0.0) if "sentiment" in merged else pd.Series(0.0, index=merged.index)
    # 过期一律归零(含 raw), 杜绝 TTL 旁路
    sent_raw = sent_raw.where(fresh, 0.0)
    out = pd.DataFrame(index=merged.index)
    out["news_sentiment"] = sent_raw * decay
    out["news_sentiment_raw"] = sent_raw
    out["news_corroboration"] = (merged.get("corroboration", 0.0)).fillna(0.0).where(fresh, 0.0)
    out["news_n_items"] = (merged.get("n_items", 0.0)).fillna(0.0).where(fresh, 0.0)
    out["news_max_authority"] = (merged.get("max_authority", 0.0)).fillna(0.0).where(fresh, 0.0)
    out["news_age_hours"] = age_h.fillna(ttl).clip(upper=ttl)
    out["has_recent_news"] = fresh.astype(float)
    out["news_sent_ema"] = out["news_sentiment"].ewm(span=ema_span, adjust=False).mean()

    # 对齐回 feat(按时间索引), 缺失填 0
    out = out.reindex(feat.index)
    for c in NEWS_FEATURE_COLS:
        feat[c] = out[c].fillna(0.0).values
    _maybe_warn_news_coverage(feat, ncfg, symbol)
    return feat
