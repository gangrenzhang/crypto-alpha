"""把新闻数值特征(情绪/互证数/条数/权威度)以无泄漏方式并入特征面板。

供所有专家(GBDT/深度时序)共享, 而不止喂给 LLM。核心纪律:
- as-of 对齐: 用 merge_asof(backward) 把每根 bar 对齐到"可用时刻 <= bar 时间"的最新
  新闻桶(新闻桶已用桶末标记 + buffer 传播缓冲), 严格无同期/未来泄漏。
- 时间衰减: 新闻的影响随时间指数衰减(halflife); 超过 TTL 视为过期 => 数值归零。
- 缺新闻: 一律填 0/中性, 保证不引入 NaN、不丢样本。
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


def add_news_features(feat: pd.DataFrame, cfg, symbol: str) -> pd.DataFrame:
    """在特征面板上追加新闻数值特征(无泄漏 as-of + 时间衰减)。"""
    ncfg = cfg["news"]
    ttl = float(ncfg.get("feature_ttl_hours", 24))
    if not ncfg.get("as_feature", True):
        return feat

    from ..data.news import load_news_panel

    news = load_news_panel(cfg, symbol)
    if news is None or len(news) == 0:
        return _empty_news_features(feat, ttl)

    buffer_min = int(ncfg.get("buffer_minutes", 5))
    halflife = float(ncfg.get("feature_halflife_hours", 6))
    ema_span = int(ncfg.get("feature_ema_span", 12))

    # 右表: 新闻桶(桶末标记) + 传播缓冲 => 可用时刻
    right = news.reset_index().rename(columns={news.index.name or "timestamp": "news_ts"})
    right["news_ts"] = pd.to_datetime(right["news_ts"], utc=True) + pd.Timedelta(minutes=buffer_min)
    keep = ["news_ts", "sentiment", "corroboration", "n_items", "max_authority"]
    right = right[[c for c in keep if c in right.columns]].sort_values("news_ts").reset_index(drop=True)

    left = pd.DataFrame({"timestamp": pd.to_datetime(feat.index, utc=True)}).sort_values("timestamp")
    # 统一时间精度, 避免 merge_asof 因 ms/us/ns 单位不一致报错
    left["timestamp"] = left["timestamp"].astype("datetime64[ns, UTC]")
    right["news_ts"] = right["news_ts"].astype("datetime64[ns, UTC]")
    merged = pd.merge_asof(left, right, left_on="timestamp", right_on="news_ts", direction="backward")
    merged = merged.set_index("timestamp")

    age_h = (merged.index - merged["news_ts"]).dt.total_seconds() / 3600.0
    fresh = (age_h.notna()) & (age_h <= ttl)
    decay = np.exp(-age_h.fillna(np.inf) / max(halflife, 1e-6))
    decay = decay.where(fresh, 0.0)

    sent_raw = merged["sentiment"].fillna(0.0) if "sentiment" in merged else pd.Series(0.0, index=merged.index)
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
    return feat
