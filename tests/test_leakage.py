"""防泄漏与标注正确性单测。

覆盖此前"文档声称但无测试"的核心闸门:
1. 新闻 as-of 对齐: 事件早于"新闻发布+缓冲"时, 绝不应看到该新闻。
2. Purged K-Fold: 训练集不含与测试区间时间重叠的样本(无一层泄漏)。
3. 合成新闻守卫:
   - 真实价格 + news.use_synthetic(未来收益造情绪) => 拒绝;
   - 真实价格 + history.providers 仅 synthetic => 拒绝;
   - 真实价格 + corpus 中混有 synthetic: 行 => 加载时过滤。
4. 三重障碍 high/low bar 内触碰: 盘中被止损打到即判亏损, 而非只看收盘。
运行: pytest -q tests/test_leakage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crypto_alpha.config import Config


# --------------------------------------------------------------------------
# 1) 新闻 as-of 对齐: 无同期/未来泄漏
# --------------------------------------------------------------------------
def test_news_asof_no_future_leak(monkeypatch):
    import crypto_alpha.data.news as news_mod
    from crypto_alpha.features.news_features import add_news_features

    cfg = Config.load()
    cfg.raw["news"]["as_feature"] = True
    cfg.raw["news"]["buffer_minutes"] = 5
    cfg.raw["news"]["feature_ttl_hours"] = 24
    cfg.raw["news"]["feature_halflife_hours"] = 6

    idx = pd.date_range("2023-01-01 08:00", periods=5, freq="1h", tz="UTC")
    # 一条新闻发布于 10:00(+5min 缓冲 => 10:05 才可用)
    news_ts = pd.DatetimeIndex([pd.Timestamp("2023-01-01 10:00", tz="UTC")])
    news_panel = pd.DataFrame(
        {"sentiment": [1.0], "corroboration": [3], "n_items": [2], "max_authority": [1.0]},
        index=news_ts,
    )
    news_panel.index.name = "timestamp"
    monkeypatch.setattr(news_mod, "ensure_news_panel", lambda cfg, symbol: news_panel)

    feat = pd.DataFrame({"close": np.arange(5, dtype=float)}, index=idx)
    out = add_news_features(feat, cfg, "BTC/USDT")

    # 决策时刻 = 开盘 + 1h; 新闻可用=10:05
    # 09:00 bar → 决策 10:00 < 10:05 → 不得看到
    assert out.loc[idx[1], "has_recent_news"] == 0.0
    assert out.loc[idx[1], "news_sentiment_raw"] == 0.0
    # 10:00 bar → 决策 11:00 ≥ 10:05 → 应看到
    assert out.loc[idx[2], "has_recent_news"] == 1.0
    assert out.loc[idx[2], "news_sentiment_raw"] > 0.0


# --------------------------------------------------------------------------
# 2) Purged K-Fold: 训练集与测试时间区间不重叠
# --------------------------------------------------------------------------
def test_purged_kfold_no_overlap():
    from crypto_alpha.validation.purged_kfold import PurgedKFold

    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    # 每个事件持有 5 根 bar => t1 与相邻事件天然重叠
    t1 = pd.Series(idx.to_series().shift(-5).fillna(idx[-1]).values, index=idx)
    t1 = pd.Series(pd.to_datetime(t1.values, utc=True), index=idx)
    X = pd.DataFrame({"f": np.arange(n)}, index=idx)

    pkf = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.01)
    for tr, te in pkf.split(X):
        assert len(set(tr) & set(te)) == 0, "训练/测试索引重叠"
        test_start = idx[te].min()
        test_end = t1.iloc[te].max()
        # 任一训练样本的 [起点, t1] 不应与测试 [start, end] 区间相交
        for i in tr:
            s_i, e_i = idx[i], t1.iloc[i]
            overlap = (s_i <= test_end) and (e_i >= test_start)
            assert not overlap, "训练样本与测试区间重叠(存在泄漏)"


# --------------------------------------------------------------------------
# 3) 合成新闻守卫: 区分「未来收益面板」与「历史语料卫生」
# --------------------------------------------------------------------------
def test_synthetic_news_guard_on_real_prices():
    from crypto_alpha.data.news import build_news_panel

    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = False   # 真实价格
    cfg.raw["news"]["use_synthetic"] = True     # 合成新闻(未来构造)
    cfg.raw["news"]["use_history"] = False

    with pytest.raises(ValueError, match="未来收益"):
        build_news_panel(cfg, "BTC/USDT")


def test_synthetic_history_providers_guard_on_real_prices():
    """真实行情下 history.providers 仅含 synthetic 亦应拒绝(研究口径, 非未来收益泄漏)。"""
    from crypto_alpha.data.news import build_news_panel

    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = False
    cfg.raw["news"]["use_synthetic"] = False
    cfg.raw["news"]["use_history"] = True
    cfg.raw["news"]["history"]["providers"] = ["synthetic"]

    with pytest.raises(ValueError, match="合成历史语料"):
        build_news_panel(cfg, "BTC/USDT")


def test_corpus_filters_synthetic_rows_on_real_prices():
    """真实行情加载 corpus 时过滤 synthetic: 行; 全合成模式保留。不改磁盘库。"""
    from crypto_alpha.data.news import _is_synthetic_news_source, _raw_to_items

    raw = pd.DataFrame([
        {
            "published_at": pd.Timestamp("2022-01-01", tz="UTC"),
            "source": "synthetic:SEC", "tier": 1,
            "title": "Bitcoin ETF 利好消息", "url": "", "symbols": "BTC/USDT",
        },
        {
            "published_at": pd.Timestamp("2022-01-02", tz="UTC"),
            "source": "CoinDesk:wire", "tier": 2,
            "title": "Bitcoin ETF approval", "url": "", "symbols": "BTC/USDT",
        },
        {
            "published_at": pd.Timestamp("2022-01-03", tz="UTC"),
            "source": "synthetic:Reuters", "tier": 1,
            "title": "ETH 利空消息", "url": "", "symbols": "ETH/USDT,BTC/USDT",
        },
    ])

    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = False
    cfg.raw["news"]["use_history"] = True
    cfg.raw["news"]["history"]["providers"] = ["cryptocompare", "gdelt"]
    cfg.raw["news"]["history"]["start"] = None
    cfg.raw["news"]["history"]["end"] = None

    items = _raw_to_items(raw, "BTC/USDT", cfg)
    assert len(items) == 1
    assert items[0]["source"] == "CoinDesk:wire"

    cfg.raw["data"]["use_synthetic"] = True
    items_syn = _raw_to_items(raw, "BTC/USDT", cfg)
    assert len(items_syn) == 3
    assert sum(_is_synthetic_news_source(i["source"]) for i in items_syn) == 2


def test_news_sentiment_raw_respects_ttl(monkeypatch):
    """过期新闻的 news_sentiment_raw 必须归零(无 TTL 旁路)。"""
    import crypto_alpha.data.news as news_mod
    from crypto_alpha.features.news_features import add_news_features

    cfg = Config.load()
    cfg.raw["news"]["as_feature"] = True
    cfg.raw["news"]["buffer_minutes"] = 0
    cfg.raw["news"]["feature_ttl_hours"] = 1.0  # 1 小时过期
    cfg.raw["news"]["feature_halflife_hours"] = 6

    idx = pd.date_range("2023-01-01 08:00", periods=6, freq="1h", tz="UTC")
    news_ts = pd.DatetimeIndex([pd.Timestamp("2023-01-01 08:00", tz="UTC")])
    news_panel = pd.DataFrame(
        {"sentiment": [1.0], "corroboration": [1], "n_items": [1], "max_authority": [1.0]},
        index=news_ts,
    )
    news_panel.index.name = "timestamp"
    monkeypatch.setattr(news_mod, "load_news_panel", lambda cfg, symbol: news_panel)

    feat = pd.DataFrame({"close": np.arange(6, dtype=float)}, index=idx)
    out = add_news_features(feat, cfg, "BTC/USDT")
    # 决策时刻=09:00 时新闻刚可用且未过期; 决策=14:00(idx[5] open 13:00+1h) 已过 TTL
    # idx[5]=13:00 → decision 14:00, age from news avail 08:00 = 6h > 1h → raw=0
    assert out.loc[idx[5], "news_sentiment_raw"] == 0.0
    assert out.loc[idx[5], "has_recent_news"] == 0.0


# --------------------------------------------------------------------------
# 4) 三重障碍: bar 内 high/low 触碰(盘中止损即判亏损)
# --------------------------------------------------------------------------
def test_triple_barrier_intrabar_stop():
    from crypto_alpha.labeling.triple_barrier import get_events, get_bins

    idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    # 收盘全程不破 entry, 末根收于 110(收盘口径会误判为盈利);
    # 但 t1 的最低价 90 已盘中击穿止损(entry=100, sl=-7.5% => 92.5)。
    close = pd.Series([100, 100, 100, 100, 110], index=idx, dtype=float)
    high = pd.Series([100, 100, 100, 100, 110], index=idx, dtype=float)
    low = pd.Series([100, 90, 100, 100, 110], index=idx, dtype=float)

    trgt = pd.Series(0.05, index=idx)
    side = pd.Series(1, index=idx)
    t_events = pd.DatetimeIndex([idx[0]])

    events = get_events(close, high, low, t_events, (1.5, 1.5), trgt, 4, side, 0.0)
    bins = get_bins(events, close, (1.5, 1.5))
    assert bins["bin"].iloc[0] == 0, "盘中止损应判为亏损(bin=0)"
    assert bins["ret"].iloc[0] < 0
    # 了结应发生在触及止损的那一根(index[1]), 而非垂直到期
    assert bins["bars_held"].iloc[0] == 1


def test_triple_barrier_vertical_profit():
    from crypto_alpha.labeling.triple_barrier import get_events, get_bins

    idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    close = pd.Series([100, 101, 102, 103, 104], index=idx, dtype=float)
    high = pd.Series([100, 101, 102, 103, 104], index=idx, dtype=float)
    low = pd.Series([100, 101, 102, 103, 104], index=idx, dtype=float)

    trgt = pd.Series(0.10, index=idx)  # 障碍很宽, 不会被触碰 => 垂直到期
    side = pd.Series(1, index=idx)
    t_events = pd.DatetimeIndex([idx[0]])

    events = get_events(close, high, low, t_events, (1.5, 1.5), trgt, 4, side, 0.0)
    bins = get_bins(events, close, (1.5, 1.5))
    assert bins["bin"].iloc[0] == 1, "垂直到期且收盘上行应判盈利"


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call(["pytest", "-q", __file__]))
