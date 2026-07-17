"""防泄漏与标注正确性单测。

覆盖此前"文档声称但无测试"的核心闸门:
1. 新闻 as-of 对齐: 事件早于"新闻发布+缓冲"时, 绝不应看到该新闻。
2. Purged K-Fold: 训练集不含与测试区间时间重叠的样本(无一层泄漏)。
3. 合成新闻守卫: 真实价格 + 合成新闻(由未来收益构造) => 拒绝, 防前视泄漏。
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
    monkeypatch.setattr(news_mod, "load_news_panel", lambda cfg, symbol: news_panel)

    feat = pd.DataFrame({"close": np.arange(5, dtype=float)}, index=idx)
    out = add_news_features(feat, cfg, "BTC/USDT")

    # 10:00 这根 bar(新闻尚未"可用")必须取不到该新闻
    assert out.loc[idx[2], "has_recent_news"] == 0.0
    assert out.loc[idx[2], "news_sentiment_raw"] == 0.0
    # 11:00 这根 bar 才应看到(已过缓冲)
    assert out.loc[idx[3], "has_recent_news"] == 1.0
    assert out.loc[idx[3], "news_sentiment_raw"] > 0.0


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
# 3) 合成新闻守卫: 真实价格下拒绝由未来收益构造的合成新闻
# --------------------------------------------------------------------------
def test_synthetic_news_guard_on_real_prices():
    from crypto_alpha.data.news import build_news_panel

    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = False   # 真实价格
    cfg.raw["news"]["use_synthetic"] = True     # 合成新闻(未来构造)
    cfg.raw["news"]["use_history"] = False

    with pytest.raises(ValueError):
        build_news_panel(cfg, "BTC/USDT")


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
