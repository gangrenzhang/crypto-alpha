"""决策 tip 新鲜度: 未收盘 bar 剔除 + lag 校验(无网络)。"""
from __future__ import annotations

import pandas as pd
import pytest

from crypto_alpha.data.fetch import (
    assert_fresh_enough,
    closed_bar_lag,
    drop_incomplete_last_bar,
    exchange_candidates,
)


def _ohlcv(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.05,
            "volume": 10.0,
        },
        index=idx,
    )


def test_drop_incomplete_last_bar_removes_open_candle():
    idx = pd.date_range("2026-07-18 06:00", periods=3, freq="1h", tz="UTC")
    df = _ohlcv(idx)
    # 现在 07:30 → 07:00 这根尚未收盘, 应剔掉
    now = pd.Timestamp("2026-07-18 07:30", tz="UTC")
    out = drop_incomplete_last_bar(df, "1h", now=now)
    assert out.index[-1] == pd.Timestamp("2026-07-18 06:00", tz="UTC")


def test_closed_bar_lag_and_assert():
    idx = pd.date_range("2026-07-18 05:00", periods=2, freq="1h", tz="UTC")
    df = drop_incomplete_last_bar(
        _ohlcv(idx), "1h", now=pd.Timestamp("2026-07-18 07:00", tz="UTC")
    )
    now = pd.Timestamp("2026-07-18 08:30", tz="UTC")
    # last open 06:00, closed 07:00, lag=1.5h → 允许 2 根通过, 1 根失败
    assert closed_bar_lag(df, "1h", now=now) == pd.Timedelta("1h30min")
    assert_fresh_enough(df, "1h", max_lag_bars=2, now=now)
    with pytest.raises(RuntimeError, match="不够新"):
        assert_fresh_enough(df, "1h", max_lag_bars=1, now=now)


def test_exchange_candidates_tip_prefers_gate():
    class _Cfg(dict):
        def __getitem__(self, k):
            return super().__getitem__(k)

    cfg = {"data": {
        "exchange": "binance",
        "tip_exchange": "gate",
        "exchange_fallbacks": ["bitget", "mexc"],
    }}
    tip = exchange_candidates(cfg, for_tip=True)
    assert tip[0] == "gate"
    assert "binance" in tip


def test_drop_incomplete_30m():
    idx = pd.date_range("2026-07-18 07:00", periods=3, freq="30min", tz="UTC")
    df = _ohlcv(idx)
    # 07:45 → 07:30 根未收盘
    now = pd.Timestamp("2026-07-18 07:45", tz="UTC")
    out = drop_incomplete_last_bar(df, "30m", now=now)
    assert out.index[-1] == pd.Timestamp("2026-07-18 07:00", tz="UTC")


def test_cache_path_30m_does_not_use_legacy_1h(tmp_path):
    from crypto_alpha.data.fetch import raw_cache_path, resolve_raw_cache_path

    cfg = {
        "data": {"timeframe": "30m"},
        "data_dir": tmp_path,
    }
    # 伪造 Config 鸭子类型
    class _C:
        def __getitem__(self, k):
            return cfg[k] if k != "data" else cfg["data"]

        @property
        def data_dir(self):
            return tmp_path

    c = _C()
    legacy = tmp_path / "raw"
    legacy.mkdir()
    (legacy / "BTC_USDT.parquet").write_bytes(b"not-used")
    preferred = raw_cache_path(c, "BTC/USDT", "30m")
    assert preferred.name == "BTC_USDT__30m.parquet"
    # 30m 不得回退到无后缀 1h 遗留文件
    assert resolve_raw_cache_path(c, "BTC/USDT", "30m") == preferred
    assert not preferred.exists()
