"""清算信息源: 桶对齐、特征降级、不污染 funding/OI 路径。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_liq_bucket_and_notional():
    from crypto_alpha.data.fetch import _liq_bucket, _liq_notional

    assert _liq_bucket("SELL") == "long"
    assert _liq_bucket("buy") == "short"
    assert _liq_bucket(None) is None
    assert _liq_bucket("unknown") is None
    assert _liq_notional({"quoteValue": 1500.0}) == 1500.0
    assert _liq_notional({"price": 100.0, "amount": 2.0}) == 200.0


def test_aggregate_liquidations_no_lookahead():
    """事件归入开盘桶; 收盘后事件不计入当根; 首笔前为 NaN。"""
    from crypto_alpha.data.fetch import _aggregate_liquidations

    idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    delta = pd.Timedelta(hours=1)
    # bar0 [00:00,01:00): sell@00:30 → long; buy@01:00 属于 bar1 开盘 → short on bar1
    rows = [
        {"timestamp": int(pd.Timestamp("2024-01-01 00:30", tz="UTC").timestamp() * 1000),
         "side": "sell", "quoteValue": 100.0},
        {"timestamp": int(pd.Timestamp("2024-01-01 01:00", tz="UTC").timestamp() * 1000),
         "side": "buy", "quoteValue": 50.0},
        # 恰好 bar0 收盘时刻 = bar1 开盘, 已在上条; 再加 bar0 内边界
        {"timestamp": int(pd.Timestamp("2024-01-01 00:59:59", tz="UTC").timestamp() * 1000),
         "side": "sell", "quoteValue": 10.0},
    ]
    lng, sht = _aggregate_liquidations(rows, idx, delta)
    assert float(lng.iloc[0]) == pytest.approx(110.0)
    assert float(sht.iloc[0]) == pytest.approx(0.0)
    assert float(sht.iloc[1]) == pytest.approx(50.0)
    assert float(lng.iloc[1]) == pytest.approx(0.0)


def test_aggregate_masks_unknown_prefix_before_first_liq():
    """首笔清算之前不得记 0(假安静), 应为 NaN。"""
    from crypto_alpha.data.fetch import _aggregate_liquidations

    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    rows = [
        {"timestamp": int(pd.Timestamp("2024-01-01 03:10", tz="UTC").timestamp() * 1000),
         "side": "buy", "quoteValue": 9.0},
    ]
    lng, sht = _aggregate_liquidations(rows, idx, pd.Timedelta(hours=1))
    assert lng.iloc[:3].isna().all()
    assert sht.iloc[:3].isna().all()
    assert float(sht.iloc[3]) == pytest.approx(9.0)
    assert float(lng.iloc[3]) == pytest.approx(0.0)


def test_liquidations_sparse_degradation_when_prefix_nan():
    from crypto_alpha.config import Config
    from crypto_alpha.features.build import build_feature_matrix

    cfg = Config.load()
    cfg.raw["features"]["mtf_enabled"] = False
    n = 2500
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ll = pd.Series(np.nan, index=idx)
    ls = pd.Series(np.nan, index=idx)
    # 仅末 10% 有观测
    cut = int(n * 0.9)
    ll.iloc[cut:] = 0.0
    ls.iloc[cut:] = 0.0
    ls.iloc[-1] = 1e5
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1.0,
        "liq_long": ll, "liq_short": ls,
    }, index=idx)
    feat = build_feature_matrix(df, cfg, symbol=None)
    deg = feat.attrs.get("degradations") or []
    assert any(str(t).startswith("derivatives_liquidations_sparse") for t in deg)


def test_fetch_derivatives_liq_failure_keeps_funding_oi(monkeypatch):
    """清算拉取抛错时 funding/OI 仍应写入(互不影响)。"""
    import types

    from crypto_alpha.data import fetch as fetch_mod

    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")

    class _Ex:
        has = {
            "fetchFundingRateHistory": True,
            "fetchOpenInterestHistory": True,
            "fetchLiquidations": True,
        }

        def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
            return [{"timestamp": int(idx[0].timestamp() * 1000), "fundingRate": 0.001}]

        def fetch_open_interest_history(self, *args, **kwargs):
            return [{"timestamp": int(idx[0].timestamp() * 1000), "openInterestAmount": 1e6}]

        def fetch_liquidations(self, *args, **kwargs):
            raise RuntimeError("liq down")

    fake_ccxt = types.SimpleNamespace(binance=lambda cfg=None: _Ex())
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)

    out = fetch_mod.fetch_derivatives("binance", "BTC/USDT", idx, include_liquidations=True)
    assert float(out["funding_rate"].iloc[-1]) == pytest.approx(0.001)
    assert float(out["open_interest"].iloc[-1]) == pytest.approx(1e6)
    assert out["liq_long"].isna().all()
    assert out["liq_short"].isna().all()


def test_liquidations_nan_does_not_wipe_samples_and_tags_degradation():
    from crypto_alpha.config import Config
    from crypto_alpha.features.build import build_feature_matrix, feature_columns

    cfg = Config.load()
    cfg.raw["features"]["mtf_enabled"] = False
    n = 2500
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.uniform(1, 10, n),
        "funding_rate": np.nan, "open_interest": np.nan,
        "liq_long": np.nan, "liq_short": np.nan,
    }, index=idx)
    feat = build_feature_matrix(df, cfg, symbol=None)
    assert (feat["liq_imbalance"] == 0.0).all()
    assert (feat["liq_imbalance_z"] == 0.0).all()
    assert (feat["liq_total_z"] == 0.0).all()
    deg = feat.attrs.get("degradations") or []
    assert "derivatives_liquidations_unavailable" in deg
    assert "derivatives_funding_unavailable" in deg
    fcols = feature_columns(feat)
    assert "liq_long" not in fcols and "liq_short" not in fcols
    assert "liq_imbalance" in fcols
    assert feat[fcols].notna().all(axis=1).sum() > 50


def test_liq_features_from_real_buckets():
    from crypto_alpha.features.technical import add_technical_features

    idx = pd.date_range("2024-01-01", periods=80, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100, 110, 80), index=idx)
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1.0,
        "liq_long": 0.0, "liq_short": 0.0,
    }, index=idx)
    df.loc[idx[50], "liq_short"] = 1e6  # 空头爆仓
    df.loc[idx[51], "liq_long"] = 1e6
    feat = add_technical_features(df, windows=[14], vol_window=20, oi_change_bars=24)
    assert feat.loc[idx[50], "liq_imbalance"] > 0
    assert feat.loc[idx[51], "liq_imbalance"] < 0


def test_ensure_liquidation_columns_idempotent():
    from crypto_alpha.data.fetch import ensure_liquidation_columns

    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    out = ensure_liquidation_columns(df)
    assert "liq_long" in out.columns and out["liq_long"].isna().all()
    out2 = ensure_liquidation_columns(out)
    assert list(out2.columns) == list(out.columns)


def test_attach_liquidations_from_store_onto_bars(tmp_path):
    """事件库按 bar 时间对齐后, 特征非全 0。"""
    from crypto_alpha.config import Config
    from crypto_alpha.data.liquidations import (
        attach_liquidations_to_ohlcv,
        save_liquidation_events,
    )
    from crypto_alpha.features.build import build_feature_matrix

    cfg = Config.load()
    cfg.raw["project"]["data_dir"] = str(tmp_path)
    cfg.raw["features"]["mtf_enabled"] = False
    cfg.raw["data"]["fetch_liquidations"] = True
    cfg.raw["data"]["liquidations"] = {"store_dir": "liquidations", "auto_attach": True}

    idx = pd.date_range("2024-06-01", periods=10, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(60000, 61000, 10), index=idx)
    ohlcv = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": 1.0,
        "liq_long": np.nan, "liq_short": np.nan,
    }, index=idx)

    events = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2024-06-01 02:15", tz="UTC"),
            pd.Timestamp("2024-06-01 05:40", tz="UTC"),
        ],
        "side_bucket": ["long", "short"],
        "notional": [1000.0, 2000.0],
        "exchange": ["test", "test"],
        "symbol": ["BTC/USDT", "BTC/USDT"],
    })
    save_liquidation_events(cfg, "BTC/USDT", events)

    attached = attach_liquidations_to_ohlcv(ohlcv, cfg, "BTC/USDT", bar_delta=pd.Timedelta(hours=1))
    assert float(attached.loc[idx[2], "liq_long"]) == pytest.approx(1000.0)
    assert float(attached.loc[idx[5], "liq_short"]) == pytest.approx(2000.0)
    # 首笔前未知
    assert pd.isna(attached.loc[idx[0], "liq_long"])

    feat = build_feature_matrix(ohlcv, cfg, symbol="BTC/USDT")
    assert "liq_imbalance" in feat.columns
    assert float(feat["liq_imbalance"].abs().max()) > 0
    assert feat.attrs.get("liquidations_attached_from_store") is True
    deg = feat.attrs.get("degradations") or []
    assert "derivatives_liquidations_unavailable" not in deg


def test_attach_keeps_panel_when_store_empty(tmp_path):
    """事件库为空时保留面板已有 liq; 有落入面板的事件则以事件库为准覆盖。"""
    from crypto_alpha.config import Config
    from crypto_alpha.data.liquidations import (
        attach_liquidations_to_ohlcv,
        save_liquidation_events,
    )

    cfg = Config.load()
    cfg.raw["project"]["data_dir"] = str(tmp_path)
    cfg.raw["data"]["fetch_liquidations"] = True
    cfg.raw["data"]["liquidations"] = {"store_dir": "liquidations", "auto_attach": True}
    idx = pd.date_range("2024-06-01", periods=3, freq="1h", tz="UTC")
    ohlcv = pd.DataFrame({
        "close": [1.0, 2.0, 3.0],
        "open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0], "low": [1.0, 2.0, 3.0],
        "volume": 1.0,
        "liq_long": [5.0, 0.0, 0.0],
        "liq_short": [0.0, 0.0, 0.0],
    }, index=idx)
    out = attach_liquidations_to_ohlcv(ohlcv, cfg, "BTC/USDT", bar_delta=pd.Timedelta(hours=1))
    assert float(out["liq_long"].iloc[0]) == pytest.approx(5.0)

    events = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-06-01 00:10", tz="UTC")],
        "side_bucket": ["long"],
        "notional": [99.0],
        "exchange": ["test"],
        "symbol": ["BTC/USDT"],
    })
    save_liquidation_events(cfg, "BTC/USDT", events)
    out2 = attach_liquidations_to_ohlcv(ohlcv, cfg, "BTC/USDT", bar_delta=pd.Timedelta(hours=1))
    assert float(out2["liq_long"].iloc[0]) == pytest.approx(99.0)


def test_attach_outside_panel_does_not_wipe(tmp_path):
    """事件全在面板窗外: 不覆盖面板, 记 liquidations_outside_panel。"""
    from crypto_alpha.config import Config
    from crypto_alpha.data.liquidations import (
        attach_liquidations_to_ohlcv,
        save_liquidation_events,
    )

    cfg = Config.load()
    cfg.raw["project"]["data_dir"] = str(tmp_path)
    cfg.raw["data"]["fetch_liquidations"] = True
    cfg.raw["data"]["liquidations"] = {"store_dir": "liquidations", "auto_attach": True}
    idx = pd.date_range("2024-06-01", periods=3, freq="1h", tz="UTC")
    ohlcv = pd.DataFrame({
        "close": [1.0, 2.0, 3.0],
        "open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0], "low": [1.0, 2.0, 3.0],
        "volume": 1.0,
        "liq_long": [5.0, 0.0, 0.0],
        "liq_short": [0.0, 0.0, 0.0],
    }, index=idx)
    events = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-01-01 00:10", tz="UTC")],
        "side_bucket": ["long"],
        "notional": [99.0],
        "exchange": ["test"],
        "symbol": ["BTC/USDT"],
    })
    save_liquidation_events(cfg, "BTC/USDT", events)
    out = attach_liquidations_to_ohlcv(ohlcv, cfg, "BTC/USDT", bar_delta=pd.Timedelta(hours=1))
    assert float(out["liq_long"].iloc[0]) == pytest.approx(5.0)
    assert out.attrs.get("liquidations_outside_panel") is True
    assert out.attrs.get("liquidations_attached_from_store") is False


def test_aggregate_events_outside_panel_stay_nan():
    """事件全在面板窗外 → 全 NaN, 不得填 0。"""
    from crypto_alpha.data.fetch import _aggregate_liquidations

    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    rows = [
        {"timestamp": int(pd.Timestamp("2024-02-01 00:10", tz="UTC").timestamp() * 1000),
         "side": "buy", "quoteValue": 9.0},
    ]
    lng, sht = _aggregate_liquidations(rows, idx, pd.Timedelta(hours=1))
    assert lng.isna().all() and sht.isna().all()


def test_paginate_liquidations_tip_without_since():
    """Gate 类: 带 since 空、无 since 有数据 → 仍应返回 tip。"""
    from crypto_alpha.data.fetch import _paginate_liquidations

    class _Ex:
        has = {"fetchLiquidations": True}

        def fetch_liquidations(self, symbol, since=None, limit=1000):
            if since is not None:
                return []
            return [
                {
                    "timestamp": 1_700_000_000_000,
                    "side": "buy",
                    "quoteValue": 1234.0,
                    "symbol": symbol,
                }
            ]

    rows = _paginate_liquidations(
        _Ex(), "BTC/USDT", start_ms=1_600_000_000_000, end_ms=1_800_000_000_000
    )
    assert len(rows) == 1
    assert float(rows[0]["quoteValue"]) == pytest.approx(1234.0)


def test_import_liquidation_events_frame(tmp_path):
    from crypto_alpha.config import Config
    from crypto_alpha.data.liquidations import (
        attach_liquidations_to_ohlcv,
        import_liquidation_events_frame,
    )

    cfg = Config.load()
    cfg.raw["project"]["data_dir"] = str(tmp_path)
    cfg.raw["data"]["fetch_liquidations"] = True
    cfg.raw["data"]["liquidations"] = {"store_dir": "liquidations", "auto_attach": True}
    idx = pd.date_range("2024-06-01", periods=4, freq="1h", tz="UTC")
    ohlcv = pd.DataFrame({
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
        "liq_long": np.nan, "liq_short": np.nan,
    }, index=idx)
    frame = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-06-01 01:20", tz="UTC")],
        "side": ["sell"],
        "notional": [500.0],
    })
    import_liquidation_events_frame(cfg, "BTC/USDT", frame, exchange="csv")
    attached = attach_liquidations_to_ohlcv(ohlcv, cfg, "BTC/USDT", bar_delta=pd.Timedelta(hours=1))
    assert float(attached.loc[idx[1], "liq_long"]) == pytest.approx(500.0)
