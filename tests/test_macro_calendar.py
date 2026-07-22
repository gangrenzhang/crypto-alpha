"""宏观日历: PIT、surprise 分流、特征定点绑定、读盘开关。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_alpha.data.macro_calendar import (
    attach_surprise_column,
    compute_surprise,
    import_macro_events_frame,
    normalize_macro_events,
    visible_events_at,
)
from crypto_alpha.features.macro_calendar import (
    MACRO_FEATURE_COLS,
    add_macro_calendar_features,
    build_macro_feature_matrix,
)


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "name": "CPI YoY",
            "country": "US",
            "category": "inflation",
            "importance": 5,
            "scheduled_at": "2024-01-11T13:30:00Z",
            "released_at": "2024-01-11T13:30:00Z",
            "previous": 3.1,
            "forecast": 3.2,
            "actual": 3.4,
            "unit": "%",
            "source": "test",
        },
        {
            "name": "Speech",
            "country": "US",
            "category": "speech",
            "importance": 3,
            "scheduled_at": "2024-01-12T15:00:00Z",
            "released_at": "2024-01-12T15:00:00Z",
            "previous": np.nan,
            "forecast": np.nan,
            "actual": np.nan,
            "unit": "",
            "source": "test",
        },
    ])


def test_compute_surprise_scale():
    s = compute_surprise(3.1, 3.2, 3.4)
    assert abs(s - (3.4 - 3.2) / max(3.2, 3.1, 1.0)) < 1e-9
    assert np.isnan(compute_surprise(1.0, 1.0, np.nan))


def test_visible_events_respects_release_buffer():
    ev = normalize_macro_events(_sample_events())
    before = pd.Timestamp("2024-01-11T13:30:00Z") - pd.Timedelta(minutes=1)
    vis = visible_events_at(ev, before, buffer_minutes=5)
    assert len(vis) == 0
    after = pd.Timestamp("2024-01-11T13:30:00Z") + pd.Timedelta(minutes=5)
    vis2 = visible_events_at(ev, after, buffer_minutes=5)
    assert "CPI YoY" in set(vis2["name"])


def test_feature_matrix_no_lookahead_surprise():
    ev = attach_surprise_column(normalize_macro_events(_sample_events()))
    dec = pd.DatetimeIndex([pd.Timestamp("2024-01-11T13:29:00Z")])
    feat = build_macro_feature_matrix(
        dec, ev, buffer_minutes=5, ttl_hours=72, min_importance=3,
    )
    assert float(feat["macro_surprise_raw"].iloc[0]) == 0.0
    assert float(feat["has_recent_macro"].iloc[0]) == 0.0
    dec2 = pd.DatetimeIndex([pd.Timestamp("2024-01-11T13:35:00Z")])
    feat2 = build_macro_feature_matrix(
        dec2, ev, buffer_minutes=5, ttl_hours=72, min_importance=3,
    )
    expected = (3.4 - 3.2) / max(3.2, 3.1, 1.0)
    assert abs(float(feat2["macro_surprise_raw"].iloc[0]) - expected) < 1e-9
    assert float(feat2["has_recent_macro"].iloc[0]) == 1.0


def test_hours_to_next_uses_schedule_before_release():
    ev = normalize_macro_events(_sample_events())
    dec = pd.DatetimeIndex([pd.Timestamp("2024-01-11T14:00:00Z")])
    feat = build_macro_feature_matrix(
        dec, ev, buffer_minutes=5, ttl_hours=72, horizon_hours=168, min_importance=3,
    )
    assert 20 < float(feat["macro_hours_to_next"].iloc[0]) < 30
    assert float(feat["macro_next_importance"].iloc[0]) == 3 / 5


def test_speech_does_not_wash_numeric_surprise():
    """讲话(无 forecast/actual)不得把 CPI surprise 冲成 0。"""
    ev = attach_surprise_column(normalize_macro_events(pd.DataFrame([
        {
            "name": "CPI YoY", "country": "US", "category": "inflation", "importance": 5,
            "scheduled_at": "2024-01-11T13:30:00Z", "released_at": "2024-01-11T13:30:00Z",
            "previous": 3.1, "forecast": 3.2, "actual": 3.4, "unit": "%", "source": "t",
        },
        {
            "name": "Speech", "country": "US", "category": "speech", "importance": 3,
            "scheduled_at": "2024-01-11T15:00:00Z", "released_at": "2024-01-11T15:00:00Z",
            "previous": np.nan, "forecast": np.nan, "actual": np.nan, "unit": "", "source": "t",
        },
    ])))
    expected = (3.4 - 3.2) / max(3.2, 3.1, 1.0)
    after_cpi = build_macro_feature_matrix(
        pd.DatetimeIndex([pd.Timestamp("2024-01-11T13:40:00Z")]),
        ev, buffer_minutes=5, ttl_hours=72,
    )
    after_speech = build_macro_feature_matrix(
        pd.DatetimeIndex([pd.Timestamp("2024-01-11T15:10:00Z")]),
        ev, buffer_minutes=5, ttl_hours=72,
    )
    assert abs(float(after_cpi["macro_surprise_raw"].iloc[0]) - expected) < 1e-9
    # surprise 通道仍指向 CPI; 注意力通道跟讲话
    assert abs(float(after_speech["macro_surprise_raw"].iloc[0]) - expected) < 1e-9
    assert float(after_speech["macro_importance"].iloc[0]) == 3 / 5
    assert float(after_speech["has_recent_macro"].iloc[0]) == 1.0
    assert float(after_speech["macro_surprise_abs_max"].iloc[0]) > 0.0


def test_ttl_expires_recent_and_surprise():
    ev = attach_surprise_column(normalize_macro_events(pd.DataFrame([{
        "name": "CPI", "country": "US", "category": "inflation", "importance": 5,
        "scheduled_at": "2024-01-11T13:30:00Z", "released_at": "2024-01-11T13:30:00Z",
        "previous": 3.1, "forecast": 3.2, "actual": 3.4, "unit": "%", "source": "t",
    }])))
    t_edge = pd.Timestamp("2024-01-11T13:35:00Z") + pd.Timedelta(hours=72)
    t_past = t_edge + pd.Timedelta(seconds=1)
    f_edge = build_macro_feature_matrix(
        pd.DatetimeIndex([t_edge]), ev, buffer_minutes=5, ttl_hours=72,
    )
    f_past = build_macro_feature_matrix(
        pd.DatetimeIndex([t_past]), ev, buffer_minutes=5, ttl_hours=72,
    )
    assert float(f_edge["has_recent_macro"].iloc[0]) == 1.0
    assert float(f_edge["macro_surprise_raw"].iloc[0]) != 0.0
    assert float(f_past["has_recent_macro"].iloc[0]) == 0.0
    assert float(f_past["macro_surprise_raw"].iloc[0]) == 0.0


def test_delayed_release_awaiting_and_no_surprise():
    """schedule 已过、尚未 available: awaiting=1, surprise=0, hours_to_next 不指向已过 schedule。"""
    ev = attach_surprise_column(normalize_macro_events(pd.DataFrame([{
        "name": "Delayed NFP", "country": "US", "category": "employment", "importance": 5,
        "scheduled_at": "2024-01-10T13:30:00Z",
        "released_at": "2024-01-12T13:30:00Z",
        "previous": 100.0, "forecast": 110.0, "actual": 150.0, "unit": "k", "source": "t",
    }])))
    mid = pd.Timestamp("2024-01-11T12:00:00Z")
    feat = build_macro_feature_matrix(
        pd.DatetimeIndex([mid]), ev, buffer_minutes=5, ttl_hours=72, horizon_hours=168,
    )
    assert float(feat["macro_surprise_raw"].iloc[0]) == 0.0
    assert float(feat["has_recent_macro"].iloc[0]) == 0.0
    assert float(feat["macro_awaiting_release"].iloc[0]) == 1.0
    assert float(feat["macro_hours_to_next"].iloc[0]) == 168.0
    after = pd.Timestamp("2024-01-12T13:35:00Z")
    feat2 = build_macro_feature_matrix(
        pd.DatetimeIndex([after]), ev, buffer_minutes=5, ttl_hours=72,
    )
    expected = (150.0 - 110.0) / max(110.0, 100.0, 1.0)
    assert abs(float(feat2["macro_surprise_raw"].iloc[0]) - expected) < 1e-9
    assert float(feat2["macro_awaiting_release"].iloc[0]) == 0.0


def test_add_macro_features_respects_as_feature_flag(tmp_path):
    class Cfg(dict):
        root = tmp_path

    cfg = Cfg({
        "data": {"timeframe": "30m"},
        "macro_calendar": {
            "as_feature": False,
            "store_dir": str(tmp_path / "macro"),
        },
    })
    idx = pd.date_range("2024-01-11", periods=5, freq="30min", tz="UTC")
    feat = pd.DataFrame({"close": np.arange(5.0)}, index=idx)
    out = add_macro_calendar_features(feat, cfg, "BTC/USDT")
    for c in MACRO_FEATURE_COLS:
        assert c not in out.columns


def test_add_macro_features_as_feature_loads_store(tmp_path):
    class Cfg(dict):
        root = tmp_path

    store = tmp_path / "macro"
    cfg = Cfg({
        "data": {"timeframe": "30m"},
        "macro_calendar": {
            "as_feature": True,
            "store_dir": str(store),
            "buffer_minutes": 5,
            "feature_ttl_hours": 72,
            "feature_halflife_hours": 24,
            "horizon_hours": 168,
            "min_importance": 3,
            "min_coverage_warn": 0.0,
        },
    })
    import_macro_events_frame(cfg, _sample_events(), replace=True)
    # bar 开盘 13:00 → decision 13:30; CPI available 13:35 → 下一根 13:30 开盘 decision=14:00 可见
    idx = pd.date_range("2024-01-11 13:00", periods=4, freq="30min", tz="UTC")
    feat = pd.DataFrame({"close": np.arange(4.0)}, index=idx)
    out = add_macro_calendar_features(feat.copy(), cfg, "BTC/USDT")
    for c in MACRO_FEATURE_COLS:
        assert c in out.columns
    # decision_at = index + 30m; 第三根 14:00 开盘 → decision 14:30, CPI 已可见
    assert float(out["has_recent_macro"].iloc[2]) == 1.0
    assert float(out["macro_surprise_raw"].iloc[2]) != 0.0


def test_empty_store_marks_unavailable(tmp_path):
    class Cfg(dict):
        root = tmp_path

    cfg = Cfg({
        "data": {"timeframe": "30m"},
        "macro_calendar": {
            "as_feature": True,
            "store_dir": str(tmp_path / "empty_macro"),
            "feature_ttl_hours": 72,
            "horizon_hours": 168,
        },
    })
    idx = pd.date_range("2024-01-11", periods=3, freq="30min", tz="UTC")
    feat = pd.DataFrame({"close": np.arange(3.0)}, index=idx)
    out = add_macro_calendar_features(feat, cfg, "BTC/USDT")
    assert "macro_calendar_unavailable" in list(out.attrs.get("degradations") or [])
    assert float(out["has_recent_macro"].sum()) == 0.0
