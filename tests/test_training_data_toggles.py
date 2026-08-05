"""training_data.yaml 开关: 合并、同步遗留键、特征面裁剪。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from crypto_alpha.config import (
    TRAINING_DATA_DEFAULTS,
    apply_training_data_toggles,
    resolve_training_data_toggles,
)
from crypto_alpha.features.build import build_feature_matrix, feature_columns
from crypto_alpha.features.macro_calendar import MACRO_FEATURE_COLS, add_macro_calendar_features
from crypto_alpha.features.news_features import NEWS_FEATURE_COLS, add_news_features
from crypto_alpha.features.technical import add_technical_features


def _ohlcv(n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.uniform(1, 10, n),
            "funding_rate": rng.normal(0, 1e-4, n),
            "open_interest": 1e6 + np.cumsum(rng.normal(0, 1e3, n)),
            "liq_long": rng.uniform(0, 100, n),
            "liq_short": rng.uniform(0, 100, n),
        },
        index=idx,
    )


def test_resolve_file_overrides_inline(tmp_path):
    raw = {"training_data": {"news": True, "macro_calendar": False}}
    p = tmp_path / "training_data.yaml"
    p.write_text(yaml.dump({"news": False, "funding": False}), encoding="utf-8")
    td = resolve_training_data_toggles(raw, training_data_path=p)
    assert td["news"] is False          # 文件优先
    assert td["funding"] is False
    assert td["macro_calendar"] is False  # inline 保留
    assert td["ohlcv"] is True


def test_ohlcv_false_raises(tmp_path):
    p = tmp_path / "training_data.yaml"
    p.write_text("ohlcv: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ohlcv"):
        resolve_training_data_toggles({}, training_data_path=p)


def test_apply_syncs_legacy_keys(tmp_path):
    p = tmp_path / "training_data.yaml"
    p.write_text(
        yaml.dump(
            {
                "news": True,
                "macro_calendar": False,
                "mtf": False,
                "liquidations": True,
                "funding": False,
                "open_interest": False,
            }
        ),
        encoding="utf-8",
    )
    raw = {
        "features": {"mtf_enabled": True},
        "news": {"as_feature": False},
        "macro_calendar": {"as_feature": True},
        "data": {"fetch_liquidations": False},
    }
    apply_training_data_toggles(raw, training_data_path=p)
    assert raw["news"]["as_feature"] is True
    assert raw["macro_calendar"]["as_feature"] is False
    assert raw["features"]["mtf_enabled"] is False
    assert raw["features"]["use_funding"] is False
    assert raw["features"]["use_open_interest"] is False
    assert raw["features"]["use_liquidations"] is True
    assert raw["data"]["fetch_liquidations"] is True
    assert raw["training_data"]["news"] is True


def test_technical_respects_funding_oi_liq_flags():
    df = _ohlcv()
    on = add_technical_features(
        df, [14], 20, use_funding=True, use_open_interest=True, use_liquidations=True
    )
    assert "funding_z" in on.columns and "oi_change" in on.columns
    assert "liq_imbalance" in on.columns

    off = add_technical_features(
        df, [14], 20, use_funding=False, use_open_interest=False, use_liquidations=False
    )
    assert "funding_z" not in off.columns
    assert "oi_change" not in off.columns
    assert "liq_imbalance" not in off.columns
    assert "rsi_14" in off.columns  # OHLCV 技术指标仍在


def test_build_feature_matrix_honors_training_toggles():
    df = _ohlcv()
    cfg = {
        "data": {"timeframe": "30m", "fetch_liquidations": False},
        "features": {
            "windows": [14],
            "vol_window": 20,
            "frac_diff_d": 0.4,
            "frac_diff_thres": 1e-4,
            "mtf_enabled": False,
            "use_funding": False,
            "use_open_interest": False,
            "use_liquidations": False,
        },
        "news": {"as_feature": False},
        "macro_calendar": {"as_feature": False},
        "training_data": dict(TRAINING_DATA_DEFAULTS),
    }
    feat = build_feature_matrix(df, cfg, symbol=None, aux_frames={})
    cols = set(feature_columns(feat))
    assert "funding_z" not in cols
    assert "oi_change" not in cols
    assert "liq_imbalance" not in cols


def test_news_macro_as_feature_still_gate(tmp_path):
    df = _ohlcv()
    feat = df.copy()
    cfg_off = {
        "news": {"as_feature": False},
        "macro_calendar": {"as_feature": False, "store_dir": str(tmp_path)},
    }
    out = add_news_features(feat, cfg_off, "BTC/USDT")
    out = add_macro_calendar_features(out, cfg_off, "BTC/USDT")
    for c in NEWS_FEATURE_COLS:
        assert c not in out.columns
    for c in MACRO_FEATURE_COLS:
        assert c not in out.columns
