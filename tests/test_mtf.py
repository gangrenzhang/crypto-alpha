"""多周期(MTF)特征严谨性测试: 无前视 + 合成重采样一致性 + 接入主干。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crypto_alpha.config import Config
from crypto_alpha.data.fetch import (
    generate_synthetic_ohlcv,
    resample_ohlcv,
    timeframe_delta,
    load_aux_timeframes,
)
from crypto_alpha.features.mtf import _align_one_tf
from crypto_alpha.features.build import build_feature_matrix, mtf_columns
from crypto_alpha.pipeline import prepare_dataset


def _make_cfg(**overrides) -> Config:
    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = True
    cfg.raw["data"]["synthetic_bars"] = 3000
    cfg.raw["data"]["timeframe"] = "1h"
    cfg.raw["data"]["aux_timeframes"] = ["4h", "1d"]
    cfg.raw["features"]["mtf_enabled"] = True
    cfg.raw["news"]["as_feature"] = False  # 本测试不依赖新闻
    cfg.raw["news"]["use_synthetic"] = True
    for k, v in overrides.items():
        # 支持 "data.aux_timeframes" 式覆盖
        if "." in k:
            a, b = k.split(".", 1)
            cfg.raw[a][b] = v
        else:
            cfg.raw[k] = v
    return cfg


def test_resample_consistent_with_main():
    """辅周期由主周期重采样: 完整 4h 窗的 OHLC 必须与窗内 1h 聚合一致。"""
    main = generate_synthetic_ohlcv("BTC/USDT", n_bars=500, timeframe="1h", seed=42)
    h4 = resample_ohlcv(main, "4h")
    assert len(h4) > 50
    delta = timeframe_delta("4h")
    found = False
    for t0 in h4.index:
        window = main.loc[(main.index >= t0) & (main.index < t0 + delta)]
        if len(window) != 4:
            continue
        assert abs(h4.loc[t0, "close"] - window["close"].iloc[-1]) < 1e-9
        assert abs(h4.loc[t0, "open"] - window["open"].iloc[0]) < 1e-9
        assert abs(h4.loc[t0, "high"] - window["high"].max()) < 1e-9
        assert abs(h4.loc[t0, "low"] - window["low"].min()) < 1e-9
        found = True
        break
    assert found, "未找到含完整 4 根 1h 的 4h 窗口"


def test_mtf_no_lookahead_alignment():
    """核心防泄漏: 主 bar 决策时刻看不到尚未收盘的 4h 特征。

    构造: 4h bar 开盘 08:00, available_at=12:00, 特征标记值=99(哨兵)。
    - 主 1h 开盘 09:00 → 决策 10:00 < 12:00 → 不得看到 99
    - 主 1h 开盘 12:00 → 决策 13:00 ≥ 12:00 → 必须看到 99
    """
    idx_1h = pd.date_range("2024-01-01 08:00", periods=8, freq="1h", tz="UTC")
    main = pd.DataFrame({"close": np.arange(100.0, 108.0)}, index=idx_1h)

    # 仅一根 4h: 08:00–12:00
    aux_open = pd.Timestamp("2024-01-01 08:00", tz="UTC")
    aux_feat = pd.DataFrame({"tf4h_sentinel": [99.0]}, index=pd.DatetimeIndex([aux_open]))

    aligned = _align_one_tf(main.index, aux_feat, "1h", "4h")

    # 09:00 主 bar: decision=10:00, 4h 尚未收盘
    v_0900 = aligned.loc[pd.Timestamp("2024-01-01 09:00", tz="UTC"), "tf4h_sentinel"]
    assert pd.isna(v_0900) or v_0900 != 99.0, f"09:00 不应看到未收盘 4h, 得到 {v_0900}"

    # 11:00 主 bar: decision=12:00, 刚好 4h 收盘 → 可见
    v_1100 = aligned.loc[pd.Timestamp("2024-01-01 11:00", tz="UTC"), "tf4h_sentinel"]
    assert v_1100 == 99.0, f"11:00(决策12:00)应看到已收盘 4h, 得到 {v_1100}"

    # 08:00 主 bar: decision=09:00 < 12:00 → 不可见
    v_0800 = aligned.loc[pd.Timestamp("2024-01-01 08:00", tz="UTC"), "tf4h_sentinel"]
    assert pd.isna(v_0800) or v_0800 != 99.0


def test_mtf_reject_finer_than_main():
    """方案B拒绝把更细周期当辅周期塞进特征。"""
    cfg = _make_cfg()
    cfg.raw["data"]["aux_timeframes"] = ["15m", "4h"]
    main = generate_synthetic_ohlcv("BTC/USDT", n_bars=400, timeframe="1h", seed=1)
    # 15m 不能从 1h resample 出更细数据; load_aux 应跳过 15m
    aux = load_aux_timeframes(cfg, "BTC/USDT", main_df=main)
    assert "15m" not in aux
    assert "4h" in aux


def test_build_feature_matrix_includes_mtf():
    cfg = _make_cfg()
    main = generate_synthetic_ohlcv("BTC/USDT", n_bars=2000, timeframe="1h", seed=7)
    feat = build_feature_matrix(main, cfg, symbol="BTC/USDT")
    cols = mtf_columns(feat)
    assert any(c.startswith("tf4h_") for c in cols), cols
    assert any(c.startswith("tf1d_") for c in cols), cols
    assert "mtf_confluence" in cols
    # 无泄漏填充后不应大片 NaN 阻断建模
    assert feat[cols].isna().mean().max() == 0.0


def test_mtf_disabled():
    cfg = _make_cfg(**{"features.mtf_enabled": False})
    # 上面 overrides 写法不对 — 手动设
    cfg = _make_cfg()
    cfg.raw["features"]["mtf_enabled"] = False
    main = generate_synthetic_ohlcv("BTC/USDT", n_bars=800, timeframe="1h", seed=3)
    feat = build_feature_matrix(main, cfg, symbol="BTC/USDT")
    assert mtf_columns(feat) == []


def test_prepare_dataset_with_mtf_smoke():
    """端到端: prepare_dataset 含 MTF 特征且事件数正常。"""
    cfg = _make_cfg()
    cfg.raw["data"]["synthetic_bars"] = 4000
    cfg.raw["experts"]["enabled"] = ["gbdt"]
    ds = prepare_dataset(cfg, "BTC/USDT")
    mtf_in_x = [c for c in ds.feature_cols if c.startswith("tf4h_") or c.startswith("tf1d_") or c == "mtf_confluence"]
    assert len(mtf_in_x) > 0, "数据集特征应含多周期列"
    assert len(ds.y) > 100
    assert "side" in ds.feature_cols
    assert "side" in ds.X.columns
    assert ds.X.shape[1] == len(ds.feature_cols)  # side 已在 feature_cols 内


if __name__ == "__main__":
    test_resample_consistent_with_main()
    print("OK resample")
    test_mtf_no_lookahead_alignment()
    print("OK no-lookahead")
    test_mtf_reject_finer_than_main()
    print("OK reject finer")
    test_build_feature_matrix_includes_mtf()
    print("OK build matrix")
    test_mtf_disabled()
    print("OK disabled")
    test_prepare_dataset_with_mtf_smoke()
    print("OK prepare_dataset")
    print("所有 MTF 测试通过。")
