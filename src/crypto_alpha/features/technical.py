"""技术指标与波动率特征。全部严格因果(仅用 t 时刻及之前信息)。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """平均真实波幅, 用于波动率自适应止损与三重障碍。"""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def realized_volatility(close: pd.Series, window: int = 50) -> pd.Series:
    """基于对数收益的滚动已实现波动率(日/bar 波动), 作为标签的目标尺度。"""
    logret = np.log(close).diff()
    return logret.rolling(window).std()


def add_technical_features(df: pd.DataFrame, windows: list[int], vol_window: int) -> pd.DataFrame:
    """在原始 OHLCV(+衍生品) 上追加一组技术指标特征。"""
    out = df.copy()
    close = out["close"]
    logret = np.log(close).diff()
    out["logret_1"] = logret

    for w in windows:
        out[f"ret_{w}"] = close.pct_change(w)
        out[f"mom_{w}"] = close / close.shift(w) - 1.0
        out[f"vol_{w}"] = logret.rolling(w).std()
        out[f"rsi_{w}"] = _rsi(close, w)
        ma = close.rolling(w).mean()
        std = close.rolling(w).std()
        out[f"zscore_{w}"] = (close - ma) / (std + 1e-12)
        out[f"bb_pos_{w}"] = (close - ma) / (2 * std + 1e-12)  # 布林带内位置
        out[f"vol_ratio_{w}"] = out["volume"] / (out["volume"].rolling(w).mean() + 1e-12)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["atr_14"] = atr(out, 14)
    out["rv"] = realized_volatility(close, vol_window)

    # 衍生品衍生特征(若存在)
    if "funding_rate" in out.columns:
        out["funding_z"] = (
            (out["funding_rate"] - out["funding_rate"].rolling(vol_window).mean())
            / (out["funding_rate"].rolling(vol_window).std() + 1e-12)
        )
    if "open_interest" in out.columns:
        out["oi_change"] = out["open_interest"].pct_change(24)

    return out
