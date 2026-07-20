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


def add_technical_features(
    df: pd.DataFrame,
    windows: list[int],
    vol_window: int,
    *,
    oi_change_bars: int = 24,
) -> pd.DataFrame:
    """在原始 OHLCV(+衍生品) 上追加一组技术指标特征。

    ``oi_change_bars``: OI 变化的回看 **bar 数**(应按墙钟≈24h 由调用方换算;
    默认 24 兼容旧 1h 主周期; 30m 主周期应由 build 传入 48)。
    """
    out = df.copy()
    close = out["close"]
    logret = np.log(close).diff()
    out["logret_1"] = logret
    oi_bars = max(int(oi_change_bars), 1)

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

    # MACD: 归一化为相对价格量纲(÷close), 避免多年价格量级漂移导致的非平稳
    # (BTC 从 ~1万 到 ~6万, 绝对 MACD 量级会翻数倍, 破坏跨 regime/实盘泛化)。
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_abs = ema12 - ema26
    macd_signal_abs = macd_abs.ewm(span=9, adjust=False).mean()
    out["macd"] = macd_abs / (close + 1e-12)
    out["macd_signal"] = macd_signal_abs / (close + 1e-12)
    out["macd_hist"] = (macd_abs - macd_signal_abs) / (close + 1e-12)

    # atr_14 保持**绝对**量纲: 标注(_barrier_target)与实盘 decide 需要绝对 ATR 距离;
    # 建模改用相对 ATR(atr_norm=atr/close), atr_14 由 feature_columns 排除, 不直接进模型。
    out["atr_14"] = atr(out, 14)
    out["atr_norm"] = out["atr_14"] / (close + 1e-12)
    out["rv"] = realized_volatility(close, vol_window)

    # 衍生品衍生特征(若存在)。拉取失败时源列为全 NaN —— 必须 fillna(0),
    # 否则 prepare_dataset 的 notna().all 会清空全部建模样本(与「优雅降级」冲突)。
    if "funding_rate" in out.columns:
        out["funding_z"] = (
            (out["funding_rate"] - out["funding_rate"].rolling(vol_window).mean())
            / (out["funding_rate"].rolling(vol_window).std() + 1e-12)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "open_interest" in out.columns:
        out["oi_change"] = (
            out["open_interest"].pct_change(oi_bars)
            .replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    return out
