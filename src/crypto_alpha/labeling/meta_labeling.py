"""主信号 + 元标签构建。

流程:
1. primary_signal: 用一个简单、可解释的主策略(动量/均值回归)决定方向 side(+1/-1)。
2. 三重障碍在该方向上给出 bin(该信号是否盈利)。
3. 二层模型(四专家集成)学习 "该不该执行主信号 + 概率", 即元标签。
这正是 "做多做空概率 + 止损" 的标准范式。

障碍宽度与实盘 decide 共用同一波动度量(默认相对 ATR = atr/close),
避免 "标签用 rv、下单用 ATR" 的训练/执行错位。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .triple_barrier import cusum_filter, get_events, get_bins


def primary_signal(close: pd.Series, kind: str = "momentum", lookback: int = 24) -> pd.Series:
    """生成方向信号 side ∈ {+1, -1}。

    momentum: 近 lookback 收益为正 => 做多, 否则做空。
    meanrev : 价格高于均值 => 做空(回归), 否则做多。
    """
    if kind == "momentum":
        mom = close.pct_change(lookback)
        side = np.sign(mom).replace(0, 1)
    elif kind == "meanrev":
        ma = close.rolling(lookback).mean()
        side = -np.sign(close - ma).replace(0, 1)
    else:
        raise ValueError(f"未知主信号类型: {kind}")
    return side.rename("side")


def _barrier_target(df: pd.DataFrame, close: pd.Series, lc: dict, vol_window: int) -> pd.Series:
    """障碍单位波动: 与实盘止损同一度量, 并换算到对数收益空间。

    - atr: (atr_14 / close), 近似相对波幅, 与 decide 的 ATR 倍数一致。
    - rv: 已实现对数收益波动(旧口径, 可配置回退)。
    """
    kind = str(lc.get("barrier_vol", "atr")).lower()
    if kind == "rv":
        if "rv" in df.columns:
            trgt = df["rv"]
        else:
            trgt = np.log(close).diff().rolling(vol_window).std()
    else:  # atr (default)
        if "atr_14" in df.columns:
            atr_abs = df["atr_14"]
        else:
            from ..features.technical import atr as atr_fn

            atr_abs = atr_fn(df, 14)
        trgt = atr_abs / close.replace(0, np.nan)
    trgt = trgt.reindex(close.index).ffill()  # 仅前向填充, 避免 bfill 把未来波动灌到首部
    return trgt


def build_meta_labels(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """端到端生成元标签数据集。

    返回索引为事件时间戳的 DataFrame, 含: ret, bin, side, t1, trgt。
    """
    lc = cfg["labeling"]
    try:
        vol_window = int(cfg["features"]["vol_window"])
    except Exception:
        vol_window = 50

    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close

    side = primary_signal(close, kind=lc["primary_signal"], lookback=int(lc["primary_lookback"]))
    trgt = _barrier_target(df, close, lc, vol_window)

    # CUSUM 事件采样: 阈值取目标波动中位数
    thr = float(np.nanmedian(trgt.values)) or 0.005
    t_events = cusum_filter(close, threshold=thr)
    min_events = int(lc.get("min_cusum_events", 50))
    if len(t_events) < min_events:  # 事件太少则退回全量采样
        t_events = close.index[int(lc["primary_lookback"]) :]

    pt_sl = tuple(lc["pt_sl"])
    events = get_events(
        close=close,
        high=high,
        low=low,
        t_events=t_events,
        pt_sl=pt_sl,
        trgt=trgt,
        vertical_bars=int(lc["vertical_barrier_bars"]),
        side=side,
        min_ret=float(lc["min_ret"]),
    )
    events["trgt"] = trgt.reindex(events.index)
    bins = get_bins(events, close, pt_sl)
    bins["trgt"] = events["trgt"].values
    return bins.dropna(subset=["bin", "t1"])
