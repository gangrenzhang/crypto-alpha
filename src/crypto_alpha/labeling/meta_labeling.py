"""主信号 + 元标签构建。

流程:
1. primary_signal: 用一个简单、可解释的主策略(动量/均值回归)决定方向 side(+1/-1)。
2. 三重障碍在该方向上给出 bin(该信号是否盈利)。
3. 二层模型(四专家集成)学习 "该不该执行主信号 + 概率", 即元标签。
这正是 "做多做空概率 + 止损" 的标准范式。

障碍宽度与实盘 decide 共用同一波动度量(默认相对 ATR = atr/close)与
加性价格公式 entry ± side×mult×atr(多空均对齐; 持仓收益再映射为对数 ret)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .triple_barrier import (
    causal_cusum_threshold,
    cusum_filter,
    get_events,
    get_bins,
)


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
    """障碍单位波动: 与实盘止损同一度量, 并换算到相对价格(atr/close)。

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


def resolve_event_times(close: pd.Series, trgt: pd.Series, lc: dict) -> tuple[pd.DatetimeIndex, bool]:
    """与 build_meta_labels 相同的事件采样逻辑; 返回 (事件索引, 是否全量回退)。

    全量回退时实盘可对每根合格 bar 决策; 否则实盘必须落在 CUSUM 事件上。
    """
    min_periods = int(lc.get("cusum_min_periods", 50))
    thr = causal_cusum_threshold(trgt, min_periods=min_periods)
    t_events = cusum_filter(close, threshold=thr)
    min_events = int(lc.get("min_cusum_events", 50))
    full_sampling = False
    lookback = int(lc.get("primary_lookback", 24))
    if len(t_events) < min_events:
        t_events = close.index[lookback:]
        full_sampling = True
    return pd.DatetimeIndex(t_events), full_sampling


def build_meta_labels(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """端到端生成元标签数据集。

    返回索引为事件时间戳的 DataFrame, 含: ret, bin, side, t1, trgt。
    attrs: cusum_full_sampling(bool) — 供实盘决策对齐。
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

    t_events, full_sampling = resolve_event_times(close, trgt, lc)

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
    if len(events) == 0:
        out = pd.DataFrame(columns=["ret", "bin", "side", "t1", "trgt", "bars_held"])
        out.attrs["cusum_full_sampling"] = full_sampling
        return out

    events["trgt"] = trgt.reindex(events.index)
    bins = get_bins(events, close, pt_sl)
    bins["trgt"] = events["trgt"].reindex(bins.index).values
    out = bins.dropna(subset=["bin", "t1"])
    out.attrs["cusum_full_sampling"] = full_sampling
    return out
