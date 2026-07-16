"""主信号 + 元标签构建。

流程:
1. primary_signal: 用一个简单、可解释的主策略(动量/均值回归)决定方向 side(+1/-1)。
2. 三重障碍在该方向上给出 bin(该信号是否盈利)。
3. 二层模型(四专家集成)学习 "该不该执行主信号 + 概率", 即元标签。
这正是 "做多做空概率 + 止损" 的标准范式。
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


def build_meta_labels(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """端到端生成元标签数据集。

    返回索引为事件时间戳的 DataFrame, 含: ret, bin, side, t1, trgt。
    """
    lc = cfg["labeling"]
    close = df["close"]

    side = primary_signal(close, kind=lc["primary_signal"], lookback=int(lc["primary_lookback"]))

    # 目标波动: 用已实现波动率(若无则用滚动 std)
    trgt = df["rv"] if "rv" in df.columns else np.log(close).diff().rolling(50).std()
    trgt = trgt.reindex(close.index).ffill().bfill()

    # CUSUM 事件采样: 阈值取目标波动中位数
    thr = float(np.nanmedian(trgt.values)) or 0.005
    t_events = cusum_filter(close, threshold=thr)
    if len(t_events) < 50:  # 事件太少则退回全量采样
        t_events = close.index[int(lc["primary_lookback"]) :]

    events = get_events(
        close=close,
        t_events=t_events,
        pt_sl=tuple(lc["pt_sl"]),
        trgt=trgt,
        vertical_bars=int(lc["vertical_barrier_bars"]),
        side=side,
        min_ret=float(lc["min_ret"]),
    )
    events["trgt"] = trgt.reindex(events.index)
    bins = get_bins(events, close)
    bins["trgt"] = events["trgt"].values
    return bins.dropna(subset=["bin", "t1"])
