"""样本权重: 唯一性(uniqueness) + 收益幅度 + 时间衰减。

出处: López de Prado, AFML, ch.4。
目的: 相邻交易的持有期在时间上重叠, 标签并不独立。用并发度归一化得到
"平均唯一性", 降低重叠样本的权重, 避免高相关样本主导训练。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def num_concurrent_events(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """计算每根 bar 上并发的(未平仓)事件数量。"""
    t1 = t1.dropna()
    idx = bar_index
    count = pd.Series(0, index=idx, dtype=float)
    for t0, t1_ in t1.items():
        count.loc[t0:t1_] += 1.0
    return count


def average_uniqueness(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """每个事件的平均唯一性 = 其持有期内 (1/并发数) 的均值。"""
    conc = num_concurrent_events(bar_index, t1)
    conc = conc.replace(0, np.nan)
    out = {}
    for t0, t1_ in t1.dropna().items():
        seg = 1.0 / conc.loc[t0:t1_]
        out[t0] = float(seg.mean())
    return pd.Series(out).reindex(t1.index)


def sample_weights_by_return(events: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.Series:
    """结合唯一性与收益幅度的样本权重(绝对收益越大, 信息量越高)。"""
    u = average_uniqueness(bar_index, events["t1"]).fillna(1e-6)
    w = (events["ret"].abs() * u).fillna(0.0)
    w = w / (w.mean() + 1e-12)  # 归一化到均值 1
    return w.rename("w")


def time_decay_weights(events: pd.DataFrame, last_weight: float = 0.5) -> pd.Series:
    """线性时间衰减: 越新的样本权重越高(最旧样本权重 = last_weight)。"""
    order = np.argsort(events.index.values)
    ranks = np.empty(len(order))
    ranks[order] = np.arange(len(order))
    frac = ranks / max(len(order) - 1, 1)
    decay = last_weight + (1.0 - last_weight) * frac
    return pd.Series(decay, index=events.index, name="w_decay")
