"""样本权重: 唯一性(uniqueness) + 收益幅度 + 时间衰减。

出处: López de Prado, AFML, ch.4。
目的: 相邻交易的持有期在时间上重叠, 标签并不独立。用并发度归一化得到
"平均唯一性", 降低重叠样本的权重, 避免高相关样本主导训练。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bar_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """UTC ns 整数时间戳, 供 searchsorted(与 label 闭区间语义对齐)。"""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.asi8.astype(np.int64, copy=False)


def num_concurrent_events(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """计算每根 bar 上并发的(未平仓)事件数量。

    语义: 对每个事件闭区间 ``[t0, t1]``(label, 含端点)上的 bar +1。
    实现用差分数组 + ``searchsorted``, 与逐事件 ``count.loc[t0:t1] += 1`` 等价,
    但避免 O(事件×持有期) 的 pandas 切片写回。
    """
    t1 = t1.dropna()
    idx = pd.DatetimeIndex(bar_index)
    n = len(idx)
    if n == 0:
        return pd.Series(dtype=float)
    if len(t1) == 0:
        return pd.Series(0.0, index=idx, dtype=float)

    bars = _bar_ns(idx)
    starts = _bar_ns(pd.DatetimeIndex(t1.index))
    ends = _bar_ns(pd.DatetimeIndex(pd.to_datetime(t1.values, utc=True)))

    # [t0, t1] label 闭区间 ↔ bars 上 first>=t0 .. first>t1 的半开区间
    left = np.searchsorted(bars, starts, side="left")
    right = np.searchsorted(bars, ends, side="right")
    # t1 < t0 等畸形输入: pandas loc[t0:t1] 为空; 必须跳过, 否则差分会污染全程计数
    valid = right > left
    if not np.any(valid):
        return pd.Series(0.0, index=idx, dtype=float)
    left = left[valid]
    right = right[valid]

    diff = np.zeros(n + 1, dtype=float)
    np.add.at(diff, left, 1.0)
    np.add.at(diff, right, -1.0)
    count = np.cumsum(diff[:-1])
    return pd.Series(count, index=idx, dtype=float)


def average_uniqueness(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """每个事件的平均唯一性 = 其持有期内 (1/并发数) 的均值。

    与旧实现一致: 并发为 0 的 bar 视为 NaN 后对段内做 nanmean
    (正常路径上事件覆盖的 bar 并发 ≥ 1)。
    """
    if len(t1) == 0:
        return pd.Series(dtype=float)

    t1_valid = t1.dropna()
    if len(t1_valid) == 0:
        return pd.Series(np.nan, index=t1.index, dtype=float)

    conc = num_concurrent_events(bar_index, t1_valid)
    idx = pd.DatetimeIndex(bar_index)
    bars = _bar_ns(idx)
    conc_v = conc.to_numpy(dtype=float)
    inv = np.divide(
        1.0, conc_v,
        out=np.full_like(conc_v, np.nan, dtype=float),
        where=conc_v > 0,
    )

    starts = _bar_ns(pd.DatetimeIndex(t1_valid.index))
    ends = _bar_ns(pd.DatetimeIndex(pd.to_datetime(t1_valid.values, utc=True)))
    left = np.searchsorted(bars, starts, side="left")
    right = np.searchsorted(bars, ends, side="right")

    out = np.full(len(t1_valid), np.nan, dtype=float)
    for i in range(len(t1_valid)):
        lo, hi = int(left[i]), int(right[i])
        if hi <= lo:
            out[i] = np.nan
            continue
        out[i] = float(np.nanmean(inv[lo:hi]))

    return pd.Series(out, index=t1_valid.index, dtype=float).reindex(t1.index)


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
