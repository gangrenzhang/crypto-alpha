"""安全 rolling: 绕过 pandas 在长序列上的全 NaN 缺陷。

pandas 2.3.x(本机实测)对 ``len(series) >= 32768`` 的整数窗 ``rolling``
会错误地返回全 NaN(疑似窗口索引 int16 溢出)。训练用多年 30m 面板远超该阈值。
本模块在超长序列上按块计算并重叠 ``window-1``, 结果与短序列 pandas 口径一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 本机 pandas 2.3.3: n>=32768 时 rolling 全 NaN; 留 1 的安全边距
_PANDAS_ROLLING_SAFE_MAX = 32767
_CHUNK = 30000


def _rolling_op(s: pd.Series, window: int, op: str) -> pd.Series:
    window = int(window)
    if window <= 0:
        raise ValueError(f"rolling window 必须 >0, 得到 {window}")
    if len(s) == 0:
        return s.astype(float).copy()
    if len(s) <= _PANDAS_ROLLING_SAFE_MAX:
        r = s.rolling(window)
        return getattr(r, op)()

    values = s.to_numpy(dtype=float, copy=False)
    out = np.full(len(values), np.nan, dtype=float)
    overlap = window - 1
    start = 0
    n = len(values)
    while start < n:
        end = min(start + _CHUNK, n)
        left = max(0, start - overlap)
        chunk = pd.Series(values[left:end])
        part = getattr(chunk.rolling(window), op)().to_numpy()
        out[start:end] = part[start - left :]
        start = end
    return pd.Series(out, index=s.index, name=getattr(s, "name", None))


def rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return _rolling_op(s, window, "mean")


def rolling_std(s: pd.Series, window: int) -> pd.Series:
    return _rolling_op(s, window, "std")
