"""分数阶差分 (Fractional Differentiation), 固定宽度窗口法 (FFD)。

出处: Marcos López de Prado, Advances in Financial Machine Learning, ch.5。
目的: 价格序列是非平稳的(整数阶差分虽平稳但抹掉记忆)。分数阶差分在
"保持平稳" 与 "保留长记忆" 之间取得平衡, 让特征既能被模型使用又不失预测力。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_weights_ffd(d: float, thres: float, max_size: int = 10_000) -> np.ndarray:
    """计算固定宽度窗口的权重, 当权重绝对值小于 thres 时截断。"""
    w = [1.0]
    k = 1
    while k < max_size:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """对单列做 FFD 分数阶差分。返回与输入对齐(前部因窗口不足为 NaN)的序列。"""
    w = get_weights_ffd(d, thres).flatten()
    width = len(w) - 1
    vals = series.values.astype(float)
    out = np.full(len(vals), np.nan)
    for i in range(width, len(vals)):
        window = vals[i - width : i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(np.dot(w, window))
    return pd.Series(out, index=series.index, name=f"{series.name}_fd{d}")
