"""Purged K-Fold + Embargo。

出处: López de Prado, AFML, ch.7。
目的: 金融样本标签在时间上重叠且强自相关。普通 KFold 会把与测试集重叠的
样本留在训练集里, 造成信息泄漏、虚高分数。Purged 清除重叠样本, Embargo 在
测试集之后再禁用一小段样本, 彻底切断泄漏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_embargo_times(times: pd.DatetimeIndex, pct: float) -> pd.Series:
    """为每个样本时间返回其禁运截止时间。"""
    step = int(times.shape[0] * pct)
    if step == 0:
        return pd.Series(times, index=times)
    ahead = pd.Series(times[step:], index=times[: -step])
    tail = pd.Series([times[-1]] * step, index=times[-step:])
    return pd.concat([ahead, tail])


class PurgedKFold:
    """带清洗与禁运的 KFold。samples_info_sets 为每个样本的标签结束时间 t1。"""

    def __init__(self, n_splits: int, t1: pd.Series, embargo_pct: float = 0.0):
        self.n_splits = n_splits
        self.t1 = t1  # index=样本开始时间, value=标签结束时间
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame):
        if not X.index.equals(self.t1.index):
            raise ValueError("X 与 t1 的索引必须一致")
        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.embargo_pct)
        test_ranges = [
            (i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)
        ]
        times = self.t1.index

        for start, end in test_ranges:
            test_idx = indices[start:end]
            t0 = times[start]  # 测试段起始时间
            test_end_time = self.t1.iloc[test_idx].max()  # 测试段标签最晚结束时间

            train_mask = np.ones(X.shape[0], dtype=bool)
            train_mask[test_idx] = False

            # 清洗: 训练样本若其标签区间 [t_start, t1] 与测试段区间重叠, 剔除
            train_t1 = self.t1
            overlap = (train_t1 >= t0).values & (train_t1.index <= test_end_time)
            train_mask &= ~overlap

            # 禁运: 测试段之后 embargo 根样本也剔除
            if embargo > 0 and end + embargo <= X.shape[0]:
                train_mask[end : end + embargo] = False

            yield indices[train_mask], test_idx
