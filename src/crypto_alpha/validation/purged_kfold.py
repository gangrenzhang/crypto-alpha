"""Purged K-Fold + Embargo。

出处: López de Prado, AFML, ch.7。
目的: 金融样本标签在时间上重叠且强自相关。普通 KFold 会把与测试集重叠的
样本留在训练集里, 造成信息泄漏、虚高分数。Purged 清除重叠样本, Embargo 在
测试集之后再禁用一小段样本, 彻底切断泄漏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def resolve_embargo_size(n: int, pct: float) -> int:
    """禁运样本数 = ``int(n*pct)``, 但 ``pct>0`` 时至少 1 根。

    ``int()`` 截断会让小样本(如 n=80、pct=0.01)静默得到 0 —— 配置写了禁运却完全没有
    禁运, 恰好发生在最容易过拟合的小样本上。向上保底 1 根既不改大样本行为
    (n*pct≥1 时结果不变), 也不会让禁运在小样本上无声失效。
    """
    if pct is None or float(pct) <= 0 or int(n) <= 0:
        return 0
    return max(int(int(n) * float(pct)), 1)


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
        # 允许仅分辨率不同(ms vs ns)但时刻相同的索引; 统一到 UTC ns 再比
        def _utc_ns(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
            out = pd.DatetimeIndex(idx)
            if out.tz is None:
                out = out.tz_localize("UTC")
            else:
                out = out.tz_convert("UTC")
            try:
                return out.as_unit("ns")
            except Exception:
                return pd.DatetimeIndex(out.asi8, tz="UTC")

        x_idx = _utc_ns(X.index)
        t_idx = _utc_ns(self.t1.index)
        if not x_idx.equals(t_idx):
            raise ValueError("X 与 t1 的索引必须一致")
        # 后续用对齐后的轴, 避免单位混用
        if not X.index.equals(x_idx) or not self.t1.index.equals(t_idx):
            X = X.copy()
            X.index = x_idx
            self.t1 = self.t1.copy()
            self.t1.index = t_idx
        indices = np.arange(X.shape[0])
        embargo = resolve_embargo_size(X.shape[0], self.embargo_pct)
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

            # 禁运: 从测试段标签最晚结束 max(t1) **之后**起算(AFML), 而非折边界下标。
            # 取随后最多 embargo 根样本; 不足则 clamp(近末折不得整段跳过)。
            if embargo > 0:
                after = np.where(times > test_end_time)[0]
                if len(after):
                    train_mask[after[:embargo]] = False

            yield indices[train_mask], test_idx
