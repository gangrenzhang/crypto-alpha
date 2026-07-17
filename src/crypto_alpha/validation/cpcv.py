"""组合式净化交叉验证 (Combinatorial Purged Cross-Validation, CPCV)。

出处: López de Prado, AFML, ch.12。
目的: 把样本切成 N 组, 每次取 k 组作为测试集, 遍历 C(N,k) 种组合。
相比单一路径的 walk-forward, CPCV 能生成 φ = C(N,k)*k/N 条互不相同的回测路径,
从而得到夏普比率的分布, 用于计算去偏夏普(DSR)与回测过拟合概率(PBO)。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


class CombinatorialPurgedCV:
    def __init__(self, n_splits: int, n_test_groups: int, t1: pd.Series, embargo_pct: float = 0.0):
        assert n_test_groups < n_splits, "测试组数必须小于总组数"
        self.N = n_splits
        self.k = n_test_groups
        self.t1 = t1
        self.embargo_pct = embargo_pct

    @property
    def n_paths(self) -> int:
        from math import comb

        return comb(self.N, self.k) * self.k // self.N

    def _group_indices(self, n: int) -> list[np.ndarray]:
        return np.array_split(np.arange(n), self.N)

    def split(self, X: pd.DataFrame):
        """逐个产出 (train_idx, test_idx, test_group_ids)。"""
        if not X.index.equals(self.t1.index):
            raise ValueError("X 与 t1 的索引必须一致")
        n = X.shape[0]
        groups = self._group_indices(n)
        embargo = int(n * self.embargo_pct)
        times = self.t1.index

        for combo in combinations(range(self.N), self.k):
            test_idx = np.concatenate([groups[g] for g in combo])
            test_idx.sort()

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False

            # 对每个测试组分别做清洗 + 禁运
            for g in combo:
                gi = groups[g]
                t0 = times[gi[0]]
                test_end_time = self.t1.iloc[gi].max()
                overlap = (self.t1 >= t0).values & (self.t1.index <= test_end_time)
                train_mask &= ~overlap
                end = gi[-1] + 1
                # 与 PurgedKFold 一致: 禁运带 clamp 到 n, 近末组也不得整段跳过
                if embargo > 0 and end < n:
                    train_mask[end : min(end + embargo, n)] = False

            yield np.where(train_mask)[0], test_idx, combo
