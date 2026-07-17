"""专家模型统一接口。

所有专家(GBDT / 深度时序 / 时序基础模型 / LLM)都实现同一套接口, 便于 Stacking
与 CPCV 统一调度。约定:
    - X: 以事件时间戳为索引的特征 DataFrame(rows=事件, cols=特征)。
    - 需要原始时序面板的专家(深度时序/TSFM)通过 set_panel 获取完整特征面板,
      再用 X.index 切出对应的历史窗口。
    - predict_proba 返回正类(该下注/盈利) 概率, 形状 (n_samples,)。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseExpert(ABC):
    name: str = "base"
    #: 是否需要完整时序面板(而非仅事件行)
    needs_panel: bool = False
    #: True: fit 不按折重训(如只加载固定权重); Stacking 默认将其排除出元学习器
    pseudo_oof: bool = False

    def __init__(self, cfg: dict, feature_cols: list[str], seed: int = 42):
        self.cfg = cfg
        self.feature_cols = feature_cols
        self.seed = seed
        self._panel: pd.DataFrame | None = None

    def set_panel(self, panel: pd.DataFrame) -> None:
        """提供完整特征面板(索引=每根 bar 的时间戳)。"""
        self._panel = panel

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None,
        **fit_params,
    ) -> "BaseExpert":
        """拟合专家。可选 ``fit_params``(如 DeepTS 的 ``es_cutoff_time``)由调用方传入; 未知键应忽略。"""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def clone(self) -> "BaseExpert":
        """返回未训练的同配置实例(用于 CV 每折重新训练)。"""
        obj = self.__class__(self.cfg, self.feature_cols, self.seed)
        obj._panel = self._panel
        return obj
