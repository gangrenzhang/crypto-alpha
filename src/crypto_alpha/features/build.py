"""特征矩阵装配: 原始数据 -> 技术指标 + 分数阶差分 + 多周期特征。

产出的 DataFrame 索引为主时间框架时间戳, 列为全部特征, 保证无未来信息泄漏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .frac_diff import frac_diff_ffd
from .technical import add_technical_features


def build_feature_matrix(df: pd.DataFrame, cfg) -> pd.DataFrame:
    fcfg = cfg["features"]
    windows = fcfg["windows"]
    vol_window = int(fcfg["vol_window"])

    feat = add_technical_features(df, windows, vol_window)

    # 分数阶差分(对 log 价格), 平稳且保留记忆
    logprice = np.log(df["close"])
    logprice.name = "logprice"
    fd = frac_diff_ffd(logprice, d=float(fcfg["frac_diff_d"]), thres=float(fcfg["frac_diff_thres"]))
    feat[fd.name] = fd

    # 保留 OHLC 原始列供标注/回测使用, 但特征列不用绝对价格(非平稳)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """用于建模的特征列: 排除原始价格/成交量等非平稳绝对量。"""
    exclude = {"open", "high", "low", "close", "volume", "open_interest"}
    return [c for c in feat.columns if c not in exclude]
