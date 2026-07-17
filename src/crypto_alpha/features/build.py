"""特征矩阵装配: 原始数据 -> 技术指标 + 分数阶差分 + 多周期(MTF)特征。

产出的 DataFrame 索引为主时间框架时间戳, 列为全部特征, 保证无未来信息泄漏。
多周期: 辅周期(4h/1d 等)仅作高周期上下文, as-of 对齐进主面板(见 features/mtf.py)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .frac_diff import frac_diff_ffd
from .technical import add_technical_features
from .mtf import add_mtf_features, MTF_COL_RE


def build_feature_matrix(
    df: pd.DataFrame,
    cfg,
    symbol: str | None = None,
    aux_frames: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """构建主周期特征, 并按需无泄漏并入辅周期上下文。

    Parameters
    ----------
    df : 主周期 OHLCV(+可选衍生品)
    cfg : 全局配置
    symbol : 币种; 若提供且未显式传入 aux_frames, 将自动 load_aux_timeframes
    aux_frames : 可选预加载的 `{timeframe: OHLCV}`; 传入则可避免重复 IO
    """
    fcfg = cfg["features"]
    windows = fcfg["windows"]
    vol_window = int(fcfg["vol_window"])

    feat = add_technical_features(df, windows, vol_window)

    # 分数阶差分(对 log 价格), 平稳且保留记忆
    logprice = np.log(df["close"])
    logprice.name = "logprice"
    fd = frac_diff_ffd(logprice, d=float(fcfg["frac_diff_d"]), thres=float(fcfg["frac_diff_thres"]))
    feat[fd.name] = fd

    # --- 多周期上下文(方案B) ---
    if fcfg.get("mtf_enabled", True):
        frames = aux_frames
        if frames is None and symbol is not None:
            from ..data.fetch import load_aux_timeframes

            frames = load_aux_timeframes(cfg, symbol, main_df=df)
        if frames:
            feat = add_mtf_features(feat, frames, cfg, main_tf=cfg["data"]["timeframe"])

    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """用于建模的特征列: 排除原始价格/成交量等非平稳绝对量。"""
    exclude = {"open", "high", "low", "close", "volume", "open_interest", "funding_rate"}
    return [c for c in feat.columns if c not in exclude]


def mtf_columns(feat: pd.DataFrame) -> list[str]:
    """诊断用: 列出多周期特征列。"""
    cols = [c for c in feat.columns if MTF_COL_RE.match(c)]
    if "mtf_confluence" in feat.columns:
        cols.append("mtf_confluence")
    return cols
