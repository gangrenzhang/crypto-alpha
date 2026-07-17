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

    # 衍生品不可用时特征已填 0; 记 degradations 供看板/审计(不阻断主流程)
    degradations: list[str] = list(getattr(feat, "attrs", {}).get("degradations") or [])
    if "funding_rate" in df.columns and df["funding_rate"].isna().all():
        tag = "derivatives_funding_unavailable"
        if tag not in degradations:
            degradations.append(tag)
    if "open_interest" in df.columns and df["open_interest"].isna().all():
        tag = "derivatives_oi_unavailable"
        if tag not in degradations:
            degradations.append(tag)

    # 分数阶差分(对 log 价格), 平稳且保留记忆
    logprice = np.log(df["close"])
    logprice.name = "logprice"
    fd = frac_diff_ffd(logprice, d=float(fcfg["frac_diff_d"]), thres=float(fcfg["frac_diff_thres"]))
    feat[fd.name] = fd

    # --- 多周期上下文(方案B): 独立辅周期 OHLCV → as-of 对齐, 无前视 ---
    if fcfg.get("mtf_enabled", True):
        frames = aux_frames
        if frames is None and symbol is not None:
            from ..data.fetch import load_aux_timeframes

            frames = load_aux_timeframes(cfg, symbol, main_df=df)
        if frames:
            feat = add_mtf_features(feat, frames, cfg, main_tf=cfg["data"]["timeframe"])

    feat = feat.replace([np.inf, -np.inf], np.nan)
    if degradations:
        feat.attrs["degradations"] = degradations
    return feat


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """用于建模的特征列: 排除原始价格/成交量等非平稳绝对量。

    额外排除 atr_14: 它是**绝对**价格量纲(供标注/decide 计算止损距离用), 直接入模会随
    价格量级漂移而非平稳; 建模用其相对版本 atr_norm(见 add_technical_features)。

    ``side``(+1/-1) 若已写入面板则**保留**: 元标签需要显式方向, 供 GBDT/DeepTS 等共享
    (由 pipeline.prepare_dataset 注入, 与 labels.side 对齐)。
    """
    exclude = {
        "open", "high", "low", "close", "volume", "open_interest", "funding_rate",
        "atr_14",  # 绝对 ATR, 仅供标注/风控; 建模用 atr_norm
    }
    return [c for c in feat.columns if c not in exclude]


def mtf_columns(feat: pd.DataFrame) -> list[str]:
    """诊断用: 列出多周期特征列。"""
    cols = [c for c in feat.columns if MTF_COL_RE.match(c)]
    if "mtf_confluence" in feat.columns:
        cols.append("mtf_confluence")
    return cols
