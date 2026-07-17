"""特征矩阵装配: 原始数据 -> 技术指标 + 分数阶差分 + 多周期特征。

产出的 DataFrame 索引为主时间框架时间戳, 列为全部特征, 保证无未来信息泄漏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .frac_diff import frac_diff_ffd
from .technical import add_technical_features

_PANDAS_RULE = {"1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1D", "1w": "1W"}


def add_multitf_features(df: pd.DataFrame, aux_timeframes: list[str]) -> pd.DataFrame:
    """由主时间框架 OHLCV **因果地**重采样出更高周期特征(4h/1d 等)。

    纪律: 用 label='right'/closed='right' 使每根高周期 bar 以其**收盘时刻**为索引,
    再以 as-of(ffill) 对齐回主 bar —— 主 bar t 只会看到收盘时刻 <= t 的高周期 bar,
    绝不使用"仍在形成中"的高周期 bar, 故无未来泄漏。
    """
    out = pd.DataFrame(index=df.index)
    agg_map = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for tf in aux_timeframes or []:
        rule = _PANDAS_RULE.get(tf, tf)
        try:
            agg = df.resample(rule, label="right", closed="right").agg(agg_map).dropna(subset=["close"])
        except Exception:
            continue
        if len(agg) < 15:
            continue
        c = agg["close"]
        ret = np.log(c / c.shift(1))
        vol = ret.rolling(14).std()
        ma = c.rolling(10).mean()
        trend = np.sign(c - ma)
        feats = pd.DataFrame(
            {f"aux_{tf}_ret": ret, f"aux_{tf}_vol": vol, f"aux_{tf}_trend": trend}
        )
        # as-of 对齐到主 bar: 只取收盘 <= 当前主 bar 时间的高周期 bar
        aligned = feats.reindex(feats.index.union(df.index)).ffill().reindex(df.index)
        out = out.join(aligned)
    return out


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

    # 多周期(aux)因果特征: 4h/1d 等由主 bar 重采样, as-of 对齐, 无泄漏
    aux_tfs = cfg["data"].get("aux_timeframes", [])
    if aux_tfs:
        mtf = add_multitf_features(df, aux_tfs)
        for col in mtf.columns:
            feat[col] = mtf[col]

    # 保留 OHLC 原始列供标注/回测使用, 但特征列不用绝对价格(非平稳)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """用于建模的特征列: 排除原始价格/成交量等非平稳绝对量。"""
    exclude = {"open", "high", "low", "close", "volume", "open_interest"}
    return [c for c in feat.columns if c not in exclude]
