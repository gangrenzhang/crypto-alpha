"""三重障碍标注法 (Triple-Barrier Method)。

出处: López de Prado, AFML, ch.3。
思想: 对每个候选入场时点, 设三条障碍:
    - 上障碍(止盈): 价格上行触及 => 盈利
    - 下障碍(止损): 价格下行触及 => 亏损
    - 垂直障碍(时间): 到期未触碰 => 按到期收益/超时处理
止盈/止损宽度由 "当时波动率 * 倍数" 决定, 因此止损天然内建于标签, 无需模型猜测。
配合 side(方向) 即构成元标签所需的二分类目标。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cusum_filter(close: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """对称 CUSUM 事件采样: 只在累计偏移超过阈值时产生事件, 降低样本冗余。

    目的: 不对每根 bar 都建仓, 而是在出现显著价格偏移时才采样, 使样本更独立。
    """
    t_events, s_pos, s_neg = [], 0.0, 0.0
    diff = np.log(close).diff().fillna(0.0)
    for t, d in diff.items():
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        if s_neg < -threshold:
            s_neg = 0.0
            t_events.append(t)
        elif s_pos > threshold:
            s_pos = 0.0
            t_events.append(t)
    return pd.DatetimeIndex(t_events)


def get_vertical_barriers(close: pd.Series, t_events: pd.DatetimeIndex, n_bars: int) -> pd.Series:
    """为每个事件设定垂直障碍(最大持有 n_bars 根 bar 后的时间戳)。"""
    idx = close.index
    loc = idx.get_indexer(t_events)
    end_loc = np.minimum(loc + n_bars, len(idx) - 1)
    t1 = pd.Series(idx[end_loc], index=t_events)
    # 若事件已在末尾无法满足持有期, 置 NaT
    t1[loc + n_bars > len(idx) - 1] = pd.NaT
    return t1


def apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float],
) -> pd.DataFrame:
    """对每个事件, 在 [t0, t1] 区间内寻找止盈/止损首次触碰时间。

    events 需含列: t1(垂直障碍), trgt(目标波动), side(方向 +1/-1)。
    """
    pt_mult, sl_mult = pt_sl
    idx = events.index
    t1_list, pt_list, sl_list = [], [], []
    log_close = np.log(close)
    for t0 in idx:
        t1 = events.at[t0, "t1"]
        if pd.isna(t1):
            t1 = close.index[-1]
        path = log_close.loc[t0:t1]
        # side 调整后的路径对数收益(做多 side=+1, 做空 side=-1)
        ret = (path - log_close.loc[t0]) * events.at[t0, "side"]
        trgt = events.at[t0, "trgt"]
        pt, sl = pt_mult * trgt, -sl_mult * trgt
        sl_touch = ret[ret < sl].index.min()  # 止损首次触碰(无则 NaT)
        pt_touch = ret[ret > pt].index.min()  # 止盈首次触碰(无则 NaT)
        t1_list.append(t1)
        pt_list.append(pt_touch)
        sl_list.append(sl_touch)
    return pd.DataFrame({"t1": t1_list, "pt": pt_list, "sl": sl_list}, index=idx)


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float],
    trgt: pd.Series,
    vertical_bars: int,
    side: pd.Series,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """组装事件表并计算首次触碰(止盈/止损/垂直)时间 t1_touch。"""
    trgt = trgt.reindex(t_events).ffill()
    t_events = trgt[trgt > min_ret].index  # 过滤目标过小的事件

    vb = get_vertical_barriers(close, t_events, vertical_bars)
    events = pd.DataFrame(
        {"t1": vb, "trgt": trgt.reindex(t_events), "side": side.reindex(t_events)}
    ).dropna(subset=["trgt", "side"])

    touches = apply_pt_sl_on_t1(close, events, pt_sl)
    events["t1_touch"] = touches[["t1", "pt", "sl"]].min(axis=1)
    return events


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """由触碰结果生成元标签。

    返回列:
        ret  : 该笔交易(含方向)的对数收益
        bin  : 元标签 {0,1} —— 是否盈利(1=该下注, 0=不该下注)
    """
    out = pd.DataFrame(index=events.index)
    t1 = events["t1_touch"].fillna(events["t1"])
    px_t0 = close.reindex(events.index)
    px_t1 = close.reindex(t1).values
    ret = (np.log(px_t1) - np.log(px_t0.values)) * events["side"].values
    out["ret"] = ret
    out["bin"] = (ret > 0).astype(int)
    out["side"] = events["side"].values
    out["t1"] = t1  # 保留 tz-aware 时间(勿用 .values, 否则丢失时区)
    return out
