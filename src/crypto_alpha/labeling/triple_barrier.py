"""三重障碍标注法 (Triple-Barrier Method)。

出处: López de Prado, AFML, ch.3。
思想: 对每个候选入场时点, 设三条障碍:
    - 上障碍(止盈): 价格上行触及 => 盈利
    - 下障碍(止损): 价格下行触及 => 亏损
    - 垂直障碍(时间): 到期未触碰 => 按到期收益/超时处理
止盈/止损宽度由 "当时波动率 * 倍数" 决定, 因此止损天然内建于标签, 无需模型猜测。
配合 side(方向) 即构成元标签所需的二分类目标。

障碍触碰用 **bar 内 high/low** 判定(而非仅收盘路径), 与实盘"止损/止盈常在 bar 内被
最高/最低价先打到"的真实执行一致, 避免系统性高估胜率、错估持有期。
同一根 bar 内若止盈止损同时被触及, 无法确定盘中先后, 保守判为止损(悲观)。
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
    high: pd.Series,
    low: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float],
) -> pd.DataFrame:
    """对每个事件, 在 [t0, t1] 区间内用 bar 内 high/low 寻找止盈/止损首次触碰时间。

    events 需含列: t1(垂直障碍), trgt(目标波动), side(方向 +1/-1)。
    返回列: t1(垂直), pt(止盈触碰时间), sl(止损触碰时间)。
    """
    pt_mult, sl_mult = pt_sl
    idx = events.index
    t1_list, pt_list, sl_list = [], [], []
    log_close = np.log(close)
    log_high = np.log(high)
    log_low = np.log(low)
    for t0 in idx:
        t1 = events.at[t0, "t1"]
        if pd.isna(t1):
            t1 = close.index[-1]
        side = events.at[t0, "side"]
        trgt = events.at[t0, "trgt"]
        pt, sl = pt_mult * trgt, -sl_mult * trgt
        entry = log_close.loc[t0]
        # 顺方向的有利极值用 high(多)/low(空); 逆方向的不利极值用 low(多)/high(空)
        if side > 0:
            fav = log_high.loc[t0:t1] - entry
            adv = log_low.loc[t0:t1] - entry
        else:
            fav = -(log_low.loc[t0:t1] - entry)
            adv = -(log_high.loc[t0:t1] - entry)
        pt_touch = fav[fav >= pt].index.min()   # 止盈首次触碰(无则 NaT)
        sl_touch = adv[adv <= sl].index.min()    # 止损首次触碰(无则 NaT)
        t1_list.append(t1)
        pt_list.append(pt_touch)
        sl_list.append(sl_touch)
    return pd.DataFrame({"t1": t1_list, "pt": pt_list, "sl": sl_list}, index=idx)


def get_events(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float],
    trgt: pd.Series,
    vertical_bars: int,
    side: pd.Series,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """组装事件表并计算首次触碰(止盈/止损/垂直)时间。"""
    trgt = trgt.reindex(t_events).ffill()
    t_events = trgt[trgt > min_ret].index  # 过滤目标过小的事件

    vb = get_vertical_barriers(close, t_events, vertical_bars)
    events = pd.DataFrame(
        {"t1": vb, "trgt": trgt.reindex(t_events), "side": side.reindex(t_events)}
    ).dropna(subset=["trgt", "side"])

    touches = apply_pt_sl_on_t1(close, high, low, events, pt_sl)
    events["pt_touch"] = touches["pt"]
    events["sl_touch"] = touches["sl"]
    # 首次触碰时间(垂直/止盈/止损三者最早)
    events["t1_touch"] = touches[["t1", "pt", "sl"]].min(axis=1)
    return events


def get_bins(events: pd.DataFrame, close: pd.Series, pt_sl: tuple[float, float]) -> pd.DataFrame:
    """由触碰结果生成元标签。

    先判定哪条障碍最先被触及(止盈/止损/垂直), 同 bar 平局判止损(悲观):
      - 止盈先到 => ret = +pt_mult*trgt, bin=1
      - 止损先到 => ret = -sl_mult*trgt, bin=0
      - 垂直到期 => ret = side*log(close[t1]/close[t0]), bin=(ret>0)

    返回列: ret / bin / side / t1(实际了结时间) / bars_held。
    """
    pt_mult, sl_mult = pt_sl
    idx_pos = {ts: i for i, ts in enumerate(close.index)}
    log_close = np.log(close)

    rets, bins, t1_real, bars_held = [], [], [], []
    for t0 in events.index:
        vt = events.at[t0, "t1"]
        if pd.isna(vt):
            vt = close.index[-1]
        pt_t = events.at[t0, "pt_touch"]
        sl_t = events.at[t0, "sl_touch"]
        side = events.at[t0, "side"]
        trgt = events.at[t0, "trgt"]

        # 各障碍触碰时间(缺失记为极大)
        big = pd.Timestamp.max.tz_localize("UTC") if getattr(close.index, "tz", None) is not None else pd.Timestamp.max
        pt_time = pt_t if pd.notna(pt_t) else big
        sl_time = sl_t if pd.notna(sl_t) else big

        # 谁最先: 平局(含同 bar)判止损
        if sl_time <= pt_time and sl_time <= vt:
            end = sl_time
            ret = -sl_mult * trgt
            b = 0
        elif pt_time < sl_time and pt_time <= vt:
            end = pt_time
            ret = pt_mult * trgt
            b = 1
        else:  # 垂直到期
            end = vt
            ret = float((log_close.loc[end] - log_close.loc[t0]) * side)
            b = int(ret > 0)

        rets.append(float(ret))
        bins.append(int(b))
        t1_real.append(end)
        bars_held.append(int(idx_pos.get(end, idx_pos[vt]) - idx_pos[t0]))

    out = pd.DataFrame(index=events.index)
    out["ret"] = rets
    out["bin"] = bins
    out["side"] = events["side"].values
    out["t1"] = t1_real  # 实际了结时间(tz-aware)
    out["bars_held"] = np.maximum(bars_held, 1)
    return out
