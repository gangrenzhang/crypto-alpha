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
入场价取事件 bar(t0)的收盘价, 触碰扫描从 t0 的**下一根** bar 开始: t0 自身的盘中极值
发生在入场之前, 入场后已不可成交, 计入会用"入场前价格"误判触碰、使标签偏乐观。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 冷启动先验: 扩展中位数未就绪时使用的固定相对波动阈值(约 0.5%), 不含全样本信息。
_CUSUM_PRIOR: float = 0.005


def cusum_filter(
    close: pd.Series, threshold: float | pd.Series,
) -> pd.DatetimeIndex:
    """对称 CUSUM 事件采样: 只在累计偏移超过阈值时产生事件, 降低样本冗余。

    threshold 可为标量, 或与 close 对齐的**因果**阈值序列(逐 bar 取值)。
    Series 路径仅 ffill + 固定先验填补, **禁止 bfill / 全样本 nanmedian**(防前视)。
    """
    t_events, s_pos, s_neg = [], 0.0, 0.0
    diff = np.log(close).diff().fillna(0.0)
    if isinstance(threshold, pd.Series):
        thr_s = threshold.reindex(close.index).ffill().fillna(_CUSUM_PRIOR)
    else:
        thr_s = None
        thr_scalar = float(threshold) if threshold and np.isfinite(threshold) else _CUSUM_PRIOR
    for t, d in diff.items():
        thr = float(thr_s.loc[t]) if thr_s is not None else thr_scalar
        if not np.isfinite(thr) or thr <= 0:
            thr = _CUSUM_PRIOR
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        if s_neg < -thr:
            s_neg = 0.0
            t_events.append(t)
        elif s_pos > thr:
            s_pos = 0.0
            t_events.append(t)
    return pd.DatetimeIndex(t_events)


def causal_cusum_threshold(
    trgt: pd.Series, min_periods: int = 50, prior: float = _CUSUM_PRIOR,
) -> pd.Series:
    """CUSUM 阈值: 目标波动的**扩展中位数**(仅用当时及之前信息)。

    冷启动: 仅 ffill 已算出的扩展值; 仍为 NaN 时用固定 ``prior``(默认 0.5%),
    **不用**全样本 nanmedian / bfill。
    """
    mp = max(int(min_periods), 5)
    thr = trgt.expanding(min_periods=mp).median()
    return thr.ffill().fillna(float(prior))


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
    """对每个事件, 在 (t0, t1] 用 bar 内 high/low 寻找止盈/止损首次触碰时间。

    价格障碍与 ``decide`` / ``atr_stop`` / ``atr_take_profit`` **同一加性公式**
    (``trgt = atr/close`` 时 atr_abs ≈ trgt×entry)::

        take_profit = entry + side × pt_mult × atr_abs
        stop_loss   = entry - side × sl_mult × atr_abs

    多头: TP=entry(1+pt·trgt), SL=entry(1-sl·trgt);
    空头: TP=entry(1-pt·trgt), SL=entry(1+sl·trgt)。
    不再对空头使用对数空间对称翻转(那会得到几何价 entry/(1±x), 与挂单不一致)。

    events 需含列: t1(垂直障碍), trgt(目标波动), side(方向 +1/-1)。
    返回列: t1(垂直), pt(止盈触碰时间), sl(止损触碰时间)。
    """
    pt_mult, sl_mult = pt_sl
    idx = events.index
    t1_list, pt_list, sl_list = [], [], []
    for t0 in idx:
        t1 = events.at[t0, "t1"]
        if pd.isna(t1):
            # 调用方应已 drop 不完整垂直障碍; 此处防御性跳过(记空触碰)
            t1_list.append(pd.NaT)
            pt_list.append(pd.NaT)
            sl_list.append(pd.NaT)
            continue
        side = float(events.at[t0, "side"])
        trgt = float(events.at[t0, "trgt"])
        entry_px = float(close.loc[t0])
        atr_abs = max(trgt, 0.0) * entry_px
        tp_price = entry_px + side * pt_mult * atr_abs
        sl_price = entry_px - side * sl_mult * atr_abs
        # 入场价 = t0 收盘; t0 盘中极值发生在入场前 → 从下一根扫描
        path_high = high.loc[t0:t1].iloc[1:]
        path_low = low.loc[t0:t1].iloc[1:]
        if side > 0:
            pt_touch = path_high[path_high >= tp_price].index.min()
            sl_touch = path_low[path_low <= sl_price].index.min()
        else:
            # 空头: 价格下跌触 TP, 上涨触 SL
            pt_touch = path_low[path_low <= tp_price].index.min()
            sl_touch = path_high[path_high >= sl_price].index.min()
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
    ).dropna(subset=["trgt", "side", "t1"])  # 丢弃无法满足垂直持有期的截断样本

    if len(events) == 0:
        return events

    touches = apply_pt_sl_on_t1(close, high, low, events, pt_sl)
    events["pt_touch"] = touches["pt"]
    events["sl_touch"] = touches["sl"]
    # 首次触碰时间(垂直/止盈/止损三者最早); 显式跳过 NaT, 避免 object/float 混型 min 报错
    t1_touch = []
    for i in range(len(touches)):
        cands = [touches["t1"].iloc[i]]
        for col in ("pt", "sl"):
            v = touches[col].iloc[i]
            if pd.notna(v):
                cands.append(v)
        t1_touch.append(min(cands))
    events["t1_touch"] = t1_touch
    return events


def _barrier_log_returns(
    side: float, pt_mult: float, sl_mult: float, trgt: float,
) -> tuple[float, float]:
    """触碰加性价格障碍时的持仓对数收益, 与 decide 挂单价一致。

    多头 TP/SL: log(1±mult·trgt);
    空头 TP(价跌至 entry(1-pt·trgt)): -log(1-pt·trgt);
    空头 SL(价涨至 entry(1+sl·trgt)): -log(1+sl·trgt)。
    """
    pt_frac = max(pt_mult * trgt, 0.0)
    sl_frac = min(max(sl_mult * trgt, 0.0), 1.0 - 1e-6)
    if side > 0:
        return float(np.log1p(pt_frac)), float(np.log1p(-sl_frac))
    # 空头: 盈利时价格下跌, 亏损时价格上涨
    pt_down = min(pt_frac, 1.0 - 1e-6)
    return float(-np.log1p(-pt_down)), float(-np.log1p(sl_frac))


def get_bins(events: pd.DataFrame, close: pd.Series, pt_sl: tuple[float, float]) -> pd.DataFrame:
    """由触碰结果生成元标签。

    先判定哪条障碍最先被触及(止盈/止损/垂直), 同 bar 平局判止损(悲观):
      - 止盈先到 => ret = 持仓对数收益(加性障碍), bin=1
      - 止损先到 => ret = 持仓对数收益(加性障碍), bin=0
      - 垂直到期 => ret = side*log(close[t1]/close[t0]), bin=(ret>0)

    返回列: ret / bin / side / t1(实际了结时间) / bars_held。
    """
    pt_mult, sl_mult = pt_sl
    idx_pos = {ts: i for i, ts in enumerate(close.index)}
    log_close = np.log(close)

    rets, bins, t1_real, bars_held, kept = [], [], [], [], []
    for t0 in events.index:
        vt = events.at[t0, "t1"]
        if pd.isna(vt):
            continue  # 不完整垂直障碍已在 get_events 丢弃; 防御性跳过
        pt_t = events.at[t0, "pt_touch"]
        sl_t = events.at[t0, "sl_touch"]
        side = float(events.at[t0, "side"])
        trgt = float(events.at[t0, "trgt"])
        ret_pt, ret_sl = _barrier_log_returns(side, pt_mult, sl_mult, trgt)

        # 未触碰用显式分支, 避免依赖 Timestamp.max 哨兵
        has_sl = pd.notna(sl_t)
        has_pt = pd.notna(pt_t)
        if has_sl and (not has_pt or sl_t <= pt_t) and sl_t <= vt:
            end, ret, b = sl_t, ret_sl, 0
        elif has_pt and pt_t <= vt:
            end, ret, b = pt_t, ret_pt, 1
        else:  # 垂直到期
            end = vt
            ret = float((log_close.loc[end] - log_close.loc[t0]) * side)
            b = int(ret > 0)

        kept.append(t0)
        rets.append(float(ret))
        bins.append(int(b))
        t1_real.append(end)
        bars_held.append(int(idx_pos.get(end, idx_pos[vt]) - idx_pos[t0]))

    out = pd.DataFrame(index=pd.DatetimeIndex(kept))
    out["ret"] = rets
    out["bin"] = bins
    out["side"] = events.loc[kept, "side"].values
    out["t1"] = t1_real  # 实际了结时间(tz-aware)
    out["bars_held"] = np.maximum(bars_held, 1)
    return out
