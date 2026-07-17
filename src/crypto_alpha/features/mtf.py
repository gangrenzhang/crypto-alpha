"""多周期(MTF)特征: 高周期上下文 as-of 对齐到主周期。

方案B(主周期建模 + 辅周期特征):
- 主周期(默认 1h)负责事件采样 / 标注 / 训练索引;
- 辅周期(如 4h / 1d)只提供**已收盘**K线算出的趋势/波动上下文;
- 用 merge_asof(backward) 对齐, 严格无前视。

防泄漏铁律(与交易所惯例一致: K线时间戳 = 开盘时刻):
- 辅周期 bar 开盘时刻 = u、周期长度 = Δ_aux  ⇒ 可用时刻 available_at = u + Δ_aux
  (该根 K 线的 OHLCV 在收盘后才完整可知);
- 主周期 bar 开盘时刻 = t、周期长度 = Δ_main ⇒ 决策时刻 decision_at = t + Δ_main
  (与主面板「索引 t 的特征已含该 bar 的 close」口径一致);
- 对齐条件: available_at ≤ decision_at, 取不超过决策时刻的最新已收盘辅周期特征。

这样在 09:00 的 1h bar(决策时刻 10:00)绝看不到 08:00–12:00 那根未收盘的 4h 特征。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..data.fetch import timeframe_delta, timeframe_to_prefix
from .technical import _rsi, atr


# 并入主面板后的多周期特征列名前缀匹配(供测试/诊断)
MTF_COL_RE = re.compile(r"^tf[0-9]+[a-z]+_")


def _mtf_cfg(cfg) -> dict:
    fcfg = cfg["features"]
    return {
        "enabled": bool(fcfg.get("mtf_enabled", True)),
        "lookbacks": list(fcfg.get("mtf_lookbacks", [1, 3, 7])),
        "rsi_window": int(fcfg.get("mtf_rsi_window", 14)),
        "vol_window": int(fcfg.get("mtf_vol_window", 14)),
        "include_confluence": bool(fcfg.get("mtf_include_confluence", True)),
    }


def build_higher_tf_features(aux_df: pd.DataFrame, timeframe: str, cfg) -> pd.DataFrame:
    """在辅周期 OHLCV 上计算紧凑特征集; 索引仍为开盘时刻。

    刻意控制列数, 避免特征爆炸: 收益回看 + RSI/波动/zscore + MACD柱 + 归一化ATR + 趋势符号。
    """
    mcfg = _mtf_cfg(cfg)
    prefix = timeframe_to_prefix(timeframe)
    close = aux_df["close"].astype(float)
    logret = np.log(close).diff()

    out = pd.DataFrame(index=aux_df.index)
    for w in mcfg["lookbacks"]:
        w = int(w)
        out[f"{prefix}_ret_{w}"] = close.pct_change(w)
        out[f"{prefix}_mom_{w}"] = close / close.shift(w) - 1.0

    rw = mcfg["rsi_window"]
    vw = mcfg["vol_window"]
    out[f"{prefix}_rsi_{rw}"] = _rsi(close, rw)
    out[f"{prefix}_vol_{vw}"] = logret.rolling(vw).std()
    ma = close.rolling(vw).mean()
    std = close.rolling(vw).std()
    out[f"{prefix}_zscore_{vw}"] = (close - ma) / (std + 1e-12)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out[f"{prefix}_macd_hist"] = macd - signal

    atr_s = atr(aux_df, 14)
    out[f"{prefix}_atr_norm"] = atr_s / (close + 1e-12)
    # 趋势符号: 近 3 根辅周期收益的方向(0 视为中性 0)
    trend_lb = 3 if 3 in mcfg["lookbacks"] else mcfg["lookbacks"][min(1, len(mcfg["lookbacks"]) - 1)]
    out[f"{prefix}_trend"] = np.sign(close.pct_change(int(trend_lb))).replace(0.0, 0.0)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _align_one_tf(
    main_index: pd.DatetimeIndex,
    aux_feat: pd.DataFrame,
    main_tf: str,
    aux_tf: str,
) -> pd.DataFrame:
    """把一组辅周期特征 as-of 对齐到主周期索引(无泄漏)。"""
    main_delta = timeframe_delta(main_tf)
    aux_delta = timeframe_delta(aux_tf)

    main_idx = pd.DatetimeIndex(pd.to_datetime(main_index, utc=True))
    if main_idx.tz is None:
        main_idx = main_idx.tz_localize("UTC")

    # 决策时刻 = 主 bar 收盘; 辅特征可用时刻 = 辅 bar 收盘
    left = pd.DataFrame({
        "decision_at": (main_idx + main_delta).astype("datetime64[ns, UTC]"),
        "main_ts": main_idx,
    }).sort_values("decision_at")

    right = aux_feat.copy()
    right_idx = pd.DatetimeIndex(pd.to_datetime(right.index, utc=True))
    if right_idx.tz is None:
        right_idx = right_idx.tz_localize("UTC")
    right = right.copy()
    right["available_at"] = (right_idx + aux_delta).astype("datetime64[ns, UTC]")
    right = right.reset_index(drop=True).sort_values("available_at")

    feat_cols = [c for c in right.columns if c != "available_at"]
    merged = pd.merge_asof(
        left,
        right,
        left_on="decision_at",
        right_on="available_at",
        direction="backward",
    )
    aligned = merged.set_index("main_ts")[feat_cols]
    # 与传入 main_index 顺序对齐
    return aligned.reindex(main_idx)


def add_mtf_features(
    main_feat: pd.DataFrame,
    aux_frames: dict[str, pd.DataFrame],
    cfg,
    main_tf: str | None = None,
) -> pd.DataFrame:
    """将辅周期特征无泄漏并入主特征面板。

    Parameters
    ----------
    main_feat : 主周期特征面板(索引=主 K 线开盘时刻)
    aux_frames : {timeframe: OHLCV DataFrame}
    cfg : 全局配置
    main_tf : 主周期字符串; 默认读 config.data.timeframe
    """
    mcfg = _mtf_cfg(cfg)
    if not mcfg["enabled"] or not aux_frames:
        return main_feat

    if main_tf is None:
        main_tf = cfg["data"]["timeframe"]

    out = main_feat.copy()
    aligned_trend_cols: list[str] = []

    for tf, raw in aux_frames.items():
        if raw is None or len(raw) == 0:
            continue
        if tf == main_tf:
            continue
        # 辅周期不得细于主周期(方案B: 高周期上下文); 细周期应另作主周期
        if timeframe_delta(tf) < timeframe_delta(main_tf):
            print(f"[warn] 跳过辅周期 {tf}: 细于主周期 {main_tf}, 方案B只用更高周期上下文。")
            continue

        aux_feat = build_higher_tf_features(raw, tf, cfg)
        aligned = _align_one_tf(out.index, aux_feat, main_tf, tf)
        for c in aligned.columns:
            out[c] = aligned[c].values
        trend_c = f"{timeframe_to_prefix(tf)}_trend"
        if trend_c in out.columns:
            aligned_trend_cols.append(trend_c)

    if mcfg["include_confluence"] and aligned_trend_cols:
        # 与主信号同口径的动量方向做共振(缺主列则跳过)
        main_side = None
        for cand in ("mom_24", "mom_28", "mom_14", "ret_24", "ret_14"):
            if cand in out.columns:
                main_side = np.sign(out[cand]).replace(0.0, 0.0)
                break
        if main_side is not None:
            votes = []
            for tc in aligned_trend_cols:
                agree = (main_side * out[tc].fillna(0.0) > 0).astype(float)
                # 任一方为 0(中性) → 记 0, 不强制同意/反对
                neutral = (main_side == 0) | (out[tc].fillna(0.0) == 0)
                agree = agree.where(~neutral, 0.0)
                name = tc.replace("_trend", "_agree")
                out[name] = agree
                votes.append(agree)
            if votes:
                out["mtf_confluence"] = sum(votes) / len(votes)

    # 对齐初期无高周期历史 → 填 0/中性, 与新闻特征策略一致, 不丢主样本
    mtf_cols = [c for c in out.columns if MTF_COL_RE.match(c) or c == "mtf_confluence"]
    for c in mtf_cols:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def mtf_feature_columns(feat: pd.DataFrame) -> list[str]:
    """返回面板中的多周期特征列名。"""
    cols = [c for c in feat.columns if MTF_COL_RE.match(c)]
    if "mtf_confluence" in feat.columns:
        cols.append("mtf_confluence")
    return cols
