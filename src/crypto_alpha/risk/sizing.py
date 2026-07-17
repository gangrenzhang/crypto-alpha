"""风控: 分数 Kelly 仓位 + 波动率(ATR)止损, 以及统一决策输出。

概率(校准后) + 盈亏比 -> Kelly 最优下注比例; 取分数 Kelly 并封顶以控制回撤。
止损/止盈由 **与三重障碍相同的 pt_sl 倍数 × ATR** 定义, 保证训练标签与实盘执行口径一致。
"""
from __future__ import annotations

import numpy as np


def kelly_fraction(p: float, payoff: float) -> float:
    """二元 Kelly: f* = (p*(b+1) - 1) / b, b=盈亏比。负值表示不下注。"""
    b = max(payoff, 1e-6)
    f = (p * (b + 1) - 1) / b
    return float(max(f, 0.0))


def position_size(p: float, payoff: float, kelly_fraction_mult: float, max_pct: float) -> float:
    f = kelly_fraction(p, payoff) * kelly_fraction_mult
    return float(min(f, max_pct))


def atr_stop(entry_price: float, atr: float, side: int, mult: float) -> float:
    """做多止损在下方, 做空止损在上方。"""
    return float(entry_price - side * mult * atr)


def atr_take_profit(entry_price: float, atr: float, side: int, mult: float) -> float:
    """做多止盈在上方, 做空止盈在下方。"""
    return float(entry_price + side * mult * atr)


def decide(
    prob: float, side: int, entry_price: float, atr: float, risk_cfg: dict,
    prob_threshold: float = 0.55, payoff: float | None = None,
    confident: bool = True, pt_sl: tuple[float, float] | list[float] | None = None,
) -> dict:
    """把概率+方向+价格+ATR 汇总为一条结构化交易决策。

    - confident: 保形预测是否高置信; False 时强制 HOLD(不确定则观望)。
    - HOLD 时不输出 stop_loss/take_profit(避免被误当作可执行挂单)。
    - pt_sl: 与 labeling.pt_sl 一致的 (止盈倍数, 止损倍数); 缺省时回退 risk.atr_stop_mult。
    """
    if pt_sl is not None:
        pt_mult, sl_mult = float(pt_sl[0]), float(pt_sl[1])
    else:
        sl_mult = float(risk_cfg.get("atr_stop_mult", 1.5))
        pt_mult = sl_mult
    if payoff is None:
        payoff = pt_mult / max(sl_mult, 1e-6)

    signal = "HOLD"
    size = 0.0
    reason = None
    if not confident:
        reason = "low_confidence_conformal"
    elif side == 0:
        reason = "no_side"
    elif prob < prob_threshold:
        reason = "prob_below_threshold"
    else:
        signal = "LONG" if side > 0 else "SHORT"
        size = position_size(
            prob, payoff, float(risk_cfg.get("kelly_fraction", 0.5)),
            float(risk_cfg.get("max_position_pct", 0.3)),
        )

    out = {
        "signal": signal,
        "win_probability": round(float(prob), 4),
        "entry_price": round(float(entry_price), 2),
        "suggested_position_pct": round(size, 4),
        "atr": round(float(atr), 4),
        "confident": bool(confident),
        "pt_mult": pt_mult,
        "sl_mult": sl_mult,
    }
    if signal == "HOLD":
        out["stop_loss"] = None
        out["take_profit"] = None
        out["reason"] = reason
    else:
        out["stop_loss"] = round(atr_stop(entry_price, atr, side, sl_mult), 2)
        out["take_profit"] = round(atr_take_profit(entry_price, atr, side, pt_mult), 2)
    return out
