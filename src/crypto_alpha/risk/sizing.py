"""风控: 分数 Kelly 仓位 + 波动率(ATR)止损, 以及统一决策输出。

概率(校准后) + 盈亏比 -> Kelly 最优下注比例; 取分数 Kelly 并封顶以控制回撤。
止损由 ATR 定义, 与三重障碍标注口径一致。
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


def decide(
    prob: float, side: int, entry_price: float, atr: float, risk_cfg: dict,
    prob_threshold: float = 0.55, payoff: float | None = None,
    confident: bool = True,
) -> dict:
    """把概率+方向+价格+ATR 汇总为一条结构化交易决策。

    - confident: 保形预测是否高置信; False 时强制 HOLD(不确定则观望)。
    - HOLD 时不输出 stop_loss/take_profit(避免被误当作可执行挂单)。
    """
    if payoff is None:
        payoff = risk_cfg.get("pt_sl_ratio", 1.0)
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
    }
    if signal == "HOLD":
        out["stop_loss"] = None
        out["take_profit"] = None
        out["reason"] = reason
    else:
        mult = float(risk_cfg.get("atr_stop_mult", 1.5))
        out["stop_loss"] = round(atr_stop(entry_price, atr, side, mult), 2)
        out["take_profit"] = round(entry_price + side * payoff * mult * atr, 2)
    return out
