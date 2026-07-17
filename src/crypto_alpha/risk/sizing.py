"""风控: 分数 Kelly 仓位 + 波动率(ATR)止损, 以及统一决策输出。

概率(校准后) + 盈亏比 -> Kelly 最优下注比例; 取分数 Kelly 并封顶以控制回撤。
止损/止盈由 **与三重障碍相同的 pt_sl 倍数 × ATR** 定义, 保证训练标签与实盘执行口径一致。

⚠️ 仓位口径: 把 f* 直接当作名义仓位比例(并封顶), 而非 growth-optimal 连续 Kelly;
另可扣减往返成本使阈值附近不轻易开仓。详见 ARCHITECTURE §12。
"""
from __future__ import annotations

import numpy as np


def resolve_roundtrip_cost(
    risk_cfg: dict, fee: float = 0.0, slip: float = 0.0,
) -> float:
    """往返成本分数: ``roundtrip_cost_frac`` 为 null/缺失时回退 ``2*(fee+slip)``。

    与回测 engine 口径一致。注意 YAML ``null`` 会使键存在且值为 None,
    ``dict.setdefault`` **不会**覆盖, 必须用本函数显式解析。
    """
    if "roundtrip_cost_frac" not in risk_cfg or risk_cfg["roundtrip_cost_frac"] is None:
        return float(2.0 * (fee + slip))
    return float(risk_cfg["roundtrip_cost_frac"])


def kelly_fraction(p: float, payoff: float, cost: float = 0.0) -> float:
    """二元 Kelly: f* = (p*(b+1) - 1 - cost) / b。

    cost: 相对仓位名义的往返成本分数(如 2*(fee+slip)); 忽略垂直结局可变赔率时
    仍用固定 b=payoff=pt/sl 作为启发式。
    """
    b = max(payoff, 1e-6)
    f = (p * (b + 1) - 1.0 - max(cost, 0.0)) / b
    return float(max(f, 0.0))


def position_size(
    p: float, payoff: float, kelly_fraction_mult: float, max_pct: float,
    cost: float = 0.0,
) -> float:
    f = kelly_fraction(p, payoff, cost=cost) * kelly_fraction_mult
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
    fee: float = 0.0, slip: float = 0.0,
) -> dict:
    """把概率+方向+价格+ATR 汇总为一条结构化交易决策。

    - confident: 保形预测是否高置信; False 时强制 HOLD(不确定则观望)。
    - HOLD 时不输出 stop_loss/take_profit(避免被误当作可执行挂单)。
    - pt_sl: 与 labeling.pt_sl 一致的 (止盈倍数, 止损倍数); 缺省时回退 risk.atr_stop_mult。
    - fee/slip: 单边成本分数; 仅当 ``roundtrip_cost_frac`` 为 null 时用于回退 2*(fee+slip)。
    """
    if pt_sl is not None:
        pt_mult, sl_mult = float(pt_sl[0]), float(pt_sl[1])
    else:
        sl_mult = float(risk_cfg.get("atr_stop_mult", 1.5))
        pt_mult = sl_mult
    if payoff is None:
        payoff = pt_mult / max(sl_mult, 1e-6)

    rt_cost = resolve_roundtrip_cost(risk_cfg, fee=fee, slip=slip)

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
            cost=rt_cost,
        )
        if size <= 0.0:
            signal = "HOLD"
            reason = "kelly_non_positive_after_cost"
            size = 0.0

    out = {
        "signal": signal,
        "win_probability": round(float(prob), 4),
        "entry_price": round(float(entry_price), 2),
        "suggested_position_pct": round(size, 4),
        "atr": round(float(atr), 4),
        "confident": bool(confident),
        "pt_mult": pt_mult,
        "sl_mult": sl_mult,
        "execution_assumption": str(risk_cfg.get("execution_assumption", "close_fill")),
        "sizing_note": "confidence_to_position_heuristic",
    }
    if signal == "HOLD":
        out["stop_loss"] = None
        out["take_profit"] = None
        out["reason"] = reason
    else:
        out["stop_loss"] = round(atr_stop(entry_price, atr, side, sl_mult), 2)
        out["take_profit"] = round(atr_take_profit(entry_price, atr, side, pt_mult), 2)
    return out
