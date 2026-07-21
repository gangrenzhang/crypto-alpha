"""降级环境护栏: 多 degradations 叠加时识别低置信环境并强制 HOLD。

校准器在「正常数据」条件下拟合; 当 tip 跨所、合成降级、衍生品全缺、新闻稀疏等
同时出现时, 模型输出的 win_probability 不再具有校准语义。与 ``feature_schema_mismatch``
强制 HOLD 同一哲学: 分布外则弃权, 而非静默开仓。

仅用于实盘/最新决策路径; 研究回测仍保留 degradations 列表供看板, 不自动砍仓
(多年回测常年有新闻稀疏等标签, 硬 HOLD 会使研究不可用)。

重要: **只对「本次推理数据面」标签计分**。训练期的校准回退 / 剪枝 / nested-OOF
等研究质量标签不得计入环境分, 否则会「训过一次小样本 → 永远 HOLD」。
"""
from __future__ import annotations

from typing import Iterable

# 仅「当下数据/装配环境」相关标签参与环境分(子串匹配)。
_LIVE_ENV_KEYS: tuple[str, ...] = (
    "feature_schema_mismatch",
    "ohlcv_synthetic_fallback",
    "ohlcv_tip_exchange_fallback",
    "aux_tip_resample_seam",
    "news_features_sparse",
    "derivatives_funding_unavailable",
    "derivatives_oi_unavailable",
    "derivatives_liquidations_unavailable",
    "derivatives_liquidations_sparse",
)

# 标签前缀/子串 → 严重度权重。未列入 _LIVE_ENV_KEYS 的不计环境分。
_SEVERITY: list[tuple[str, int]] = [
    ("feature_schema_mismatch", 100),
    ("ohlcv_synthetic_fallback", 40),
    ("ohlcv_tip_exchange_fallback", 25),
    ("aux_tip_resample_seam", 15),
    ("news_features_sparse", 15),
    ("derivatives_funding_unavailable", 10),
    ("derivatives_oi_unavailable", 10),
    ("derivatives_liquidations_unavailable", 10),
    ("derivatives_liquidations_sparse", 5),
]


def is_live_environment_tag(tag: str) -> bool:
    """是否属于「本次推理数据环境」降级(可计入环境 HOLD)。"""
    t = str(tag or "")
    return any(k in t for k in _LIVE_ENV_KEYS)


def filter_live_environment_tags(tags: Iterable[str] | None) -> list[str]:
    return [str(t) for t in (tags or []) if is_live_environment_tag(str(t))]


def degradation_severity(tag: str) -> int:
    """单条 **live** degradation 的严重度权重; 非 live 标签返回 0。"""
    t = str(tag or "")
    if not is_live_environment_tag(t):
        return 0
    for key, w in _SEVERITY:
        if key in t:
            return int(w)
    return 5


def score_degradations(tags: Iterable[str] | None) -> tuple[int, list[str]]:
    """累计严重度; 只计 live 环境标签; 同前缀只计一次最高权。"""
    best: dict[str, int] = {}
    for raw in filter_live_environment_tags(tags):
        t = str(raw)
        key = t
        for pref, _ in _SEVERITY:
            if pref in t:
                key = pref
                break
        w = degradation_severity(t)
        if w <= 0:
            continue
        best[key] = max(best.get(key, 0), w)
    matched = sorted(best.keys())
    return int(sum(best.values())), matched


def should_hold_for_environment(
    tags: Iterable[str] | None,
    *,
    threshold: float | int | None,
) -> tuple[bool, int, str | None]:
    """累计分 ≥ threshold 时建议强制 HOLD。

    ``threshold`` ≤0 或 None → 关闭护栏, 永不因环境 HOLD。
    返回 ``(hold, score, reason_tag)``。
    """
    if threshold is None:
        return False, 0, None
    thr = float(threshold)
    if thr <= 0:
        return False, 0, None
    score, matched = score_degradations(tags)
    if score >= thr:
        tag = f"low_confidence_environment(score={score}>={int(thr)};{','.join(matched[:6])})"
        if len(matched) > 6:
            tag = tag[:-1] + f",+{len(matched) - 6}more)"
        return True, score, tag
    return False, score, None


def hold_for_environment(
    *,
    symbol: str,
    score: int,
    reason_tag: str,
    risk_cfg: dict,
    timestamp=None,
    close: float | None = None,
    data_source: str | None = None,
    degradations: list[str] | None = None,
) -> dict:
    """低置信环境强制 HOLD(不推理、不吐 SL/TP)。"""
    from ..risk.sizing import resolve_execution_assumption
    from ..serve.notifier import attach_decision_description

    deg = list(degradations or [])
    if reason_tag and reason_tag not in deg:
        deg.append(reason_tag)
    out = {
        "signal": "HOLD",
        "symbol": symbol,
        "reason": "low_confidence_environment",
        "win_probability": None,
        "suggested_position_pct": 0.0,
        "stop_loss": None,
        "take_profit": None,
        "confident": False,
        "execution_assumption": resolve_execution_assumption(risk_cfg),
        "degradations": deg,
        "env_degradation_score": int(score),
    }
    if timestamp is not None:
        out["timestamp"] = str(timestamp)
    if close is not None:
        out["close"] = float(close)
    if data_source is not None:
        out["data_source"] = data_source
    return attach_decision_description(out)
