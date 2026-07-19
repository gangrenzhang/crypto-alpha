"""开仓门控诊断与阈值解析。

与回测/decide 共用, 回答:
- 测试窗上有多少事件过 prob 阈值、保形、二者交集;
- 校准是否把过线率显著抬高、概率是否塌成少数台阶;
- 有效阈值如何从 fixed / quantile / max_of 解析(仅用**参考分**估计, 禁止用评估窗刷阈值)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_threshold_reference_mask(
    eval_mask: np.ndarray,
    report_mask: np.ndarray,
    *,
    min_ref: int = 20,
) -> tuple[np.ndarray, list[str]]:
    """构造阈值参考窗掩码(与报告窗互斥优先)。

    优先 ``eval & ~report``(多专家前半选型窗)。若不足 ``min_ref``
    (单专家时常发生: report≡eval), **禁止**回退到全 ``eval_mask``
    (会把报告窗灌进分位估计), 改为时间序上 eval 事件的前半段, 并打标
    ``prob_threshold_ref_fallback_time_half``。若仍过少则返回空掩码,
    由 ``resolve_prob_threshold`` 回退 fixed。
    """
    tags: list[str] = []
    eval_m = np.asarray(eval_mask, dtype=bool)
    report_m = np.asarray(report_mask, dtype=bool)
    if eval_m.shape != report_m.shape:
        raise ValueError(
            f"eval_mask/report_mask shape mismatch: {eval_m.shape} vs {report_m.shape}"
        )
    ref = eval_m & ~report_m
    if int(ref.sum()) >= int(min_ref):
        return ref, tags

    pos = np.flatnonzero(eval_m)
    tags.append(f"prob_threshold_ref_fallback_time_half(n_eval={int(len(pos))})")
    if len(pos) < 2:
        return np.zeros_like(eval_m, dtype=bool), tags

    n_half = max(len(pos) // 2, 1)
    ref = np.zeros_like(eval_m, dtype=bool)
    ref[pos[:n_half]] = True
    return ref, tags


def resolve_prob_threshold(
    bt_cfg: dict,
    reference_probs: np.ndarray | None = None,
) -> tuple[float, list[str]]:
    """解析有效概率阈值。

    Modes (``backtest.prob_threshold_mode``):
    - ``fixed``: 仅用 ``prob_threshold``(默认, 旧行为)
    - ``quantile``: 用 reference_probs 的 ``prob_quantile`` 分位
    - ``max_of``: ``max(fixed, quantile)`` — 推荐部署默认

    ``reference_probs`` 须来自训练/选型窗(或部署校准基), **不得**用测试窗估计。
    """
    tags: list[str] = []
    base = float(bt_cfg.get("prob_threshold", 0.55))
    mode = str(bt_cfg.get("prob_threshold_mode", "fixed") or "fixed").strip().lower()
    if mode in ("fixed", "none", ""):
        return base, tags

    ref = None if reference_probs is None else np.asarray(reference_probs, dtype=float)
    if ref is not None:
        ref = ref[np.isfinite(ref)]
    if ref is None or len(ref) < 20:
        tags.append(f"prob_threshold_{mode}_fallback_fixed(n_ref={0 if ref is None else len(ref)})")
        return base, tags

    q = float(bt_cfg.get("prob_quantile", 0.98))
    q = float(np.clip(q, 0.5, 0.999))
    q_thr = float(np.quantile(ref, q))

    target_rate = bt_cfg.get("target_trade_rate", None)
    if target_rate is not None:
        tr = float(target_rate)
        if 0.0 < tr < 1.0:
            q_thr = max(q_thr, float(np.quantile(ref, 1.0 - tr)))
            tags.append(f"prob_threshold_target_trade_rate({tr:.4f})")

    if mode == "quantile":
        thr = q_thr
    elif mode == "max_of":
        thr = max(base, q_thr)
    else:
        tags.append(f"prob_threshold_unknown_mode({mode})_fallback_fixed")
        return base, tags

    tags.append(f"prob_threshold_resolved(mode={mode},base={base:.4f},effective={thr:.4f})")
    return float(thr), tags


def raise_threshold_if_inflated(
    thr: float,
    raw_ref: np.ndarray,
    cal_ref: np.ndarray,
    bt_cfg: dict,
    *,
    pass_rate_inflate_max: float = 1.5,
    enabled: bool = True,
) -> tuple[float, list[str]]:
    """参考窗上若校准显著抬高过线率, 用更严分位抬高 thr 并冻结。

    **仅允许在阈值参考窗上调用**; 报告/测试窗只做 ``assess_calibration_pass_health``
    告警, 不得据此改 thr(避免评估窗刷阈值)。
    """
    tags: list[str] = []
    if not enabled:
        return float(thr), tags

    health = assess_calibration_pass_health(
        raw_ref,
        cal_ref,
        float(thr),
        pass_rate_inflate_max=float(pass_rate_inflate_max),
        min_unique_levels=0,
    )
    inflate = [t for t in health if "calibration_inflates_pass_rate" in t]
    if not inflate:
        return float(thr), tags
    tags.extend(inflate)

    cal = np.asarray(cal_ref, dtype=float)
    cal = cal[np.isfinite(cal)]
    if len(cal) < 20:
        tags.append("prob_threshold_inflate_raise_skipped(n_ref<20)")
        return float(thr), tags

    q_base = float(bt_cfg.get("prob_quantile", 0.98))
    q_cfg = bt_cfg.get("inflate_raise_quantile", None)
    q_raise = float(q_cfg) if q_cfg is not None else max(q_base, 0.99)
    q_raise = float(np.clip(q_raise, 0.5, 0.999))
    cand = float(np.quantile(cal, q_raise))
    new_thr = max(float(thr), cand)
    if new_thr > float(thr) + 1e-12:
        tags.append(
            f"prob_threshold_raised_on_inflate("
            f"from={float(thr):.4f},to={new_thr:.4f},q={q_raise:.4f})"
        )
        return float(new_thr), tags
    tags.append(
        f"prob_threshold_inflate_no_raise_room("
        f"thr={float(thr):.4f},q={q_raise:.4f},cand={cand:.4f})"
    )
    return float(thr), tags


def freeze_threshold_on_reference(
    bt_cfg: dict,
    raw_ref: np.ndarray,
    cal_ref: np.ndarray,
    *,
    pass_rate_inflate_max: float = 1.5,
    tag_prefix: str = "",
) -> tuple[float, list[str]]:
    """在参考窗上 resolve + 可选 inflate 抬升, 返回冻结阈值与标签。

    ``cal_ref`` 必须与后续开门控的概率**同校准器/同尺度**:
    - 研究路径: 交叉拟合 ``oof_cal[ref]``
    - 部署/decide: ``deploy_cal.transform(raw_oof[ref])`` (禁止传入已 CF 校准分再 transform)
    """
    thr, tags = resolve_prob_threshold(bt_cfg, reference_probs=cal_ref)
    thr, raise_tags = raise_threshold_if_inflated(
        thr,
        raw_ref,
        cal_ref,
        bt_cfg,
        pass_rate_inflate_max=float(pass_rate_inflate_max),
        enabled=bool(bt_cfg.get("raise_thr_on_inflate", True)),
    )
    tags = list(tags) + list(raise_tags)
    if tag_prefix:
        tags = [f"{tag_prefix}{t}" if not t.startswith(tag_prefix) else t for t in tags]
    return float(thr), tags


def assess_calibration_pass_health(
    raw_proba: np.ndarray,
    cal_proba: np.ndarray,
    thr: float,
    *,
    pass_rate_inflate_max: float = 1.5,
    min_unique_levels: int = 20,
) -> list[str]:
    """校准后过线率/台阶健康检查; 仅返回 degradations 标签, 不改概率。"""
    tags: list[str] = []
    raw = np.asarray(raw_proba, dtype=float)
    cal = np.asarray(cal_proba, dtype=float)
    m = np.isfinite(raw) & np.isfinite(cal)
    if m.sum() < 20:
        return tags
    raw, cal = raw[m], cal[m]
    fr = float(np.mean(raw >= thr))
    fc = float(np.mean(cal >= thr))
    inflate_max = float(pass_rate_inflate_max)
    if inflate_max > 0 and fr > 1e-9 and (fc / fr) > inflate_max:
        tags.append(
            f"calibration_inflates_pass_rate(raw={fr:.4f},cal={fc:.4f},ratio={fc / fr:.2f})"
        )
    elif inflate_max > 0 and fr <= 1e-9 and fc > 0.02:
        tags.append(
            f"calibration_inflates_pass_rate(raw={fr:.4f},cal={fc:.4f},ratio=inf)"
        )

    n_unique = int(len(np.unique(np.round(cal, 6))))
    min_u = int(min_unique_levels)
    if min_u > 0 and n_unique < min_u:
        tags.append(f"calibration_low_unique_levels(n={n_unique},min={min_u})")
    return tags


def gate_diagnostics(
    event_index: pd.Index,
    raw_te: np.ndarray,
    prob_te: np.ndarray,
    confident: np.ndarray,
    detail: pd.DataFrame,
    thr: float,
    conf_obj=None,
) -> dict:
    """测试/报告窗开仓门控诊断。"""
    raw_te = np.asarray(raw_te, dtype=float)
    prob_te = np.asarray(prob_te, dtype=float)
    confident = np.asarray(confident, dtype=bool)
    n = int(len(prob_te))
    pass_thr = prob_te >= thr
    pass_conf = confident
    pass_both = pass_thr & pass_conf

    gate = pd.DataFrame(
        {
            "prob": prob_te,
            "confident": confident,
            "pass_thr": pass_thr,
            "pass_both": pass_both,
        },
        index=event_index,
    )
    if detail is not None and "size" in getattr(detail, "columns", []):
        gate = gate.join(detail[["size"]], how="left")
        gate["size"] = gate["size"].fillna(0.0)
    else:
        gate["size"] = 0.0
    opened = gate["size"].to_numpy(dtype=float) > 0.0
    pass_both_arr = gate["pass_both"].to_numpy(dtype=bool)

    rounded = np.round(prob_te, 6)
    uniq, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(-counts)
    top_levels = [
        {"prob": float(uniq[i]), "n": int(counts[i]), "frac": float(counts[i] / max(n, 1))}
        for i in order[:12]
    ]

    qhat = float(getattr(conf_obj, "qhat_", float("nan"))) if conf_obj is not None else float("nan")
    margin = float(getattr(conf_obj, "min_margin", 0.0) or 0.0) if conf_obj is not None else 0.0

    return {
        "prob_threshold": float(thr),
        "n_events": n,
        "raw_proba": {
            "mean": float(np.nanmean(raw_te)) if n else float("nan"),
            "std": float(np.nanstd(raw_te)) if n else float("nan"),
            "min": float(np.nanmin(raw_te)) if n else float("nan"),
            "max": float(np.nanmax(raw_te)) if n else float("nan"),
            "frac_ge_threshold": float(np.mean(raw_te >= thr)) if n else 0.0,
        },
        "calibrated_proba": {
            "mean": float(np.nanmean(prob_te)) if n else float("nan"),
            "std": float(np.nanstd(prob_te)) if n else float("nan"),
            "min": float(np.nanmin(prob_te)) if n else float("nan"),
            "max": float(np.nanmax(prob_te)) if n else float("nan"),
            "n_unique": int(len(uniq)),
            "top_levels": top_levels,
        },
        "gates": {
            "n_prob_ge_threshold": int(pass_thr.sum()),
            "frac_prob_ge_threshold": float(pass_thr.mean()) if n else 0.0,
            "n_confident": int(pass_conf.sum()),
            "frac_confident": float(pass_conf.mean()) if n else 0.0,
            "n_prob_and_confident": int(pass_both.sum()),
            "frac_prob_and_confident": float(pass_both.mean()) if n else 0.0,
            "n_opened_size_gt_0": int(opened.sum()),
            "frac_opened_size_gt_0": float(opened.mean()) if n else 0.0,
            "n_pass_gate_but_size_0": int((pass_both_arr & ~opened).sum()),
        },
        "conformal_qhat": qhat,
        "conformal_min_margin": margin,
        "note": (
            "frac_prob_and_confident 为阈值+保形可开仓上限(未计资金); "
            "若其≈frac_prob_ge_threshold 且 n_unique 很小, 说明保形未独立过滤且校准台阶化。"
        ),
    }
