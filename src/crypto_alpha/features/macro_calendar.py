"""宏观日历数值特征: 无泄漏定点绑定到主周期 K 线决策时刻。

纪律(相对新闻情绪流更「事件化」):
- 决策时刻 = 主 bar 开盘 + 主周期(与 MTF / news 一致);
- actual/surprise 仅当 ``released_at + buffer <= decision_at``;
- ``hours_to_next`` 仅用 ``scheduled_at > decision_at``(日历事先公开);
- ``macro_surprise*`` 只取窗内 **有限 surprise** 的最近一场(讲话等无数值事件不冲刷);
- ``macro_importance`` / ``has_recent`` / ``hours_since`` 仍跟 **最近任意可见事件**(含讲话);
- ``macro_awaiting_release``: ``scheduled_at <= decision < available_at``(延迟公布等待);
- 超过 TTL 的冲击衰减归零; 缺事件库时填 0 + degradations 告警。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MACRO_FEATURE_COLS = [
    "macro_surprise",
    "macro_surprise_raw",
    "macro_importance",
    "macro_hours_since",
    "macro_n_events_window",
    "macro_surprise_abs_max",
    "macro_hours_to_next",
    "macro_next_importance",
    "has_recent_macro",
    "macro_awaiting_release",
]


def _empty_frame(n: int, ttl_hours: float, horizon_hours: float) -> pd.DataFrame:
    return pd.DataFrame({
        "macro_surprise": np.zeros(n, dtype=float),
        "macro_surprise_raw": np.zeros(n, dtype=float),
        "macro_importance": np.zeros(n, dtype=float),
        "macro_hours_since": np.full(n, float(ttl_hours), dtype=float),
        "macro_n_events_window": np.zeros(n, dtype=float),
        "macro_surprise_abs_max": np.zeros(n, dtype=float),
        "macro_hours_to_next": np.full(n, float(horizon_hours), dtype=float),
        "macro_next_importance": np.zeros(n, dtype=float),
        "has_recent_macro": np.zeros(n, dtype=float),
        "macro_awaiting_release": np.zeros(n, dtype=float),
    })


def _empty_macro_features(
    feat: pd.DataFrame, ttl_hours: float, horizon_hours: float,
) -> pd.DataFrame:
    empty = _empty_frame(len(feat), ttl_hours, horizon_hours)
    for c in MACRO_FEATURE_COLS:
        feat[c] = empty[c].to_numpy()
    return feat


def _maybe_warn_macro_coverage(feat: pd.DataFrame, mcfg: dict, symbol: str) -> None:
    thr = float(mcfg.get("min_coverage_warn", 0.0))
    if thr <= 0:
        return
    if "has_recent_macro" in feat.columns:
        cov = float(pd.Series(feat["has_recent_macro"]).fillna(0.0).astype(float).mean())
    else:
        cov = 0.0
    feat.attrs["macro_feature_coverage"] = cov
    if cov >= thr:
        return
    tag = f"macro_features_sparse(coverage={cov:.4f},threshold={thr:.4f})"
    deg = list(feat.attrs.get("degradations") or [])
    if tag not in deg:
        deg.append(tag)
    feat.attrs["degradations"] = deg
    print(
        f"[macro] WARN: {symbol} 宏观日历特征覆盖率过低 coverage={cov:.2%} "
        f"(阈值 {thr:.0%})。请导入 events.parquet 或关闭 macro_calendar.as_feature。"
    )
    if bool(mcfg.get("require_min_coverage", False)):
        raise ValueError(
            f"{symbol}: 宏观特征覆盖率 {cov:.2%} < {thr:.0%} 且 "
            f"macro_calendar.require_min_coverage=true。"
        )


def build_macro_feature_matrix(
    decision_at: pd.DatetimeIndex,
    events: pd.DataFrame,
    *,
    buffer_minutes: float = 5.0,
    ttl_hours: float = 72.0,
    halflife_hours: float = 24.0,
    horizon_hours: float = 168.0,
    min_importance: int = 3,
) -> pd.DataFrame:
    """对每个 decision_at 计算宏观特征。可供测试直接调用。

    分流纪律:
    - surprise 通道: 最近一场 **finite surprise**(数值公布);
    - 注意力通道: 最近一场 **任意可见** 事件(含讲话) → importance / hours_since / has_recent;
    - awaiting: 已到 scheduled、尚未 available 的延迟公布。
    """
    from ..data.macro_calendar import attach_surprise_column, normalize_macro_events

    n = len(decision_at)
    out = _empty_frame(n, ttl_hours, horizon_hours)
    if events is None or len(events) == 0 or n == 0:
        return out

    ev = attach_surprise_column(normalize_macro_events(events))
    ev = ev.loc[ev["importance"] >= int(min_importance)].copy()
    if ev.empty:
        return out

    ev["available_at"] = ev["released_at"] + pd.Timedelta(minutes=float(buffer_minutes))
    # 可见窗按 available 排序; schedule 另序用于 next / awaiting
    ev = ev.sort_values(["available_at", "scheduled_at"]).reset_index(drop=True)

    avail = pd.DatetimeIndex(pd.to_datetime(ev["available_at"], utc=True)).as_unit("ns").asi8
    sched = pd.DatetimeIndex(pd.to_datetime(ev["scheduled_at"], utc=True)).as_unit("ns").asi8
    sur_raw = ev["surprise"].to_numpy(dtype=float)
    sur_finite = np.isfinite(sur_raw)
    # abs_max 用有限 surprise; 非有限视为不参与(置 0 且 mask 掉)
    sur_for_abs = np.where(sur_finite, sur_raw, 0.0)
    imp_arr = ev["importance"].to_numpy(dtype=float)

    order_s = np.argsort(sched, kind="mergesort")
    sched_sorted = sched[order_s]
    imp_sorted = imp_arr[order_s]

    dec = pd.DatetimeIndex(pd.to_datetime(decision_at, utc=True)).as_unit("ns").asi8
    ttl_ns = int(float(ttl_hours) * 3600 * 1_000_000_000)
    horizon_ns = int(float(horizon_hours) * 3600 * 1_000_000_000)
    hl = max(float(halflife_hours), 1e-6)
    ns_per_hour = 3_600_000_000_000.0

    surprise_a = out["macro_surprise"].to_numpy(copy=True)
    surprise_raw = out["macro_surprise_raw"].to_numpy(copy=True)
    importance_a = out["macro_importance"].to_numpy(copy=True)
    hours_since = out["macro_hours_since"].to_numpy(copy=True)
    n_events = out["macro_n_events_window"].to_numpy(copy=True)
    surprise_abs_max = out["macro_surprise_abs_max"].to_numpy(copy=True)
    hours_to_next = out["macro_hours_to_next"].to_numpy(copy=True)
    next_imp = out["macro_next_importance"].to_numpy(copy=True)
    has_recent = out["has_recent_macro"].to_numpy(copy=True)
    awaiting = out["macro_awaiting_release"].to_numpy(copy=True)

    for i, t in enumerate(dec):
        # --- 下一场(严格未来 schedule) ---
        j_next = int(np.searchsorted(sched_sorted, t, side="right"))
        if j_next < len(sched_sorted):
            dt_ns = int(sched_sorted[j_next] - t)
            if 0 < dt_ns <= horizon_ns:
                hours_to_next[i] = dt_ns / ns_per_hour
                next_imp[i] = float(imp_sorted[j_next]) / 5.0

        # --- 延迟公布: 已到点尚未可见 ---
        awaiting_mask = (sched <= t) & (avail > t)
        if bool(awaiting_mask.any()):
            awaiting[i] = 1.0

        # --- 已可见事件窗 [t-ttl, t] ---
        right = int(np.searchsorted(avail, t, side="right"))
        if right <= 0:
            continue
        left = int(np.searchsorted(avail, t - ttl_ns, side="left"))
        left = max(0, min(left, right))
        if left >= right:
            continue

        # 注意力: 最近任意可见事件
        last_any = right - 1
        last_age_any = float((t - avail[last_any]) / ns_per_hour)
        importance_a[i] = float(imp_arr[last_any]) / 5.0
        hours_since[i] = min(last_age_any, float(ttl_hours))
        n_events[i] = min((right - left) / 10.0, 1.0)
        has_recent[i] = 1.0

        # surprise: 最近一场有限 surprise(数值公布); 讲话 NaN 不冲刷
        finite_local = np.flatnonzero(sur_finite[left:right])
        if len(finite_local):
            k_num = left + int(finite_local[-1])
            last_s = float(sur_raw[k_num])
            last_age_num = float((t - avail[k_num]) / ns_per_hour)
            last_decay = float(np.exp(-last_age_num / hl))
            surprise_raw[i] = last_s
            surprise_a[i] = last_s * last_decay

        # abs_max: 仅有限 surprise
        sl_finite = sur_finite[left:right]
        if bool(sl_finite.any()):
            sl_sur = sur_for_abs[left:right]
            ages_h = (t - avail[left:right]) / ns_per_hour
            decay = np.exp(-ages_h / hl)
            # 非有限位置 decay 无关; abs 为 0
            scored = np.where(sl_finite, np.abs(sl_sur) * decay, -1.0)
            k = int(np.argmax(scored))
            if scored[k] >= 0.0:
                surprise_abs_max[i] = float(scored[k])

    return pd.DataFrame({
        "macro_surprise": surprise_a,
        "macro_surprise_raw": surprise_raw,
        "macro_importance": importance_a,
        "macro_hours_since": hours_since,
        "macro_n_events_window": n_events,
        "macro_surprise_abs_max": surprise_abs_max,
        "macro_hours_to_next": hours_to_next,
        "macro_next_importance": next_imp,
        "has_recent_macro": has_recent,
        "macro_awaiting_release": awaiting,
    })


def add_macro_calendar_features(
    feat: pd.DataFrame, cfg, symbol: str | None = None,
) -> pd.DataFrame:
    """在特征面板追加宏观日历特征。``as_feature=false`` 时原样返回。"""
    mcfg = dict(cfg.get("macro_calendar", {}) or {})
    ttl = float(mcfg.get("feature_ttl_hours", 72))
    horizon = float(mcfg.get("horizon_hours", 168))
    if not bool(mcfg.get("as_feature", False)):
        return feat

    from ..data.fetch import timeframe_delta
    from ..data.macro_calendar import load_macro_events

    sym = symbol or "?"
    events = load_macro_events(cfg)
    if events is None or len(events) == 0:
        feat = _empty_macro_features(feat, ttl, horizon)
        tag = "macro_calendar_unavailable"
        deg = list(feat.attrs.get("degradations") or [])
        if tag not in deg:
            deg.append(tag)
        feat.attrs["degradations"] = deg
        _maybe_warn_macro_coverage(feat, mcfg, sym)
        return feat

    buffer_min = float(mcfg.get("buffer_minutes", 5))
    halflife = float(mcfg.get("feature_halflife_hours", 24))
    min_imp = int(mcfg.get("min_importance", 3))
    main_tf = cfg["data"]["timeframe"]
    main_delta = timeframe_delta(main_tf)

    main_idx = pd.DatetimeIndex(pd.to_datetime(feat.index, utc=True))
    decision_at = main_idx + main_delta

    built = build_macro_feature_matrix(
        decision_at,
        events,
        buffer_minutes=buffer_min,
        ttl_hours=ttl,
        halflife_hours=halflife,
        horizon_hours=horizon,
        min_importance=min_imp,
    )
    built.index = main_idx
    built = built.reindex(feat.index)
    for c in MACRO_FEATURE_COLS:
        if c in ("macro_hours_since", "macro_hours_to_next"):
            fill = ttl if c == "macro_hours_since" else horizon
            feat[c] = built[c].fillna(fill).astype(float).values
        else:
            feat[c] = built[c].fillna(0.0).astype(float).values

    _maybe_warn_macro_coverage(feat, mcfg, sym)
    return feat
