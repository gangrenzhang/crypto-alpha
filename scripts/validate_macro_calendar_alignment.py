#!/usr/bin/env python3
"""校验宏观日历 PIT: actual/surprise 不得在 released_at+buffer 之前进入特征。"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data.fetch import timeframe_delta
from crypto_alpha.data.macro_calendar import (
    attach_surprise_column,
    load_macro_events,
    normalize_macro_events,
    visible_events_at,
)
from crypto_alpha.features.macro_calendar import build_macro_feature_matrix


def main() -> int:
    cfg = Config.load()
    events = load_macro_events(cfg)
    if events.empty:
        print("[fail] 事件库为空。请先: python scripts/14_import_macro_calendar.py --csv ...")
        return 2

    mcfg = cfg.get("macro_calendar", {}) or {}
    buf = float(mcfg.get("buffer_minutes", 5))
    ttl = float(mcfg.get("feature_ttl_hours", 72))
    hl = float(mcfg.get("feature_halflife_hours", 24))
    horizon = float(mcfg.get("horizon_hours", 168))
    min_imp = int(mcfg.get("min_importance", 3))

    ev = attach_surprise_column(normalize_macro_events(events))
    n_bad = 0
    checked = 0
    for row in ev.itertuples(index=False):
        if int(row.importance) < min_imp:
            continue
        if not np.isfinite(getattr(row, "surprise", np.nan)):
            continue
        t_before = pd.Timestamp(row.released_at) - pd.Timedelta(minutes=1)
        vis = visible_events_at(ev, t_before, buffer_minutes=buf)
        if str(row.event_id) in set(vis["event_id"].astype(str)):
            print(f"[fail] 公布前可见: {row.event_id} @ {t_before}")
            n_bad += 1
        checked += 1

        t_after = pd.Timestamp(row.released_at) + pd.Timedelta(minutes=buf)
        vis2 = visible_events_at(ev, t_after, buffer_minutes=buf)
        if str(row.event_id) not in set(vis2["event_id"].astype(str)):
            print(f"[fail] 公布后不可见: {row.event_id} @ {t_after}")
            n_bad += 1

        # 单事件隔离: 排除「更早事件碰巧同 surprise」干扰, 断言 raw 在 available 前必为 0
        single = ev.loc[ev["event_id"].astype(str) == str(row.event_id)].copy()
        feat_b = build_macro_feature_matrix(
            pd.DatetimeIndex([t_before]), single,
            buffer_minutes=buf, ttl_hours=ttl, halflife_hours=hl,
            horizon_hours=horizon, min_importance=min_imp,
        )
        if abs(float(feat_b["macro_surprise_raw"].iloc[0])) > 1e-12:
            print(
                f"[fail] 单事件隔离: 公布前 surprise_raw≠0 "
                f"event={row.event_id} raw={feat_b['macro_surprise_raw'].iloc[0]}"
            )
            n_bad += 1
        feat_a = build_macro_feature_matrix(
            pd.DatetimeIndex([t_after]), single,
            buffer_minutes=buf, ttl_hours=ttl, halflife_hours=hl,
            horizon_hours=horizon, min_importance=min_imp,
        )
        expected = float(row.surprise)
        got = float(feat_a["macro_surprise_raw"].iloc[0])
        if abs(got - expected) > 1e-9:
            print(
                f"[fail] 单事件隔离: 公布后 surprise_raw 不符 "
                f"event={row.event_id} got={got} expected={expected}"
            )
            n_bad += 1

    # 决策口径与主周期一致的抽样
    delta = timeframe_delta(cfg["data"]["timeframe"])
    sample_t = ev["released_at"].iloc[len(ev) // 2]
    decision = pd.Timestamp(sample_t) + delta
    _ = build_macro_feature_matrix(
        pd.DatetimeIndex([decision]), ev,
        buffer_minutes=buf, ttl_hours=ttl, halflife_hours=hl,
        horizon_hours=horizon, min_importance=min_imp,
    )

    if n_bad:
        print(f"[fail] PIT 违规 {n_bad} 处 (检查了 {checked} 条带 surprise 的事件)")
        return 1
    print(
        f"[ok] 宏观日历 PIT 通过: {len(ev)} 事件, 检查 {checked} 条数值公布; "
        f"buffer={buf}m ttl={ttl}h (含单事件隔离 surprise_raw)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
