"""宏观经济日历事件库: 结构化公布/讲话 → 无泄漏对齐到决策时刻。

与 ``data/news.py``(文章情绪流)互补:
- 本模块存 **事件**(CPI/失业率/ZEW/讲话等), 含前值/预测/公布/重要性;
- 特征只在 ``released_at + buffer <= 决策时刻`` 后暴露 actual/surprise;
- ``scheduled_at`` 可在公布前用于 ``hours_to_next``(日历事先公开, 非前视);
- **禁止**在公布前把 actual 写入特征。

存储: ``{root}/{store_dir}/events.parquet``(全局事件表, 非按币种拆分)。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "event_id",
    "name",
    "country",
    "category",
    "importance",
    "scheduled_at",
    "released_at",
    "previous",
    "forecast",
    "actual",
    "unit",
    "source",
    "print_kind",       # first_print | current_vintage | n/a
    "schedule_source",  # bls_official | heuristic | forexfactory | federalreserve | import
]

# 日历源只给出日期、未给盘中时刻时的标记(FF 归档的 All Day / Tentative / 缺时刻行)
DATE_ONLY_SCHEDULE_SOURCE = "forexfactory_dateonly"

REQUIRED_COLUMNS = [
    "name",
    "scheduled_at",
    "released_at",
    "importance",
]


def _macro_cfg(cfg) -> dict:
    return dict(cfg.get("macro_calendar", {}) or {})


def macro_store_dir(cfg) -> Path:
    m = _macro_cfg(cfg)
    rel = str(m.get("store_dir", "data/macro_calendar"))
    p = Path(rel)
    return p if p.is_absolute() else (cfg.root / p)


def macro_events_path(cfg) -> Path:
    return macro_store_dir(cfg) / "events.parquet"


def _to_utc_ts(series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def date_only_release_ts(local_date) -> pd.Timestamp | None:
    """源只给出日期(无盘中时刻)时的保守可见时刻: 该本地日 **UTC 日终**。

    绝不可用本地午夜: 本地 00:00 换算成 UTC 常落到**前一天**, 会让 actual/surprise
    比真实公布早十几小时暴露给特征(前视泄漏)。任何时区的本地日 D 最晚也在
    ``D 23:59:59 UTC`` 之前结束(UTC+14 的 D 末 = D 09:59:59 UTC), 因此取 UTC 日终
    可保证「不早于真实公布」, 代价仅是把该事件推迟到当日收尾才可见。
    """
    try:
        d = pd.Timestamp(local_date)
    except Exception:
        return None
    if pd.isna(d):
        return None
    if d.tzinfo is not None:
        # 取**本地**日历日: tz_convert(None) 会先折算到 UTC, 恰好把本地午夜推回前一天
        d = d.tz_localize(None)
    return d.normalize().tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)


def normalize_macro_events(df: pd.DataFrame) -> pd.DataFrame:
    """规范化事件表: UTC 时间、重要性裁剪、缺省列、去重。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    out = df.copy()
    for c in EVENT_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    out["name"] = out["name"].astype(str).str.strip()
    out["country"] = out["country"].fillna("").astype(str)
    out["category"] = out["category"].fillna("other").astype(str)
    out["unit"] = out["unit"].fillna("").astype(str)
    out["source"] = out["source"].fillna("import").astype(str)
    out["print_kind"] = out["print_kind"].fillna("n/a").astype(str)
    out["schedule_source"] = out["schedule_source"].fillna("import").astype(str)

    out["scheduled_at"] = _to_utc_ts(out["scheduled_at"])
    out["released_at"] = _to_utc_ts(out["released_at"])
    miss_rel = out["released_at"].isna() & out["scheduled_at"].notna()
    out.loc[miss_rel, "released_at"] = out.loc[miss_rel, "scheduled_at"]

    out["importance"] = pd.to_numeric(out["importance"], errors="coerce").fillna(1)
    out["importance"] = out["importance"].clip(1, 5).astype(int)
    for c in ("previous", "forecast", "actual"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["event_id"] = out["event_id"].astype(object)
    eid = out["event_id"]
    need_id = eid.isna() | (eid.astype(str).str.strip() == "") | (eid.astype(str) == "nan")
    if bool(need_id.any()):
        # 含 source: 跨源同名同刻行在去重前必须共存, 供字段级合并调查 forecast
        gen = (
            out["country"].astype(str) + "|"
            + out["name"].astype(str) + "|"
            + out["scheduled_at"].dt.strftime("%Y%m%dT%H%M%SZ").fillna("") + "|"
            + out["source"].astype(str)
        )
        out.loc[need_id, "event_id"] = gen.loc[need_id].astype(str).values
    out["event_id"] = out["event_id"].astype(str)

    out = out.dropna(subset=["scheduled_at", "released_at", "name"])
    out = out.drop_duplicates(subset=["event_id"], keep="last")
    out = out.sort_values("released_at").reset_index(drop=True)
    return out[EVENT_COLUMNS]


def compute_surprise(previous, forecast, actual) -> float:
    """标准化 surprise = (actual - forecast) / scale; 缺数则 NaN。

    scale = max(|forecast|, |previous|, 1.0), 避免除零与量纲爆炸。
    """
    if actual is None or (isinstance(actual, float) and np.isnan(actual)):
        return float("nan")
    if forecast is None or (isinstance(forecast, float) and np.isnan(forecast)):
        return float("nan")
    a = float(actual)
    f = float(forecast)
    if previous is None or (isinstance(previous, float) and np.isnan(previous)):
        p = 0.0
    else:
        p = float(previous)
    scale = max(abs(f), abs(p), 1.0)
    return (a - f) / scale


def attach_surprise_column(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    surprises = [
        compute_surprise(p, f, a)
        for p, f, a in zip(out["previous"], out["forecast"], out["actual"])
    ]
    out["surprise"] = np.asarray(surprises, dtype=float)
    return out


def load_macro_events(cfg) -> pd.DataFrame:
    path = macro_events_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.read_parquet(path, engine="pyarrow")
    return normalize_macro_events(df)


def save_macro_events(cfg, events: pd.DataFrame) -> Path:
    path = macro_events_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = normalize_macro_events(events)
    df.to_parquet(path, engine="pyarrow", index=False)
    meta = {
        "n_events": int(len(df)),
        "min_released_at": None if df.empty else df["released_at"].min().isoformat(),
        "max_released_at": None if df.empty else df["released_at"].max().isoformat(),
    }
    (path.parent / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return path


def import_macro_events_frame(cfg, frame: pd.DataFrame, *, replace: bool = False) -> tuple[int, int]:
    """导入 DataFrame 到事件库。返回 (新 event_id 数, 库总量)。"""
    incoming = normalize_macro_events(frame)
    if incoming.empty:
        cur = load_macro_events(cfg)
        return 0, len(cur)
    if replace:
        save_macro_events(cfg, incoming)
        return len(incoming), len(incoming)
    cur = load_macro_events(cfg)
    if cur.empty:
        save_macro_events(cfg, incoming)
        return len(incoming), len(incoming)
    before_ids = set(cur["event_id"].astype(str))
    merged = normalize_macro_events(pd.concat([cur, incoming], ignore_index=True))
    save_macro_events(cfg, merged)
    after_ids = set(merged["event_id"].astype(str))
    added = len(after_ids - before_ids)
    return added, len(merged)


def import_macro_events_csv(cfg, csv_path, *, replace: bool = False) -> tuple[int, int]:
    path = Path(csv_path)
    df = pd.read_csv(path)
    return import_macro_events_frame(cfg, df, replace=replace)


def visible_events_at(
    events: pd.DataFrame,
    decision_at: pd.Timestamp,
    *,
    buffer_minutes: float = 5.0,
) -> pd.DataFrame:
    """返回在决策时刻已「可交易可见」的事件(released_at + buffer <= decision_at)。"""
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS + ["surprise", "available_at"])
    df = attach_surprise_column(normalize_macro_events(events))
    t = pd.Timestamp(decision_at)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    df["available_at"] = df["released_at"] + pd.Timedelta(minutes=float(buffer_minutes))
    return df.loc[df["available_at"] <= t].copy()
