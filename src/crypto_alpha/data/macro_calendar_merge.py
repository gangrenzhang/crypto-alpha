"""跨源去重: BLS/ALFRED 与 FF 历史等同事件合并, 保留更可信一行。"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .macro_calendar import EVENT_COLUMNS, normalize_macro_events

# 来源可信度(越高越优先)
_SOURCE_SCORE = {
    "alfred_bls": 100,
    "bls": 90,
    "forexfactory_hist": 55,
    "forexfactory_week": 50,
    "federalreserve": 40,
    "import": 10,
    "test": 5,
}

_SCHEDULE_SCORE = {
    "bls_official": 30,
    "federalreserve": 15,
    "forexfactory": 10,
    "heuristic": 0,
    "import": 0,
}

_PRINT_SCORE = {
    "first_print": 25,
    "current_vintage": 10,
    "n/a": 0,
}

_CANONICAL = (
    # ADP 非农 ≠ BLS Nonfarm Payrolls
    (re.compile(r"(?<!adp )non[- ]?farm employment change$|nonfarm payrolls?", re.I), "Nonfarm Payrolls"),
    (re.compile(r"^adp non[- ]?farm", re.I), "ADP Nonfarm Employment Change"),
    (re.compile(r"unemployment rate", re.I), "Unemployment Rate"),
    (re.compile(r"^cpi\s*y/?y$|^cpi\s*yoy$|consumer price index.*y/?y", re.I), "CPI YoY"),
    (re.compile(r"core cpi", re.I), "Core CPI YoY"),
    (re.compile(r"fomc.*meeting|fomc rate decision", re.I), "FOMC Rate Decision"),
    (re.compile(r"fomc minutes", re.I), "FOMC Minutes"),
    (re.compile(r"beige book", re.I), "Beige Book"),
)


def canonical_macro_name(name: str) -> str:
    s = str(name or "").strip()
    for pat, canon in _CANONICAL:
        if pat.search(s):
            return canon
    return s[:120]


def _event_score(row) -> float:
    src = _SOURCE_SCORE.get(str(row.get("source") or ""), 0)
    sched = _SCHEDULE_SCORE.get(str(row.get("schedule_source") or ""), 0)
    pk = _PRINT_SCORE.get(str(row.get("print_kind") or ""), 0)
    imp = float(row.get("importance") or 0) * 0.5
    # 有完整数值略加分
    num = 0.0
    for c in ("previous", "forecast", "actual"):
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            num += 1.0
    return src + sched + pk + imp + num * 0.3


def _dedupe_key(row) -> tuple:
    country = str(row.get("country") or "")
    canon = canonical_macro_name(str(row.get("name") or ""))
    ts = pd.Timestamp(row.get("released_at"))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    # 同一公布窗口: 按 ET 小时桶(就业/CPI 通常同一时刻)
    hour_bucket = ts.floor("h")
    cat = str(row.get("category") or "")
    if cat in ("employment", "inflation", "central_bank"):
        return country, canon, hour_bucket.isoformat()
    # 讲话/其它: 名称 + 小时桶
    return country, canon, hour_bucket.isoformat()


def dedupe_cross_source_events(df: pd.DataFrame) -> pd.DataFrame:
    """合并多源重复事件; 每组保留综合得分最高的一行。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    out = normalize_macro_events(df)
    rows = []
    for r in out.to_dict("records"):
        r["_score"] = _event_score(r)
        r["_key"] = _dedupe_key(r)
        rows.append(r)
    tmp = pd.DataFrame(rows)
    tmp = tmp.sort_values(["_key", "_score"], ascending=[True, False])
    tmp = tmp.drop_duplicates(subset=["_key"], keep="first")
    tmp = tmp.drop(columns=["_score", "_key"], errors="ignore")
    return normalize_macro_events(tmp)


def filter_events_for_features(
    events: pd.DataFrame,
    *,
    prefer_first_print: bool = True,
    numeric_print_kind: str = "first_print",
) -> pd.DataFrame:
    """特征层可选: 数值事件优先 first_print; 无首印时保留 current_vintage。"""
    if events is None or len(events) == 0:
        return events
    ev = normalize_macro_events(events)
    if not prefer_first_print:
        return ev
    want = str(numeric_print_kind or "first_print")
    numeric_mask = ev[["previous", "forecast", "actual"]].notna().any(axis=1)
    if not bool(numeric_mask.any()):
        return ev
    # 对每个 canonical+country+release hour, 若有 first_print 则丢 current_vintage
    keep_idx = []
    ev = ev.copy()
    ev["_canon"] = ev["name"].map(canonical_macro_name)
    ev["_hour"] = pd.to_datetime(ev["released_at"], utc=True).dt.floor("h")
    for key, grp in ev.loc[numeric_mask].groupby(["country", "_canon", "_hour"], sort=False):
        if want in set(grp["print_kind"].astype(str)):
            keep_idx.extend(grp.loc[grp["print_kind"].astype(str) == want].index.tolist())
        else:
            keep_idx.extend(grp.index.tolist())
    non_num = ev.loc[~numeric_mask].index.tolist()
    sel = sorted(set(keep_idx + non_num))
    out = ev.loc[sel].drop(columns=["_canon", "_hour"], errors="ignore")
    return normalize_macro_events(out)
