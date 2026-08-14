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
    "forexfactory_hf": 45,
    "forexfactory_scrape": 48,  # 本机 cloudscraper 日页; 补 HF 冻结后缺口
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
    # 源只给了日期、时刻是我们保守推的日终: 同事件若另有精确时刻源, 必须让位
    "forexfactory_dateonly": -20,
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
    # minutes 必须先于 meeting 命中: "FOMC Meeting Minutes" 含 "meeting",
    # 顺序颠倒会把纪要误标为 Rate Decision(且与 fed 手工 Minutes 无法去重)
    (re.compile(r"fomc.*minutes", re.I), "FOMC Minutes"),
    (re.compile(r"fomc.*meeting|fomc rate decision", re.I), "FOMC Rate Decision"),
    # FF 命名: 决议同刻公布的声明/利率行 → 与 Fed 官网 FOMC Rate Decision 合并
    (re.compile(r"^federal funds rate$|^fomc statement$", re.I), "FOMC Rate Decision"),
    (re.compile(r"beige book", re.I), "Beige Book"),
    # 非美央行利率决议: 统一 FF 与手工日程的命名差异
    (re.compile(r"ecb.*rate|ecb.*interest rate|ecb.*refinancing", re.I), "ECB Rate Decision"),
    (re.compile(r"^main refinancing rate$|^minimum bid rate$|^deposit facility rate$", re.I), "ECB Rate Decision"),
    (re.compile(r"boe.*rate|boe.*interest rate|boe.*bank rate", re.I), "BOE Rate Decision"),
    (re.compile(r"^official bank rate$", re.I), "BOE Rate Decision"),
    (re.compile(r"boj.*rate|boj.*interest rate|boj.*policy rate", re.I), "BOJ Rate Decision"),
    (re.compile(r"^boj monetary policy statement$|^boj policy rate$", re.I), "BOJ Rate Decision"),
)


def canonical_macro_name(name: str) -> str:
    s = str(name or "").strip()
    for pat, canon in _CANONICAL:
        if pat.search(s):
            return canon
    return s[:120]


def _finite(v) -> bool:
    if v is None:
        return False
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _is_survey_source(source: str) -> bool:
    """ForexFactory 等调查中位数源(有真实 forecast), 非 BLS/ALFRED naive。"""
    return str(source or "").startswith("forexfactory")


def _is_naive_forecast(row) -> bool:
    """BLS/ALFRED 常把 forecast≡previous; 这不是市场调查中位数。"""
    if _is_survey_source(str(row.get("source") or "")):
        return False
    if not _finite(row.get("forecast")):
        return True
    if not _finite(row.get("previous")):
        return False
    return bool(np.isclose(float(row["forecast"]), float(row["previous"]), rtol=0.0, atol=1e-9))


def _event_score(row) -> float:
    src = _SOURCE_SCORE.get(str(row.get("source") or ""), 0)
    sched = _SCHEDULE_SCORE.get(str(row.get("schedule_source") or ""), 0)
    pk = _PRINT_SCORE.get(str(row.get("print_kind") or ""), 0)
    imp = float(row.get("importance") or 0) * 0.5
    # 有完整数值略加分; 调查 forecast 额外加分(避免 ALFRED 空 forecast 被 FF 数值行碾压时丢 actual)
    num = 0.0
    for c in ("previous", "forecast", "actual"):
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            num += 1.0
    survey_bonus = 2.0 if (_is_survey_source(str(row.get("source") or "")) and _finite(row.get("forecast")) and not _is_naive_forecast(row)) else 0.0
    return src + sched + pk + imp + num * 0.3 + survey_bonus


def _pick_survey_forecast(group: pd.DataFrame) -> float:
    """同组内取最佳调查 forecast(优先非 naive 的 FF 行)。"""
    best_f = float("nan")
    best_score = -1e18
    for r in group.to_dict("records"):
        if not _is_survey_source(str(r.get("source") or "")):
            continue
        if not _finite(r.get("forecast")) or _is_naive_forecast(r):
            continue
        sc = _event_score(r)
        if sc > best_score:
            best_score = sc
            best_f = float(r["forecast"])
    return best_f


def _merge_winner_fields(winner: dict, group: pd.DataFrame) -> dict:
    """胜出行保留 ALFRED/BLS 的 actual/首印, 但 forecast 优先用同组调查源。

    ALFRED/BLS 的 forecast≡previous 无市场语义; 若胜出后仍是 naive/缺失,
    用 FF 调查中位数覆盖; 仍无调查值则 forecast=NaN(surprise 通道关闭,
    注意力通道仍可用 importance/hours)。
    """
    out = dict(winner)
    survey_f = _pick_survey_forecast(group)
    if _finite(survey_f):
        out["forecast"] = survey_f
    elif _is_naive_forecast(out) or not _finite(out.get("forecast")):
        out["forecast"] = float("nan")
    return out


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
    """合并多源重复事件; 每组保留综合得分最高的一行, 并字段级合并调查 forecast。

    硬规则(与 filter_events_for_features 语义对齐):
    1. 组内若存在 print_kind=first_print 且带数值的行, 则仅在 first_print 行中
       按得分选胜者 —— current_vintage 不得覆盖首印;
    2. 同为 first_print 时 ALFRED 仍可胜出(更好的 actual/日程), 但 **forecast**
       优先取同组 ForexFactory 调查中位数; ALFRED/BLS 的 forecast≡previous
       不得当作市场 surprise 分母。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    out = normalize_macro_events(df)
    rows = []
    for r in out.to_dict("records"):
        r["_score"] = _event_score(r)
        r["_key"] = _dedupe_key(r)
        rows.append(r)
    tmp = pd.DataFrame(rows)
    is_fp = tmp["print_kind"].astype(str).eq("first_print")
    has_num = tmp[["previous", "forecast", "actual"]].apply(
        pd.to_numeric, errors="coerce",
    ).notna().any(axis=1)
    grp_has_fp_num = (is_fp & has_num).groupby(tmp["_key"]).transform("max")
    # 组内有首印数值行 → 淘汰全部非首印行, 首印行间再按得分竞争
    tmp = tmp.loc[~(grp_has_fp_num & ~is_fp)].copy()
    merged_rows: list[dict] = []
    for _, grp in tmp.groupby("_key", sort=False):
        g = grp.sort_values("_score", ascending=False)
        winner = _merge_winner_fields(g.iloc[0].to_dict(), g)
        merged_rows.append(winner)
    tmp = pd.DataFrame(merged_rows)
    tmp = tmp.drop(columns=["_score", "_key"], errors="ignore")
    return normalize_macro_events(tmp)


def filter_events_for_features(
    events: pd.DataFrame,
    *,
    prefer_first_print: bool = True,
    numeric_print_kind: str = "first_print",
    ban_current_vintage_surprise: bool = True,
    strip_naive_forecast: bool = True,
) -> pd.DataFrame:
    """特征层: 数值事件优先 first_print; 禁止修订版/naive forecast 进 surprise。

    - 有 first_print 时丢同窗 current_vintage 行(修订值软前视)。
    - 仅剩 current_vintage 时仍保留行供注意力通道, 但清空 previous/forecast/actual
      → surprise=NaN(``ban_current_vintage_surprise``)。
    - 非调查源且 forecast≡previous 时清空 forecast(``strip_naive_forecast``)。
    """
    if events is None or len(events) == 0:
        return events
    ev = normalize_macro_events(events)
    if prefer_first_print:
        want = str(numeric_print_kind or "first_print")
        numeric_mask = ev[["previous", "forecast", "actual"]].notna().any(axis=1)
        if bool(numeric_mask.any()):
            keep_idx = []
            ev = ev.copy()
            ev["_canon"] = ev["name"].map(canonical_macro_name)
            ev["_hour"] = pd.to_datetime(ev["released_at"], utc=True).dt.floor("h")
            for _, grp in ev.loc[numeric_mask].groupby(["country", "_canon", "_hour"], sort=False):
                if want in set(grp["print_kind"].astype(str)):
                    keep_idx.extend(grp.loc[grp["print_kind"].astype(str) == want].index.tolist())
                else:
                    keep_idx.extend(grp.index.tolist())
            non_num = ev.loc[~numeric_mask].index.tolist()
            sel = sorted(set(keep_idx + non_num))
            ev = ev.loc[sel].drop(columns=["_canon", "_hour"], errors="ignore")
            ev = normalize_macro_events(ev)

    out = ev.copy()
    if ban_current_vintage_surprise:
        cv = out["print_kind"].astype(str).eq("current_vintage")
        if bool(cv.any()):
            out.loc[cv, ["previous", "forecast", "actual"]] = np.nan
    if strip_naive_forecast:
        for i, row in out.iterrows():
            if _is_naive_forecast(row) and _finite(row.get("forecast")):
                out.at[i, "forecast"] = np.nan
    return normalize_macro_events(out)
