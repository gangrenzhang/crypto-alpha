"""构建宏观日历事件表: 官方日程 + 首印/现行 + 全球历史。

数据源:
- BLS Public API: 数值(无 FRED_API_KEY 时为 current_vintage)
- ALFRED/FRED(需 FRED_API_KEY): 首印 first_print
- BLS 官方 release schedule(直播/Wayback): 精确 released_at
- Federal Reserve calendar.json: FOMC/讲话日程
- ForexFactory GitHub 归档 2020–2023: 全球 actual/forecast/previous
- ForexFactory 本周 JSON: 近端补齐

元数据列:
- print_kind: first_print | current_vintage | n/a
- schedule_source: bls_official | heuristic | forexfactory | federalreserve
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .macro_calendar import EVENT_COLUMNS, normalize_macro_events
from .macro_calendar_alfred import FRED_SERIES, load_or_fetch_first_prints
from .macro_calendar_bls_schedule import (
    enrich_schedule_from_ff_hist,
    load_or_build_bls_schedule,
)
from .macro_calendar_global_ff import fetch_ff_historical_events
from .macro_calendar_merge import dedupe_cross_source_events

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

BLS_SERIES = {
    "CES0000000001": "Nonfarm Payrolls",
    "LNS14000000": "Unemployment Rate",
    "CUUR0000SA0": "CPI YoY",
    "CUUR0000SA0L1E": "Core CPI YoY",
}

IMPACT_MAP = {"High": 5, "Medium": 3, "Low": 1, "Holiday": 1}


def _curl_bytes(url: str, *, data: bytes | None = None, timeout: float = 90.0) -> bytes:
    import os
    import subprocess

    cmd = [
        "curl", "-sL", "-A", "Mozilla/5.0 (crypto-alpha macro calendar)",
        "--connect-timeout", "20", "--max-time", str(int(timeout)), "-k",
    ]
    proxy = (
        os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    )
    if not proxy:
        try:
            from .news import _resolve_http_proxies
            proxies = _resolve_http_proxies()
            proxy = proxies.get("https") or proxies.get("http")
        except Exception:
            proxy = None
    if proxy:
        cmd.extend(["-x", proxy])
    if data is not None:
        cmd.extend(["-H", "Content-Type: application/json", "--data-binary", "@-"])
    cmd.append(url)
    proc = subprocess.run(
        cmd, input=data, capture_output=True, timeout=timeout + 5, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"curl 非零/空响应({proc.returncode}): {url} {err}")
    return proc.stdout


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    return _curl_bytes(url, timeout=timeout)


def _http_post_json(url: str, payload: dict, timeout: float = 90.0) -> dict:
    raw = _curl_bytes(url, data=json.dumps(payload).encode("utf-8"), timeout=timeout)
    return json.loads(raw.decode("utf-8"))


def _parse_et(year: int, month: int, day: int, hour: int, minute: int = 0) -> pd.Timestamp:
    local = datetime(year, month, day, hour, minute, tzinfo=ET)
    return pd.Timestamp(local.astimezone(UTC))


def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    offset = (4 - d.weekday()) % 7
    return d + timedelta(days=offset)


def _second_wednesday(year: int, month: int) -> date:
    d = date(year, month, 1)
    offset = (2 - d.weekday()) % 7
    return d + timedelta(days=offset) + timedelta(days=7)


def _add_month(y: int, m: int, delta: int = 1) -> tuple[int, int]:
    m2 = m + delta
    y2 = y + (m2 - 1) // 12
    m2 = (m2 - 1) % 12 + 1
    return y2, m2


def fetch_bls_monthly(start_year: int, end_year: int) -> dict[str, pd.DataFrame]:
    payload = {
        "seriesid": list(BLS_SERIES.keys()),
        "startyear": str(int(start_year)),
        "endyear": str(int(end_year)),
    }
    data = _http_post_json("https://api.bls.gov/publicAPI/v2/timeseries/data/", payload)
    if str(data.get("status", "")).upper() != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API 失败: {data.get('message') or data.get('status')}")
    out: dict[str, pd.DataFrame] = {}
    for series in data.get("Results", {}).get("series", []):
        sid = series["seriesID"]
        rows = []
        for obs in series.get("data", []):
            per = str(obs.get("period", ""))
            if not per.startswith("M"):
                continue
            month = int(per[1:])
            year = int(obs["year"])
            try:
                val = float(obs["value"])
            except (TypeError, ValueError):
                continue
            rows.append({"year": year, "month": month, "value": val})
        out[sid] = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)
    return out


def _release_ts(
    schedule: pd.DataFrame,
    release_key: str,
    ref_year: int,
    ref_month: int,
    *,
    heuristic: str,
) -> tuple[pd.Timestamp, str]:
    """返回 (timestamp, schedule_source)。"""
    if schedule is not None and len(schedule):
        hit = schedule.loc[
            (schedule["release_key"] == release_key)
            & (schedule["ref_year"] == int(ref_year))
            & (schedule["ref_month"] == int(ref_month))
        ]
        if not hit.empty:
            row = hit.iloc[-1]
            return pd.Timestamp(row["released_at"]), str(row.get("schedule_source") or "bls_official")
    ry, rm = _add_month(ref_year, ref_month, 1)
    if heuristic == "first_friday":
        rel = _first_friday(ry, rm)
    else:
        rel = _second_wednesday(ry, rm)
    return _parse_et(rel.year, rel.month, rel.day, 8, 30), "heuristic"


def _alfred_level(first_prints: pd.DataFrame, series_id: str, year: int, month: int) -> float:
    if first_prints is None or len(first_prints) == 0:
        return float("nan")
    sub = first_prints.loc[first_prints["series_id"] == series_id]
    if sub.empty:
        return float("nan")
    # obs_date 为月初
    target = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    hit = sub.loc[sub["obs_date"] == target]
    if hit.empty:
        # 容差: 同月任意日
        hit = sub.loc[
            (sub["obs_date"].dt.year == year) & (sub["obs_date"].dt.month == month)
        ]
    if hit.empty:
        return float("nan")
    return float(hit.iloc[-1]["value"])


def _bls_numeric_events(
    series_map: dict[str, pd.DataFrame],
    schedule: pd.DataFrame,
    first_prints: pd.DataFrame,
    print_status: str,
) -> list[dict]:
    events: list[dict] = []
    use_first = print_status.startswith("first_print") and first_prints is not None and len(first_prints) > 0
    print_kind = "first_print" if use_first else "current_vintage"

    # NFP MoM from PAYEMS or CES
    nfp = series_map.get("CES0000000001")
    if nfp is not None and len(nfp) >= 2:
        for i in range(1, len(nfp)):
            y, m = int(nfp.loc[i, "year"]), int(nfp.loc[i, "month"])
            if use_first:
                cur = _alfred_level(first_prints, "PAYEMS", y, m)
                prev_lvl = _alfred_level(first_prints, "PAYEMS", *_add_month(y, m, -1))
                prev2 = _alfred_level(first_prints, "PAYEMS", *_add_month(y, m, -2))
                if not (np.isfinite(cur) and np.isfinite(prev_lvl)):
                    cur = float(nfp.loc[i, "value"])
                    prev_lvl = float(nfp.loc[i - 1, "value"])
                    prev2 = float(nfp.loc[i - 2, "value"]) if i >= 2 else np.nan
                    pk = "current_vintage"
                else:
                    pk = "first_print"
            else:
                cur = float(nfp.loc[i, "value"])
                prev_lvl = float(nfp.loc[i - 1, "value"])
                prev2 = float(nfp.loc[i - 2, "value"]) if i >= 2 else np.nan
                pk = print_kind
            actual = cur - prev_lvl
            prev = (prev_lvl - prev2) if np.isfinite(prev2) else np.nan
            ts, sched_src = _release_ts(
                schedule, "employment_situation", y, m, heuristic="first_friday",
            )
            events.append({
                "event_id": f"US|Nonfarm Payrolls|{ts.strftime('%Y%m%dT%H%M%SZ')}",
                "name": "Nonfarm Payrolls",
                "country": "US",
                "category": "employment",
                "importance": 5,
                "scheduled_at": ts,
                "released_at": ts,
                "previous": prev,
                "forecast": prev,
                "actual": actual,
                "unit": "k",
                "source": "bls" if pk == "current_vintage" else "alfred_bls",
                "print_kind": pk,
                "schedule_source": sched_src,
            })

    ur = series_map.get("LNS14000000")
    if ur is not None and len(ur) >= 1:
        for i in range(len(ur)):
            y, m = int(ur.loc[i, "year"]), int(ur.loc[i, "month"])
            if use_first:
                actual = _alfred_level(first_prints, "UNRATE", y, m)
                prev = _alfred_level(first_prints, "UNRATE", *_add_month(y, m, -1))
                pk = "first_print" if np.isfinite(actual) else "current_vintage"
                if not np.isfinite(actual):
                    actual = float(ur.loc[i, "value"])
                    prev = float(ur.loc[i - 1, "value"]) if i >= 1 else np.nan
            else:
                actual = float(ur.loc[i, "value"])
                prev = float(ur.loc[i - 1, "value"]) if i >= 1 else np.nan
                pk = print_kind
            ts, sched_src = _release_ts(
                schedule, "employment_situation", y, m, heuristic="first_friday",
            )
            events.append({
                "event_id": f"US|Unemployment Rate|{ts.strftime('%Y%m%dT%H%M%SZ')}",
                "name": "Unemployment Rate",
                "country": "US",
                "category": "employment",
                "importance": 5,
                "scheduled_at": ts,
                "released_at": ts,
                "previous": prev,
                "forecast": prev,
                "actual": actual,
                "unit": "%",
                "source": "bls" if pk == "current_vintage" else "alfred_bls",
                "print_kind": pk,
                "schedule_source": sched_src,
            })

    for sid, name, fred_id in (
        ("CUUR0000SA0", "CPI YoY", "CPIAUCSL"),
        ("CUUR0000SA0L1E", "Core CPI YoY", "CPILFESL"),
    ):
        cpi = series_map.get(sid)
        if cpi is None or len(cpi) < 13:
            continue
        # prefer alfred index for YoY if available
        lookup = {(int(r.year), int(r.month)): float(r.value) for r in cpi.itertuples()}
        if use_first:
            for r in first_prints.loc[first_prints["series_id"] == fred_id].itertuples():
                lookup[(int(r.obs_date.year), int(r.obs_date.month))] = float(r.value)
        for i in range(len(cpi)):
            y, m = int(cpi.loc[i, "year"]), int(cpi.loc[i, "month"])
            cur = lookup.get((y, m))
            base = lookup.get((y - 1, m))
            if cur is None or base in (None, 0):
                continue
            yoy = (cur / base - 1.0) * 100.0
            py, pm = _add_month(y, m, -1)
            cur_p, base_p = lookup.get((py, pm)), lookup.get((py - 1, pm))
            prev_yoy = (
                (cur_p / base_p - 1.0) * 100.0
                if cur_p is not None and base_p not in (None, 0) else np.nan
            )
            ts, sched_src = _release_ts(schedule, "cpi", y, m, heuristic="second_wednesday")
            pk = "first_print" if use_first else "current_vintage"
            events.append({
                "event_id": f"US|{name}|{ts.strftime('%Y%m%dT%H%M%SZ')}",
                "name": name,
                "country": "US",
                "category": "inflation",
                "importance": 5,
                "scheduled_at": ts,
                "released_at": ts,
                "previous": prev_yoy,
                "forecast": prev_yoy,
                "actual": yoy,
                "unit": "%",
                "source": "bls" if pk == "current_vintage" else "alfred_bls",
                "print_kind": pk,
                "schedule_source": sched_src,
            })
    return events


def _parse_fed_time(month: str, days: str, time_str: str) -> pd.Timestamp | None:
    try:
        y, m = map(int, str(month).split("-"))
    except Exception:
        return None
    day_raw = str(days or "").strip()
    if not day_raw:
        return None
    parts = re.split(r"\s*[-–—]\s*", day_raw)
    try:
        day = int(re.sub(r"[^\d]", "", parts[-1]) or "0")
    except ValueError:
        return None
    if day <= 0:
        return None
    t = (time_str or "2:00 p.m.").strip().lower()
    hour, minute = 14, 0
    mobj = re.match(r"(\d{1,2}):(\d{2})\s*([ap]\.?m\.?)", t)
    if mobj:
        hour = int(mobj.group(1))
        minute = int(mobj.group(2))
        ampm = mobj.group(3).replace(".", "")
        if ampm.startswith("p") and hour != 12:
            hour += 12
        if ampm.startswith("a") and hour == 12:
            hour = 0
    try:
        return _parse_et(y, m, day, hour, minute)
    except ValueError:
        return None


def fetch_fed_calendar_events(start: str = "2020-01-01") -> list[dict]:
    raw = _http_get("https://www.federalreserve.gov/json/calendar.json")
    payload = json.loads(raw.decode("utf-8-sig"))
    start_m = pd.Timestamp(start, tz="UTC").strftime("%Y-%m")
    events: list[dict] = []
    for e in payload.get("events", []):
        month = str(e.get("month") or "")
        if month < start_m:
            continue
        typ = str(e.get("type") or "").strip()
        title = str(e.get("title") or "").strip()
        if typ not in ("FOMC", "Speeches", "Testimony", "Beige"):
            continue
        if "Press Conference" in title and "Meeting" not in title:
            continue
        ts = _parse_fed_time(month, str(e.get("days") or ""), str(e.get("time") or ""))
        if ts is None:
            continue
        if typ == "FOMC" and "Meeting" in title:
            name, category, importance = "FOMC Rate Decision", "central_bank", 5
        elif typ == "FOMC" and "Minutes" in title:
            name, category, importance = "FOMC Minutes", "central_bank", 4
        elif typ == "Beige":
            name, category, importance = "Beige Book", "central_bank", 3
        else:
            name = title[:80] or "Fed Speech"
            category, importance = "speech", 3
        events.append({
            "event_id": f"US|{name}|{ts.strftime('%Y%m%dT%H%M%SZ')}",
            "name": name,
            "country": "US",
            "category": category,
            "importance": importance,
            "scheduled_at": ts,
            "released_at": ts,
            "previous": np.nan,
            "forecast": np.nan,
            "actual": np.nan,
            "unit": "",
            "source": "federalreserve",
            "print_kind": "n/a",
            "schedule_source": "federalreserve",
        })
    return events


def _parse_ff_number(text) -> float:
    if text is None:
        return float("nan")
    s = str(text).strip()
    if not s or s.lower() in ("null", "none", "-", "n/a"):
        return float("nan")
    s = s.replace(",", "")
    s = re.sub(r"[%KkMmBb]+$", "", s).strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def fetch_ff_thisweek_events() -> list[dict]:
    raw = _http_get("https://nfs.faireconomy.media/ff_calendar_thisweek.json")
    rows = json.loads(raw.decode("utf-8"))
    ccy_map = {
        "USD": "US", "EUR": "EU", "GBP": "GB", "JPY": "JP", "CNY": "CN",
        "AUD": "AU", "CAD": "CA", "CHF": "CH", "NZD": "NZ",
    }
    events: list[dict] = []
    for r in rows:
        impact = IMPACT_MAP.get(str(r.get("impact") or ""), 1)
        if impact < 3:
            continue
        title = str(r.get("title") or "").strip()
        if not title:
            continue
        ccy = str(r.get("country") or "").upper()
        country = ccy_map.get(ccy, ccy[:2] if ccy else "XX")
        ts = pd.Timestamp(r.get("date"))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        prev = _parse_ff_number(r.get("previous"))
        fc = _parse_ff_number(r.get("forecast"))
        act = _parse_ff_number(r.get("actual")) if "actual" in r else float("nan")
        cat = "other"
        low = title.lower()
        if any(k in low for k in ("cpi", "inflation", "pce", "ppi")):
            cat = "inflation"
        elif any(k in low for k in ("payroll", "unemployment", "nfp", "jobless", "employment")):
            cat = "employment"
        elif any(k in low for k in ("fomc", "rate decision", "interest rate")):
            cat = "central_bank"
        elif "speech" in low or "speak" in low:
            cat = "speech"
        events.append({
            "event_id": f"{country}|{title}|{ts.strftime('%Y%m%dT%H%M%SZ')}",
            "name": title[:120],
            "country": country,
            "category": cat,
            "importance": impact,
            "scheduled_at": ts,
            "released_at": ts,
            "previous": prev,
            "forecast": fc,
            "actual": act,
            "unit": "",
            "source": "forexfactory_week",
            "print_kind": "first_print" if np.isfinite(act) else "n/a",
            "schedule_source": "forexfactory",
        })
    return events


def build_macro_calendar_frame(
    *,
    start: str = "2020-01-01",
    end_year: int | None = None,
    store_dir=None,
    include_fed: bool = True,
    include_bls: bool = True,
    include_ff_week: bool = True,
    include_ff_hist: bool = True,
    refresh_bls_schedule: bool = False,
    refresh_alfred: bool = False,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    ey = int(end_year or pd.Timestamp.utcnow().year)
    from pathlib import Path
    sdir = Path(store_dir) if store_dir else Path("data/macro_calendar")

    chunks: list[dict] = []
    meta = {"print_status": "n/a", "bls_schedule_rows": 0, "ff_hist": 0}

    schedule = pd.DataFrame()
    first_prints = pd.DataFrame()
    print_status = "missing_api_key"
    ff_hist_events: list[dict] = []

    if include_ff_hist:
        ff_hist_events = fetch_ff_historical_events(
            sdir / "ff_hist_cache",
            start_year=max(2020, start_ts.year),
            end_year=min(2023, ey),
            min_importance=3,
        )
        meta["ff_hist"] = len(ff_hist_events)

    if include_bls:
        schedule = load_or_build_bls_schedule(sdir, refresh=refresh_bls_schedule)
        if ff_hist_events:
            schedule = enrich_schedule_from_ff_hist(schedule, ff_hist_events)
            # 持久化增强后的日程
            sched_path = sdir / "bls_official_releases.parquet"
            schedule.to_parquet(sched_path, engine="pyarrow", index=False)
        meta["bls_schedule_rows"] = int(len(schedule))
        first_prints, print_status = load_or_fetch_first_prints(
            sdir, refresh=refresh_alfred, observation_start="2019-01-01",
        )
        meta["print_status"] = print_status
        if print_status == "missing_api_key":
            print(
                "[macro-build] WARN: 未设置 FRED_API_KEY → BLS 数值用 current_vintage"
                " (非首印)。申请: https://fred.stlouisfed.org/docs/api/api_key.html",
                flush=True,
            )
        series = fetch_bls_monthly(start_ts.year - 1, ey)
        chunks.extend(_bls_numeric_events(series, schedule, first_prints, print_status))

    if include_fed:
        chunks.extend(fetch_fed_calendar_events(start=start))

    if include_ff_hist:
        chunks.extend(ff_hist_events)

    if include_ff_week:
        try:
            chunks.extend(fetch_ff_thisweek_events())
        except Exception as e:
            print(f"[macro-build] WARN: FF 本周拉取失败({e})", flush=True)

    if not chunks:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    df = normalize_macro_events(pd.DataFrame(chunks))
    before = len(df)
    df = dedupe_cross_source_events(df)
    if len(df) < before:
        print(f"[macro-build] dedupe: {before} → {len(df)} (-{before - len(df)})", flush=True)
    df = df.loc[pd.to_datetime(df["scheduled_at"], utc=True) >= start_ts].copy()
    # 写构建元数据
    meta_path = sdir / "build_meta.json"
    sdir.mkdir(parents=True, exist_ok=True)
    meta.update({
        "n_events": int(len(df)),
        "official_schedule_share": float(
            (df["schedule_source"] == "bls_official").mean()
        ) if len(df) else 0.0,
        "first_print_share": float(
            (df["print_kind"] == "first_print").mean()
        ) if len(df) else 0.0,
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[macro-build] meta {meta}", flush=True)
    return df.reset_index(drop=True)


def build_and_save_macro_calendar(cfg, **kwargs) -> tuple[pd.DataFrame, int]:
    from .macro_calendar import macro_store_dir, save_macro_events

    store = macro_store_dir(cfg)
    kwargs.setdefault("store_dir", store)
    df = build_macro_calendar_frame(**kwargs)
    save_macro_events(cfg, df)
    return df, len(df)
