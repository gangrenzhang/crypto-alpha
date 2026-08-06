#!/usr/bin/env python3
"""用 cloudscraper 抓 Forex Factory 日历页, 补 HF 数据集冻结后的缺口。

HF ``Ehsanrs2/Forex_Factory_Calendar`` 停在 2025-04-07; BLS API 在本机网络
被 Akamai 403。本脚本按日抓 ``calendar?day=monD.YYYY`` HTML, 解析
actual/forecast/previous, 经 ``import_macro_events_frame`` 并入 events.parquet。

用法:
  HTTPS_PROXY=http://127.0.0.1:7890 PYTHONPATH=src \\
    python -u scripts/21_scrape_ff_calendar.py --start 2025-04-08 --end 2026-07-31
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from crypto_alpha.config import Config
from crypto_alpha.data.macro_calendar import (
    DATE_ONLY_SCHEDULE_SOURCE,
    date_only_release_ts,
    import_macro_events_frame,
    load_macro_events,
)
from crypto_alpha.data.macro_calendar_global_ff import CCY_MAP, _category, _parse_num

ET = ZoneInfo("America/New_York")

IMPACT_CLASS = {
    "icon--ff-impact-red": 5,
    "icon--ff-impact-ora": 3,
    "icon--ff-impact-yel": 1,
    "icon--ff-impact-gra": 0,
}

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _day_slug(d: date) -> str:
    # FF: day=may7.2025 / day=aug12.2024
    mon = d.strftime("%b").lower()
    return f"{mon}{d.day}.{d.year}"


def _parse_impact(td) -> int:
    if td is None:
        return 1
    span = td.select_one("span.icon")
    if span is None:
        return 1
    classes = span.get("class") or []
    for c in classes:
        if c in IMPACT_CLASS:
            return IMPACT_CLASS[c]
    return 1


def _parse_ff_time(day: date, time_s: str) -> tuple[pd.Timestamp | None, bool]:
    """FF 墙钟为 America/New_York。空时刻 → UTC 日终(防前视)。"""
    ts = (time_s or "").strip().lower()
    if not ts or ts in ("all day", "tentative", "day"):
        return date_only_release_ts(day), True
    try:
        # 12:34am / 2:01am / 8:30am / 1:15pm
        local = pd.to_datetime(f"{day.isoformat()} {ts}", format="mixed")
        if local.tzinfo is None:
            local = local.tz_localize(ET, ambiguous="NaT", nonexistent="shift_forward")
        if pd.isna(local):
            return date_only_release_ts(day), True
        return pd.Timestamp(local.tz_convert("UTC")), False
    except Exception:
        return date_only_release_ts(day), True


def parse_day_html(html: str, day: date, *, min_importance: int) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("tr.calendar__row")
    events: list[dict] = []
    carry_time = ""
    for tr in rows:
        # 日头行无 event
        title_el = tr.select_one(".calendar__event-title")
        if title_el is None:
            continue
        title = title_el.get_text(" ", strip=True)
        if not title:
            continue
        cur_td = tr.select_one("td.calendar__currency")
        ccy = (cur_td.get_text(strip=True) if cur_td else "").upper()
        if not ccy:
            continue
        time_td = tr.select_one("td.calendar__time")
        t_raw = time_td.get_text(" ", strip=True) if time_td else ""
        if t_raw:
            carry_time = t_raw
        impact = _parse_impact(tr.select_one("td.calendar__impact"))
        if impact < int(min_importance):
            continue

        def _num(sel: str) -> float:
            td = tr.select_one(sel)
            return _parse_num(td.get_text(" ", strip=True) if td else None)

        ts, date_only = _parse_ff_time(day, carry_time)
        if ts is None:
            continue
        country = CCY_MAP.get(ccy, ccy[:2] if ccy else "XX")
        act = _num("td.calendar__actual")
        fc = _num("td.calendar__forecast")
        prev = _num("td.calendar__previous")
        events.append({
            "event_id": f"{country}|{title}|{ts.strftime('%Y%m%dT%H%M%SZ')}",
            "name": title[:120],
            "country": country,
            "category": _category(title),
            "importance": impact,
            "scheduled_at": ts,
            "released_at": ts,
            "previous": prev,
            "forecast": fc,
            "actual": act,
            "unit": "",
            "source": "forexfactory_scrape",
            "print_kind": "first_print" if np.isfinite(act) else "n/a",
            "schedule_source": (
                DATE_ONLY_SCHEDULE_SOURCE if date_only else "forexfactory"
            ),
        })
    return events


def ensure_eastern_timezone(scraper) -> str:
    """把会话时区钉到 America/New_York, 避免代理 IP 落在 Asia/Tokyo 把 CPI 显示成 9:30pm。"""
    import re

    try:
        scraper.post(
            "https://www.forexfactory.com/timezone",
            data={"timezone": "America/New_York"},
            timeout=60,
        )
    except Exception as e:
        print(f"[warn] timezone POST 失败: {e}", flush=True)
    try:
        r = scraper.get("https://www.forexfactory.com/calendar", timeout=60)
        m = re.search(r"Calendar Time Zone:\s*([^<(]+)", r.text or "")
        tz_label = (m.group(1).strip() if m else "?")
    except Exception:
        tz_label = "?"
    print(f"[tz] Calendar Time Zone: {tz_label}", flush=True)
    return tz_label


def fetch_day(scraper, day: date, *, retries: int = 4) -> str | None:
    url = f"https://www.forexfactory.com/calendar?day={_day_slug(day)}"
    last = None
    for i in range(retries):
        try:
            r = scraper.get(url, timeout=90)
            if r.status_code == 200 and "calendar__row" in r.text:
                return r.text
            last = f"HTTP {r.status_code} len={len(r.text)}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * (i + 1))
    print(f"[warn] {day}: {last}", flush=True)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-04-08")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--min-importance", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=1.2, help="日与日之间休眠秒")
    ap.add_argument("--proxy", default="http://127.0.0.1:7890")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        import cloudscraper  # noqa: F401
        from bs4 import BeautifulSoup  # noqa: F401
    except ImportError:
        print("[err] 需要: pip install cloudscraper beautifulsoup4 lxml", flush=True)
        return 2

    import cloudscraper

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False},
    )
    if args.proxy:
        scraper.proxies = {"http": args.proxy, "https": args.proxy}
    ensure_eastern_timezone(scraper)

    import json

    ckpt_dir = Path("data/macro_calendar/ff_hist_cache")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ff_scrape_checkpoint.parquet"
    days_path = ckpt_dir / "ff_scrape_days.json"
    all_rows: list[dict] = []
    done_days: set[str] = set()
    if days_path.exists():
        try:
            done_days = set(json.loads(days_path.read_text(encoding="utf-8")))
        except Exception:
            done_days = set()
    if ckpt_path.exists() and not args.dry_run:
        try:
            prev = pd.read_parquet(ckpt_path)
            if len(prev):
                all_rows = prev.to_dict("records")
                print(f"[ckpt] 载入 {len(all_rows)} 条, 已抓日={len(done_days)}", flush=True)
        except Exception as e:
            print(f"[ckpt] 忽略损坏检查点: {e}", flush=True)

    def _persist():
        if args.dry_run:
            return
        if all_rows:
            pd.DataFrame(all_rows).to_parquet(ckpt_path, index=False)
        days_path.write_text(json.dumps(sorted(done_days)), encoding="utf-8")

    d = start
    n_days = (end - start).days + 1
    for i in range(n_days):
        if d.isoformat() in done_days:
            d += timedelta(days=1)
            continue
        html = fetch_day(scraper, d)
        n_part = 0
        if html:
            part = parse_day_html(html, d, min_importance=args.min_importance)
            n_part = len(part)
            all_rows.extend(part)
            done_days.add(d.isoformat())
        if (i + 1) % 7 == 0 or i == 0 or i == n_days - 1 or not html:
            print(
                f"[ff] {d} day_events={n_part} cumulative={len(all_rows)} "
                f"days={i+1}/{n_days} ok={bool(html)}",
                flush=True,
            )
        if html and ((i + 1) % 7 == 0):
            _persist()
        d += timedelta(days=1)
        time.sleep(max(0.2, float(args.sleep)))

    if not all_rows:
        print("[err] 未解析到任何事件", flush=True)
        return 1

    _persist()
    print(f"[ckpt] 已写 {ckpt_path} n={len(all_rows)} days={len(done_days)}", flush=True)

    frame = pd.DataFrame(all_rows)
    t = pd.to_datetime(frame["released_at"], utc=True)
    ready = frame["forecast"].notna() & frame["actual"].notna()
    print(
        f"[parse] n={len(frame)} {t.min()} → {t.max()} "
        f"surprise就绪={int(ready.sum())} ({ready.mean():.1%})",
        flush=True,
    )
    ym = t.dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    print(f"       月份 {ym}", flush=True)

    if args.dry_run:
        print("[dry-run] 未写库", flush=True)
        return 0

    cfg = Config.load()
    before = len(load_macro_events(cfg))
    n_new, total = import_macro_events_frame(cfg, frame, replace=False)
    print(f"[ok] 新增 event_id≈{n_new}; 库 {before} → {total}", flush=True)
    cur = load_macro_events(cfg)
    tt = pd.to_datetime(cur["released_at"], utc=True)
    e3 = cur[cur["importance"] >= 3]
    for y, g in e3.groupby(tt.loc[e3.index].dt.year):
        ok = g["forecast"].notna() & g["actual"].notna()
        print(f"  {y}: n={len(g)} surprise可算={int(ok.sum())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
