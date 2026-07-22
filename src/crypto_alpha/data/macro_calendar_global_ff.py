"""全球宏观日历历史: ForexFactory 社区归档 CSV(含 actual/forecast/previous)。

来源: https://github.com/spoluan/forex-factory-scraper/tree/master/datasets
覆盖约 2010–2023; 近端用官方 FF 本周 JSON 补齐。

时间解释: 该归档的钟点与 Asia/Shanghai 对齐后可还原 ET 公布(如 21:30 CST = 08:30 ET)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FF_RAW_BASE = (
    "https://raw.githubusercontent.com/spoluan/forex-factory-scraper/"
    "master/datasets/forex_factory_calendar_{year}.csv"
)

CCY_MAP = {
    "USD": "US", "EUR": "EU", "GBP": "GB", "JPY": "JP", "CNY": "CN",
    "AUD": "AU", "CAD": "CA", "CHF": "CH", "NZD": "NZ",
}
IMPACT_MAP = {
    "High": 5, "Medium": 3, "Low": 1,
    "high": 5, "medium": 3, "low": 1,
    "Non-economic": 1, "Holiday": 1,
}


def _curl_bytes(url: str, timeout: float = 90.0) -> bytes:
    import os
    import subprocess

    cmd = [
        "curl", "-sL", "-A", "Mozilla/5.0 (crypto-alpha ff hist)",
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
    cmd.append(url)
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout + 5, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"download failed: {url}")
    return proc.stdout


def _parse_num(text) -> float:
    import re
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return float("nan")
    s = str(text).strip()
    if not s or s.lower() in ("null", "none", "-", "n/a", "nan"):
        return float("nan")
    s = s.replace(",", "")
    s = re.sub(r"[%KkMmBb]+$", "", s).strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _category(title: str) -> str:
    low = title.lower()
    if any(k in low for k in ("cpi", "inflation", "pce", "ppi")):
        return "inflation"
    if any(k in low for k in ("payroll", "unemployment", "nfp", "jobless", "employment", "non-farm")):
        return "employment"
    if any(k in low for k in ("fomc", "rate decision", "interest rate", "federal funds")):
        return "central_bank"
    if "speech" in low or "speak" in low or "testify" in low:
        return "speech"
    return "other"


def _parse_ff_datetime(date_s, time_s) -> pd.Timestamp | None:
    """归档时间为 Asia/Shanghai 墙钟 → UTC。"""
    ds = str(date_s).strip()
    ts = str(time_s).strip()
    if not ds or ds.lower() == "nan":
        return None
    if not ts or ts.lower() in ("all day", "nan", "tentative", ""):
        # 全日事件: 中午 CST
        try:
            local = pd.Timestamp(f"{ds} 12:00:00").tz_localize("Asia/Shanghai")
            return local.tz_convert("UTC")
        except Exception:
            return None
    # 9:30pm / 3:00am
    try:
        local = pd.to_datetime(f"{ds} {ts}", format="mixed")
        if local.tzinfo is None:
            local = local.tz_localize("Asia/Shanghai")
        return pd.Timestamp(local.tz_convert("UTC"))
    except Exception:
        try:
            local = pd.Timestamp(f"{ds} {ts}").tz_localize("Asia/Shanghai")
            return local.tz_convert("UTC")
        except Exception:
            return None


def download_ff_year_csv(year: int, cache_dir: Path) -> Path | None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"forex_factory_calendar_{year}.csv"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    url = FF_RAW_BASE.format(year=int(year))
    try:
        raw = _curl_bytes(url, timeout=120)
    except Exception as e:
        print(f"[ff-hist] WARN {year}: {e}", flush=True)
        return None
    if raw.startswith(b"404") or len(raw) < 100:
        return None
    dest.write_bytes(raw)
    return dest


def ff_csv_to_events(path: Path, *, min_importance: int = 3) -> list[dict]:
    df = pd.read_csv(path)
    events: list[dict] = []
    for r in df.itertuples(index=False):
        impact = IMPACT_MAP.get(str(getattr(r, "Impact", "")), 1)
        if impact < int(min_importance):
            continue
        title = str(getattr(r, "Event", "") or "").strip()
        if not title or title.lower() == "nan":
            continue
        ccy = str(getattr(r, "Currency", "") or "").upper()
        country = CCY_MAP.get(ccy, ccy[:2] if ccy else "XX")
        ts = _parse_ff_datetime(getattr(r, "Date", None), getattr(r, "Time", None))
        if ts is None:
            continue
        prev = _parse_num(getattr(r, "Previous", None))
        fc = _parse_num(getattr(r, "Forecast", None))
        act = _parse_num(getattr(r, "Actual", None))
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
            "source": "forexfactory_hist",
            "print_kind": "first_print" if np.isfinite(act) else "n/a",
            "schedule_source": "forexfactory",
        })
    return events


def fetch_ff_historical_events(
    cache_dir: Path,
    *,
    start_year: int = 2020,
    end_year: int = 2023,
    min_importance: int = 3,
) -> list[dict]:
    events: list[dict] = []
    for y in range(int(start_year), int(end_year) + 1):
        path = download_ff_year_csv(y, cache_dir)
        if path is None:
            continue
        part = ff_csv_to_events(path, min_importance=min_importance)
        print(f"[ff-hist] {y}: {len(part)} events (impact>={min_importance})", flush=True)
        events.extend(part)
    return events
