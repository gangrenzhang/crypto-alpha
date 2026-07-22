"""ALFRED / FRED 首印(first print)数值。

需要环境变量 ``FRED_API_KEY``(免费申请: https://fred.stlouisfed.org/docs/api/api_key.html)。

返回 DataFrame 列: series_id, obs_date(period 月初), value, realtime_start(首印日), print_kind=first_print
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# FRED 序列 ↔ 我方事件口径
FRED_SERIES = {
    "PAYEMS": {"name": "Nonfarm Payrolls", "kind": "level_thousands"},
    "UNRATE": {"name": "Unemployment Rate", "kind": "level_pct"},
    "CPIAUCSL": {"name": "CPI YoY", "kind": "index_yoy"},
    "CPILFESL": {"name": "Core CPI YoY", "kind": "index_yoy"},
}


def _curl_bytes(url: str, timeout: float = 90.0) -> bytes:
    import subprocess

    cmd = [
        "curl", "-sL", "-A", "Mozilla/5.0 (crypto-alpha alfred)",
        "--connect-timeout", "20", "--max-time", str(int(timeout)), "-k", url,
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
        cmd = cmd[:-1] + ["-x", proxy, cmd[-1]]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout + 5, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"curl failed for FRED: {url[:80]}")
    return proc.stdout


def fred_api_key() -> str | None:
    key = (os.environ.get("FRED_API_KEY") or os.environ.get("FRED_KEY") or "").strip()
    return key or None


def fetch_series_first_release(series_id: str, api_key: str,
                               observation_start: str = "2019-01-01") -> pd.DataFrame:
    """ALFRED: realtime 窗拉宽后按 obs 取最早 realtime_start 作为首印。"""
    from urllib.parse import urlencode

    # 拉全部 vintage 片段: realtime_start 很早, realtime_end 很晚
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
        "realtime_start": "1776-07-04",
        "realtime_end": "9999-12-31",
    }
    url = "https://api.stlouisfed.org/fred/series/observations?" + urlencode(params)
    raw = _curl_bytes(url, timeout=120)
    payload = json.loads(raw.decode("utf-8"))
    if "error_code" in payload:
        raise RuntimeError(f"FRED error: {payload.get('error_message')}")
    obs = payload.get("observations") or []
    if not obs:
        return pd.DataFrame(columns=["series_id", "obs_date", "value", "realtime_start", "print_kind"])

    df = pd.DataFrame(obs)
    df["obs_date"] = pd.to_datetime(df["date"], utc=True)
    df["realtime_start"] = pd.to_datetime(df["realtime_start"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value", "obs_date", "realtime_start"])
    # 每个观测期取最早 realtime_start = 首印
    df = df.sort_values(["obs_date", "realtime_start"])
    first = df.groupby("obs_date", as_index=False).first()
    first["series_id"] = series_id
    first["print_kind"] = "first_print"
    return first[["series_id", "obs_date", "value", "realtime_start", "print_kind"]]


def load_or_fetch_first_prints(
    store_dir: Path,
    *,
    refresh: bool = False,
    observation_start: str = "2019-01-01",
) -> tuple[pd.DataFrame, str]:
    """返回 (first_prints_df, status)。

    status: first_print | missing_api_key | error:...
    """
    path = Path(store_dir) / "alfred_first_prints.parquet"
    key = fred_api_key()
    if path.exists() and not refresh:
        return pd.read_parquet(path, engine="pyarrow"), "first_print_cached"
    if not key:
        if path.exists():
            return pd.read_parquet(path, engine="pyarrow"), "first_print_cached"
        return pd.DataFrame(), "missing_api_key"

    frames = []
    for sid in FRED_SERIES:
        try:
            frames.append(fetch_series_first_release(sid, key, observation_start))
            print(f"[alfred] first_print {sid}: {len(frames[-1])} obs", flush=True)
        except Exception as e:
            print(f"[alfred] WARN {sid}: {e}", flush=True)
    if not frames:
        return pd.DataFrame(), "error:no_series"
    out = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, engine="pyarrow", index=False)
    return out, "first_print"
