#!/usr/bin/env python3
"""从 Binance Vision UM daily metrics 回填多年 open_interest。

公开 REST ``openInterestHist`` 仅约 30 天; Vision
``data/futures/um/daily/metrics/{SYM}/`` 提供 5m 粒度 OI 快照:
- BTCUSDT: 2020-09-01 → 近端
- ETHUSDT: 2021-12-01 → 近端

把日 zip 解压后的 ``sum_open_interest`` as-of 对齐到主周期 parquet,
只填原先为空的位置, 不覆盖 tip REST 已写的近端值。

用法:
  HTTPS_PROXY=http://127.0.0.1:7890 PYTHONPATH=src \\
    python -u scripts/20_backfill_oi_vision.py
  python -u scripts/20_backfill_oi_vision.py --start 2020-09-01 --end 2026-07-31 --workers 24
"""
from __future__ import annotations

import argparse
import http.client
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from crypto_alpha.config import Config
from crypto_alpha.data.fetch import raw_cache_path
from crypto_alpha.data.storage import load_parquet, save_parquet

VISION = "https://data.binance.vision/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day}.zip"

# Vision 各币种 metrics 起始日(更早 404)
VISION_START = {
    "BTCUSDT": date(2020, 9, 1),
    "ETHUSDT": date(2021, 12, 1),
}


def _opener(proxy: str | None):
    if proxy:
        return build_opener(ProxyHandler({"http": proxy, "https": proxy}))
    return build_opener()


def _symbol_binance(symbol: str) -> str:
    return symbol.split(":")[0].replace("/", "")


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _fetch_day_oi(
    opener, sym: str, day: date, *, retries: int = 5,
) -> pd.Series | None:
    url = VISION.format(sym=sym, day=day.isoformat())
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "crypto-alpha-oi-vision/1.0"})
            with opener.open(req, timeout=60) as resp:
                raw = resp.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    df = pd.read_csv(f)
            if "create_time" not in df.columns or "sum_open_interest" not in df.columns:
                return None
            ts = pd.to_datetime(df["create_time"], utc=True)
            vals = pd.to_numeric(df["sum_open_interest"], errors="coerce")
            s = pd.Series(vals.to_numpy(dtype=float), index=ts).dropna().sort_index()
            # 同一秒去重取最后
            s = s[~s.index.duplicated(keep="last")]
            return s
        except HTTPError as e:
            if e.code == 404:
                return None
            last_err = e
            time.sleep(0.4 * (attempt + 1))
        except (
            URLError,
            zipfile.BadZipFile,
            OSError,
            ValueError,
            TimeoutError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ) as e:
            last_err = e
            time.sleep(0.6 * (attempt + 1))
    if last_err is not None:
        print(f"[warn] {sym} {day}: {last_err}", flush=True)
    return None


def fetch_vision_oi(
    sym: str,
    start: date,
    end: date,
    *,
    proxy: str | None,
    workers: int,
    cache_dir: Path | None = None,
) -> pd.Series:
    """并发拉取 [start, end] 日包, 拼成 5m OI Series。可选落本地日缓存。"""
    opener = _opener(proxy)
    days = list(_daterange(start, end))
    parts: list[pd.Series] = []
    ok = miss = err = 0
    t0 = time.time()

    def one(d: date):
        if cache_dir is not None:
            cp = cache_dir / f"{sym}-metrics-{d.isoformat()}.parquet"
            if cp.exists():
                try:
                    df = pd.read_parquet(cp)
                    if len(df) and "oi" in df.columns:
                        s = pd.Series(
                            df["oi"].to_numpy(dtype=float),
                            index=pd.DatetimeIndex(pd.to_datetime(df["ts"], utc=True)),
                        )
                        return d, s, "cache"
                except Exception:
                    pass
        s = _fetch_day_oi(opener, sym, d)
        if s is not None and len(s) and cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"ts": s.index, "oi": s.to_numpy()}).to_parquet(
                cache_dir / f"{sym}-metrics-{d.isoformat()}.parquet", index=False,
            )
        return d, s, "net"

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = [pool.submit(one, d) for d in days]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                d, s, src = fut.result()
            except Exception as e:
                # 单日线程异常不应拖垮整批; 记 miss 后续可再跑补洞
                err += 1
                print(f"[warn] {sym} worker: {e}", flush=True)
                d, s, src = None, None, "err"
            if s is None or len(s) == 0:
                miss += 1
            else:
                parts.append(s)
                ok += 1
            if i % 100 == 0 or i == len(futs):
                print(
                    f"[vision] {sym} {i}/{len(futs)} ok={ok} miss={miss} err={err} "
                    f"elapsed={time.time()-t0:.0f}s last={d} via={src}",
                    flush=True,
                )
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts)
    # pandas DatetimeIndex.sort_values 偶发 IndexError(空/混类型); 用 i8 mergesort
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True)).as_unit("ns")
    vals = out.to_numpy(dtype=float)
    order = np.argsort(idx.asi8, kind="mergesort")
    idx = idx[order]
    vals = vals[order]
    # 去重保留最后: 丢掉「与后一项时间戳相同」的前项
    keep = np.ones(len(idx), dtype=bool)
    if len(idx) > 1:
        keep[:-1] = idx.asi8[:-1] != idx.asi8[1:]
    return pd.Series(vals[keep], index=idx[keep], dtype=float)


def align_to_panel(oi_5m: pd.Series, panel_index: pd.DatetimeIndex) -> pd.Series:
    """5m OI → 主周期: 取不超过 bar 开盘时刻的最近观测(as-of backward)。"""
    if oi_5m.empty or len(panel_index) == 0:
        return pd.Series(np.nan, index=panel_index, dtype=float)
    # 避开 pandas DatetimeTZ.sort_values 的偶发 IndexError: 全程用 i8 + searchsorted
    left_i8 = pd.DatetimeIndex(pd.to_datetime(panel_index, utc=True)).as_unit("ns").asi8
    right_idx = pd.DatetimeIndex(pd.to_datetime(oi_5m.index, utc=True)).as_unit("ns")
    right_i8 = right_idx.asi8
    right_vals = oi_5m.to_numpy(dtype=float)
    order = np.argsort(right_i8, kind="mergesort")
    right_i8 = right_i8[order]
    right_vals = right_vals[order]
    # as-of backward: 每个 left 取 ≤ t 的最右下标
    pos = np.searchsorted(right_i8, left_i8, side="right") - 1
    out = np.full(len(left_i8), np.nan, dtype=float)
    ok = pos >= 0
    out[ok] = right_vals[pos[ok]]
    return pd.Series(out, index=panel_index, dtype=float)


def backfill_symbol(
    cfg,
    symbol: str,
    *,
    start: date | None,
    end: date,
    proxy: str | None,
    workers: int,
    cache_dir: Path,
) -> dict:
    bn = _symbol_binance(symbol)
    path = raw_cache_path(cfg, symbol, cfg["data"]["timeframe"])
    if not path.exists():
        return {"symbol": symbol, "ok": False, "reason": f"missing {path}"}

    df = load_parquet(path)
    if "open_interest" not in df.columns:
        df["open_interest"] = np.nan
    before = float(df["open_interest"].notna().mean())

    vs = VISION_START.get(bn, date(2021, 1, 1))
    s0 = max(vs, start) if start else vs
    # 面板起点之后才有意义
    panel_min = pd.Timestamp(df.index.min()).tz_convert("UTC").date() if df.index.tz else pd.Timestamp(df.index.min()).tz_localize("UTC").date()
    s0 = max(s0, panel_min)

    print(f"[oi] {symbol} Vision {bn} {s0} → {end} workers={workers}", flush=True)
    oi = fetch_vision_oi(bn, s0, end, proxy=proxy, workers=workers, cache_dir=cache_dir / bn)
    if oi.empty:
        return {"symbol": symbol, "ok": False, "reason": "empty vision oi", "before_nn": before}

    aligned = align_to_panel(oi, pd.DatetimeIndex(pd.to_datetime(df.index, utc=True)))
    # 只填空位; tip REST 近端保留
    mask = df["open_interest"].isna() & aligned.notna()
    df.loc[mask, "open_interest"] = aligned.loc[mask].to_numpy()
    # 再对「已有任一观测」的前缀做 ffill, 把 5m→30m 缝抹平; 首个观测前保持 NaN
    first = df["open_interest"].first_valid_index()
    if first is not None:
        df.loc[first:, "open_interest"] = df.loc[first:, "open_interest"].ffill()

    after = float(df["open_interest"].notna().mean())
    save_parquet(df, path)
    return {
        "symbol": symbol,
        "ok": True,
        "path": str(path),
        "vision_points": int(len(oi)),
        "vision_span": f"{oi.index.min()} → {oi.index.max()}",
        "filled_rows": int(mask.sum()),
        "before_nn": round(before, 4),
        "after_nn": round(after, 4),
        "start": str(s0),
        "end": str(end),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="http://127.0.0.1:7890")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD; 默认各币种 Vision 起点")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD; 默认昨天 UTC")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument(
        "--cache-dir",
        default="data/raw/_oi_vision_cache",
        help="日包本地缓存, 便于断点续跑",
    )
    args = ap.parse_args()

    cfg = Config.load()
    end = date.fromisoformat(args.end) if args.end else (date.today() - timedelta(days=1))
    start = date.fromisoformat(args.start) if args.start else None
    proxy = args.proxy or None
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = cfg.root / cache_dir

    any_ok = False
    for symbol in cfg["data"]["symbols"]:
        info = backfill_symbol(
            cfg, symbol, start=start, end=end, proxy=proxy,
            workers=args.workers, cache_dir=cache_dir,
        )
        print(f"[done] {info}", flush=True)
        any_ok = any_ok or bool(info.get("ok"))

    print(
        "[note] Vision metrics 为 5m OI 快照 as-of 到主周期; "
        "BTC≈2020-09 起 / ETH≈2021-12 起。更早段落仍为 NaN→oi_change≈0。",
        flush=True,
    )
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
