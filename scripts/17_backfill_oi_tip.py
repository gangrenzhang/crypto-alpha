"""用 Binance 期货公开 OI 历史回填主周期 parquet 的 open_interest 列。

公开接口仅约最近 30 天(再早 startTime 会 400)。对多年训练窗无法补全历史;
本脚本只把**可拿到的近端**写回缓存, 消除 tip 段全 NaN, 并打印覆盖率。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from crypto_alpha.config import Config
from crypto_alpha.data.fetch import raw_cache_path, timeframe_delta
from crypto_alpha.data.storage import load_parquet, save_parquet


def _opener(proxy: str | None):
    if proxy:
        h = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        return urllib.request.build_opener(h)
    return urllib.request.build_opener()


def fetch_oi_hist(
    symbol_binance: str,
    period: str,
    *,
    proxy: str | None,
    limit: int = 500,
) -> pd.Series:
    """拉取 Binance 公开 OI 历史(仅近端)。返回 UTC 索引 Series。"""
    opener = _opener(proxy)
    url = (
        "https://fapi.binance.com/futures/data/openInterestHist"
        f"?symbol={symbol_binance}&period={period}&limit={limit}"
    )
    with opener.open(url, timeout=60) as r:
        rows = json.loads(r.read().decode())
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([int(x["timestamp"]) for x in rows], unit="ms", utc=True)
    vals = [float(x.get("sumOpenInterest") or x.get("sumOpenInterestValue") or np.nan) for x in rows]
    return pd.Series(vals, index=idx, dtype=float).sort_index()


def backfill_symbol(cfg, symbol: str, *, proxy: str | None, period: str) -> dict:
    path = raw_cache_path(cfg, symbol, cfg["data"]["timeframe"])
    if not path.exists():
        return {"symbol": symbol, "ok": False, "reason": f"missing {path}"}
    df = load_parquet(path)
    if "open_interest" not in df.columns:
        df["open_interest"] = np.nan
    before_nn = float(df["open_interest"].notna().mean())

    bn = symbol.replace("/", "").replace(":USDT", "")  # BTC/USDT → BTCUSDT
    if not bn.endswith("USDT"):
        bn = bn + "USDT" if "USDT" not in bn else bn
    # BTC/USDT → BTCUSDT
    bn = symbol.split(":")[0].replace("/", "")

    s = fetch_oi_hist(bn, period, proxy=proxy)
    if len(s) == 0:
        return {"symbol": symbol, "ok": False, "reason": "empty oi hist", "before_nn": before_nn}

    # as-of 对齐到主面板索引
    aligned = s.reindex(df.index, method="ffill")
    # 只填原先为空的位置, 不覆盖已有非空(若将来有更长历史源)
    mask = df["open_interest"].isna() & aligned.notna()
    df.loc[mask, "open_interest"] = aligned.loc[mask]
    # 再 ffill 一次覆盖对齐缝
    df["open_interest"] = df["open_interest"].ffill()
    after_nn = float(df["open_interest"].notna().mean())
    save_parquet(df, path)
    return {
        "symbol": symbol,
        "ok": True,
        "path": str(path),
        "oi_points": int(len(s)),
        "oi_span": f"{s.index.min()} → {s.index.max()}",
        "before_nn": before_nn,
        "after_nn": after_nn,
        "filled_rows": int(mask.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="http://127.0.0.1:7890")
    ap.add_argument("--period", default=None, help="默认跟随主周期(30m/1h/…)")
    args = ap.parse_args()
    cfg = Config.load()
    period = args.period or str(cfg["data"]["timeframe"])
    # Binance 支持 5m 15m 30m 1h 2h 4h 6h 12h 1d
    if period not in {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}:
        period = "1h"
        print(f"[warn] 主周期不在 Binance OI 支持列表, 回退 period={period}", flush=True)

    proxy = args.proxy or None
    print(f"[oi] period={period} proxy={proxy}", flush=True)
    any_ok = False
    for symbol in cfg["data"]["symbols"]:
        info = backfill_symbol(cfg, symbol, proxy=proxy, period=period)
        print(f"[oi] {info}", flush=True)
        any_ok = any_ok or bool(info.get("ok"))
        time.sleep(0.3)
    print(
        "[note] 公开 OI 历史仅约 30 天; 多年训练窗仍会大量 NaN→特征填 0。"
        "完整历史需付费/自建采集。",
        flush=True,
    )
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
