"""从 Binance Vision CDN 回填现货 K 线(+可选资金费率)到 data/raw。

用途: 交易所 REST(api.binance.com 等)不可达时, 用官方公开历史包拿到真实 OHLCV,
供主干训练使用。写入格式与 load_symbol_data 缓存约定一致。

用法:
    python scripts/fetch_binance_vision.py
    python scripts/fetch_binance_vision.py --start 2020-01 --end 2026-07
"""
from __future__ import annotations

import argparse
import io
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from crypto_alpha.config import Config

VISION = "https://data.binance.vision/data"
SPOT_KLINE = VISION + "/spot/monthly/klines/{sym}/{tf}/{sym}-{tf}-{ym}.zip"
FUNDING = VISION + "/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip"

TF_MAP = {"1h": "1h", "4h": "4h", "1d": "1d"}


def _month_range(start: str, end: str) -> list[str]:
    s = datetime.strptime(start, "%Y-%m")
    e = datetime.strptime(end, "%Y-%m")
    out = []
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _download_zip_csv(url: str) -> pd.DataFrame | None:
    try:
        with urlopen(url, timeout=60) as resp:
            raw = resp.read()
    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            # Vision kline CSV 无表头
            df = pd.read_csv(f, header=None)
    return df


def fetch_spot_klines(symbol_id: str, timeframe: str, months: list[str]) -> pd.DataFrame:
    frames = []
    for ym in months:
        url = SPOT_KLINE.format(sym=symbol_id, tf=timeframe, ym=ym)
        df = _download_zip_csv(url)
        if df is None or len(df) == 0:
            print(f"[skip] {symbol_id} {timeframe} {ym}")
            continue
        # open_time, open, high, low, close, volume, ...
        part = df.iloc[:, :6].copy()
        part.columns = ["ts", "open", "high", "low", "close", "volume"]
        frames.append(part)
        print(f"[ok] {symbol_id} {timeframe} {ym}: {len(part)} rows")
    if not frames:
        raise RuntimeError(f"无可用 K 线: {symbol_id} {timeframe}")
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts"])
    ts = out["ts"].astype("int64").to_numpy()
    # 逐行归一到毫秒: 微秒(>1e14) //1000; 秒(<1e12) *1000
    ts = np.where(ts > 10**14, ts // 1000, ts)
    ts = np.where(ts < 10**12, ts * 1000, ts)
    out.index = pd.to_datetime(ts, unit="ms", utc=True)
    out.index.name = "timestamp"
    out = out[["open", "high", "low", "close", "volume"]].astype(float)
    # 丢掉表头误解析/异常时间戳
    out = out[(out.index.year >= 2017) & (out.index.year <= 2030)]
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def fetch_funding(symbol_id: str, months: list[str], index: pd.DatetimeIndex) -> pd.Series:
    frames = []
    for ym in months:
        url = FUNDING.format(sym=symbol_id, ym=ym)
        df = _download_zip_csv(url)
        if df is None or len(df) == 0:
            continue
        # calc_time, funding_interval_hours, last_funding_rate
        if df.shape[1] >= 3:
            part = df.iloc[:, [0, 2]].copy()
        else:
            part = df.iloc[:, [0, 1]].copy()
        part.columns = ["ts", "funding_rate"]
        frames.append(part)
        print(f"[ok] funding {symbol_id} {ym}: {len(part)}")
    if not frames:
        return pd.Series(index=index, dtype=float, name="funding_rate")
    fr = pd.concat(frames, ignore_index=True)
    fr["ts"] = pd.to_numeric(fr["ts"], errors="coerce")
    fr = fr.dropna(subset=["ts"])
    ts = fr["ts"].astype("int64").to_numpy()
    ts = np.where(ts > 10**14, ts // 1000, ts)
    ts = np.where(ts < 10**12, ts * 1000, ts)
    s = pd.Series(
        fr["funding_rate"].astype(float).values,
        index=pd.to_datetime(ts, unit="ms", utc=True),
    )
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(index, method="ffill")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default=None, help="YYYY-MM; 默认到当前月")
    args = ap.parse_args()
    end = args.end or datetime.utcnow().strftime("%Y-%m")
    months = _month_range(args.start, end)

    cfg = Config.load()
    out_dir = Path(cfg.data_dir) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    # config symbols: BTC/USDT -> BTCUSDT
    for symbol in cfg["data"]["symbols"]:
        sym_id = symbol.replace("/", "")
        main_tf = cfg["data"]["timeframe"]
        print(f"\n===== {symbol} main {main_tf} =====")
        main = fetch_spot_klines(sym_id, TF_MAP[main_tf], months)
        main["funding_rate"] = fetch_funding(sym_id, months, main.index)
        main["open_interest"] = float("nan")  # Vision 现货包无 OI; 记缺失由特征层降级
        main.attrs["data_source"] = "real"
        path = out_dir / f"{symbol.replace('/', '_')}.parquet"
        # attrs 不进 parquet; 用文件即表示真实回填。load 时标 cache。
        main.drop(columns=[], errors="ignore").to_parquet(path)
        # 重新读确保可加载
        print(f"[saved] {path} rows={len(main)} "
              f"range={main.index.min()} -> {main.index.max()} "
              f"close0={main['close'].iloc[0]:.2f} close1={main['close'].iloc[-1]:.2f}")

        for tf in cfg["data"].get("aux_timeframes") or []:
            if tf == main_tf:
                continue
            print(f"===== {symbol} aux {tf} =====")
            aux = fetch_spot_klines(sym_id, TF_MAP[tf], months)
            apath = out_dir / f"{symbol.replace('/', '_')}__{tf}.parquet"
            aux.to_parquet(apath)
            print(f"[saved] {apath} rows={len(aux)}")

    print("\n[done] Binance Vision 回填完成。后续 04 将走 cache; REST 增量失败时仍用本缓存。")


if __name__ == "__main__":
    main()
