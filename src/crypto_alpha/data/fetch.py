"""数据采集: 交易所 OHLCV + 衍生品(资金费率/持仓量), 并提供合成数据兜底。

设计目的:
- 用 ccxt 统一接口拉真实数据; 但为了让整条流水线在无网络/无 API key
  的情况下也能一键跑通(便于开发与冒烟测试), 提供高保真的合成数据生成器。
- 所有输出统一为带 UTC DatetimeIndex 的 DataFrame, 列: open/high/low/close/volume,
  衍生品列(若有): funding_rate, open_interest。
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .storage import load_parquet, save_parquet

_TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _to_ms(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def fetch_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str = "1h",
    since: str = "2020-01-01T00:00:00Z",
    limit_per_call: int = 1000,
    max_calls: int = 10_000,
) -> pd.DataFrame:
    """用 ccxt 分页拉取 OHLCV。失败时抛异常(由上层决定是否降级到合成数据)。"""
    import ccxt  # 延迟导入, 避免无依赖时报错

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    tf_ms = _TF_MS[timeframe]
    since_ms = _to_ms(since)
    now_ms = int(time.time() * 1000)

    rows: list[list[float]] = []
    calls = 0
    while since_ms < now_ms and calls < max_calls:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit_per_call)
        calls += 1
        if not batch:
            break
        rows.extend(batch)
        since_ms = batch[-1][0] + tf_ms
        if len(batch) < limit_per_call:
            break

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "timestamp"
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_derivatives(exchange: str, symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """尝试拉取资金费率与持仓量, 对齐到给定索引。任何失败都优雅降级为 NaN 列。"""
    out = pd.DataFrame(index=index)
    out["funding_rate"] = np.nan
    out["open_interest"] = np.nan
    try:
        import ccxt

        ex = getattr(ccxt, exchange)({"enableRateLimit": True})
        if ex.has.get("fetchFundingRateHistory"):
            fr = ex.fetch_funding_rate_history(symbol, limit=1000)
            if fr:
                s = pd.Series(
                    {pd.to_datetime(r["timestamp"], unit="ms", utc=True): r["fundingRate"] for r in fr}
                )
                out["funding_rate"] = s.reindex(index, method="ffill")
    except Exception:
        pass  # 衍生品缺失不阻断主流程
    return out


def generate_synthetic_ohlcv(
    symbol: str,
    n_bars: int = 20_000,
    timeframe: str = "1h",
    seed: int = 42,
) -> pd.DataFrame:
    """生成带趋势/波动聚集(GARCH 味道)/regime 切换的合成价格, 用于跑通流水线。

    目的: 合成数据刻意包含可学习的结构(动量 + 波动聚集 + regime), 这样
    模型/回测/验证代码能被真实地检验, 而不是对纯随机游走束手无策。
    """
    rng = np.random.default_rng(seed + hash(symbol) % 10_000)
    tf_ms = _TF_MS[timeframe]

    # regime: 0=震荡, 1=牛, 2=熊; 用马尔可夫链切换
    trans = np.array([[0.98, 0.01, 0.01], [0.02, 0.97, 0.01], [0.02, 0.01, 0.97]])
    drift = {0: 0.0, 1: 0.0004, 2: -0.0004}
    regime = 0

    # 波动聚集
    vol = 0.01
    omega, alpha, beta = 1e-6, 0.08, 0.9

    prices = [10000.0 if "BTC" in symbol else 2000.0]
    rets = []
    for _ in range(n_bars):
        regime = rng.choice(3, p=trans[regime])
        shock = rng.standard_normal()
        vol = np.sqrt(omega + alpha * (vol * (rets[-1] if rets else 0.0)) ** 2 + beta * vol**2)
        r = drift[regime] + vol * shock
        rets.append(r)
        prices.append(prices[-1] * np.exp(r))
    prices = np.array(prices[1:])

    # 由收盘构造 OHLCV
    close = prices
    high = close * (1 + np.abs(rng.standard_normal(n_bars)) * 0.003)
    low = close * (1 - np.abs(rng.standard_normal(n_bars)) * 0.003)
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.lognormal(mean=8.0, sigma=0.5, size=n_bars)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - n_bars * tf_ms
    idx = pd.to_datetime(np.arange(start_ms, end_ms, tf_ms)[:n_bars], unit="ms", utc=True)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )
    df.index.name = "timestamp"
    # 合成衍生品: 资金费率与近期收益弱相关(模拟多头拥挤)
    df["funding_rate"] = pd.Series(rets, index=idx).rolling(8).mean().fillna(0.0) * 0.5
    df["open_interest"] = (volume.cumsum() % 1e6) + 1e5
    return df


def raw_cache_path(cfg, symbol: str):
    from pathlib import Path

    return Path(cfg.data_dir) / "raw" / (symbol.replace("/", "_") + ".parquet")


def _fetch_real(cfg, symbol: str) -> pd.DataFrame:
    d = cfg["data"]
    df = fetch_ohlcv(d["exchange"], symbol, timeframe=d["timeframe"], since=d["since"])
    if d.get("fetch_derivatives", False):
        df = df.join(fetch_derivatives(d["exchange"], symbol, df.index))
    return df


def _incremental_update(cfg, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """只拉取缓存最后一根 bar 之后的新数据并合并(多年回测的低成本增量刷新)。"""
    d = cfg["data"]
    if df is None or len(df) == 0:
        return _fetch_real(cfg, symbol)
    last = df.index[-1]
    new = fetch_ohlcv(d["exchange"], symbol, timeframe=d["timeframe"], since=last.isoformat())
    if d.get("fetch_derivatives", False) and len(new):
        new = new.join(fetch_derivatives(d["exchange"], symbol, new.index))
    if len(new) == 0:
        return df
    merged = pd.concat([df, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def load_symbol_data(cfg, symbol: str) -> pd.DataFrame:
    """按配置加载单个币种数据。

    - 合成模式: 确定性重建(不缓存, 尊重 synthetic_bars 改动)。
    - 真实模式: 优先读本地 parquet 缓存, 并按需**增量更新**(只拉新 bar);
      首次或无缓存时全量分页拉取多年历史并落盘。网络失败降级合成。
    """
    d = cfg["data"]
    if d.get("use_synthetic", True):
        return generate_synthetic_ohlcv(
            symbol, n_bars=int(d.get("synthetic_bars", 20_000)),
            timeframe=d["timeframe"], seed=cfg.seed,
        )

    cache = raw_cache_path(cfg, symbol)
    use_cache = bool(d.get("cache", True))
    if use_cache and cache.exists():
        df = load_parquet(cache)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if d.get("incremental_update", True):
            try:
                df = _incremental_update(cfg, symbol, df)
                save_parquet(df, cache)
            except Exception as e:
                print(f"[warn] 增量更新失败({e}); 使用现有缓存。")
        return df

    try:
        df = _fetch_real(cfg, symbol)
    except Exception as e:  # 网络/接口失败 -> 合成兜底
        print(f"[warn] 真实数据拉取失败 ({e}); 降级为合成数据。")
        return generate_synthetic_ohlcv(
            symbol, n_bars=int(d.get("synthetic_bars", 20_000)),
            timeframe=d["timeframe"], seed=cfg.seed,
        )
    if use_cache and len(df):
        save_parquet(df, cache)
    return df
