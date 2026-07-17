"""数据采集: 交易所 OHLCV + 衍生品(资金费率/持仓量), 并提供合成数据兜底。

设计目的:
- 用 ccxt 统一接口拉真实数据; 但为了让整条流水线在无网络/无 API key
  的情况下也能一键跑通(便于开发与冒烟测试), 提供高保真的合成数据生成器。
- 所有输出统一为带 UTC DatetimeIndex 的 DataFrame, 列: open/high/low/close/volume,
  衍生品列(若有): funding_rate, open_interest。
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .storage import load_parquet, save_parquet


def stable_symbol_offset(symbol: str, mod: int = 10_000) -> int:
    """基于 symbol 的**确定性**偏移(跨进程/机器一致)。

    不能用内置 hash(): 它对 str 带进程级随机盐(PYTHONHASHSEED), 会让"同 seed 的合成数据"
    在不同会话下不一致, 破坏可复现性。此处用 md5 派生稳定整数。
    """
    digest = hashlib.md5(symbol.encode("utf-8")).hexdigest()
    return int(digest, 16) % mod

_TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# pandas resample 规则(与交易所开盘对齐: label/closed=left)
_TF_RESAMPLE = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    """周期长度。用于「开盘时间戳 + 周期 = 收盘可用时刻」。"""
    if timeframe not in _TF_MS:
        raise ValueError(f"不支持的 timeframe: {timeframe}; 可选 {list(_TF_MS)}")
    return pd.Timedelta(milliseconds=_TF_MS[timeframe])


def timeframe_to_prefix(timeframe: str) -> str:
    """特征列前缀, 如 4h -> tf4h, 1d -> tf1d。"""
    return f"tf{timeframe}"


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


def _paginate_funding(ex, symbol: str, start_ms: int, end_ms: int, max_pages: int = 500) -> list:
    """分页拉取资金费率历史, 覆盖 [start_ms, end_ms], 避免单次 limit=1000 截断多年数据。"""
    rows, since, pages = [], int(start_ms), 0
    while pages < max_pages and since <= end_ms:
        batch = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        pages += 1
        if not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1]["timestamp"])
        if last_ts >= end_ms or len(batch) < 1000:
            break
        nxt = last_ts + 1
        if nxt <= since:  # 防护: 交易所若不前进时间戳则停止
            break
        since = nxt
    return rows


def _paginate_oi(ex, symbol: str, start_ms: int, end_ms: int, max_pages: int = 500) -> list:
    """分页拉取持仓量历史。"""
    rows, since, pages = [], int(start_ms), 0
    while pages < max_pages and since <= end_ms:
        kwargs = {"symbol": symbol, "timeframe": "1h", "since": since, "limit": 1000}
        try:
            batch = ex.fetch_open_interest_history(**kwargs)
        except TypeError:
            batch = ex.fetch_open_interest_history(symbol, "1h", since, 1000)
        pages += 1
        if not batch:
            break
        rows.extend(batch)
        ts_list = [r.get("timestamp") for r in batch if r.get("timestamp")]
        if not ts_list:
            break
        last_ts = int(max(ts_list))
        if last_ts >= end_ms or len(batch) < 1000:
            break
        nxt = last_ts + 1
        if nxt <= since:
            break
        since = nxt
    return rows


def fetch_derivatives(exchange: str, symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """尝试拉取资金费率与持仓量, 对齐到给定索引。任何失败都优雅降级为 NaN 列。

    对多年回测做分页拉取(不再单次 limit=1000 截断)。
    """
    out = pd.DataFrame(index=index)
    out["funding_rate"] = np.nan
    out["open_interest"] = np.nan
    if len(index) == 0:
        return out
    start_ms = int(pd.Timestamp(index[0]).timestamp() * 1000)
    end_ms = int(pd.Timestamp(index[-1]).timestamp() * 1000)

    try:
        import ccxt

        ex = getattr(ccxt, exchange)({"enableRateLimit": True})
        if ex.has.get("fetchFundingRateHistory"):
            fr = _paginate_funding(ex, symbol, start_ms, end_ms)
            if fr:
                s = pd.Series(
                    {pd.to_datetime(r["timestamp"], unit="ms", utc=True): r["fundingRate"] for r in fr}
                )
                out["funding_rate"] = s.reindex(index, method="ffill")
    except Exception:
        pass  # 衍生品缺失不阻断主流程

    try:
        import ccxt

        ex = getattr(ccxt, exchange)({"enableRateLimit": True})
        if ex.has.get("fetchOpenInterestHistory"):
            oi = _paginate_oi(ex, symbol, start_ms, end_ms)
            if oi:
                def _oi_val(r):
                    return r.get("openInterestAmount") or r.get("openInterestValue") or (
                        (r.get("info") or {}).get("sumOpenInterest")
                    )
                s = pd.Series(
                    {pd.to_datetime(r["timestamp"], unit="ms", utc=True): _oi_val(r) for r in oi if r.get("timestamp")}
                ).astype(float)
                out["open_interest"] = s.reindex(index, method="ffill")
    except Exception:
        pass
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
    rng = np.random.default_rng(seed + stable_symbol_offset(symbol, 10_000))
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


def raw_cache_path(cfg, symbol: str, timeframe: str | None = None):
    """主周期沿用 `SYMBOL.parquet`; 辅周期为 `SYMBOL__4h.parquet` 等, 避免互相覆盖。"""
    from pathlib import Path

    main_tf = cfg["data"]["timeframe"]
    tf = timeframe or main_tf
    base = symbol.replace("/", "_")
    name = f"{base}.parquet" if tf == main_tf else f"{base}__{tf}.parquet"
    return Path(cfg.data_dir) / "raw" / name


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """把更细周期 OHLCV 重采样到目标周期。

    约定与 ccxt 一致: 时间戳 = 该根 K 线**开盘**时刻(label/closed=left)。
    用于合成模式下由主周期派生辅周期, 保证价格路径一致(禁止独立再生成一套假行情)。
    """
    if target_tf not in _TF_RESAMPLE:
        raise ValueError(f"无法重采样到 {target_tf}")
    rule = _TF_RESAMPLE[target_tf]
    cols = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # 仅聚合存在的 OHLCV 列; 衍生品列不跨周期简单求和
    present = {k: v for k, v in cols.items() if k in df.columns}
    out = df.resample(rule, label="left", closed="left").agg(present).dropna(subset=["open", "close"])
    out.index = pd.DatetimeIndex(out.index).tz_convert("UTC") if out.index.tz else out.index.tz_localize("UTC")
    out.index.name = "timestamp"
    return out.astype(float)


def drop_incomplete_last_bar(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """若最后一根 K 线尚未收盘, 剔除之(实盘/增量场景防用到半成品 OHLC)。"""
    if df is None or len(df) == 0:
        return df
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    last_open = df.index[-1]
    available_at = last_open + timeframe_delta(timeframe)
    if available_at > now:
        return df.iloc[:-1]
    return df


def _fetch_real(cfg, symbol: str, timeframe: str) -> pd.DataFrame:
    d = cfg["data"]
    df = fetch_ohlcv(d["exchange"], symbol, timeframe=timeframe, since=d["since"])
    # 衍生品只挂在主周期面板上(辅周期不重复拉, 避免接口浪费与错位)
    if timeframe == d["timeframe"] and d.get("fetch_derivatives", False):
        df = df.join(fetch_derivatives(d["exchange"], symbol, df.index))
    return drop_incomplete_last_bar(df, timeframe)


def _incremental_update(cfg, symbol: str, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """只拉取缓存最后一根 bar 之后的新数据并合并(多年回测的低成本增量刷新)。"""
    d = cfg["data"]
    if df is None or len(df) == 0:
        return _fetch_real(cfg, symbol, timeframe)
    last = df.index[-1]
    new = fetch_ohlcv(d["exchange"], symbol, timeframe=timeframe, since=last.isoformat())
    if timeframe == d["timeframe"] and d.get("fetch_derivatives", False) and len(new):
        new = new.join(fetch_derivatives(d["exchange"], symbol, new.index))
    if len(new) == 0:
        return drop_incomplete_last_bar(df, timeframe)
    merged = pd.concat([df, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return drop_incomplete_last_bar(merged, timeframe)


def _load_synthetic(cfg, symbol: str, timeframe: str) -> pd.DataFrame:
    """合成数据: 始终先生成主周期, 辅周期由主周期重采样得到(路径一致)。"""
    d = cfg["data"]
    main_tf = d["timeframe"]
    main = generate_synthetic_ohlcv(
        symbol,
        n_bars=int(d.get("synthetic_bars", 20_000)),
        timeframe=main_tf,
        seed=cfg.seed,
    )
    if timeframe == main_tf:
        return main
    # 辅周期更粗: 重采样; 若误配更细周期, 拒绝静默生成错误数据
    if timeframe_delta(timeframe) < timeframe_delta(main_tf):
        raise ValueError(
            f"合成辅周期 {timeframe} 细于主周期 {main_tf}; "
            f"方案B要求辅周期为更高周期上下文。"
        )
    return resample_ohlcv(main, timeframe)


def _tag_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """在 DataFrame.attrs 标记数据来源, 供看板/审计读取(不改变列数据)。"""
    out = df
    try:
        out.attrs["data_source"] = source
    except Exception:
        pass
    return out


def load_symbol_data(cfg, symbol: str, timeframe: str | None = None) -> pd.DataFrame:
    """按配置加载单个币种、指定周期的 OHLCV。

    - 合成模式: 主周期确定性生成; 辅周期由主周期 resample(价格路径一致, 不缓存)。
    - 真实模式: 每周期独立 parquet 缓存 + 增量更新; 网络失败时主周期可降级合成
      (``synthetic_fallback``); 此时 ``load_aux_timeframes`` 会强制从主面板重采样辅周期,
      辅周期独立拉取失败则跳过(不拖垮主流程)。
    - 产物 `df.attrs["data_source"]` ∈ {synthetic, real, cache, synthetic_fallback}。
    """
    d = cfg["data"]
    main_tf = d["timeframe"]
    tf = timeframe or main_tf

    if d.get("use_synthetic", False):
        return _tag_source(_load_synthetic(cfg, symbol, tf), "synthetic")

    cache = raw_cache_path(cfg, symbol, tf)
    use_cache = bool(d.get("cache", True))
    if use_cache and cache.exists():
        df = load_parquet(cache)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if d.get("incremental_update", True):
            try:
                df = _incremental_update(cfg, symbol, df, tf)
                save_parquet(df, cache)
            except Exception as e:
                print(f"[warn] {symbol} {tf} 增量更新失败({e}); 使用现有缓存。")
                df = drop_incomplete_last_bar(df, tf)
        else:
            df = drop_incomplete_last_bar(df, tf)
        return _tag_source(df, "cache")

    try:
        df = _fetch_real(cfg, symbol, tf)
    except Exception as e:
        if tf == main_tf:
            allow = bool(d.get("allow_synthetic_fallback", True))
            print(
                f"[warn] 真实数据拉取失败 ({e}); "
                + ("降级为合成数据(data_source=synthetic_fallback)。" if allow else "且禁止降级。")
            )
            if not allow:
                raise
            return _tag_source(_load_synthetic(cfg, symbol, tf), "synthetic_fallback")
        raise
    if use_cache and len(df):
        save_parquet(df, cache)
    return _tag_source(df, "real")


def _main_requires_aux_resample(main_df: pd.DataFrame | None, use_synthetic: bool) -> bool:
    """主路径为合成(含 synthetic_fallback)时, 辅周期必须从 main 重采样。

    避免主面板已降级合成、辅周期仍命中真实缓存/拉数导致价格路径混用。
    """
    if use_synthetic:
        return main_df is not None
    if main_df is None:
        return False
    src = str(getattr(main_df, "attrs", {}).get("data_source", "") or "")
    return src in {"synthetic", "synthetic_fallback"}


def load_aux_timeframes(
    cfg,
    symbol: str,
    main_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """加载配置中的辅周期 OHLCV 字典 `{tf: df}`。

    - 跳过与主周期相同的项、空列表、细于主周期的项;
    - ``use_synthetic`` 或主 ``data_source`` ∈ {synthetic, synthetic_fallback} 时,
      **强制**用 ``main_df`` 重采样(价格路径与主面板一致, 禁止混入真实辅周期);
    - 单个辅周期失败不阻断: 打印 warn 并跳过。
    """
    d = cfg["data"]
    main_tf = d["timeframe"]
    aux_list = list(d.get("aux_timeframes") or [])
    out: dict[str, pd.DataFrame] = {}
    force_resample = _main_requires_aux_resample(main_df, bool(d.get("use_synthetic", False)))
    for tf in aux_list:
        if not tf or tf == main_tf:
            continue
        try:
            if timeframe_delta(tf) < timeframe_delta(main_tf):
                print(f"[warn] 跳过辅周期 {tf}: 细于主周期 {main_tf}。")
                continue
            if force_resample:
                aux = resample_ohlcv(main_df, tf)
                src = str(getattr(main_df, "attrs", {}).get("data_source", "synthetic") or "synthetic")
                out[tf] = _tag_source(aux, src)
                if src == "synthetic_fallback":
                    print(
                        f"[warn] 主行情为 synthetic_fallback; 辅周期 {tf} 已从主面板重采样"
                        f"(避免混入真实高周期)。"
                    )
            else:
                out[tf] = load_symbol_data(cfg, symbol, timeframe=tf)
        except Exception as e:
            print(f"[warn] 辅周期 {symbol} {tf} 加载失败({e}); 跳过。")
    return out
