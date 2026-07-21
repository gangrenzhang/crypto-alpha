"""数据采集: 交易所 OHLCV + 衍生品(资金费率/持仓量/清算), 并提供合成数据兜底。

设计目的:
- 用 ccxt 统一接口拉真实数据; 但为了让整条流水线在无网络/无 API key
  的情况下也能一键跑通(便于开发与冒烟测试), 提供高保真的合成数据生成器。
- 所有输出统一为带 UTC DatetimeIndex 的 DataFrame, 列: open/high/low/close/volume,
  衍生品列(若有): funding_rate, open_interest, liq_long, liq_short。
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
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# pandas resample / Grouper 规则(与交易所开盘对齐: label/closed=left)
# 注意: 不可把 "30m" 直接交给 Grouper——pandas offset 里 m/M 表示月, 分钟须用 min
_TF_RESAMPLE = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1D",
}


def supported_timeframes() -> list[str]:
    return list(_TF_MS.keys())


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    """周期长度。用于「开盘时间戳 + 周期 = 收盘可用时刻」。"""
    if timeframe not in _TF_MS:
        raise ValueError(f"不支持的 timeframe: {timeframe}; 可选 {list(_TF_MS)}")
    return pd.Timedelta(milliseconds=_TF_MS[timeframe])


def timeframe_to_pandas_freq(timeframe: str) -> str:
    """ccxt 风格周期 → pandas 安全 freq(如 30m → 30min)。"""
    if timeframe in _TF_RESAMPLE:
        return _TF_RESAMPLE[timeframe]
    # 已是 pandas 写法(如 30min)则原样返回
    return str(timeframe)


def timeframe_to_prefix(timeframe: str) -> str:
    """特征列前缀, 如 4h -> tf4h, 1d -> tf1d, 30m -> tf30m。"""
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
    timeout_ms: int = 15_000,
    enable_rate_limit: bool = True,
) -> pd.DataFrame:
    """用 ccxt 分页拉取 OHLCV。失败时抛异常(由上层决定是否降级到合成数据)。"""
    import ccxt  # 延迟导入, 避免无依赖时报错

    ex = getattr(ccxt, exchange)({
        "enableRateLimit": bool(enable_rate_limit),
        "timeout": int(timeout_ms),
    })
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


def exchange_candidates(cfg, *, for_tip: bool = False) -> list[str]:
    """主交易所 + 备用列表(去重保序)。

    ``for_tip=True``(决策增量): 优先 ``tip_exchange`` / fallbacks, 再试主所 —
    避免主所 REST 长时间超时拖死「刷到当下」路径。
    """
    d = cfg["data"]
    primary = str(d.get("exchange") or "binance")
    tip = str(d.get("tip_exchange") or "").strip()
    fallbacks = [str(x).strip() for x in list(d.get("exchange_fallbacks") or []) if str(x).strip()]
    if for_tip:
        ordered = ([tip] if tip else []) + fallbacks + [primary]
    else:
        ordered = [primary, *fallbacks]
        if tip:
            ordered.append(tip)
    out: list[str] = []
    for name in ordered:
        n = str(name or "").strip()
        if n and n not in out:
            out.append(n)
    return out


def fetch_ohlcv_resilient(
    cfg,
    symbol: str,
    timeframe: str,
    since: str,
    *,
    max_calls: int = 10_000,
    for_tip: bool = False,
) -> tuple[pd.DataFrame, str]:
    """按 exchange_candidates 依次尝试拉取 OHLCV, 返回 (df, 实际使用的交易所)。"""
    errors: list[str] = []
    d = cfg["data"]
    timeout_ms = int(d.get("rest_timeout_ms", 8_000 if for_tip else 15_000))
    # tip 增量通常只有数百根, 关 rateLimit 显著加速(仍受 timeout 保护)
    rate_limit = bool(d.get("rest_rate_limit", not for_tip))
    for ex_name in exchange_candidates(cfg, for_tip=for_tip):
        try:
            df = fetch_ohlcv(
                ex_name,
                symbol,
                timeframe=timeframe,
                since=since,
                max_calls=max_calls,
                timeout_ms=timeout_ms,
                enable_rate_limit=rate_limit,
            )
            return df, ex_name
        except Exception as e:
            errors.append(f"{ex_name}: {e}")
            print(f"[warn] OHLCV {ex_name} 失败, 尝试下一所: {e}")
    raise RuntimeError("全部交易所 OHLCV 拉取失败: " + "; ".join(errors))


def closed_bar_lag(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.Timedelta:
    """相对「最后一根已收盘 bar」的落后时长。df 应已 drop_incomplete_last_bar。"""
    if df is None or len(df) == 0:
        return pd.Timedelta.max
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    closed_at = df.index[-1] + timeframe_delta(timeframe)
    lag = now - closed_at
    return lag if lag > pd.Timedelta(0) else pd.Timedelta(0)


def assert_fresh_enough(
    df: pd.DataFrame,
    timeframe: str,
    *,
    max_lag_bars: int = 2,
    now: pd.Timestamp | None = None,
    context: str = "",
) -> None:
    """若最后一根已收盘 bar 落后超过 max_lag_bars 根主周期, 抛错。"""
    lag = closed_bar_lag(df, timeframe, now=now)
    max_lag = max(1, int(max_lag_bars)) * timeframe_delta(timeframe)
    if lag > max_lag:
        last = df.index[-1] if df is not None and len(df) else None
        raise RuntimeError(
            f"{context}行情不够新: 最后 bar_open={last}, lag={lag}, "
            f"允许最大 lag={max_lag} ({max_lag_bars} 根 {timeframe})。"
            "请检查网络/交易所可达性后重试。"
        )


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


def _swap_symbol_candidates(symbol: str) -> list[str]:
    """现货风格 BTC/USDT → 另试永续 BTC/USDT:USDT(清算接口常用统一合约符)。"""
    s = str(symbol or "").strip()
    out = [s] if s else []
    if s and ":" not in s and "/" in s:
        base, quote = s.split("/", 1)
        alt = f"{base}/{quote}:{quote}"
        if alt not in out:
            out.append(alt)
    return out


def _liq_notional(row: dict) -> float:
    """清算名义额(优先 quoteValue)。失败返回 nan。"""
    for k in ("quoteValue", "cost", "notional"):
        v = row.get(k)
        if v is not None:
            try:
                x = float(v)
                if np.isfinite(x):
                    return x
            except (TypeError, ValueError):
                pass
    price, amount = row.get("price"), row.get("amount")
    if amount is None:
        amount = row.get("contracts")
    try:
        if price is not None and amount is not None:
            x = float(price) * float(amount)
            if np.isfinite(x):
                return abs(x)
    except (TypeError, ValueError):
        pass
    return float("nan")


def _liq_bucket(side) -> str | None:
    """清算订单 side → 仓位桶。

    约定与 Binance forceOrder 一致(订单方向):
    - SELL = 多头被强平(被动卖出) → liq_long
    - BUY  = 空头被强平(被动买入) → liq_short
    缺 side / 歧义值时返回 None(丢弃该笔, 避免污染不平衡)。
    """
    s = str(side or "").strip().lower()
    if s in ("sell", "seller"):
        return "long"
    if s in ("buy", "buyer"):
        return "short"
    return None


def _parse_liq_ts(ts_raw) -> pd.Timestamp | None:
    """清算事件时间戳 → UTC Timestamp; 无法解析返回 None。"""
    try:
        if isinstance(ts_raw, pd.Timestamp):
            ts = ts_raw
        elif isinstance(ts_raw, (int, float, np.integer, np.floating)):
            v = float(ts_raw)
            # ccxt 一般为 ms; 若数值像秒级则用 s(防误把秒当 ms 甩到 1970)
            unit = "ms" if v >= 1e11 else "s"
            ts = pd.Timestamp(v, unit=unit, tz="UTC")
        else:
            ts = pd.to_datetime(ts_raw, utc=True)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


def _liq_exchange_fallbacks(primary: str) -> list[str]:
    """清算常需 U 本位合约所; 在主所无数据时另试映射所(不影响 funding/OI 所用实例)。"""
    p = str(primary or "").strip()
    mapped = {
        "binance": "binanceusdm",
        "binancecoinm": "binanceusdm",
    }
    out = [p] if p else []
    alt = mapped.get(p.lower())
    if alt and alt not in out:
        out.append(alt)
    return out


def _paginate_liquidations(
    ex, symbol: str, start_ms: int, end_ms: int, max_pages: int = 200,
) -> list:
    """分页拉取公开清算; 不支持/失败返回空列表(由上层降级 NaN)。

    注意: 多数所 REST 仅覆盖**近端**清算, 多年全历史常不完整 → 聚合后会把
    「首笔之前」置 NaN, 并由特征层记 ``derivatives_liquidations_sparse`` /
    ``derivatives_liquidations_unavailable``。
    """
    if not (getattr(ex, "has", None) or {}).get("fetchLiquidations"):
        return []
    fetch_fn = getattr(ex, "fetch_liquidations", None) or getattr(ex, "fetchLiquidations", None)
    if fetch_fn is None:
        return []

    for sym in _swap_symbol_candidates(symbol):
        rows: list = []
        since, pages = int(start_ms), 0
        try:
            while pages < max_pages and since <= end_ms:
                batch = fetch_fn(sym, since=since, limit=1000)
                pages += 1
                if not batch:
                    break
                rows.extend(batch)
                ts_list = [int(r["timestamp"]) for r in batch if r.get("timestamp") is not None]
                if not ts_list:
                    break
                last_ts = max(ts_list)
                if last_ts >= end_ms or len(batch) < 1000:
                    break
                nxt = last_ts + 1
                if nxt <= since:
                    break
                since = nxt
            if rows:
                return rows
        except Exception:
            continue
    return []


def _aggregate_liquidations(
    rows: list, index: pd.DatetimeIndex, bar_delta: pd.Timedelta,
) -> tuple[pd.Series, pd.Series]:
    """事件 → 主周期 bar 桶内名义额合计(开盘时刻索引)。

    事件时刻 τ 归入开盘 t 满足 t ≤ τ < t+Δ 的 bar(与 OHLCV volume 同属当根已收盘信息)。
    决策时刻为 t+Δ 时可用该根合计 → 无前视。

    有事件时: 首笔活动 bar **之前**置 NaN(未知历史, 禁止把缺史当成「零清算」);
    首笔至末笔之间无事件的 bar 记 0(真安静); 无任何成功事件时由调用方保持全 NaN。
    """
    long_a = np.zeros(len(index), dtype=float)
    short_a = np.zeros(len(index), dtype=float)
    if len(index) == 0:
        return (
            pd.Series(long_a, index=index, dtype=float),
            pd.Series(short_a, index=index, dtype=float),
        )
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    delta = pd.Timedelta(bar_delta)
    for r in rows:
        ts = _parse_liq_ts(r.get("timestamp"))
        if ts is None:
            continue
        pos = int(idx.searchsorted(ts, side="right") - 1)
        if pos < 0:
            continue
        # 超出该 bar 收盘 → 不属于已收盘面板(防把未完成 bar 或未来事件计入)
        if ts >= idx[pos] + delta:
            continue
        bucket = _liq_bucket(r.get("side"))
        notion = _liq_notional(r)
        if bucket is None or not np.isfinite(notion) or notion <= 0:
            continue
        if bucket == "long":
            long_a[pos] += notion
        else:
            short_a[pos] += notion
    active = (long_a > 0) | (short_a > 0)
    if active.any():
        first_i = int(np.flatnonzero(active)[0])
        if first_i > 0:
            long_a[:first_i] = np.nan
            short_a[:first_i] = np.nan
    return (
        pd.Series(long_a, index=index, dtype=float),
        pd.Series(short_a, index=index, dtype=float),
    )


def fetch_derivatives(
    exchange: str,
    symbol: str,
    index: pd.DatetimeIndex,
    *,
    include_liquidations: bool = True,
    bar_delta: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """尝试拉取资金费率、持仓量与(可选)清算, 对齐到给定索引。

    任何一路失败都优雅降级为 NaN 列, **互不影响**(funding / OI / liquidations 各自 try)。
    对多年回测做分页拉取(不再单次 limit=1000 截断)。

    ``include_liquidations=False`` 时仍写出 ``liq_long``/``liq_short`` 全 NaN,
    便于特征层统一降级, 且不改变旧调用方对 funding/OI 的行为。
    """
    out = pd.DataFrame(index=index)
    out["funding_rate"] = np.nan
    out["open_interest"] = np.nan
    out["liq_long"] = np.nan
    out["liq_short"] = np.nan
    if len(index) == 0:
        return out
    start_ms = int(pd.Timestamp(index[0]).timestamp() * 1000)
    end_ms = int(pd.Timestamp(index[-1]).timestamp() * 1000)

    try:
        import ccxt

        ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    except Exception:
        return out  # 无 ccxt / 无法建所 → 列保持 NaN

    try:
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

    # --- 清算: 独立 try, 失败不影响 funding/OI; 主所无数据时另试合约所映射 ---
    if include_liquidations:
        try:
            if bar_delta is None:
                if len(index) >= 2:
                    bar_delta = pd.Timedelta(pd.Series(index).diff().median())
                else:
                    bar_delta = pd.Timedelta(hours=1)
            liq_rows: list = []
            if not pd.isna(bar_delta) and bar_delta > pd.Timedelta(0):
                liq_rows = _paginate_liquidations(ex, symbol, start_ms, end_ms)
                if not liq_rows:
                    import ccxt as _ccxt_mod

                    for alt_name in _liq_exchange_fallbacks(exchange)[1:]:
                        try:
                            alt_ex = getattr(_ccxt_mod, alt_name)({"enableRateLimit": True})
                        except Exception:
                            continue
                        liq_rows = _paginate_liquidations(alt_ex, symbol, start_ms, end_ms)
                        if liq_rows:
                            break
                if liq_rows:
                    lng, sht = _aggregate_liquidations(liq_rows, index, bar_delta)
                    out["liq_long"] = lng
                    out["liq_short"] = sht
        except Exception:
            pass
    return out


def ensure_liquidation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """旧缓存可能无清算列: 补 NaN 列, 不改已有值(特征层可统一降级)。"""
    if df is None or len(df.columns) == 0:
        return df
    out = df
    for col in ("liq_long", "liq_short"):
        if col not in out.columns:
            out = out.copy()
            out[col] = np.nan
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
    # 合成清算: 下跌时多头清算、上涨时空头清算(弱相关, 仅供冒烟/离线路径有列)
    rets_s = pd.Series(rets, index=idx)
    shock = volume * np.abs(rets_s)
    df["liq_long"] = shock.where(rets_s < 0, 0.0).fillna(0.0)
    df["liq_short"] = shock.where(rets_s > 0, 0.0).fillna(0.0)
    return df


def raw_cache_path(cfg, symbol: str, timeframe: str | None = None):
    """统一 `SYMBOL__{tf}.parquet`(含主周期)。

    旧版主周期无后缀 `SYMBOL.parquet` 仅通过 ``resolve_raw_cache_path`` 在
    ``tf==1h`` 时兼容读取, **禁止** 30m 等周期误读 1h 文件。
    """
    from pathlib import Path

    tf = timeframe or cfg["data"]["timeframe"]
    base = symbol.replace("/", "_")
    return Path(cfg.data_dir) / "raw" / f"{base}__{tf}.parquet"


def resolve_raw_cache_path(cfg, symbol: str, timeframe: str | None = None):
    """解析实际可读缓存路径: 优先带周期后缀; 仅 1h 回退无后缀遗留文件。"""
    from pathlib import Path

    preferred = raw_cache_path(cfg, symbol, timeframe)
    if preferred.exists():
        return preferred
    tf = timeframe or cfg["data"]["timeframe"]
    if tf == "1h":
        legacy = Path(cfg.data_dir) / "raw" / f"{symbol.replace('/', '_')}.parquet"
        if legacy.exists():
            return legacy
    return preferred


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
    """若尾部 K 线尚未收盘, 逐根剔除(实盘/增量场景防用到半成品 OHLC)。"""
    if df is None or len(df) == 0:
        return df
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    out = df
    delta = timeframe_delta(timeframe)
    while len(out) and (out.index[-1] + delta) > now:
        out = out.iloc[:-1]
    return out


def _fetch_real(cfg, symbol: str, timeframe: str) -> pd.DataFrame:
    d = cfg["data"]
    df, ex_used = fetch_ohlcv_resilient(
        cfg, symbol, timeframe, since=d["since"], for_tip=False
    )
    # 衍生品只挂在主周期面板上(辅周期不重复拉, 避免接口浪费与错位)
    if timeframe == d["timeframe"] and d.get("fetch_derivatives", False):
        try:
            df = df.join(
                fetch_derivatives(
                    ex_used,
                    symbol,
                    df.index,
                    include_liquidations=bool(d.get("fetch_liquidations", True)),
                    bar_delta=timeframe_delta(timeframe),
                )
            )
        except Exception as e:
            print(f"[warn] {symbol} 衍生品拉取失败({ex_used}: {e}); 继续仅用 OHLCV。")
    if timeframe == d["timeframe"]:
        df = ensure_liquidation_columns(df)
    out = drop_incomplete_last_bar(df, timeframe)
    try:
        out.attrs["exchange_used"] = ex_used
    except Exception:
        pass
    return out


def _incremental_update(cfg, symbol: str, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """只拉取缓存最后一根 bar 之后的新数据并合并(多年回测的低成本增量刷新)。"""
    d = cfg["data"]
    if df is None or len(df) == 0:
        return _fetch_real(cfg, symbol, timeframe)
    last = df.index[-1]
    # 从最后一根开盘时刻重拉, 覆盖可能尚未定稿/被修正的 tip, 再与缓存合并
    new, ex_used = fetch_ohlcv_resilient(
        cfg,
        symbol,
        timeframe,
        since=last.isoformat(),
        max_calls=50,
        for_tip=True,
    )
    # tip 增量默认不拉衍生品(慢且易失败); 新 tip 的 funding/OI/清算可为 NaN, 特征侧已 fillna
    pull_deriv = bool(d.get("fetch_derivatives", False)) and bool(
        d.get("fetch_derivatives_on_tip", False)
    )
    if timeframe == d["timeframe"] and pull_deriv and len(new):
        try:
            new = new.join(
                fetch_derivatives(
                    ex_used,
                    symbol,
                    new.index,
                    include_liquidations=bool(d.get("fetch_liquidations", True)),
                    bar_delta=timeframe_delta(timeframe),
                )
            )
        except Exception as e:
            print(f"[warn] {symbol} 增量衍生品失败({ex_used}: {e}); 继续仅用 OHLCV。")
    if len(new) == 0:
        out = drop_incomplete_last_bar(df, timeframe)
    else:
        merged = pd.concat([df, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        # tip 默认不重拉衍生品:
        # - funding/OI 为**状态**量 → ffill 合理
        # - liq_* 为**流量**(当根名义额) → 禁止 ffill(否则把上根爆仓额复制到新 tip, 假信号)
        #   新 tip 无重拉时保持 NaN, 由特征层 fillna(0) 降级
        for col in ("funding_rate", "open_interest"):
            if col in merged.columns:
                merged[col] = merged[col].ffill()
        if timeframe == d["timeframe"]:
            merged = ensure_liquidation_columns(merged)
        out = drop_incomplete_last_bar(merged, timeframe)
    try:
        out.attrs["exchange_used"] = ex_used
        primary = str(d.get("exchange") or "")
        if ex_used and primary and ex_used != primary:
            out.attrs["tip_exchange_mismatch"] = True
    except Exception:
        pass
    return out


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


def load_symbol_data(
    cfg,
    symbol: str,
    timeframe: str | None = None,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """按配置加载单个币种、指定周期的 OHLCV。

    - 合成模式: 主周期确定性生成; 辅周期由主周期 resample(价格路径一致, 不缓存)。
    - 真实模式: 每周期独立 parquet 缓存 + 增量更新; 网络失败时主周期可降级合成
      (``synthetic_fallback``); 此时 ``load_aux_timeframes`` 会强制从主面板重采样辅周期,
      辅周期独立拉取失败则跳过(不拖垮主流程)。
    - ``force_refresh=True``: 无视 ``incremental_update`` 开关, 必须增量到当下已收盘 tip
      (决策路径用; 训练/回测可继续读冷缓存)。
    - 产物 `df.attrs["data_source"]` ∈ {synthetic, real, cache, synthetic_fallback}。
    """
    d = cfg["data"]
    main_tf = d["timeframe"]
    tf = timeframe or main_tf

    if d.get("use_synthetic", False):
        return _tag_source(_load_synthetic(cfg, symbol, tf), "synthetic")

    cache_read = resolve_raw_cache_path(cfg, symbol, tf)
    cache_write = raw_cache_path(cfg, symbol, tf)
    use_cache = bool(d.get("cache", True))
    # 缺省 False: 与 config.yaml / 架构「训练读冷缓存」一致; 决策用 force_refresh
    do_incremental = bool(force_refresh) or bool(d.get("incremental_update", False))
    if use_cache and cache_read.exists():
        df = load_parquet(cache_read)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if tf == main_tf:
            df = ensure_liquidation_columns(df)
        if do_incremental:
            try:
                df = _incremental_update(cfg, symbol, df, tf)
                save_parquet(df, cache_write)
                src = "real" if force_refresh else "cache"
            except Exception as e:
                if force_refresh:
                    raise RuntimeError(
                        f"{symbol} {tf} 决策前强制刷新失败: {e}"
                    ) from e
                print(f"[warn] {symbol} {tf} 增量更新失败({e}); 使用现有缓存。")
                df = drop_incomplete_last_bar(df, tf)
                src = "cache"
        else:
            df = drop_incomplete_last_bar(df, tf)
            src = "cache"
            # 冷读遗留无后缀 1h 时, 顺便迁移到带后缀路径(不改数据)
            if cache_read != cache_write and len(df):
                try:
                    save_parquet(df, cache_write)
                except Exception:
                    pass
        return _tag_source(df, src)

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
        save_parquet(df, cache_write)
    return _tag_source(df, "real")


def refresh_market_data(cfg, symbol: str) -> pd.DataFrame:
    """决策前刷新主周期(+辅周期)到当下已收盘最新 bar, 写回缓存并校验新鲜度。

    返回刷新后的主周期 DataFrame。失败或过旧时按配置 fail-fast。
    """
    d = cfg["data"]
    main_tf = d["timeframe"]
    if d.get("use_synthetic", False):
        return load_symbol_data(cfg, symbol, force_refresh=False)

    print(f"[refresh] {symbol}: 拉取至当下已收盘 {main_tf} tip …")
    main = load_symbol_data(cfg, symbol, timeframe=main_tf, force_refresh=True)
    ex_used = str(getattr(main, "attrs", {}).get("exchange_used", "") or "")
    lag = closed_bar_lag(main, main_tf)
    last = main.index[-1] if len(main) else None
    print(
        f"[refresh] {symbol}: last_bar_open={last}, lag={lag}, "
        f"exchange={ex_used or d.get('exchange')}"
    )

    # 辅周期: 用已刷新主面板 resample 对齐 tip(避免再打多路 REST; 历史仍保留原缓存)
    for tf in list(d.get("aux_timeframes") or []):
        if not tf or tf == main_tf:
            continue
        try:
            if timeframe_delta(tf) < timeframe_delta(main_tf):
                continue
            resampled = drop_incomplete_last_bar(resample_ohlcv(main, tf), tf)
            cache_read = resolve_raw_cache_path(cfg, symbol, tf)
            cache_write = raw_cache_path(cfg, symbol, tf)
            if bool(d.get("cache", True)) and cache_read.exists():
                old = load_parquet(cache_read)
                if old.index.tz is None:
                    old.index = old.index.tz_localize("UTC")
                # 只把主面板 resample 的 tip 并入, 保留 Vision 辅周期历史
                if len(old) and len(resampled):
                    tip = resampled.loc[resampled.index >= old.index[-1]]
                    merged = pd.concat([old, tip])
                else:
                    merged = resampled if len(resampled) else old
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                merged = drop_incomplete_last_bar(merged, tf)
                save_parquet(merged, cache_write)
            elif bool(d.get("cache", True)) and len(resampled):
                save_parquet(resampled, cache_write)
        except Exception as e:
            print(f"[warn] {symbol} 辅周期 {tf} tip 对齐失败({e}); 跳过。")

    if bool(d.get("require_fresh_for_decide", True)):
        assert_fresh_enough(
            main,
            main_tf,
            max_lag_bars=int(d.get("max_closed_bar_lag", 2)),
            context=f"{symbol} ",
        )
    return main


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
