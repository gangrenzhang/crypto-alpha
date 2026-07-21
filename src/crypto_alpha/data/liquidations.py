"""清算事件库: 与 OHLCV 缓存解耦, 按主周期 bar 开盘时刻桶对齐喂入面板。

设计要点
--------
1. OHLCV 冷缓存不含清算时, 训练仍可从 ``data/liquidations/<SYMBOL>.parquet`` 事件库
   按每根 K 线时间桶聚合出 ``liq_long``/``liq_short``, 再进 technical 衍生特征。
2. 事件时刻 τ → 开盘 t 满足 ``t ≤ τ < t+Δ``(与 volume 同属当根收盘信息; 决策在 t+Δ)。
3. 拉取与 OHLCV 增量解耦: 回填脚本/全量拉衍生品时可 append 事件库; tip 流量不 ffill。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import (
    _aggregate_liquidations,
    _liq_bucket,
    _liq_exchange_fallbacks,
    _liq_notional,
    _paginate_liquidations,
    _parse_liq_ts,
    exchange_candidates,
    timeframe_delta,
)
from .storage import load_parquet, save_parquet


_EVENT_COLS = ("timestamp", "side_bucket", "notional", "exchange", "symbol")


def liquidations_store_dir(cfg) -> Path:
    d = cfg["data"]
    sub = (d.get("liquidations") or {}).get("store_dir") or "liquidations"
    return Path(cfg.data_dir) / str(sub)


def liquidations_path(cfg, symbol: str) -> Path:
    base = symbol.replace("/", "_").replace(":", "_")
    return liquidations_store_dir(cfg) / f"{base}.parquet"


def _normalize_rows(rows: list, *, exchange: str, symbol: str) -> pd.DataFrame:
    recs = []
    for r in rows or []:
        ts = _parse_liq_ts(r.get("timestamp"))
        bucket = _liq_bucket(r.get("side"))
        notion = _liq_notional(r)
        if ts is None or bucket is None or not np.isfinite(notion) or notion <= 0:
            continue
        recs.append(
            {
                "timestamp": ts,
                "side_bucket": bucket,
                "notional": float(notion),
                "exchange": str(exchange),
                "symbol": str(symbol),
            }
        )
    if not recs:
        return pd.DataFrame(columns=list(_EVENT_COLS))
    out = pd.DataFrame.from_records(recs)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def load_liquidation_events(cfg, symbol: str) -> pd.DataFrame:
    path = liquidations_path(cfg, symbol)
    if not path.exists():
        return pd.DataFrame(columns=list(_EVENT_COLS))
    df = load_parquet(path)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=list(_EVENT_COLS))
    out = df.copy()
    if "timestamp" not in out.columns and out.index.name in ("timestamp", "time"):
        out = out.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for c in _EVENT_COLS:
        if c not in out.columns:
            out[c] = np.nan if c == "notional" else ""
    return out[list(_EVENT_COLS)].sort_values("timestamp").reset_index(drop=True)


def save_liquidation_events(cfg, symbol: str, events: pd.DataFrame) -> Path:
    path = liquidations_path(cfg, symbol)
    if events is None or len(events) == 0:
        empty = pd.DataFrame(columns=list(_EVENT_COLS))
        return save_parquet(empty, path)
    out = events.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp")
    # 去重: 同毫秒+同向+同名义+同所
    out = out.drop_duplicates(
        subset=["timestamp", "side_bucket", "notional", "exchange"], keep="last"
    )
    return save_parquet(out[list(_EVENT_COLS)].reset_index(drop=True), path)


def append_liquidation_events(cfg, symbol: str, new_events: pd.DataFrame) -> pd.DataFrame:
    """合并写入事件库, 返回合并后全表。"""
    old = load_liquidation_events(cfg, symbol)
    if new_events is None or len(new_events) == 0:
        return old
    if len(old) == 0:
        save_liquidation_events(cfg, symbol, new_events)
        return load_liquidation_events(cfg, symbol)
    merged = pd.concat([old, new_events], ignore_index=True)
    save_liquidation_events(cfg, symbol, merged)
    return load_liquidation_events(cfg, symbol)


def events_to_agg_rows(events: pd.DataFrame) -> list[dict]:
    """转为 ``_aggregate_liquidations`` 可消费的 row 列表(side=buy/sell)。"""
    if events is None or len(events) == 0:
        return []
    rows = []
    for _, r in events.iterrows():
        bucket = str(r.get("side_bucket") or "")
        side = "sell" if bucket == "long" else ("buy" if bucket == "short" else None)
        if side is None:
            continue
        ts = r.get("timestamp")
        ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)
        rows.append({"timestamp": ts_ms, "side": side, "quoteValue": float(r["notional"])})
    return rows


def aggregate_events_to_bars(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    bar_delta: pd.Timedelta,
) -> pd.DataFrame:
    """事件库 → 与主周期索引对齐的 liq_long/liq_short。

    无事件: 两列全 NaN(调用方记 unavailable)。
    有事件: 首笔前 NaN, 其后无事件 bar 为 0(与 fetch._aggregate_liquidations 一致)。
    """
    out = pd.DataFrame(index=index)
    out["liq_long"] = np.nan
    out["liq_short"] = np.nan
    rows = events_to_agg_rows(events)
    if not rows or len(index) == 0:
        return out
    lng, sht = _aggregate_liquidations(rows, index, bar_delta)
    out["liq_long"] = lng
    out["liq_short"] = sht
    return out


def attach_liquidations_to_ohlcv(
    df: pd.DataFrame,
    cfg,
    symbol: str | None,
    *,
    bar_delta: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """把事件库清算按 bar 时间点并入 OHLCV 面板。

    - ``fetch_liquidations=false``: 只保证列存在(NaN), 不读库。
    - 事件库为空: 保持面板原值。
    - 事件库非空: **以事件库聚合覆盖**面板 liq 列(事件库由 tip/全量拉取 append,
      是跨冷缓存与 tip 刷新的权威源; 避免「面板已有全 NaN/旧 tip」挡住新事件)。
    """
    if df is None or len(df) == 0:
        return df
    d = cfg["data"]
    out = df.copy()
    for col in ("liq_long", "liq_short"):
        if col not in out.columns:
            out[col] = np.nan

    if not bool(d.get("fetch_liquidations", True)):
        return out

    if not symbol:
        return out

    liq_cfg = d.get("liquidations") or {}
    if not bool(liq_cfg.get("auto_attach", True)):
        return out

    events = load_liquidation_events(cfg, symbol)
    if len(events) == 0:
        return out

    if bar_delta is None:
        try:
            bar_delta = timeframe_delta(d["timeframe"])
        except Exception:
            if len(out.index) >= 2:
                bar_delta = pd.Timedelta(pd.Series(out.index).diff().median())
            else:
                bar_delta = pd.Timedelta(hours=1)

    bars = aggregate_events_to_bars(events, out.index, bar_delta)
    hit = bars["liq_long"].notna() | bars["liq_short"].notna()
    if not bool(hit.any()):
        # 事件库非空但无一落入本面板(冷缓存末 bar 早于 tip 清算等)
        # → 不把全 NaN 盖掉面板已有值; 记 outside 供审计
        try:
            out.attrs["liquidations_attached_from_store"] = False
            out.attrs["liquidations_outside_panel"] = True
            out.attrs["liquidations_n_events"] = int(len(events))
        except Exception:
            pass
        return out

    out["liq_long"] = bars["liq_long"].reindex(out.index)
    out["liq_short"] = bars["liq_short"].reindex(out.index)
    try:
        out.attrs["liquidations_attached_from_store"] = True
        out.attrs["liquidations_outside_panel"] = False
        out.attrs["liquidations_n_events"] = int(len(events))
        out.attrs["liquidations_bar_coverage"] = float(hit.mean())
    except Exception:
        pass
    return out


def import_liquidation_events_frame(
    cfg,
    symbol: str,
    frame: pd.DataFrame,
    *,
    exchange: str = "import",
) -> pd.DataFrame:
    """从外部表导入清算事件(多年历史/第三方源)。

    需要列: ``timestamp``; ``side_bucket``(long/short) 或 ``side``(buy/sell);
    ``notional`` 或 ``quoteValue``。
    """
    if frame is None or len(frame) == 0:
        return load_liquidation_events(cfg, symbol)
    df = frame.copy()
    if "timestamp" not in df.columns and df.index.name in ("timestamp", "time"):
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "side_bucket" not in df.columns:
        if "side" not in df.columns:
            raise ValueError("导入清算需要 side_bucket 或 side 列")
        df["side_bucket"] = df["side"].map(
            lambda s: _liq_bucket(s) if _liq_bucket(s) else np.nan
        )
    if "notional" not in df.columns:
        if "quoteValue" in df.columns:
            df["notional"] = pd.to_numeric(df["quoteValue"], errors="coerce")
        else:
            raise ValueError("导入清算需要 notional 或 quoteValue 列")
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    df = df.dropna(subset=["timestamp", "side_bucket", "notional"])
    df = df[df["notional"] > 0]
    df["exchange"] = str(exchange)
    df["symbol"] = str(symbol)
    part = df[list(_EVENT_COLS)]
    return append_liquidation_events(cfg, symbol, part)


def fetch_and_store_liquidations(
    cfg,
    symbol: str,
    *,
    since: str | pd.Timestamp | None = None,
    until: str | pd.Timestamp | None = None,
    exchanges: list[str] | None = None,
    tip_only: bool = False,
) -> dict:
    """从交易所拉取清算事件并 append 到本地事件库。

    返回 ``{n_fetched, n_stored_total, path, exchanges_tried, ok}``。
    多数所仅近端历史 → n_fetched 可能远小于面板跨度(属预期);
    Gate 等需 **无 since tip 拉取**(见 ``_paginate_liquidations``)。

    ``tip_only=True``: 只扫 ``liquidations.exchanges``(默认 gate), 缩短决策 tip 路径,
    避免备用所 20s×N 超时拖死 ``refresh_market_data``。
    """
    d = cfg["data"]
    start = pd.Timestamp(since or d.get("since") or "2020-01-01T00:00:00Z")
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    end = pd.Timestamp(until or pd.Timestamp.now(tz="UTC"))
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    tried: list[str] = []
    all_new: list[pd.DataFrame] = []
    liq_cfg = d.get("liquidations") or {}
    preferred = [str(x).strip() for x in list(liq_cfg.get("exchanges") or []) if str(x).strip()]
    if not preferred:
        preferred = ["gate"]

    # 候选: tip_only → 仅优先所; 否则优先所 → tip 候选 → 映射/增补
    cands: list[str] = []
    if exchanges:
        cands.extend(exchanges)
    elif tip_only:
        cands.extend(preferred)
    else:
        for name in preferred:
            if name not in cands:
                cands.append(name)
        for name in exchange_candidates(cfg, for_tip=True):
            if name not in cands:
                cands.append(name)
            for alt in _liq_exchange_fallbacks(name):
                if alt not in cands:
                    cands.append(alt)
        for extra in ("binanceusdm", "bybit", "okx", "htx", "bitmex"):
            if extra not in cands:
                cands.append(extra)

    try:
        import ccxt
    except Exception as e:
        return {
            "ok": False,
            "error": f"ccxt import failed: {e}",
            "n_fetched": 0,
            "n_stored_total": len(load_liquidation_events(cfg, symbol)),
            "path": str(liquidations_path(cfg, symbol)),
            "exchanges_tried": tried,
        }

    # 默认拿到第一所有效数据即停(Gate tip 通常几秒); merge_all 才扫全所增补
    merge_all = bool(liq_cfg.get("merge_all_exchanges", False)) and not tip_only
    timeout_ms = 8_000 if tip_only else 20_000

    for name in cands:
        name = str(name or "").strip()
        if not name or name in tried:
            continue
        tried.append(name)
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True, "timeout": timeout_ms})
        except Exception:
            continue
        rows = _paginate_liquidations(ex, symbol, start_ms, end_ms)
        if rows:
            part = _normalize_rows(rows, exchange=name, symbol=symbol)
            if len(part):
                all_new.append(part)
                if not merge_all:
                    break

    if not all_new:
        stored = load_liquidation_events(cfg, symbol)
        return {
            "ok": False,
            "error": "no liquidations returned from any exchange",
            "n_fetched": 0,
            "n_stored_total": len(stored),
            "path": str(liquidations_path(cfg, symbol)),
            "exchanges_tried": tried,
            "tip_only": bool(tip_only),
        }

    new_df = pd.concat(all_new, ignore_index=True)
    n_fetched = len(new_df)
    merged = append_liquidation_events(cfg, symbol, new_df)
    return {
        "ok": True,
        "n_fetched": int(n_fetched),
        "n_stored_total": int(len(merged)),
        "path": str(liquidations_path(cfg, symbol)),
        "exchanges_tried": tried,
        "tip_only": bool(tip_only),
        "t_min": str(merged["timestamp"].min()) if len(merged) else None,
        "t_max": str(merged["timestamp"].max()) if len(merged) else None,
    }
