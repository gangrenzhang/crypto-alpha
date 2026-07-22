"""校验新闻语料/面板与主周期 K 线的时间点搭配(PIT、桶末、覆盖区间)。

用法:
  PYTHONPATH=src python scripts/validate_news_alignment.py
  PYTHONPATH=src python scripts/validate_news_alignment.py --symbol BTC/USDT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data.fetch import timeframe_delta, timeframe_to_pandas_freq
from crypto_alpha.data.news import (
    _is_synthetic_news_source,
    _load_raw_store,
    _raw_to_items,
    build_news_panel,
    load_news_panel,
)
from crypto_alpha.features.news_features import add_news_features
from crypto_alpha.data import load_symbol_data
from crypto_alpha.features.build import build_feature_matrix


def _fail(msg: str, errors: list[str]) -> None:
    print(f"[FAIL] {msg}")
    errors.append(msg)


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def validate_corpus(cfg, errors: list[str]) -> pd.DataFrame | None:
    raw = _load_raw_store(cfg)
    if raw is None or len(raw) == 0:
        _fail("corpus.parquet 为空或不存在 — 请先跑 09_backfill_news", errors)
        return None
    raw = raw.copy()
    raw["published_at"] = pd.to_datetime(raw["published_at"], utc=True)
    if raw["published_at"].isna().any():
        _fail("corpus 存在无法解析的 published_at", errors)
    if (raw["published_at"].dt.tz is None):
        _fail("corpus published_at 缺少 UTC tz", errors)
    n_syn = sum(_is_synthetic_news_source(s) for s in raw["source"].astype(str))
    allow_syn = bool(cfg["data"].get("use_synthetic", False))
    if n_syn and not allow_syn:
        print(f"[WARN] corpus 含 {n_syn} 条 synthetic(真实行情下加载时会过滤)")
    h = cfg["news"].get("history") or {}
    start = pd.Timestamp(h.get("start") or "2020-01-01", tz="UTC")
    end = pd.Timestamp(h.get("end") or pd.Timestamp.now(tz="UTC"), tz="UTC")
    span_min, span_max = raw["published_at"].min(), raw["published_at"].max()
    _ok(f"corpus n={len(raw)} span={span_min.date()}~{span_max.date()} "
        f"synthetic={n_syn}")
    # 覆盖应触及配置起点附近(允许 GDELT 早期稀疏)
    if span_min > start + pd.Timedelta(days=120):
        _fail(
            f"语料起点 {span_min.date()} 远晚于配置 start {start.date()} "
            f"(>120 天) — 回填可能未从训练窗开始或早期窗失败",
            errors,
        )
    if span_max < end - pd.Timedelta(days=14):
        print(
            f"[WARN] 语料终点 {span_max.date()} 距配置 end {end.date()} "
            f">14 天 — 可能仍在回填中或 tip 缺口"
        )
    if (raw["published_at"] < start - pd.Timedelta(days=1)).any():
        print("[WARN] 存在早于 history.start 的条目(可接受, 加载时会裁剪)")
    return raw


def validate_panel(cfg, symbol: str, errors: list[str]) -> pd.DataFrame | None:
    panel = load_news_panel(cfg, symbol)
    if panel is None or len(panel) == 0:
        # 尝试现建
        cfg["news"]["use_history"] = True
        panel = build_news_panel(cfg, symbol)
    if panel is None or len(panel) == 0:
        _fail(f"{symbol}: 新闻面板为空", errors)
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(panel.index, utc=True))
    if idx.tz is None:
        _fail(f"{symbol}: 面板索引无 UTC tz", errors)
    bucket = str(cfg["news"].get("bucket", "30m"))
    freq = timeframe_to_pandas_freq(bucket)
    td = pd.to_timedelta(freq)
    # 桶末对齐: 时间戳应对齐到 freq 网格
    # 允许 ns 级浮点; 用模运算检查
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    mods = ((idx - epoch) % td).total_seconds()
    bad = int((mods > 1e-6).sum())
    if bad:
        _fail(f"{symbol}: {bad}/{len(idx)} 个面板时间戳未对齐桶末网格({bucket})", errors)
    else:
        _ok(f"{symbol}: 面板 {len(idx)} 桶对齐 {bucket} 网格 "
            f"span={idx.min().date()}~{idx.max().date()}")
    return panel


def validate_pit(cfg, symbol: str, panel: pd.DataFrame, errors: list[str]) -> None:
    """抽样验证: 新闻可用时刻 = 桶末+buffer 不得晚于决策时刻(开盘+主周期)。"""
    bars = load_symbol_data(cfg, symbol)
    main_delta = timeframe_delta(cfg["data"]["timeframe"])
    buffer_min = int(cfg["news"].get("buffer_minutes", 5))
    # 取面板与 K 线重叠区中段 200 根
    main_idx = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    overlap = main_idx[(main_idx >= panel.index.min()) & (main_idx <= panel.index.max())]
    if len(overlap) < 10:
        _fail(f"{symbol}: 面板与 K 线几乎无重叠, 无法验证 PIT", errors)
        return
    sample = overlap[len(overlap) // 2: len(overlap) // 2 + 200]
    right = panel.copy()
    right_ts = pd.DatetimeIndex(pd.to_datetime(right.index, utc=True)) + pd.Timedelta(minutes=buffer_min)
    # 对每个 sample bar, merge_asof backward 的新闻可用时刻应 <= decision_at
    left = pd.DataFrame({
        "decision_at": (sample + main_delta).astype("datetime64[ns, UTC]"),
        "main_ts": sample,
    }).sort_values("decision_at")
    right_df = pd.DataFrame({"news_ts": right_ts.astype("datetime64[ns, UTC]")}).sort_values("news_ts")
    merged = pd.merge_asof(left, right_df, left_on="decision_at", right_on="news_ts", direction="backward")
    leaked = merged["news_ts"].notna() & (merged["news_ts"] > merged["decision_at"])
    n_leak = int(leaked.sum())
    if n_leak:
        _fail(f"{symbol}: PIT 泄漏 {n_leak} 条 (news_ts+buffer > decision_at)", errors)
    else:
        hit = int(merged["news_ts"].notna().sum())
        _ok(f"{symbol}: PIT 抽样通过 (n={len(sample)}, 命中新闻 {hit}, buffer={buffer_min}m, "
            f"decision=+{main_delta})")


def validate_feature_coverage(cfg, symbol: str, errors: list[str]) -> None:
    cfg = cfg  # noqa
    cfg["news"]["use_history"] = True
    cfg["news"]["as_feature"] = True
    bars = load_symbol_data(cfg, symbol)
    # 只用与 WF 训练重叠的一段, 避免全量过慢
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-03-01", tz="UTC")
    bars = bars.loc[(bars.index >= start) & (bars.index < end)]
    if len(bars) < 100:
        print(f"[WARN] {symbol}: 覆盖率抽查样本过少, 跳过")
        return
    feat = build_feature_matrix(cfg, symbol, bars)
    feat = add_news_features(feat, cfg, symbol)
    cov = float(feat["has_recent_news"].mean()) if "has_recent_news" in feat.columns else 0.0
    thr = float(cfg["news"].get("min_coverage_warn", 0.05))
    msg = f"{symbol}: 2024Q1 新闻特征覆盖率={cov:.2%} (warn阈值 {thr:.0%})"
    if cov < thr:
        print(f"[WARN] {msg}")
    else:
        _ok(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    args = ap.parse_args()
    cfg = Config.load()
    symbols = [args.symbol] if args.symbol else list(cfg["data"]["symbols"])
    errors: list[str] = []
    print("=== 新闻时间点对齐校验 ===")
    raw = validate_corpus(cfg, errors)
    if raw is None:
        return 1
    for sym in symbols:
        print(f"\n--- {sym} ---")
        # 过滤后真实条目数
        items = _raw_to_items(raw, sym, cfg)
        _ok(f"加载后真实相关条目 {len(items)} (已按 use_synthetic={cfg['data'].get('use_synthetic')} 过滤)")
        panel = validate_panel(cfg, sym, errors)
        if panel is not None and len(panel):
            validate_pit(cfg, sym, panel, errors)
            try:
                validate_feature_coverage(cfg, sym, errors)
            except Exception as e:
                print(f"[WARN] 特征覆盖率抽查失败: {e}")
    print("\n=== 汇总 ===")
    if errors:
        for e in errors:
            print(f" - {e}")
        return 1
    print("全部关键校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
