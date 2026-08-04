#!/usr/bin/env python3
"""GDELT GAL 全量新闻回填入口(绕开 DOC API 429)。

按年分片调用, 控制内存; 可用 setsid/nohup 脱离终端长跑。

用法:
  PYTHONUNBUFFERED=1 PYTHONPATH=src SKIP_PANELS=1 \\
    python -u scripts/run_news_gal_backfill.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data.news import (
    _load_raw_store,
    _parse_dt,
    backfill_news,
    build_news_panel,
    save_news_panel,
)


def main() -> int:
    cfg = Config.load()
    if cfg["data"].get("use_synthetic", False):
        print("[err] data.use_synthetic=true — 拒绝与真实新闻回填混跑", flush=True)
        return 2

    h = cfg["news"].setdefault("history", {})
    h["gdelt_backend"] = "gal"
    h["gdelt_gal_workers"] = int(os.environ.get("GDELT_GAL_WORKERS", h.get("gdelt_gal_workers", 32)))
    h["gdelt_gal_flush_minutes"] = int(
        os.environ.get("GDELT_GAL_FLUSH_MINUTES", h.get("gdelt_gal_flush_minutes", 120))
    )

    campaign_start = _parse_dt(h.get("start"), default=_parse_dt("2020-01-01T00:00:00Z"))
    campaign_end = _parse_dt(h.get("end"), default=datetime.now(timezone.utc))

    print(
        f"[start] GAL backfill {campaign_start.date()} -> {campaign_end.date()} "
        f"workers={h['gdelt_gal_workers']} flush={h['gdelt_gal_flush_minutes']}min "
        f"pid={os.getpid()}",
        flush=True,
    )
    t0 = datetime.now(timezone.utc)

    # 按年分片: 每片结束后进程内内存可回收, checkpoint 跨片续跑
    year = campaign_start.year
    while year <= campaign_end.year:
        y0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        y1 = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        if y0 < campaign_start:
            y0 = campaign_start
        if y1 > campaign_end:
            y1 = campaign_end
        if y0 >= y1:
            year += 1
            continue
        print(f"[year] {y0.isoformat()} -> {y1.isoformat()}", flush=True)
        stats = backfill_news(
            cfg,
            start=y0.isoformat(),
            end=y1.isoformat(),
            providers=["gdelt"],
        )
        print(f"[year-done] {year} stats={stats}", flush=True)
        year += 1

    raw = _load_raw_store(cfg)
    total = 0 if raw is None else len(raw)
    print(
        f"[ok] corpus total={total} "
        f"elapsed={(datetime.now(timezone.utc) - t0).total_seconds():.0f}s",
        flush=True,
    )

    if os.environ.get("SKIP_PANELS", "").strip() in ("1", "true", "yes"):
        return 0

    cfg["news"]["use_history"] = True
    for symbol in cfg["data"]["symbols"]:
        df = build_news_panel(cfg, symbol)
        if len(df) == 0:
            print(f"[warn] {symbol}: 面板为空", flush=True)
            continue
        path = save_news_panel(cfg, symbol, df)
        print(f"[ok] panel {symbol} -> {path} rows={len(df)}", flush=True)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
