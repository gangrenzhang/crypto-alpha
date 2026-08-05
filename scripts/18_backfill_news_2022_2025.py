#!/usr/bin/env python3
"""耐久回填 2022–2025 GDELT GAL 新闻空洞 + 重建面板。

用法:
  HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \\
  PYTHONUNBUFFERED=1 PYTHONPATH=src \\
    nohup python -u scripts/18_backfill_news_2022_2025.py \\
      > artifacts/logs/news_backfill_2022_2025.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from crypto_alpha.config import Config
from crypto_alpha.data.news import (
    _checkpoint_path,
    _load_raw_store,
    backfill_news,
    build_news_panel,
    save_news_panel,
)


def _reset_cursor_to(cfg, iso: str, range_end: str) -> None:
    """把 GAL cursor 拉回战役起点, 避免冒烟窗口把进度推到年中而跳段。"""
    cp = _checkpoint_path(cfg)
    ckpt: dict = {}
    if cp.exists():
        try:
            ckpt = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            ckpt = {}
    ckpt["gdelt_gal_cursor"] = iso
    ckpt["range"] = [iso, range_end]
    ckpt["gdelt_backend"] = "gal"
    ckpt["note"] = "18_backfill_news_2022_2025 durable campaign"
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    print(f"[ckpt] cursor -> {iso}", flush=True)


def main() -> int:
    cfg = Config.load()
    if cfg["data"].get("use_synthetic", False):
        print("[err] data.use_synthetic=true — 拒绝混跑", flush=True)
        return 2

    h = cfg.raw["news"].setdefault("history", {})
    h["gdelt_backend"] = "gal"
    h["gdelt_gal_workers"] = int(os.environ.get("GDELT_GAL_WORKERS", h.get("gdelt_gal_workers", 32)))
    h["gdelt_gal_flush_minutes"] = int(
        os.environ.get("GDELT_GAL_FLUSH_MINUTES", h.get("gdelt_gal_flush_minutes", 180))
    )
    h["gdelt_gal_sparse_minutes"] = True
    h["providers"] = ["gdelt"]

    # 仅在显式要求时重置; 默认可续跑
    if os.environ.get("RESET_GAL_CURSOR", "").strip() in ("1", "true", "yes"):
        _reset_cursor_to(cfg, "2022-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")

    years = [2022, 2023, 2024, 2025]
    for y in years:
        y0 = f"{y}-01-01T00:00:00Z"
        y1 = f"{y + 1}-01-01T00:00:00Z"
        print(f"[year-start] {y0} -> {y1}", flush=True)
        stats = backfill_news(cfg, start=y0, end=y1, providers=["gdelt"])
        print(f"[year-done] {y} stats={stats}", flush=True)
        raw = _load_raw_store(cfg)
        if raw is not None and len(raw):
            ts = pd.to_datetime(raw["published_at"], utc=True)
            print("[corpus-years]", ts.dt.year.value_counts().sort_index().to_dict(), flush=True)

    raw = _load_raw_store(cfg)
    print(f"[corpus] total={0 if raw is None else len(raw)}", flush=True)

    if os.environ.get("SKIP_PANELS", "").strip() in ("1", "true", "yes"):
        print("[done] skipped panels", flush=True)
        return 0

    print("[panels] building…", flush=True)
    cfg.raw["news"]["use_history"] = True
    for sym in cfg["data"]["symbols"]:
        try:
            panel = build_news_panel(cfg, sym)
            if panel is not None and len(panel):
                path = save_news_panel(cfg, sym, panel)
                print(f"[panel-ok] {sym} n={len(panel)} -> {path}", flush=True)
            else:
                print(f"[panel-empty] {sym}", flush=True)
        except Exception as e:
            print(f"[panel-err] {sym}: {type(e).__name__}: {e}", flush=True)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
