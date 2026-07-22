#!/usr/bin/env python3
"""稳健新闻历史回填入口: 真实源 + 可续跑 + 结束后重建面板并做时点校验。

设计:
- 默认 providers=gdelt(+cryptocompare 若有 CRYPTOCOMPARE_KEY)
- 隔离/拒绝 synthetic 与真实行情混用
- GDELT 满 250 条自动切分窗口, 避免 datedesc 截断窗前半段
- 窗级 append + gdelt_cursor 续跑
- 结束后重建 30m 桶末面板, 并跑 validate_news_alignment

用法:
  PYTHONUNBUFFERED=1 PYTHONPATH=src python -u scripts/run_news_backfill_robust.py
  PYTHONUNBUFFERED=1 PYTHONPATH=src python -u scripts/run_news_backfill_robust.py \\
      --start 2020-01-01 --end 2026-07-22 --providers gdelt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data.news import (
    _is_synthetic_news_source,
    _load_raw_store,
    backfill_news,
    build_news_panel,
    save_news_panel,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--providers", nargs="*", default=None)
    ap.add_argument("--skip-panels", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="忽略 gdelt_cursor, 从 --start 重新扫(仍按 title+url 去重)")
    ap.add_argument("--purge-synthetic", action="store_true",
                    help="回填前从 corpus 删除 synthetic: 行(真实行情推荐)")
    args = ap.parse_args()

    cfg = Config.load()
    if cfg["data"].get("use_synthetic", False):
        print("[err] data.use_synthetic=true — 拒绝与真实新闻回填混跑", flush=True)
        return 2

    if args.no_resume:
        cp = cfg.root / cfg["news"]["history"]["raw_dir"] / "backfill_state.json"
        if cp.exists():
            import json
            try:
                st = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                st = {}
            st.pop("gdelt_cursor", None)
            st["range"] = [
                args.start or cfg["news"]["history"].get("start"),
                args.end or "now",
            ]
            cp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[info] 已清除 gdelt_cursor 以强制从起点扫描 -> {cp}", flush=True)

    providers = args.providers
    if providers is None:
        providers = list(cfg["news"].get("history", {}).get("providers") or ["gdelt"])
    # 无 key 时去掉 cryptocompare, 避免空跑
    if "cryptocompare" in providers and not os.environ.get("CRYPTOCOMPARE_KEY"):
        print("[info] CRYPTOCOMPARE_KEY 未设置, 本次仅跑 gdelt", flush=True)
        providers = [p for p in providers if p != "cryptocompare"]
    if "synthetic" in providers:
        print("[err] 真实行情禁止 providers 含 synthetic", flush=True)
        return 2

    if args.purge_synthetic:
        raw = _load_raw_store(cfg)
        if raw is not None and len(raw):
            mask = ~raw["source"].astype(str).map(_is_synthetic_news_source)
            cleaned = raw.loc[mask].copy()
            if len(cleaned) < len(raw):
                path = cfg.root / cfg["news"]["history"]["raw_dir"] / "corpus.parquet"
                cleaned.to_parquet(path, engine="pyarrow", index=False)
                print(f"[ok] 已清除 synthetic: {len(raw)-len(cleaned)} 条, 剩余 {len(cleaned)}",
                      flush=True)

    print(
        f"[start] backfill providers={providers} "
        f"start={args.start or cfg['news']['history'].get('start')} "
        f"end={args.end or 'now'} "
        f"rate={cfg['news']['history'].get('rate_limit_sec')}s",
        flush=True,
    )
    t0 = datetime.now(timezone.utc)
    stats = backfill_news(cfg, start=args.start, end=args.end, providers=providers)
    for k, v in stats.items():
        print(f"  - {k}: {v}", flush=True)
    print(
        f"[ok] corpus total={stats.get('_total')} added={stats.get('_added')} "
        f"elapsed={(datetime.now(timezone.utc)-t0).total_seconds():.0f}s",
        flush=True,
    )

    if args.skip_panels:
        return 0

    cfg["news"]["use_history"] = True
    for symbol in cfg["data"]["symbols"]:
        df = build_news_panel(cfg, symbol)
        if len(df) == 0:
            print(f"[warn] {symbol}: 面板为空", flush=True)
            continue
        path = save_news_panel(cfg, symbol, df)
        print(
            f"[ok] panel {symbol}: {len(df)} buckets "
            f"{df.index.min()} -> {df.index.max()} -> {path}",
            flush=True,
        )

    if args.skip_validate:
        return 0
    script = Path(__file__).resolve().parent / "validate_news_alignment.py"
    print("[start] validate_news_alignment", flush=True)
    rc = subprocess.call(
        [sys.executable, str(script)],
        env={**os.environ, "PYTHONPATH": str(cfg.root / "src")},
    )
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
