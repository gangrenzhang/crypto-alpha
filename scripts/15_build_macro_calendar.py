#!/usr/bin/env python3
"""从官方源构建完整宏观日历 → data/macro_calendar/events.parquet。

用法:
  PYTHONPATH=src python scripts/15_build_macro_calendar.py --export-csv
  FRED_API_KEY=xxx PYTHONPATH=src python scripts/15_build_macro_calendar.py --refresh-alfred
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data.macro_calendar import load_macro_events, macro_events_path
from crypto_alpha.data.macro_calendar_sources import build_and_save_macro_calendar


def main() -> int:
    ap = argparse.ArgumentParser(description="构建并导入完整宏观日历")
    ap.add_argument("--start", default="2020-01-01", help="事件起点(UTC 日期)")
    ap.add_argument("--no-fed", action="store_true")
    ap.add_argument("--no-bls", action="store_true")
    ap.add_argument("--no-ff", action="store_true", help="跳过 FF 本周")
    ap.add_argument("--no-ff-hist", action="store_true", help="跳过 FF GitHub 历史归档")
    ap.add_argument("--refresh-bls-schedule", action="store_true", help="强制重拉 BLS 官方日程")
    ap.add_argument("--refresh-alfred", action="store_true", help="强制重拉 ALFRED 首印(需 FRED_API_KEY)")
    ap.add_argument("--export-csv", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    df, n = build_and_save_macro_calendar(
        cfg,
        start=str(args.start),
        include_fed=not args.no_fed,
        include_bls=not args.no_bls,
        include_ff_week=not args.no_ff,
        include_ff_hist=not args.no_ff_hist,
        refresh_bls_schedule=bool(args.refresh_bls_schedule),
        refresh_alfred=bool(args.refresh_alfred),
    )
    path = macro_events_path(cfg)
    print(f"[ok] 宏观日历已写入 {n} 条 → {path}", flush=True)
    if n:
        print(
            f"     区间 {df['scheduled_at'].min()} → {df['scheduled_at'].max()}",
            flush=True,
        )
        print(f"     来源 {df['source'].value_counts().to_dict()}", flush=True)
        print(f"     类别 {df['category'].value_counts().to_dict()}", flush=True)
        print(f"     print_kind {df['print_kind'].value_counts().to_dict()}", flush=True)
        print(f"     schedule_source {df['schedule_source'].value_counts().to_dict()}", flush=True)
        n_num = int(df[["previous", "forecast", "actual"]].notna().all(axis=1).sum())
        print(f"     含完整 previous/forecast/actual: {n_num}", flush=True)
        n_non_us = int((df["country"] != "US").sum())
        print(f"     非美事件: {n_non_us}", flush=True)
    if args.export_csv and n:
        csv_path = path.parent / "events_full.csv"
        df.to_csv(csv_path, index=False)
        print(f"[ok] CSV → {csv_path}", flush=True)

    cur = load_macro_events(cfg)
    print(f"[ok] 回读校验 {len(cur)} 条", flush=True)
    print(
        "[next] PYTHONPATH=src python scripts/validate_macro_calendar_alignment.py",
        flush=True,
    )
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
