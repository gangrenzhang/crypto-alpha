#!/usr/bin/env python3
"""导入宏观经济日历 CSV → data/macro_calendar/events.parquet。

CSV 列(必填): name, scheduled_at, released_at, importance
推荐: country, category, previous, forecast, actual, unit, source, event_id

用法:
  PYTHONPATH=src python scripts/14_import_macro_calendar.py \\
      --csv data/macro_calendar/sample_events.csv
  PYTHONPATH=src python scripts/14_import_macro_calendar.py --csv my.csv --replace
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data.macro_calendar import (
    import_macro_events_csv,
    load_macro_events,
    macro_events_path,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="导入宏观日历事件 CSV")
    ap.add_argument("--csv", required=True, help="事件 CSV 路径")
    ap.add_argument(
        "--replace", action="store_true",
        help="用本次 CSV 覆盖整个事件库(默认增量合并)",
    )
    args = ap.parse_args()

    cfg = Config.load()
    added, total = import_macro_events_csv(cfg, args.csv, replace=bool(args.replace))
    path = macro_events_path(cfg)
    ev = load_macro_events(cfg)
    print(f"[ok] 导入完成: +{added} 新 id / 库总量 {total} -> {path}", flush=True)
    if len(ev):
        print(
            f"     时间范围 {ev['released_at'].min()} → {ev['released_at'].max()} "
            f"| 重要性≥3: {int((ev['importance'] >= 3).sum())}",
            flush=True,
        )
    print(
        "[next] 校验: PYTHONPATH=src python scripts/validate_macro_calendar_alignment.py",
        flush=True,
    )
    print(
        "[next] 完整库构建: PYTHONPATH=src python scripts/15_build_macro_calendar.py --export-csv",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
