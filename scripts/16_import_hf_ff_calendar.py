"""把 HuggingFace Forex Factory 日历缓存导入 events.parquet。

补齐 FF GitHub 归档(仅 2020–2023)停更后的 2024+ 缺口。
源文件: data/macro_calendar/ff_hist_cache/hf_forex_factory_cache.csv
(数据集 Ehsanrs2/Forex_Factory_Calendar, 时区 Asia/Tehran)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from crypto_alpha.config import Config
from crypto_alpha.data.macro_calendar import (
    DATE_ONLY_SCHEDULE_SOURCE,
    date_only_release_ts,
    import_macro_events_frame,
    load_macro_events,
)
from crypto_alpha.data.macro_calendar_global_ff import CCY_MAP, _category, _parse_num


IMPACT_MAP = {
    "High Impact Expected": 5,
    "Medium Impact Expected": 3,
    "Low Impact Expected": 1,
    "Non-Economic": 0,
    "High": 5,
    "Medium": 3,
    "Low": 1,
}


def hf_csv_to_events(
    path: Path,
    *,
    start: str = "2024-01-01",
    end: str | None = None,
    min_importance: int = 3,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else None
    rows: list[dict] = []
    for chunk in pd.read_csv(path, chunksize=100_000):
        for r in chunk.itertuples(index=False):
            impact = IMPACT_MAP.get(str(getattr(r, "Impact", "")).strip(), 1)
            if impact < int(min_importance):
                continue
            title = str(getattr(r, "Event", "") or "").strip()
            if not title or title.lower() == "nan":
                continue
            ccy = str(getattr(r, "Currency", "") or "").upper()
            country = CCY_MAP.get(ccy, ccy[:2] if ccy else "XX")
            try:
                local = pd.Timestamp(getattr(r, "DateTime"))
                if local.tzinfo is None:
                    local = local.tz_localize("Asia/Tehran")
            except Exception:
                continue
            # 该数据集对「无盘中时刻」的行写本地 00:00:00 或 23:59:59。本地午夜换算成
            # UTC 会落到前一天(如 2024-01-11T00:00+03:30 → 2024-01-10 20:30Z), 使 CPI
            # 等 actual 比真实公布早约 17h 可见 —— 前视泄漏。统一按 UTC 日终保守处理。
            date_only = local.strftime("%H:%M:%S") in ("00:00:00", "23:59:59")
            if date_only:
                ts = date_only_release_ts(local.tz_localize(None).date())
                if ts is None:
                    continue
            else:
                ts = local.tz_convert("UTC")
            if ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            prev = _parse_num(getattr(r, "Previous", None))
            fc = _parse_num(getattr(r, "Forecast", None))
            act = _parse_num(getattr(r, "Actual", None))
            rows.append({
                "event_id": f"{country}|{title}|{ts.strftime('%Y%m%dT%H%M%SZ')}",
                "name": title[:120],
                "country": country,
                "category": _category(title),
                "importance": impact,
                "scheduled_at": ts,
                "released_at": ts,
                "previous": prev,
                "forecast": fc,
                "actual": act,
                "unit": "",
                "source": "forexfactory_hf",
                "print_kind": "first_print" if np.isfinite(act) else "n/a",
                "schedule_source": (
                    DATE_ONLY_SCHEDULE_SOURCE if date_only else "forexfactory"
                ),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="data/macro_calendar/ff_hist_cache/hf_forex_factory_cache.csv",
    )
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-importance", type=int, default=3)
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"[err] 找不到 {path}", flush=True)
        return 2

    print(f"[import] 解析 {path} (≥{args.start}, impact>={args.min_importance}) …", flush=True)
    frame = hf_csv_to_events(
        path, start=args.start, end=args.end, min_importance=args.min_importance,
    )
    print(f"[import] 解析出 {len(frame)} 条", flush=True)
    if len(frame) == 0:
        return 1
    t = pd.to_datetime(frame["released_at"], utc=True)
    print(f"         区间 {t.min()} → {t.max()}", flush=True)
    print(f"         年份 {t.dt.year.value_counts().sort_index().to_dict()}", flush=True)
    ready = frame["forecast"].notna() & frame["actual"].notna()
    print(f"         surprise 就绪 {int(ready.sum())} ({ready.mean():.1%})", flush=True)

    cfg = Config.load()
    before = len(load_macro_events(cfg))
    n_new, total = import_macro_events_frame(cfg, frame, replace=False)
    print(f"[ok] 导入新增 event_id≈{n_new}; 库 {before} → {total}", flush=True)
    cur = load_macro_events(cfg)
    tt = pd.to_datetime(cur["released_at"], utc=True)
    print(f"     全库年份 {tt.dt.year.value_counts().sort_index().to_dict()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
