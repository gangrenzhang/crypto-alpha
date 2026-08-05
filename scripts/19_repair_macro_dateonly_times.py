#!/usr/bin/env python3
"""就地修复 events.parquet 中 ForexFactory 系「日期无时刻」事件的伪造时刻。

背景: 两个 FF 派生源在源数据缺盘中时刻时都编了一个时刻, 而且编早了 ——
- ``forexfactory_hf``: HF 数据集把无时刻行写成本地 00:00:00(Asia/Tehran)。换算成
  UTC 落到**前一天**傍晚(2024-01-11T00:00+03:30 → 2024-01-10 20:30Z), 使 CPI 等
  actual 比真实公布(13:30Z)早约 17 小时进入特征;
- ``forexfactory_hist``: ``All Day`` / ``Tentative`` 被归到中午 CST(=04:00 UTC),
  同样早于当天绝大多数欧美公布。

修复后这些行统一落在该本地日的 UTC 日终(23:59:59Z), 并标 ``schedule_source=
forexfactory_dateonly``, 保证「不早于真实公布」。只重建这两个源的行, 其余源
(ALFRED/BLS/Fed/央行日程/FF 本周)原样保留, 不触网。

用法:
  PYTHONPATH=src python scripts/19_repair_macro_dateonly_times.py [--dry-run]
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from crypto_alpha.config import Config
from crypto_alpha.data.macro_calendar import (
    load_macro_events,
    macro_events_path,
    normalize_macro_events,
    save_macro_events,
)
from crypto_alpha.data.macro_calendar_global_ff import ff_csv_to_events
from crypto_alpha.data.macro_calendar_merge import dedupe_cross_source_events

REBUILT_SOURCES = ("forexfactory_hist", "forexfactory_hf")


def _load_hf_importer():
    """scripts/16_ 不是合法模块名, 按路径加载以复用其已修好的解析器。"""
    path = Path(__file__).with_name("16_import_hf_ff_calendar.py")
    spec = importlib.util.spec_from_file_location("_hf_ff_importer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _time_profile(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"  [{label}] 空", flush=True)
        return
    t = pd.to_datetime(df["scheduled_at"], utc=True)
    hhmm = t.dt.strftime("%H:%M:%S").value_counts().head(5).to_dict()
    print(f"  [{label}] n={len(df)} 时刻top5={hhmm}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报告, 不写回")
    ap.add_argument("--hf-start", default="2024-01-01")
    ap.add_argument("--min-importance", type=int, default=3)
    args = ap.parse_args()

    cfg = Config.load()
    path = macro_events_path(cfg)
    if not path.exists():
        print(f"[err] 找不到 {path}; 请先跑 15_build_macro_calendar.py", flush=True)
        return 2

    cur = load_macro_events(cfg)
    is_ff = cur["source"].astype(str).isin(REBUILT_SOURCES)
    keep = cur.loc[~is_ff].copy()
    print(f"[load] 现有 {len(cur)} 条; FF 派生 {int(is_ff.sum())} 条待重建, 保留 {len(keep)} 条", flush=True)
    _time_profile(cur.loc[cur["source"] == "forexfactory_hf"], "旧 forexfactory_hf")
    _time_profile(cur.loc[cur["source"] == "forexfactory_hist"], "旧 forexfactory_hist")

    cache = path.parent / "ff_hist_cache"
    chunks: list[pd.DataFrame] = [keep]

    hist_rows: list[dict] = []
    for csv in sorted(cache.glob("forex_factory_calendar_*.csv")):
        part = ff_csv_to_events(csv, min_importance=args.min_importance)
        hist_rows.extend(part)
        print(f"[hist] {csv.name}: {len(part)} 条", flush=True)
    if hist_rows:
        hist_df = normalize_macro_events(pd.DataFrame(hist_rows))
        _time_profile(hist_df, "新 forexfactory_hist")
        chunks.append(hist_df)

    hf_csv = cache / "hf_forex_factory_cache.csv"
    if hf_csv.exists():
        hf_mod = _load_hf_importer()
        hf_df = hf_mod.hf_csv_to_events(
            hf_csv, start=args.hf_start, min_importance=args.min_importance,
        )
        if len(hf_df):
            hf_df = normalize_macro_events(hf_df)
            _time_profile(hf_df, "新 forexfactory_hf")
            chunks.append(hf_df)
    else:
        print(f"[warn] 缺 {hf_csv.name}; 2024+ FF 行将丢失", flush=True)

    merged = normalize_macro_events(pd.concat(chunks, ignore_index=True))
    merged = dedupe_cross_source_events(merged)
    merged = merged.sort_values("scheduled_at").reset_index(drop=True)

    t = pd.to_datetime(merged["scheduled_at"], utc=True)
    print(f"[new] {len(merged)} 条 {t.min()} → {t.max()}", flush=True)
    print(f"      source {merged['source'].value_counts().to_dict()}", flush=True)
    print(
        f"      schedule_source {merged['schedule_source'].value_counts().to_dict()}",
        flush=True,
    )

    if args.dry_run:
        print("[dry-run] 未写回", flush=True)
        return 0

    backup = path.with_suffix(".parquet.bak")
    shutil.copy2(path, backup)
    save_macro_events(cfg, merged)
    print(f"[ok] 已写回 {path} (备份 {backup.name})", flush=True)
    csv_path = path.parent / "events_full.csv"
    if csv_path.exists():
        merged.to_csv(csv_path, index=False)
        print(f"[ok] CSV 同步 → {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
