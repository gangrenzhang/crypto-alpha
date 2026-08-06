#!/usr/bin/env python3
"""监督缺失数据回填: 进程被杀后自动续跑直到完成。

覆盖:
1. 新闻 GDELT-GAL 2021H2 + 2022–2025 + 重建面板
2. FF 日历 scrape 2025-04-08 → 2026-07-31

用法:
  PYTHONPATH=src python -u scripts/22_supervise_backfills.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "artifacts" / "logs"
PROXY = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7890")


def _env() -> dict:
    e = os.environ.copy()
    e.update({
        "HTTPS_PROXY": PROXY,
        "HTTP_PROXY": PROXY,
        "https_proxy": PROXY,
        "http_proxy": PROXY,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "src",
        "GDELT_GAL_WORKERS": os.environ.get("GDELT_GAL_WORKERS", "32"),
    })
    return e


def _log(msg: str) -> None:
    line = f"[supervise {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    p = LOG_DIR / "supervise_backfills.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def news_done() -> bool:
    """语料覆盖到 2025 且面板已重建的粗判。"""
    try:
        import pandas as pd
        c = pd.read_parquet(ROOT / "data/news_raw/corpus.parquet")
        ts = pd.to_datetime(c["published_at"], utc=True)
        years = set(ts.dt.year.unique().tolist())
        # 至少要有 2021–2025 各年一些文章; 2025 可较少
        need = {2021, 2022, 2023, 2024}
        if not need.issubset(years):
            return False
        # 2021H2 非空
        h2 = ((ts >= "2021-07-01") & (ts < "2022-01-01")).sum()
        if int(h2) < 1000:
            return False
        for sym in ("BTC_USDT", "ETH_USDT"):
            p = ROOT / "data/news" / f"{sym}.parquet"
            if not p.exists():
                return False
            d = pd.read_parquet(p)
            idx = pd.DatetimeIndex(pd.to_datetime(d.index, utc=True))
            if int(((idx >= "2022-01-01") & (idx < "2025-01-01")).sum()) < 100:
                return False
        return True
    except Exception as e:
        _log(f"news_done check err: {e}")
        return False


def ff_done() -> bool:
    days_path = ROOT / "data/macro_calendar/ff_hist_cache/ff_scrape_days.json"
    if not days_path.exists():
        return False
    try:
        days = set(json.loads(days_path.read_text(encoding="utf-8")))
    except Exception:
        return False
    # 期望覆盖 2025-04-08 → 2026-07-31
    start = date(2025, 4, 8)
    end = date(2026, 7, 31)
    n_expect = (end - start).days + 1
    # 允许少量失败日
    return len(days) >= int(n_expect * 0.95)


def run_until(cmd: list[str], log_name: str, done_fn, *, max_restarts: int = 50) -> bool:
    log_path = LOG_DIR / log_name
    for i in range(max_restarts):
        if done_fn():
            _log(f"{log_name}: already done")
            return True
        _log(f"{log_name}: start attempt {i+1}: {' '.join(cmd)}")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n----- supervise attempt {i+1} -----\n")
            f.flush()
            p = subprocess.Popen(
                cmd, cwd=str(ROOT), env=_env(),
                stdout=f, stderr=subprocess.STDOUT,
            )
            rc = p.wait()
        _log(f"{log_name}: exit={rc}")
        if done_fn():
            _log(f"{log_name}: COMPLETE")
            return True
        time.sleep(5)
    _log(f"{log_name}: gave up after {max_restarts} restarts")
    return False


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log("supervisor start")

    # 1) 宏观 FF scrape（较快, 先做）
    ok_ff = run_until(
        ["python", "-u", "scripts/21_scrape_ff_calendar.py",
         "--start", "2025-04-08", "--end", "2026-07-31", "--sleep", "1.0"],
        "ff_scrape_2025_2026.log",
        ff_done,
    )

    # 2) 新闻: 先跑 18(含现有 2022-2025 续跑), 再单独补 2021H2+面板
    # 18 脚本已扩展 spans 含 2021H2, 但正在跑的旧进程可能没有; 这里显式再跑一遍 spans
    ok_news = run_until(
        ["python", "-u", "scripts/18_backfill_news_2022_2025.py"],
        "news_backfill_2022_2025.log",
        news_done,
        max_restarts=20,
    )

    _log(f"summary ff={ok_ff} news={ok_news}")
    return 0 if (ok_ff and ok_news) else 1


if __name__ == "__main__":
    sys.exit(main())
