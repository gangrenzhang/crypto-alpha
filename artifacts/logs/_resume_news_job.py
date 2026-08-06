
import os, json
os.environ.setdefault("HTTPS_PROXY","http://127.0.0.1:7890")
os.environ.setdefault("HTTP_PROXY","http://127.0.0.1:7890")
os.environ["PYTHONPATH"]="src"
os.environ["PYTHONUNBUFFERED"]="1"
os.environ["GDELT_GAL_WORKERS"]="32"
from crypto_alpha.config import Config
from crypto_alpha.data.news import backfill_news, build_news_panel, save_news_panel, _load_raw_store
import pandas as pd

cfg=Config.load()
h=cfg.raw["news"].setdefault("history",{})
h.update({"gdelt_backend":"gal","gdelt_gal_workers":32,"gdelt_gal_sparse_minutes":True,"providers":["gdelt"]})

spans=[
  ("2023","2023-01-01T00:00:00Z","2024-01-01T00:00:00Z"),
  ("2024","2024-01-01T00:00:00Z","2025-01-01T00:00:00Z"),
  ("2025","2025-01-01T00:00:00Z","2026-01-01T00:00:00Z"),
  ("2021H2","2021-07-01T00:00:00Z","2022-01-01T00:00:00Z"),
]
for label,a,b in spans:
    print(f"[year-start] {label} {a}->{b}", flush=True)
    stats=backfill_news(cfg, start=a, end=b, providers=["gdelt"])
    print(f"[year-done] {label} {stats}", flush=True)
    raw=_load_raw_store(cfg)
    if raw is not None and len(raw):
        ts=pd.to_datetime(raw["published_at"], utc=True)
        print("[corpus-years]", ts.dt.year.value_counts().sort_index().to_dict(), flush=True)

cfg.raw["news"]["use_history"]=True
for sym in cfg["data"]["symbols"]:
    try:
        panel=build_news_panel(cfg, sym)
        if panel is not None and len(panel):
            path=save_news_panel(cfg, sym, panel)
            print(f"[panel-ok] {sym} n={len(panel)} -> {path}", flush=True)
        else:
            print(f"[panel-empty] {sym}", flush=True)
    except Exception as e:
        print(f"[panel-err] {sym}: {e}", flush=True)
print("[resume-news] ALL DONE", flush=True)
