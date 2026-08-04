"""GDELT Article List (GAL) 静态分钟文件历史回填。

DOC API (`api.gdeltproject.org`) 对高频查询会长期 429, 官方建议改用可下载数据集。
GAL 每分钟一份:
  http://data.gdeltproject.org/gdeltv3/gal/YYYYMMDDHHMMSS.gal.json.gz
字段含 date/url/domain/title/desc/lang, 与 DOC artlist 元数据口径接近。
"""
from __future__ import annotations

import gzip
import io
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from .news import (
    _UA,
    _GAL_BASE,
    _append_raw_store,
    _keyword_hit,
    _load_raw_store,
    _news_dedupe_ts,
    _relevant_symbols,
    _resolve_http_proxies,
    _write_checkpoint,
)


def _http_get_bytes_simple(
    url: str,
    *,
    timeout: float = 45.0,
    proxies: dict[str, str] | None = None,
) -> tuple[bytes | None, int | None]:
    proxies = proxies if proxies is not None else _resolve_http_proxies()
    if proxies:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
        open_url = opener.open
    else:
        open_url = urllib.request.urlopen
    try:
        req = urllib.request.Request(url, headers=_UA)
        with open_url(req, timeout=timeout) as r:
            return r.read(), int(getattr(r, "status", 200) or 200)
    except urllib.error.HTTPError as e:
        return None, int(e.code)
    except Exception:
        return None, None


def _gal_parse_minute(
    raw_gz: bytes,
    name: str,
    tier: int,
    start: datetime,
    end: datetime,
) -> list[dict]:
    out: list[dict] = []
    try:
        text = gzip.GzipFile(fileobj=io.BytesIO(raw_gz)).read().decode(
            "utf-8", errors="replace"
        )
    except Exception:
        return out
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except Exception:
            continue
        title = str(o.get("title") or "").strip()
        desc = str(o.get("desc") or "").strip()
        syms = _relevant_symbols(f"{title} {desc}", crypto_only=True)
        if not syms:
            continue
        # GAL 全量流里再收紧一层: 至少命中显式加密词, 降低 eth/btc 短码误伤残留
        blob = f"{title} {desc}".lower()
        crypto_anchors = (
            "bitcoin", "ethereum", "btc", "eth", "crypto", "cryptocurrency",
            "比特币", "以太坊", "以太",
        )
        if not any(_keyword_hit(blob, a) for a in crypto_anchors):
            continue
        raw_dt = o.get("date") or ""
        try:
            dt = datetime.fromisoformat(str(raw_dt).replace("Z", "+00:00"))
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        if dt < start or dt > end:
            continue
        domain = str(o.get("domain") or o.get("outletName") or "unknown")
        out.append({
            "published_at": dt,
            "source": f"{name}:{domain}",
            "tier": int(tier),
            "title": title,
            "url": str(o.get("url") or ""),
            "symbols": syms,
        })
    return out


def fetch_gdelt_gal_history(
    name: str,
    tier: int,
    start: datetime,
    end: datetime,
    *,
    cfg=None,
    resume_from: datetime | None = None,
    workers: int = 24,
    flush_every_minutes: int = 180,
    day_chunk: bool = True,
) -> list[dict]:
    """并发拉取 GAL 分钟文件并过滤加密相关标题, 增量写入 corpus。"""
    if end <= start:
        return []
    cur = resume_from if resume_from is not None else start
    if cur < start:
        cur = start
    cur = cur.replace(second=0, microsecond=0)
    end_m = end.replace(second=0, microsecond=0)

    seen: set = set()
    if cfg is not None:
        existing = _load_raw_store(cfg)
        if existing is not None and len(existing):
            for r in existing.itertuples(index=False):
                pa = getattr(r, "published_at", None)
                pa_s = _news_dedupe_ts(pa) if pa is not None else ""
                seen.add((
                    str(getattr(r, "source", "")),
                    str(getattr(r, "title", "")),
                    pa_s,
                ))

    proxies = _resolve_http_proxies()
    workers = max(1, int(workers))
    flush_every = max(1, int(flush_every_minutes))
    # 小块提交: 避免「整天 1440 个 future 全完成才刷盘」, 中途崩溃会丢整天进度
    chunk_minutes = 120 if day_chunk else 60

    def _iter_stamps(t0: datetime, t1: datetime):
        t = t0
        while t <= t1:
            yield t
            t = t + timedelta(minutes=1)

    def _fetch_one(ts: datetime) -> tuple[datetime, list[dict], str]:
        stamp = ts.strftime("%Y%m%d%H%M%S")
        url = _GAL_BASE.format(stamp=stamp)
        try:
            raw, code = _http_get_bytes_simple(url, timeout=60.0, proxies=proxies)
        except Exception:
            return ts, [], "err"
        if code == 404:
            return ts, [], "404"
        if raw is None:
            return ts, [], "err"
        try:
            return ts, _gal_parse_minute(raw, name, tier, start, end), "ok"
        except Exception:
            return ts, [], "err"

    print(
        f"[hist] GDELT-GAL 从 {cur.isoformat()} 扫到 {end_m.isoformat()} "
        f"(workers={workers}, flush_every={flush_every}min, chunk={chunk_minutes}min)",
        flush=True,
    )

    out_count = 0
    pending_items: list[dict] = []
    n_ok = n_404 = n_err = n_files = 0
    last_flush_anchor = cur
    cursor = cur

    stamps_iter = _iter_stamps(cur, end_m)
    while True:
        batch: list[datetime] = []
        try:
            for _ in range(chunk_minutes):
                batch.append(next(stamps_iter))
        except StopIteration:
            pass
        if not batch:
            break

        results: dict[datetime, tuple[list[dict], str]] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_fetch_one, t) for t in batch]
            for fut in as_completed(futs):
                try:
                    t, items, status = fut.result()
                except Exception:
                    continue
                results[t] = (items, status)

        for t in sorted(results.keys()):
            items, status = results[t]
            n_files += 1
            if status == "ok":
                n_ok += 1
            elif status == "404":
                n_404 += 1
            else:
                n_err += 1
            for it in items:
                key = (
                    str(it["source"]),
                    str(it["title"]),
                    _news_dedupe_ts(it["published_at"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                pending_items.append(it)
                out_count += 1
            cursor = t + timedelta(minutes=1)

            due = (cursor - last_flush_anchor) >= timedelta(minutes=flush_every)
            if due or t == batch[-1]:
                total = 0
                if cfg is not None:
                    if pending_items:
                        added, total = _append_raw_store(cfg, pending_items)
                        print(
                            f"[hist] GDELT-GAL flush +{added} "
                            f"(batch_pending={len(pending_items)}) total={total} "
                            f"cursor={cursor.isoformat()} "
                            f"files={n_files} ok={n_ok} 404={n_404} err={n_err}",
                            flush=True,
                        )
                        pending_items = []
                    else:
                        cur_store = _load_raw_store(cfg)
                        total = 0 if cur_store is None else len(cur_store)
                    _write_checkpoint(
                        cfg, start, end, ["gdelt"], total,
                        extra={
                            "gdelt_gal_cursor": cursor.isoformat(),
                            "gdelt_backend": "gal",
                            "gdelt_gal_stats": {
                                "files": n_files,
                                "ok": n_ok,
                                "missing_404": n_404,
                                "err": n_err,
                                "matched": out_count,
                            },
                        },
                    )
                last_flush_anchor = cursor
                print(
                    f"[hist] GDELT-GAL progress cursor={cursor.isoformat()} "
                    f"matched={out_count} files={n_files}",
                    flush=True,
                )

    if cfg is not None and pending_items:
        added, total = _append_raw_store(cfg, pending_items)
        print(
            f"[hist] GDELT-GAL final flush +{added} total={total} "
            f"matched={out_count}",
            flush=True,
        )
        _write_checkpoint(
            cfg, start, end, ["gdelt"], total,
            extra={
                "gdelt_gal_cursor": (end_m + timedelta(minutes=1)).isoformat(),
                "gdelt_backend": "gal",
                "gdelt_gal_stats": {
                    "files": n_files,
                    "ok": n_ok,
                    "missing_404": n_404,
                    "err": n_err,
                    "matched": out_count,
                },
            },
        )

    print(
        f"[hist] GDELT-GAL 完成 matched={out_count} "
        f"files={n_files} ok={n_ok} 404={n_404} err={n_err} "
        f"({start.date()}~{end.date()})",
        flush=True,
    )
    # 条目已写入 store; 返回空列表避免把整年结果再堆进调用方内存
    return []
