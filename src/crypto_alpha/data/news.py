"""真实新闻多源采集 + 权威分层 + 去重互证 + 情绪打分 + 无泄漏 as-of 对齐。

设计要点(多维度考量, 详见 README 的"新闻数据源"章节):
- 权威分层(tier): 官方/监管/交易所公告(最高) > 一线财经 > 一线加密媒体 > 聚合/社媒。
  以 tier_weights 给每条新闻加权, 抑制低可信来源对判断的干扰。
- 多源互证(corroboration): 同一事件被多个独立来源报道 => 置信度上调; 单一小源 => 下调。
- 去重(dedup): 同一新闻常被多家转载, 用标题 token Jaccard 相似度归并, 避免重复计数。
- 无泄漏对齐: 只用 published_at + buffer <= bar_time 的新闻(新闻需要传播时间才可交易)。
- 兜底: 无网络/无 key 时用合成新闻把整条链路跑通(sentiment 与后续行情弱相关, 便于测试)。

所有适配器都做优雅降级: 任何来源失败都不阻断其他来源。
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 归一化 schema: 每条新闻 = dict(published_at[UTC], source, tier, title, url, symbols)
# --------------------------------------------------------------------------
_UA = {"User-Agent": "Mozilla/5.0 (crypto-alpha news fetcher)"}

# GDELT DOC API 全进程级节流/熔断(跨重试与跨窗共享, 避免「窗间 30s + 重试连打」仍超限)
_GDELT_GATE: dict = {
    "last_mono": 0.0,
    "consec_429": 0,
}


def _is_gdelt_url(url: str) -> bool:
    return "gdeltproject.org" in str(url).lower()


def _gdelt_gate_wait(min_interval_sec: float) -> None:
    """确保任意两次 GDELT 请求间隔 >= min_interval(含重试)。"""
    gap = max(0.0, float(min_interval_sec))
    # 连续 429 后强制加长间隔(熔断), 否则官方 5s 建议在实践中不够
    streak = int(_GDELT_GATE.get("consec_429") or 0)
    if streak >= 1:
        gap = max(gap, 60.0 * streak)
    if streak >= 3:
        gap = max(gap, 300.0)
    last = float(_GDELT_GATE.get("last_mono") or 0.0)
    now = time.monotonic()
    wait = last + gap - now
    if wait > 0:
        time.sleep(wait)
    _GDELT_GATE["last_mono"] = time.monotonic()


def _gdelt_note_result(*, ok: bool) -> None:
    if ok:
        _GDELT_GATE["consec_429"] = 0
    else:
        _GDELT_GATE["consec_429"] = int(_GDELT_GATE.get("consec_429") or 0) + 1


def _resolve_http_proxies() -> dict[str, str]:
    """解析 HTTP(S) 代理: 环境变量优先, 其次 macOS 系统代理(scutil)。

    Clash/Surge 等「系统代理」常只写进 macOS 设置, Python urllib 默认不读,
    会导致 curl 能通 GDELT、回填脚本却打到直连旧 IP 一直 429。
    """
    import os

    proxies = {k: v for k, v in urllib.request.getproxies().items() if v}
    # getproxies 可能只有 'all'; 归一到 http/https
    if "all" in proxies and "https" not in proxies:
        proxies["https"] = proxies["all"]
    if "all" in proxies and "http" not in proxies:
        proxies["http"] = proxies["all"]
    # urllib ProxyHandler 只稳妥支持 http/https; socks 键常导致走错或仍直连
    proxies = {k: v for k, v in proxies.items() if k in ("http", "https") and v}
    if proxies.get("http") or proxies.get("https"):
        return proxies

    # macOS SystemConfiguration
    try:
        import subprocess
        import re as _re

        out = subprocess.check_output(
            ["scutil", "--proxy"], text=True, timeout=3, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}

    def _flag(key: str) -> bool:
        m = _re.search(rf"{key}\s*:\s*(\d+)", out)
        return bool(m and m.group(1) == "1")

    def _host_port(host_key: str, port_key: str) -> str | None:
        hm = _re.search(rf"{host_key}\s*:\s*(\S+)", out)
        pm = _re.search(rf"{port_key}\s*:\s*(\d+)", out)
        if not hm or not pm:
            return None
        return f"http://{hm.group(1)}:{pm.group(1)}"

    resolved: dict[str, str] = {}
    if _flag("HTTPEnable"):
        u = _host_port("HTTPProxy", "HTTPPort")
        if u:
            resolved["http"] = u
    if _flag("HTTPSEnable"):
        u = _host_port("HTTPSProxy", "HTTPSPort")
        if u:
            resolved["https"] = u
    # HTTPS 未开时回退 HTTP 代理(多数本地代理两者同端口)
    if "https" not in resolved and "http" in resolved:
        resolved["https"] = resolved["http"]
    return resolved


def _http_get(
    url: str,
    timeout: float = 30.0,
    *,
    max_retries: int = 4,
    base_backoff_sec: float = 20.0,
    min_interval_sec: float | None = None,
) -> bytes | None:
    """GET with retries for transient HTTP errors (esp. GDELT 429).

    On 429, prefer long cool-downs over rapid retries (rapid retries worsen bans).
    Honors ``Retry-After`` when present. Returns ``None`` after exhausting retries.

    GDELT URL 自动走全进程闸门: 任意两次请求间隔受 ``min_interval_sec`` /
    连续 429 熔断约束(默认间隔至少 5s)。
    自动使用环境变量 / macOS 系统 HTTP(S) 代理。
    """
    import urllib.error

    is_gdelt = _is_gdelt_url(url)
    # GDELT: 默认少重试(窗级会再排), 避免单窗连打 3 次把 IP 打进长黑名单
    if is_gdelt and max_retries > 2:
        max_retries = 1
    interval = min_interval_sec
    if interval is None and is_gdelt:
        interval = 5.0

    proxies = _resolve_http_proxies()
    if proxies:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
        open_url = opener.open
    else:
        open_url = urllib.request.urlopen

    last_err: Exception | None = None
    for attempt in range(int(max_retries) + 1):
        if interval is not None:
            _gdelt_gate_wait(float(interval))
        try:
            req = urllib.request.Request(url, headers=_UA)
            with open_url(req, timeout=timeout) as r:
                raw = r.read()
            if is_gdelt:
                _gdelt_note_result(ok=True)
            return raw
        except urllib.error.HTTPError as e:
            last_err = e
            body_preview = ""
            try:
                body_preview = (e.read() or b"")[:160].decode("utf-8", errors="replace")
            except Exception:
                body_preview = ""
            if is_gdelt and e.code == 429:
                _gdelt_note_result(ok=False)
            retryable = e.code in (408, 425, 429, 500, 502, 503, 504)
            if (not retryable) or attempt >= int(max_retries):
                print(
                    f"[warn] 抓取失败 HTTP {e.code} {url[:60]}...: {body_preview or e}",
                    flush=True,
                )
                return None
            ra = e.headers.get("Retry-After") if e.headers else None
            try:
                if ra is not None:
                    sleep_s = float(ra)
                elif e.code == 429:
                    streak = int(_GDELT_GATE.get("consec_429") or 1)
                    # 429: 长冷却; 连续越多越久(上限 15min)
                    sleep_s = max(120.0, float(base_backoff_sec) * (attempt + 2), 90.0 * streak)
                else:
                    sleep_s = base_backoff_sec * (1.5 ** attempt)
            except Exception:
                sleep_s = 120.0 if e.code == 429 else base_backoff_sec
            sleep_s = min(max(sleep_s, 1.0), 900.0)
            print(
                f"[warn] HTTP {e.code} 重试 {attempt + 1}/{max_retries} "
                f"sleep={sleep_s:.1f}s consec_429={_GDELT_GATE.get('consec_429', 0)} "
                f"{url[:60]}...",
                flush=True,
            )
            time.sleep(sleep_s)
        except Exception as e:
            last_err = e
            if attempt >= int(max_retries):
                print(f"[warn] 抓取失败 {url[:60]}...: {e}", flush=True)
                return None
            sleep_s = min(base_backoff_sec * (1.5 ** attempt), 90.0)
            print(
                f"[warn] 网络异常重试 {attempt + 1}/{max_retries} "
                f"sleep={sleep_s:.1f}s ({e})",
                flush=True,
            )
            time.sleep(sleep_s)
    if last_err is not None:
        print(f"[warn] 抓取失败 {url[:60]}...: {last_err}", flush=True)
    return None


# 轻量情绪词典(可替换为 FinBERT/CryptoBERT 等模型)
_POS = {
    "surge", "rally", "bullish", "approve", "approval", "adopt", "adoption", "inflow",
    "record", "gain", "soar", "upgrade", "partnership", "institutional", "etf approved",
    "利好", "上涨", "批准", "通过", "增持", "利多", "突破",
}
_NEG = {
    "hack", "exploit", "ban", "banned", "lawsuit", "sue", "crackdown", "outflow", "dump",
    "crash", "plunge", "reject", "rejected", "fraud", "delist", "liquidation", "sell-off",
    "利空", "下跌", "禁止", "起诉", "抛售", "监管打击", "暴跌", "崩盘",
}

_SYMBOL_KEYWORDS = {
    "BTC/USDT": ["btc", "bitcoin", "比特币"],
    "ETH/USDT": ["eth", "ethereum", "以太坊", "以太"],
}
# 市场级关键词(对 BTC/ETH 均相关): 监管/宏观/交易所
_MARKET_KEYWORDS = ["sec", "cftc", "etf", "fed", "fomc", "cpi", "binance", "coinbase",
                    "regulation", "监管", "美联储", "加息", "降息"]


def _kw_hits(text_lower: str, words: set) -> int:
    """统计词典命中数。

    ASCII 单词用**词边界**匹配, 避免子串误判(如 "against" 含 "gain"、"banks" 含 "ban"、
    "issue" 含 "sue"); 含空格/连字符的短语与中文(无词边界)仍用子串匹配。
    """
    c = 0
    for w in words:
        if (" " in w) or ("-" in w) or not re.fullmatch(r"[a-z0-9]+", w):
            if w in text_lower:
                c += 1
        elif re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", text_lower):
            c += 1
    return c


def _score_sentiment(text: str) -> float:
    t = text.lower()
    pos = _kw_hits(t, _POS)
    neg = _kw_hits(t, _NEG)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _relevant_symbols(text: str) -> list[str]:
    t = text.lower()
    hit = [s for s, kws in _SYMBOL_KEYWORDS.items() if any(k in t for k in kws)]
    if not hit and any(k in t for k in _MARKET_KEYWORDS):
        hit = list(_SYMBOL_KEYWORDS.keys())  # 市场级新闻对两币都相关
    return hit


# --------------------------------------------------------------------------
# 来源适配器(全部 best-effort)
# --------------------------------------------------------------------------
def fetch_rss(name: str, url: str, tier: int) -> list[dict]:
    """通用 RSS/Atom 解析(标准库), 提取 title/pubDate/link。"""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    raw = _http_get(url)
    if raw is None:
        return []
    try:
        root = ET.fromstring(raw)
    except Exception:
        return []
    items = []
    for it in root.iter():
        tag = it.tag.lower().split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        title = pub = link = ""
        for c in it:
            ct = c.tag.lower().split("}")[-1]
            if ct == "title":
                title = (c.text or "").strip()
            elif ct in ("pubdate", "published", "updated"):
                pub = (c.text or "").strip()
            elif ct == "link":
                link = (c.text or c.get("href") or "").strip()
        try:
            dt = parsedate_to_datetime(pub) if pub else None
            if dt is None:
                dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
        except Exception:
            continue
        syms = _relevant_symbols(title)
        if not syms:
            continue
        items.append({"published_at": dt, "source": name, "tier": tier,
                      "title": title, "url": link, "symbols": syms})
    return items


def fetch_cryptopanic(name: str, url: str, tier: int, api_key: str) -> list[dict]:
    """CryptoPanic 聚合 API(需 free api key)。返回带来源与投票的新闻。"""
    if not api_key:
        return []
    q = urllib.parse.urlencode({"auth_token": api_key, "currencies": "BTC,ETH", "public": "true"})
    raw = _http_get(f"{url}?{q}")
    if raw is None:
        return []
    try:
        data = json.loads(raw).get("results", [])
    except Exception:
        return []
    items = []
    for r in data:
        try:
            dt = datetime.fromisoformat(r["published_at"].replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        title = r.get("title", "")
        syms = _relevant_symbols(title) or list(_SYMBOL_KEYWORDS.keys())
        items.append({"published_at": dt, "source": f"{name}:{r.get('source',{}).get('title','')}",
                      "tier": tier, "title": title, "url": r.get("url", ""), "symbols": syms})
    return items


def fetch_cryptocompare(name: str, url: str, tier: int, api_key: str = "") -> list[dict]:
    """CryptoCompare News API(有免费额度), 自带来源标注。"""
    full = url + (f"?api_key={api_key}" if api_key else "")
    raw = _http_get(full)
    if raw is None:
        return []
    try:
        data = json.loads(raw).get("Data", [])
    except Exception:
        return []
    items = []
    for r in data:
        try:
            dt = datetime.fromtimestamp(int(r["published_on"]), tz=timezone.utc)
        except Exception:
            continue
        title = r.get("title", "")
        syms = _relevant_symbols(title + " " + r.get("categories", ""))
        if not syms:
            continue
        items.append({"published_at": dt, "source": f"{name}:{r.get('source','')}",
                      "tier": tier, "title": title, "url": r.get("url", ""), "symbols": syms})
    return items


def fetch_gdelt(name: str, tier: int) -> list[dict]:
    """GDELT 全球新闻(免费, 时间戳可靠)。查询 bitcoin/ethereum 相关英文报道。"""
    url = ("https://api.gdeltproject.org/api/v2/doc/doc?query=(bitcoin%20OR%20ethereum)"
           "&mode=artlist&format=json&maxrecords=75&sort=datedesc")
    raw = _http_get(url)
    if raw is None:
        return []
    try:
        data = json.loads(raw).get("articles", [])
    except Exception:
        return []
    items = []
    for r in data:
        try:
            dt = datetime.strptime(r["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        title = r.get("title", "")
        syms = _relevant_symbols(title)
        if not syms:
            continue
        items.append({"published_at": dt, "source": f"{name}:{r.get('domain','')}",
                      "tier": tier, "title": title, "url": r.get("url", ""), "symbols": syms})
    return items


# --------------------------------------------------------------------------
# 去重 + 互证
# --------------------------------------------------------------------------
def _norm_tokens(title: str) -> set:
    return set(re.sub(r"[^a-z0-9\u4e00-\u9fff ]", " ", title.lower()).split())


def dedup_corroborate(items: list[dict], jaccard: float = 0.6,
                      window_hours: float = 48.0) -> list[dict]:
    """按标题相似度归并重复报道, **按报道时刻输出 point-in-time 快照**。

    每条报道保留自身 ``published_at``; ``corroboration`` / ``tier`` 仅反映
    **截至该报道时刻**已归入同簇的独立来源(含本条)。禁止把数小时后跟进的
    高权威源回写到首发时刻(否则 as-of 面板会在 T=0 看到未来互证)。

    仅在 window_hours 时间窗内归并: 互证是"多源近乎同时报道同一事件", 跨越数周/数月
    的同名标题不应合并。只需与"活跃簇"(最近一次更新在窗口内)比较。
    """
    items = sorted(items, key=lambda x: x["published_at"])
    win = timedelta(hours=float(window_hours))
    out: list[dict] = []
    active: list[dict] = []  # 簇状态(可变); 对外只 emit PIT 快照
    for it in items:
        toks = _norm_tokens(it["title"])
        t = it["published_at"]
        src = it["source"].split(":")[0]
        active = [c for c in active if t - c["_last"] <= win]
        placed = False
        for c in active:
            inter = len(toks & c["_toks"])
            union = len(toks | c["_toks"]) or 1
            if inter / union >= jaccard:
                c["sources"].add(src)
                c["tier"] = min(c["tier"], it["tier"])
                c["_toks"] |= toks
                c["_last"] = t
                # PIT: 快照时刻 = 本条报道时刻, 互证/权威 = 此刻已知状态
                out.append({
                    "published_at": t,
                    "title": it["title"],
                    "tier": c["tier"],
                    "symbols": it["symbols"],
                    "sources": ",".join(sorted(c["sources"])),
                    "corroboration": len(c["sources"]),
                })
                placed = True
                break
        if not placed:
            sources = {src}
            active.append({
                "tier": it["tier"], "sources": sources,
                "_toks": toks, "_last": t,
            })
            out.append({
                "published_at": t,
                "title": it["title"],
                "tier": it["tier"],
                "symbols": it["symbols"],
                "sources": src,
                "corroboration": 1,
            })
    return out


# --------------------------------------------------------------------------
# 聚合成按时间的摘要面板(供 LLM 消费)
# --------------------------------------------------------------------------
def build_news_panel(cfg, symbol: str) -> pd.DataFrame:
    """产出按 bucket(默认1h) 聚合、权威加权的新闻摘要面板。

    列: text(供 LLM 的摘要), sentiment(权威加权情绪), n_items, max_authority, corroboration
    索引: bucket 时间(UTC); 无新闻的 bucket 不出现(对齐时 ffill)。

    数据源优先级(自适应):
      1) news.use_history=true 且历史原始库非空 => 用回填的多年语料聚合(支撑几年期回测);
      2) 否则 news.use_synthetic=true => 合成新闻(离线兜底);
      3) 否则实时抓取当前快照(sources)。
    """
    clusters = _collect_clusters(cfg, symbol)
    return _aggregate_clusters(clusters, cfg)


def _collect_clusters(cfg, symbol: str) -> list[dict]:
    """按优先级选择新闻来源并归并成 clusters。"""
    ncfg = cfg["news"]
    win = float(ncfg.get("dedup_window_hours", 48.0))
    if ncfg.get("use_history", False):
        # 历史库若仅配置 synthetic provider: 真实行情下拒绝(研究口径污染)。
        # 注意: 合成*历史*语料是随机标题, 与 use_synthetic 面板路径(未来收益造情绪)不同;
        # 混回填残留另由 _raw_to_items 按行过滤。
        hist_providers = [
            str(p).lower()
            for p in (ncfg.get("history") or {}).get("providers") or []
        ]
        if (
            hist_providers
            and all(p == "synthetic" for p in hist_providers)
            and not cfg["data"].get("use_synthetic", False)
        ):
            raise ValueError(
                "检测到 data.use_synthetic=false 但 news.history.providers 仅含 synthetic: "
                "合成历史语料仅供离线打通链路(随机标题, 并非由未来收益构造), "
                "与真实行情混用会污染研究口径。"
                "请将 history.providers 改为 cryptocompare/gdelt 等真实源, "
                "或先关闭 news.use_history; 离线演示请同时打开 data.use_synthetic。"
            )
        raw = _load_raw_store(cfg)
        if raw is not None and len(raw):
            items = _raw_to_items(raw, symbol, cfg)
            if items:
                return dedup_corroborate(items, window_hours=win)
        print("[warn] news.use_history=true 但历史原始库为空或过滤后无可用条目; 请先运行 09_backfill_news。")
    if ncfg.get("use_synthetic", False):
        # 防前视泄漏: 此路径(_synthetic_clusters)情绪由**未来**收益构造。
        # 若行情为真实数据, 用它会把未来信息注入特征 -> 严重泄漏。
        if not cfg["data"].get("use_synthetic", False):
            raise ValueError(
                "检测到 data.use_synthetic=false 但 news.use_synthetic=true: "
                "news.use_synthetic 面板路径由未来收益构造情绪, 用于真实行情会造成前视泄漏。"
                "请改用 news.use_history=true(先运行 09_backfill_news)或配置真实新闻源, "
                "或将 news.as_feature 设为 false 关闭新闻特征。"
            )
        return _synthetic_clusters(cfg, symbol)
    items = []
    for s in ncfg.get("sources", []):
        items += _fetch_source(cfg, s)
    items = [it for it in items if symbol in it["symbols"]]
    return dedup_corroborate(items, window_hours=win)


def _aggregate_clusters(clusters: list[dict], cfg) -> pd.DataFrame:
    """把去重互证后的 clusters 聚合成按 bucket 的权威加权摘要面板。"""
    ncfg = cfg["news"]
    tw = {int(k): float(v) for k, v in ncfg["tier_weights"].items()}
    if not clusters:
        return pd.DataFrame(columns=["text", "sentiment", "n_items", "max_authority", "corroboration"])

    df = pd.DataFrame(clusters)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["authority"] = df["tier"].map(tw).fillna(0.1) * (1 + 0.3 * (df["corroboration"] - 1)).clip(upper=2.0)

    # 情绪打分: CryptoBERT/FinBERT(可配置), 失败回退词典
    from .sentiment import build_scorer

    scorer = build_scorer(cfg)
    df["sent"] = scorer.score(df["title"].astype(str).tolist())

    from .fetch import timeframe_to_pandas_freq

    bucket = ncfg.get("bucket", "1h")
    # ccxt 风格(30m)→pandas 安全 freq(30min); 避免 Grouper 把 m 当成「月」
    bucket_freq = timeframe_to_pandas_freq(str(bucket))
    freq_td = pd.to_timedelta(bucket_freq)  # 桶宽, 用于把桶左沿标记改为桶末(可用时刻)
    df = df.set_index("published_at").sort_index()
    top_k = int(ncfg.get("top_k", 3))

    rows = []
    for ts, g in df.groupby(pd.Grouper(freq=bucket_freq)):
        if len(g) == 0:
            continue
        # 桶 [ts, ts+freq) 内的新闻在桶末才算"可用", 以此为时间戳可根除同期前视泄漏
        ts = ts + freq_td
        g = g.sort_values("authority", ascending=False)
        w = g["authority"].values
        sent = float(np.average(g["sent"].values, weights=w)) if w.sum() > 0 else 0.0
        tone = "偏多" if sent > 0.15 else ("偏空" if sent < -0.15 else "中性")
        conf = "高" if g["tier"].min() <= 1 else ("中" if g["tier"].min() <= 2 else "低")
        heads = "; ".join(
            f"[T{int(t)}|证{int(c)}] {ti[:60]}"
            for ti, t, c in zip(g["title"].head(top_k), g["tier"].head(top_k),
                                g["corroboration"].head(top_k))
        )
        text = f"情绪:{tone}(可信度{conf}); {heads}"
        rows.append({"published_at": ts, "text": text, "sentiment": round(sent, 3),
                     "n_items": int(len(g)), "max_authority": round(float(w.max()), 3),
                     "corroboration": int(g["corroboration"].max())})
    out = pd.DataFrame(rows).set_index("published_at").sort_index()
    out.index.name = "timestamp"
    return out


def _fetch_source(cfg, s: dict) -> list[dict]:
    import os

    typ = s.get("type")
    name, tier = s["name"], int(s.get("tier", 3))
    if typ == "rss":
        return fetch_rss(name, s["url"], tier)
    if typ == "cryptopanic":
        return fetch_cryptopanic(name, s["url"], tier, os.environ.get(s.get("api_key_env", ""), ""))
    if typ == "cryptocompare":
        return fetch_cryptocompare(name, s["url"], tier, os.environ.get(s.get("api_key_env", ""), ""))
    if typ == "gdelt":
        return fetch_gdelt(name, tier)
    return []


# --------------------------------------------------------------------------
# 无泄漏 as-of 对齐
# --------------------------------------------------------------------------
def align_news_asof(
    news_df: pd.DataFrame, timestamps, buffer_minutes: int = 5,
    ttl_hours: float | None = None,
    decision_delta: pd.Timedelta | None = None,
) -> dict:
    """把新闻摘要 as-of 对齐到**决策时刻**(与数值新闻特征 / MTF 口径一致)。

    决策时刻 = 事件 bar 开盘 + ``decision_delta``(主周期长度); 仅取
    published_at + buffer <= 决策时刻 的最新摘要。输出字典仍以原始事件时间
    (bar 开盘)为键, 便于按 ``X.index`` 查表。

    ttl_hours: 若给定, 距**决策时刻**超过该时长的新闻视为过期 => 置空, 避免 ffill
    把几天前的旧新闻当作"最近新闻"一直塞进 LLM 提示。
    """
    if news_df is None or len(news_df) == 0:
        return {}
    ts_index = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True))
    if decision_delta is not None:
        decision_at = (ts_index + pd.Timedelta(decision_delta)).astype("datetime64[ns, UTC]")
    else:
        decision_at = ts_index
    # 平移新闻时间以纳入传播缓冲(新闻发布后需一定时间才可交易)
    shifted = news_df.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=buffer_minutes)
    shifted = shifted.sort_index()
    union = shifted.index.union(decision_at)
    aligned = shifted["text"].reindex(union).ffill().reindex(decision_at)
    # 记录每个决策时刻所对齐到的新闻(缓冲后)可用时刻, 用于 TTL 过期判定
    avail = pd.Series(shifted.index, index=shifted.index).reindex(union).ffill().reindex(decision_at)

    out = {}
    for orig_ts, dec_ts in zip(ts_index, decision_at):
        txt = aligned.loc[dec_ts]
        if pd.isna(txt):
            out[orig_ts] = ""
            continue
        if ttl_hours is not None and pd.notna(avail.loc[dec_ts]):
            age_h = (dec_ts - avail.loc[dec_ts]).total_seconds() / 3600.0
            if age_h > float(ttl_hours):
                out[orig_ts] = ""
                continue
        out[orig_ts] = txt
    return out


# --------------------------------------------------------------------------
# 合成新闻(离线兜底/测试): sentiment 与后续行情弱相关, 使其具有可学习信号
# --------------------------------------------------------------------------
def _synthetic_clusters(cfg, symbol: str) -> list[dict]:
    from . import load_symbol_data
    from .fetch import stable_symbol_offset

    rng = np.random.default_rng(cfg.seed + stable_symbol_offset(symbol, 1000))
    df = load_symbol_data(cfg, symbol)
    close = df["close"]
    fwd = np.log(close.shift(-6) / close)  # 未来 6 bar 收益(新闻应先于行情)
    src_pool = [("SEC", 1), ("Reuters", 1), ("CoinDesk", 2), ("TheBlock", 2),
                ("Cointelegraph", 3), ("CryptoPanic", 4)]
    asset = "Bitcoin" if "BTC" in symbol else "Ethereum"
    n = max(30, len(close) // 40)  # 约每 40 bar 一条
    locs = np.sort(rng.choice(len(close) - 6, size=min(n, len(close) - 6), replace=False))
    clusters = []
    for loc in locs:
        f = fwd.iloc[loc]
        if not np.isfinite(f):
            continue
        bullish = (f + rng.normal(0, 0.004)) > 0  # 情绪与未来收益弱相关 + 噪声
        src, tier = src_pool[rng.integers(len(src_pool))]
        corr = int(rng.integers(1, 4))
        topic = rng.choice(["ETF", "监管", "机构增持", "交易所", "链上巨鲸", "宏观"])
        word = "利好" if bullish else "利空"
        title = f"{asset} {topic}{word}消息"
        clusters.append({
            "published_at": close.index[loc], "title": title, "tier": tier,
            "symbols": [symbol], "sources": src, "corroboration": corr,
        })
    return clusters


# --------------------------------------------------------------------------
# 落盘 / 加载 / 路径解析
# --------------------------------------------------------------------------
def news_path_for(cfg, symbol: str) -> Path:
    out_dir = cfg.root / cfg["news"].get("output_dir", "data/news")
    return out_dir / (symbol.replace("/", "_") + ".parquet")


def save_news_panel(cfg, symbol: str, df: pd.DataFrame) -> Path:
    p = news_path_for(cfg, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, engine="pyarrow")
    return p


def load_news_panel(cfg, symbol: str) -> pd.DataFrame | None:
    # 优先用 experts.llm.news_path 显式指定(单文件); 否则按 news.output_dir/<symbol>
    explicit = cfg["experts"]["llm"].get("news_path")
    p = Path(cfg.root / explicit) if explicit else news_path_for(cfg, symbol)
    if not p.exists():
        return None
    return pd.read_parquet(p, engine="pyarrow")


def load_news_for_events(cfg, symbol: str, timestamps) -> dict:
    """便捷函数: 加载新闻面板并 as-of 对齐到决策时刻(供训练/推理共用)。"""
    from .fetch import timeframe_delta

    df = load_news_panel(cfg, symbol)
    buf = int(cfg["news"].get("buffer_minutes", 5))
    ttl = float(cfg["news"].get("feature_ttl_hours", 24))
    delta = timeframe_delta(cfg["data"]["timeframe"])
    return align_news_asof(
        df, timestamps, buffer_minutes=buf, ttl_hours=ttl, decision_delta=delta,
    )


def ensure_news_panel(cfg, symbol: str) -> pd.DataFrame | None:
    """加载新闻面板; 缺失时按配置自动 ``build_news_panel`` 并落盘。

    ``news.auto_build_panel`` 默认 true。构建失败或仍为空则返回 None
    (由调用方走空特征 / coverage warn; 若 ``news.require_panel`` 则抛错)。
    """
    ncfg = cfg["news"]
    df = load_news_panel(cfg, symbol)
    if df is not None and len(df):
        return df
    if not bool(ncfg.get("auto_build_panel", True)):
        if bool(ncfg.get("require_panel", False)):
            raise FileNotFoundError(
                f"新闻面板缺失且 auto_build_panel=false: {news_path_for(cfg, symbol)}"
            )
        return None
    try:
        built = build_news_panel(cfg, symbol)
    except Exception as ex:
        print(f"[warn] {symbol}: 自动构建新闻面板失败: {ex}")
        built = None
    if built is not None and len(built):
        path = save_news_panel(cfg, symbol, built)
        print(f"[news] {symbol}: 已自动构建并保存新闻面板 -> {path} ({len(built)} buckets)")
        return built
    if bool(ncfg.get("require_panel", False)):
        raise FileNotFoundError(
            f"新闻面板缺失且自动构建为空: {news_path_for(cfg, symbol)}; "
            "请运行 08_fetch_news / 09_backfill_news, 或关闭 news.require_panel。"
        )
    return None


# ==========================================================================
# 历史新闻回填(多年期回测): 分页/分窗抓取真实历史新闻 -> 追加去重到原始语料库
# --------------------------------------------------------------------------
# 设计:
# - 原始语料库(append-only, 去重): data/news_raw/corpus.parquet, 一条=一篇原始报道。
#   与聚合面板分离, 使"多年语料的抓取"可增量、可续跑, 面板可随时重建。
# - 续跑: backfill_state.json 记录各 provider 已覆盖的 [min,max] 时间; 追加天然幂等(去重)。
# - 限速+重试: 对公共 API 友好(避免被封), 任何来源失败不阻断其他来源。
# ==========================================================================
def _history_cfg(cfg) -> dict:
    h = dict(cfg["news"].get("history", {}) or {})
    h.setdefault("raw_dir", "data/news_raw")
    h.setdefault("providers", ["synthetic"])
    h.setdefault("window_days", 2)
    h.setdefault("rate_limit_sec", 90.0)
    h.setdefault("max_windows", 400)
    h.setdefault("max_pages", 400)
    h.setdefault("gdelt_query", "(bitcoin OR ethereum)")
    h.setdefault("gdelt_http_retries", 1)
    h.setdefault("gdelt_429_cooldown_sec", 300.0)
    h.setdefault("synthetic_per_day", 8)
    return h


def _raw_store_path(cfg) -> Path:
    return cfg.root / _history_cfg(cfg)["raw_dir"] / "corpus.parquet"


def _checkpoint_path(cfg) -> Path:
    return cfg.root / _history_cfg(cfg)["raw_dir"] / "backfill_state.json"


def _parse_dt(value, default=None):
    """把 ISO 字符串/None 解析为 UTC aware datetime。"""
    if value is None or value == "":
        return default
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_raw_store(cfg) -> pd.DataFrame | None:
    p = _raw_store_path(cfg)
    if not p.exists():
        return None
    df = pd.read_parquet(p, engine="pyarrow")
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def _append_raw_store(cfg, items: list[dict]) -> tuple[int, int]:
    """把新抓取的原始条目追加进语料库并去重。返回 (新增数, 总数)。"""
    if not items:
        cur = _load_raw_store(cfg)
        return 0, (0 if cur is None else len(cur))
    rows = []
    for it in items:
        syms = it.get("symbols", [])
        ts = pd.Timestamp(it["published_at"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        ts = ts.floor("s")  # 与 _news_dedupe_ts / 内存 seen 对齐
        rows.append({
            "published_at": ts,
            "source": str(it.get("source", "")),
            "tier": int(it.get("tier", 3)),
            "title": str(it.get("title", "")),
            "url": str(it.get("url", "")),
            "symbols": ",".join(syms) if isinstance(syms, (list, tuple, set)) else str(syms),
        })
    new = pd.DataFrame(rows)
    cur = _load_raw_store(cfg)
    combined = new if cur is None else pd.concat([cur, new], ignore_index=True)
    combined["published_at"] = pd.to_datetime(combined["published_at"], utc=True).dt.floor("s")
    before = 0 if cur is None else len(cur)
    combined = combined.drop_duplicates(subset=["source", "title", "published_at"]).sort_values("published_at")
    p = _raw_store_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(p, engine="pyarrow", index=False)
    return len(combined) - before, len(combined)


def _is_synthetic_news_source(source) -> bool:
    """识别合成语料 source 标记(回填为 ``synthetic:SEC`` 等; 兼容裸 ``synthetic``)。"""
    s = str(source or "").strip().lower()
    return s == "synthetic" or s.startswith("synthetic:")


def _raw_to_items(raw: pd.DataFrame, symbol: str, cfg) -> list[dict]:
    """从原始语料库筛出与 symbol 相关的条目, 转回归一化 items 供 dedup_corroborate。

    当 ``data.use_synthetic=false`` 时过滤 ``source`` 为 synthetic 的行:
    防止曾用 synthetic provider 混回填后、即便 providers 已改成真源, 脏行仍进入特征。
    ``data.use_synthetic=true`` 时保留(离线全合成链路)。不修改磁盘 corpus。
    """
    hcfg = _history_cfg(cfg)
    start = _parse_dt(hcfg.get("start"))
    end = _parse_dt(hcfg.get("end"))
    df = raw
    if start is not None:
        df = df[df["published_at"] >= start]
    if end is not None:
        df = df[df["published_at"] <= end]
    allow_synthetic = bool(cfg["data"].get("use_synthetic", False))
    items = []
    n_skip_syn = 0
    for r in df.itertuples(index=False):
        if not allow_synthetic and _is_synthetic_news_source(getattr(r, "source", "")):
            n_skip_syn += 1
            continue
        syms = [s for s in str(r.symbols).split(",") if s]
        if symbol not in syms:
            continue
        items.append({
            "published_at": r.published_at.to_pydatetime(), "source": r.source,
            "tier": int(r.tier), "title": r.title, "url": r.url, "symbols": syms,
        })
    if n_skip_syn:
        print(
            f"[warn] 真实行情下已从历史语料过滤 {n_skip_syn} 条 synthetic 条目"
            f"(source 以 synthetic: 开头); 避免离线合成新闻混入真实研究。"
            f"若需使用合成语料请设 data.use_synthetic=true。"
        )
    return items


def fetch_cryptocompare_history(name, url, tier, api_key, start, end,
                                rate_limit_sec=1.0, max_pages=400) -> list[dict]:
    """CryptoCompare 历史新闻: 用 lTs 向历史方向分页(每页约50条)直到覆盖 start。"""
    base = url.split("?")[0]
    lang = "EN"
    lts = int(end.timestamp())
    start_ts = int(start.timestamp())
    out: list[dict] = []
    seen = set()
    for _ in range(int(max_pages)):
        q = {"lang": lang, "lTs": lts}
        if api_key:
            q["api_key"] = api_key
        raw = _http_get(f"{base}?{urllib.parse.urlencode(q)}")
        if raw is None:
            break
        try:
            data = json.loads(raw).get("Data", [])
        except Exception:
            break
        if not data:
            break
        oldest = lts
        added = 0
        for r in data:
            try:
                on = int(r["published_on"])
            except Exception:
                continue
            oldest = min(oldest, on)
            if on < start_ts or r.get("id") in seen:
                continue
            seen.add(r.get("id"))
            dt = datetime.fromtimestamp(on, tz=timezone.utc)
            title = r.get("title", "")
            syms = _relevant_symbols(title + " " + r.get("categories", ""))
            if not syms:
                continue
            out.append({"published_at": dt, "source": f"{name}:{r.get('source','')}",
                        "tier": tier, "title": title, "url": r.get("url", ""), "symbols": syms})
            added += 1
        if oldest <= start_ts or oldest >= lts:
            break
        lts = oldest - 1
        time.sleep(max(0.0, float(rate_limit_sec)))
    print(f"[hist] CryptoCompare 历史抓取 {len(out)} 条 ({start.date()}~{end.date()})")
    return out


def _news_dedupe_ts(dt) -> str:
    """统一去重时间键: UTC 秒精度 ISO, 避免 datetime vs Timestamp / us 精度不一致。"""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("s").isoformat()


def fetch_gdelt_history(name, tier, start, end, window_days=7,
                        rate_limit_sec=5.0, max_windows=400,
                        query="(bitcoin OR ethereum)",
                        cfg=None, resume_from=None,
                        http_retries: int = 1,
                        cooldown_429_sec: float = 300.0) -> list[dict]:
    """GDELT 历史新闻: 把 [start,end] 切成窗口逐窗抓取。

    时间戳取 ``seendate``(GDELT 首次见到文章的 UTC; 常量化到 15min 网格)。
    API 日期参数必须为 ``YYYYMMDDHHMMSS``(无 ``T``); ``maxrecords`` 上限 250。

    **覆盖正确性**: 若某窗返回满 250 条, 说明该窗可能被截断 — 自动对半分窗重抓,
    直到不满额或窗宽 < 1 小时。避免 ``sort=datedesc`` + 宽窗系统性丢掉窗前半段。

    **限流**: DOC API 对高频极敏感。``rate_limit_sec`` 同时作为全进程最小请求间隔;
    429 后用 ``cooldown_429_sec`` 与连续失败熔断, ``http_retries`` 宜小(默认 1)。
    """
    out: list[dict] = []
    seen = set()
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
    cur = resume_from if resume_from is not None else start
    if cur < start:
        cur = start
    n = 0
    n_fail = 0
    n_split = 0
    q_enc = urllib.parse.quote(query)
    max_per_window = 250
    min_split = timedelta(hours=1)
    http_retries = max(0, int(http_retries))
    cool_429 = max(60.0, float(cooldown_429_sec))

    def _fetch_window(w0: datetime, w1: datetime) -> tuple[list[dict], int]:
        """返回 (相关条目, 原始 articles 条数); 原始条数用于判断是否截断。"""
        url = (f"https://api.gdeltproject.org/api/v2/doc/doc?query={q_enc}"
               f"&mode=artlist&format=json&maxrecords={max_per_window}&sort=datedesc"
               f"&startdatetime={w0.strftime('%Y%m%d%H%M%S')}"
               f"&enddatetime={w1.strftime('%Y%m%d%H%M%S')}")
        raw = _http_get(
            url,
            timeout=45.0,
            max_retries=http_retries,
            base_backoff_sec=max(60.0, float(rate_limit_sec)),
            min_interval_sec=float(rate_limit_sec),
        )
        if raw is None:
            return [], -1
        text = raw.decode("utf-8", errors="replace").strip()
        if not text.startswith("{") and not text.startswith("["):
            print(f"[warn] GDELT 非 JSON 响应窗 {w0.date()}~{w1.date()}: {text[:120]}")
            return [], -1
        try:
            articles = json.loads(text).get("articles", [])
        except Exception as e:
            print(f"[warn] GDELT JSON 解析失败窗 {w0.date()}: {e}")
            return [], -1
        items = []
        for r in articles:
            try:
                dt = datetime.strptime(
                    r["seendate"], "%Y%m%dT%H%M%SZ"
                ).replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if dt < start or dt > end:
                continue
            title = r.get("title", "")
            src = f"{name}:{r.get('domain', '')}"
            key = (src, title, _news_dedupe_ts(dt))
            if key in seen:
                continue
            syms = _relevant_symbols(title)
            if not syms:
                continue
            items.append({
                "published_at": dt,
                "source": src,
                "tier": tier,
                "title": title,
                "url": r.get("url", ""),
                "symbols": syms,
                # 与 _append_raw_store 去重键对齐: source+title+published_at
                "_key": key,
            })
        return items, len(articles)

    # 待处理区间队列(支持满额自动对半分窗 + 失败重试)
    from collections import deque

    win = timedelta(days=int(window_days))
    t = cur
    queue: deque[tuple[datetime, datetime, int]] = deque()  # (w0,w1,attempt)
    while t < end:
        t2 = min(t + win, end)
        queue.append((t, t2, 0))
        t = t2

    # 续跑时把上次放弃/截断窗重新入队(插到队首), 避免 cursor 前进后永久空洞
    if cfg is not None:
        st_path = _checkpoint_path(cfg)
        if st_path.exists():
            try:
                st0 = json.loads(st_path.read_text(encoding="utf-8"))
            except Exception:
                st0 = {}
            replay = []
            for pair in (st0.get("gdelt_failed_windows") or []):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                a, b = _parse_dt(pair[0]), _parse_dt(pair[1])
                if a is not None and b is not None and a < b:
                    replay.append((a, b, 0))
            for pair in (st0.get("gdelt_truncated_windows") or []):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                a, b = _parse_dt(pair[0]), _parse_dt(pair[1])
                if a is not None and b is not None and a < b:
                    replay.append((a, b, 0))
            for pair in (st0.get("gdelt_pending_windows") or []):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                a, b = _parse_dt(pair[0]), _parse_dt(pair[1])
                att = int(pair[2]) if len(pair) >= 3 else 0
                if a is not None and b is not None and a < b:
                    replay.append((a, b, max(0, att)))
            for item in reversed(replay):
                queue.appendleft(item)
            if replay:
                print(
                    f"[hist] GDELT 续跑补洞: 重新入队失败/截断/pending 窗 {len(replay)} 个",
                    flush=True,
                )
                # 清空列表, 本次跑完若再失败会重新写入
                st0["gdelt_failed_windows"] = []
                st0["gdelt_truncated_windows"] = []
                st0["gdelt_pending_windows"] = []
                st_path.write_text(
                    json.dumps(st0, ensure_ascii=False, indent=2), encoding="utf-8",
                )

    print(f"[hist] GDELT 从 {cur.isoformat()} 扫到 {end.isoformat()} "
          f"(window_days={window_days}, rate={rate_limit_sec}s, "
          f"http_retries={http_retries}, cool_429={cool_429:.0f}s, 满额自动切分)",
          flush=True)

    max_window_attempts = 5
    while queue and n < int(max_windows):
        w0, w1, attempt = queue.popleft()
        items, n_raw = _fetch_window(w0, w1)
        n += 1
        if n_raw < 0:
            n_fail += 1
            if attempt + 1 < max_window_attempts:
                streak = int(_GDELT_GATE.get("consec_429") or 1)
                cool = max(cool_429, float(rate_limit_sec) * 3, 90.0 * streak)
                cool = min(cool, 900.0)
                print(
                    f"[warn] GDELT 窗失败 {w0.isoformat()}~{w1.isoformat()} "
                    f"attempt={attempt+1}/{max_window_attempts}, "
                    f"冷却 {cool:.0f}s 后重试 (consec_429={streak})",
                    flush=True,
                )
                time.sleep(cool)
                queue.append((w0, w1, attempt + 1))
                # 立即记 pending: 否则后续成功窗推高 cursor 后进程崩溃会永久丢洞
                if cfg is not None:
                    st_path = _checkpoint_path(cfg)
                    st = {}
                    if st_path.exists():
                        try:
                            st = json.loads(st_path.read_text(encoding="utf-8"))
                        except Exception:
                            st = {}
                    pending = [
                        p for p in (st.get("gdelt_pending_windows") or [])
                        if not (
                            isinstance(p, (list, tuple)) and len(p) >= 2
                            and str(p[0]) == w0.isoformat()
                            and str(p[1]) == w1.isoformat()
                        )
                    ]
                    pending.append([w0.isoformat(), w1.isoformat(), int(attempt + 1)])
                    st["gdelt_pending_windows"] = pending[-500:]
                    st_path.write_text(
                        json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8",
                    )
            else:
                print(
                    f"[warn] GDELT 窗放弃 {w0.isoformat()}~{w1.isoformat()} "
                    f"(已重试 {max_window_attempts} 次)",
                    flush=True,
                )
                if cfg is not None:
                    # 记录空洞供后续补洞, 避免 cursor 越过失败窗后永久丢失
                    st_path = _checkpoint_path(cfg)
                    st = {}
                    if st_path.exists():
                        try:
                            st = json.loads(st_path.read_text(encoding="utf-8"))
                        except Exception:
                            st = {}
                    failed = list(st.get("gdelt_failed_windows") or [])
                    failed.append([w0.isoformat(), w1.isoformat()])
                    st["gdelt_failed_windows"] = failed[-500:]
                    # 放弃后不再留在 pending
                    st["gdelt_pending_windows"] = [
                        p for p in (st.get("gdelt_pending_windows") or [])
                        if not (
                            isinstance(p, (list, tuple)) and len(p) >= 2
                            and str(p[0]) == w0.isoformat()
                            and str(p[1]) == w1.isoformat()
                        )
                    ]
                    st_path.parent.mkdir(parents=True, exist_ok=True)
                    st_path.write_text(
                        json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8",
                    )
            if cfg is not None:
                cur_store = _load_raw_store(cfg)
                store_total = 0 if cur_store is None else len(cur_store)
                _write_checkpoint(
                    cfg, start, end, ["gdelt"], int(store_total),
                    extra={
                        "gdelt_cursor": w0.isoformat(),
                        "gdelt_windows_done": n,
                        "gdelt_fail_windows": n_fail,
                        "gdelt_split_windows": n_split,
                    },
                )
            time.sleep(max(0.0, float(rate_limit_sec)))
            continue

        # 满额且窗仍可分 → 对半切分, 本窗结果丢弃(子窗会重抓, 防 datedesc 截断)
        if n_raw >= max_per_window and (w1 - w0) > min_split * 2:
            mid = w0 + (w1 - w0) / 2
            queue.appendleft((mid, w1, 0))
            queue.appendleft((w0, mid, 0))
            n_split += 1
            print(
                f"[hist] GDELT 窗满额 {n_raw} @ {w0.isoformat()}~{w1.isoformat()} "
                f"→ 切分为 2 子窗 (累计切分 {n_split})",
                flush=True,
            )
            time.sleep(max(0.0, float(rate_limit_sec)))
            continue
        if n_raw >= max_per_window:
            print(
                f"[warn] GDELT 窗仍满额但不可再切 "
                f"{w0.isoformat()}~{w1.isoformat()} (raw={n_raw}); "
                f"datedesc 可能截断窗前半 — 记 gdelt_truncated_window",
                flush=True,
            )
            if cfg is not None:
                st_path = _checkpoint_path(cfg)
                st = {}
                if st_path.exists():
                    try:
                        st = json.loads(st_path.read_text(encoding="utf-8"))
                    except Exception:
                        st = {}
                trunc = list(st.get("gdelt_truncated_windows") or [])
                trunc.append([w0.isoformat(), w1.isoformat(), int(n_raw)])
                st["gdelt_truncated_windows"] = trunc[-200:]
                st_path.write_text(
                    json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8",
                )

        window_items = []
        for it in items:
            key = it.pop("_key")
            seen.add(key)
            window_items.append(it)
            out.append(it)

        store_total = None
        if cfg is not None and window_items:
            added, store_total = _append_raw_store(cfg, window_items)
            print(
                f"[hist] GDELT 窗 {w0.isoformat()}~{w1.isoformat()}: "
                f"raw={n_raw} +{len(window_items)} 相关 / +{added} 入库 "
                f"(库总量 {store_total})",
                flush=True,
            )
        elif cfg is not None:
            cur_store = _load_raw_store(cfg)
            store_total = 0 if cur_store is None else len(cur_store)
            if n_raw == 0:
                print(f"[hist] GDELT 窗 {w0.date()}~{w1.date()}: 空", flush=True)
        if cfg is not None:
            # 成功后从 pending 移除本窗(若曾失败重试)
            st_path = _checkpoint_path(cfg)
            if st_path.exists():
                try:
                    st = json.loads(st_path.read_text(encoding="utf-8"))
                except Exception:
                    st = {}
                pend = st.get("gdelt_pending_windows") or []
                if pend:
                    st["gdelt_pending_windows"] = [
                        p for p in pend
                        if not (
                            isinstance(p, (list, tuple)) and len(p) >= 2
                            and str(p[0]) == w0.isoformat()
                            and str(p[1]) == w1.isoformat()
                        )
                    ]
                    st_path.write_text(
                        json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8",
                    )
            _write_checkpoint(
                cfg, start, end, ["gdelt"],
                int(store_total or 0),
                extra={
                    "gdelt_cursor": w1.isoformat(),
                    "gdelt_windows_done": n,
                    "gdelt_fail_windows": n_fail,
                    "gdelt_split_windows": n_split,
                },
            )
        time.sleep(max(0.0, float(rate_limit_sec)))

    if queue:
        print(f"[warn] GDELT 达到 max_windows={max_windows}, 仍剩 {len(queue)} 个子窗未抓; "
              f"请提高 max_windows 后续跑(cursor={queue[0][0].isoformat()})",
              flush=True)
        if cfg is not None:
            _write_checkpoint(
                cfg, start, end, ["gdelt"],
                0 if _load_raw_store(cfg) is None else len(_load_raw_store(cfg)),
                extra={
                    "gdelt_cursor": queue[0][0].isoformat(),
                    "gdelt_windows_done": n,
                    "gdelt_fail_windows": n_fail,
                    "gdelt_split_windows": n_split,
                    "gdelt_incomplete": True,
                },
            )

    print(
        f"[hist] GDELT 历史抓取 {len(out)} 条 / {n} 请求 "
        f"(失败 {n_fail}, 切分 {n_split}) ({start.date()}~{end.date()})",
        flush=True,
    )
    return out


def _synthetic_history_items(cfg, start, end) -> list[dict]:
    """离线兜底: 在 [start,end] 生成多年合成语料(打通回填->聚合->回测)。

    标题情绪随机(利好/利空), **不**读取未来价格——与 ``_synthetic_clusters``
    (``news.use_synthetic`` 面板路径, 由未来收益构造)不同。真实行情下不得混用;
    ``_raw_to_items`` 在 ``data.use_synthetic=false`` 时会过滤此类 source。
    """
    hcfg = _history_cfg(cfg)
    rng = np.random.default_rng(cfg.seed + 20260716)
    per_day = int(hcfg.get("synthetic_per_day", 8))
    src_pool = [("SEC", 1), ("Reuters", 1), ("CoinDesk", 2), ("TheBlock", 2),
                ("Cointelegraph", 3), ("CryptoPanic", 4), ("金色财经", 3)]
    topics_en = ["ETF", "regulation", "institutional inflow", "exchange", "on-chain whale", "macro"]
    topics_zh = ["ETF", "监管", "机构增持", "交易所", "链上巨鲸", "宏观"]
    total_days = max(1, (end - start).days)
    out: list[dict] = []
    for d in range(total_days):
        day = start + timedelta(days=d)
        k = rng.poisson(per_day)
        for _ in range(int(k)):
            dt = day + timedelta(seconds=int(rng.integers(0, 86400)))
            src, tier = src_pool[rng.integers(len(src_pool))]
            bullish = rng.random() > 0.5
            zh = "财经" in src or rng.random() < 0.3
            asset = rng.choice(["Bitcoin", "Ethereum", "BTC", "ETH"])
            if zh:
                asset_zh = "比特币" if asset in ("Bitcoin", "BTC") else "以太坊"
                topic = topics_zh[rng.integers(len(topics_zh))]
                title = f"{asset_zh} {topic}{'利好' if bullish else '利空'}消息"
            else:
                topic = topics_en[rng.integers(len(topics_en))]
                title = f"{asset} {topic} {'surges on bullish' if bullish else 'drops on bearish'} news"
            syms = _relevant_symbols(title) or list(_SYMBOL_KEYWORDS.keys())
            out.append({"published_at": dt, "source": f"synthetic:{src}", "tier": tier,
                        "title": title, "url": "", "symbols": syms})
    print(f"[hist] 合成历史语料 {len(out)} 条 ({start.date()}~{end.date()})")
    return out


def backfill_news(cfg, start=None, end=None, providers=None) -> dict:
    """编排历史新闻回填: 按 provider 抓取 [start,end] 并追加去重到原始语料库。

    provider ∈ {synthetic, cryptocompare, gdelt}; 缺 key/网络的源自动跳过。
    返回统计: {provider: 抓取条数, "_added": 新增, "_total": 库总量}。
    """
    import os

    hcfg = _history_cfg(cfg)
    start = _parse_dt(start or hcfg.get("start"), default=_parse_dt("2020-01-01T00:00:00Z"))
    end = _parse_dt(end or hcfg.get("end"), default=datetime.now(timezone.utc))
    providers = providers or hcfg["providers"]
    rate = float(hcfg["rate_limit_sec"])
    src_by_type = {s.get("type"): s for s in cfg["news"].get("sources", [])}

    before = _load_raw_store(cfg)
    before_n = 0 if before is None else len(before)

    # 读取可续跑 cursor(仅 gdelt)
    ckpt = {}
    cp = _checkpoint_path(cfg)
    if cp.exists():
        try:
            ckpt = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            ckpt = {}
    gdelt_resume = _parse_dt(ckpt.get("gdelt_cursor"))
    ckpt_range_start = _parse_dt((ckpt.get("range") or [None])[0])
    # 仅当「同一战役起点」时续跑, 避免旧试点 cursor 把 2020 全量跳到 2024
    resume_same_campaign = (
        gdelt_resume is not None
        and ckpt_range_start is not None
        and ckpt_range_start == start
        and start <= gdelt_resume < end
        and "gdelt" in (ckpt.get("providers") or [])
    )

    stats: dict = {}
    for p in providers:
        try:
            if p == "synthetic":
                items = _synthetic_history_items(cfg, start, end)
                stats[p] = len(items)
                _append_raw_store(cfg, items)
            elif p == "cryptocompare":
                s = src_by_type.get("cryptocompare", {})
                key = os.environ.get(s.get("api_key_env", "CRYPTOCOMPARE_KEY"), "")
                if not key:
                    print("[warn] CRYPTOCOMPARE_KEY 未设置; 跳过 cryptocompare 历史回填"
                          "(免费档对历史分页常 401)。请设置环境变量后重跑该 provider。")
                    stats[p] = 0
                    continue
                items = fetch_cryptocompare_history(
                    s.get("name", "CryptoCompare"),
                    s.get("url", "https://min-api.cryptocompare.com/data/v2/news/"),
                    int(s.get("tier", 2)),
                    key,
                    start, end, rate, int(hcfg["max_pages"]))
                stats[p] = len(items)
                _append_raw_store(cfg, items)
            elif p == "gdelt":
                s = src_by_type.get("gdelt", {})
                # 已增量写入 store, 这里不再二次 append
                items = fetch_gdelt_history(
                    s.get("name", "GDELT"), int(s.get("tier", 2)),
                    start, end, int(hcfg["window_days"]), rate,
                    int(hcfg["max_windows"]), hcfg["gdelt_query"],
                    cfg=cfg,
                    resume_from=gdelt_resume if resume_same_campaign else None,
                    http_retries=int(hcfg.get("gdelt_http_retries", 1)),
                    cooldown_429_sec=float(hcfg.get("gdelt_429_cooldown_sec", 300.0)),
                )
                stats[p] = len(items)
            else:
                print(f"[warn] 未知 history provider: {p}")
                continue
        except Exception as e:
            print(f"[warn] provider {p} 回填失败: {e}")
            stats[p] = 0

    after = _load_raw_store(cfg)
    total = 0 if after is None else len(after)
    stats["_added"] = max(0, total - before_n)
    stats["_total"] = total
    # 不覆盖 gdelt_cursor: 窗级写入已是权威进度; 完整跑完时 cursor≈end
    extra = {}
    if "gdelt" not in providers and ckpt.get("gdelt_cursor"):
        extra["gdelt_cursor"] = ckpt["gdelt_cursor"]
    _write_checkpoint(cfg, start, end, providers, total, extra=extra or None)
    return stats


def _write_checkpoint(cfg, start, end, providers, total, extra: dict | None = None) -> None:
    p = _checkpoint_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if p.exists():
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["range"] = [start.isoformat(), end.isoformat()]
    state["providers"] = list(providers)
    state["total"] = int(total)
    if extra:
        state.update(extra)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")