"""GDELT 回填: 满额窗口应切分, 避免 datedesc 截断。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO

from crypto_alpha.data import news as news_mod


def test_gdelt_full_window_splits_and_covers_early_days(monkeypatch):
    """7 天窗若返回 250 条, 必须切分; 切分后应能覆盖窗前半段日期。"""
    calls: list[tuple[str, str]] = []

    def fake_http(url: str, *a, **k):
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(url).query)
        a0, a1 = qs["startdatetime"][0], qs["enddatetime"][0]
        calls.append((a0, a1))
        w0 = datetime.strptime(a0, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        w1 = datetime.strptime(a1, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        width_h = (w1 - w0).total_seconds() / 3600.0
        arts = []
        if width_h > 48:
            for i in range(250):
                dt = w1.strftime("%Y%m%dT%H%M%SZ")
                arts.append({
                    "seendate": dt,
                    "title": f"Bitcoin rally news {i} late",
                    "url": f"https://example.com/late/{i}",
                    "domain": "example.com",
                })
        else:
            dt = w0.strftime("%Y%m%dT%H%M%SZ")
            arts.append({
                "seendate": dt,
                "title": f"Bitcoin early window {a0}",
                "url": f"https://example.com/early/{a0}",
                "domain": "example.com",
            })
        return json.dumps({"articles": arts}).encode()

    monkeypatch.setattr(news_mod, "_http_get", fake_http)
    monkeypatch.setattr(news_mod.time, "sleep", lambda *_: None)

    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 8, tzinfo=timezone.utc)
    items = news_mod.fetch_gdelt_history(
        "GDELT", 2, start, end,
        window_days=7, rate_limit_sec=0, max_windows=20,
    )
    assert len(calls) >= 3, f"应发生切分, calls={calls}"
    early = [it for it in items if it["published_at"] < datetime(2024, 6, 4, tzinfo=timezone.utc)]
    assert early, f"切分后应覆盖窗前半段, got {[it['published_at'] for it in items[:5]]}"


def test_http_get_429_uses_long_cooldown(monkeypatch):
    import urllib.error

    sleeps: list[float] = []
    monkeypatch.setattr(news_mod.time, "sleep", lambda s: sleeps.append(s))
    news_mod._GDELT_GATE["last_mono"] = 0.0
    news_mod._GDELT_GATE["consec_429"] = 0
    monkeypatch.setattr(news_mod, "_resolve_http_proxies", lambda: {})

    class FakeHeaders(dict):
        def get(self, k, default=None):
            return dict.get(self, k, default)

    n = {"i": 0}

    def fake_urlopen(req, timeout=30):
        n["i"] += 1
        if n["i"] < 3:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many", FakeHeaders(), BytesIO(b"slow down")
            )

        class R:
            def read(self):
                return b'{"ok":1}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return R()

    monkeypatch.setattr(news_mod.urllib.request, "urlopen", fake_urlopen)
    # 非 GDELT URL: 不走闸门强制间隔; 429 冷却仍应 >=60s
    raw = news_mod._http_get(
        "https://example.com/x", max_retries=4, base_backoff_sec=20.0,
    )
    assert raw == b'{"ok":1}'
    assert sleeps, "应休眠"
    assert sleeps[0] >= 60.0, f"429 首次冷却应 >=60s, got {sleeps}"


def test_gdelt_http_get_enforces_min_interval(monkeypatch):
    """GDELT URL 两次成功请求之间应至少间隔 min_interval_sec。"""
    sleeps: list[float] = []
    monkeypatch.setattr(news_mod.time, "sleep", lambda s: sleeps.append(float(s)))
    news_mod._GDELT_GATE["last_mono"] = 0.0
    news_mod._GDELT_GATE["consec_429"] = 0
    monkeypatch.setattr(news_mod, "_resolve_http_proxies", lambda: {})

    # 单调时钟: 每次 urlopen 前 gate_wait 会读 monotonic
    mono = {"t": 1000.0}
    monkeypatch.setattr(news_mod.time, "monotonic", lambda: mono["t"])

    class R:
        def read(self):
            return b'{"articles":[]}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=30):
        mono["t"] += 0.01  # 请求几乎瞬时
        return R()

    monkeypatch.setattr(news_mod.urllib.request, "urlopen", fake_urlopen)
    url = "https://api.gdeltproject.org/api/v2/doc/doc?q=1"
    assert news_mod._http_get(url, max_retries=0, min_interval_sec=90.0) is not None
    # 第二次应等待 ~90s
    assert news_mod._http_get(url, max_retries=0, min_interval_sec=90.0) is not None
    assert any(s >= 89.0 for s in sleeps), f"应强制间隔, sleeps={sleeps}"
