"""BLS 官方新闻发布日程解析(Employment Situation / CPI)。

优先顺序:
1. 本地缓存 ``bls_official_releases.parquet``
2. 直播 ``empsit.htm`` / ``cpi.htm``(前瞻日程)
3. Wayback 月度 ``/{year}/{mm}_sched.htm``(历史官方表)
4. 调用方再回退启发式(并标记 schedule_source=heuristic)
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

RELEASE_ALIASES = {
    "Employment Situation": "employment_situation",
    "Consumer Price Index": "cpi",
}

CELL_DAY_RE = re.compile(r"^(?P<day>\d{1,2})")
# 单元格形如: 10Employment SituationDecember 201908:30 AM
PIECE_RE = re.compile(
    r"(?P<name>Employment Situation|Consumer Price Index|Producer Price Index)"
    r"(?P<ref_month>January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s*(?P<ref_year>\d{4})\s*"
    r"(?P<time>\d{1,2}:\d{2}\s*[AP]M)",
    re.I,
)


def _curl_bytes(url: str, timeout: float = 90.0) -> bytes:
    """抓取 BLS 发布日程; TLS 校验优先(见 data.http_curl)。"""
    from .http_curl import curl_bytes

    return curl_bytes(
        url, timeout=timeout,
        user_agent="Mozilla/5.0 (crypto-alpha bls schedule)",
        label="bls_schedule",
    )


def _parse_et(y: int, m: int, d: int, time_str: str) -> pd.Timestamp:
    t = time_str.strip().upper().replace(".", "")
    hh, mm_ampm = t.split(":")
    mm, ampm = mm_ampm.split()
    hour = int(hh)
    minute = int(mm)
    if ampm.startswith("P") and hour != 12:
        hour += 12
    if ampm.startswith("A") and hour == 12:
        hour = 0
    local = datetime(y, m, d, hour, minute, tzinfo=ET)
    return pd.Timestamp(local.astimezone(ZoneInfo("UTC")))


def parse_bls_month_schedule_html(html: str, calendar_year: int, calendar_month: int) -> list[dict]:
    """从 BLS 月度日程 HTML 提取 Employment Situation / CPI 发布时间。"""
    tds = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", html, flags=re.I | re.S)
    rows: list[dict] = []
    for raw in tds:
        cell = re.sub(r"<[^>]+>", "", raw)
        cell = re.sub(r"\s+", " ", cell).replace("\xa0", " ").strip()
        day_m = CELL_DAY_RE.match(cell)
        if not day_m:
            continue
        day = int(day_m.group("day"))
        for piece in PIECE_RE.finditer(cell):
            name = piece.group("name")
            canon = None
            key_name = name
            for k, v in RELEASE_ALIASES.items():
                if k.lower() == name.lower():
                    canon = v
                    key_name = k
                    break
            if canon is None:
                continue
            ref_month = MONTHS[piece.group("ref_month").lower()]
            ref_year = int(piece.group("ref_year"))
            # 页面底部"下月预告"块: ref 年月 >= 页面年月时, 发布日属于下个月,
            # 按当月 calendar 解析会把 07-03 错标成 06-03 → 拒绝(下月页面会正确覆盖)
            if (ref_year, ref_month) >= (calendar_year, calendar_month):
                continue
            try:
                ts = _parse_et(calendar_year, calendar_month, day, piece.group("time"))
            except ValueError:
                continue
            rows.append({
                "release_key": canon,
                "release_name": key_name,
                "ref_year": ref_year,
                "ref_month": ref_month,
                "released_at": ts,
                "schedule_source": "bls_official",
            })
    return rows


def _wayback_snapshot_url(original: str) -> str | None:
    """CDX 查最近成功快照。"""
    q = (
        "http://web.archive.org/cdx/search/cdx?"
        f"url={original}&output=json&limit=5&filter=statuscode:200&fl=timestamp,original"
    )
    try:
        raw = _curl_bytes(q, timeout=60)
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, list) or len(data) < 2:
        return None
    ts, orig = data[-1][0], data[-1][1]
    return f"https://web.archive.org/web/{ts}/{orig}"


def fetch_bls_month_schedule(year: int, month: int) -> list[dict]:
    path = f"https://www.bls.gov/schedule/{year}/{month:02d}_sched.htm"
    html = None
    try:
        html = _curl_bytes(path, timeout=45).decode("utf-8", errors="replace")
        if "Access Denied" in html or "Employment Situation" not in html:
            html = None
    except Exception:
        html = None
    if html is None:
        snap = _wayback_snapshot_url(path)
        if snap:
            try:
                html = _curl_bytes(snap, timeout=90).decode("utf-8", errors="replace")
            except Exception:
                html = None
    if not html:
        return []
    return parse_bls_month_schedule_html(html, year, month)


def fetch_bls_forward_release_tables() -> list[dict]:
    """解析 empsit.htm / cpi.htm 前瞻表(Reference Month → Release Date)。"""
    out: list[dict] = []
    specs = [
        ("https://www.bls.gov/schedule/news_release/empsit.htm", "employment_situation", "Employment Situation"),
        ("https://www.bls.gov/schedule/news_release/cpi.htm", "cpi", "Consumer Price Index"),
    ]
    row_re = re.compile(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\s+"
        r"([A-Za-z]{3}\.\s+\d{1,2},\s+\d{4})\s+(\d{1,2}:\d{2}\s*[AP]M)",
        re.I,
    )
    for url, key, name in specs:
        html = None
        try:
            html = _curl_bytes(url, timeout=60).decode("utf-8", errors="replace")
            if "Access Denied" in html:
                html = None
        except Exception:
            html = None
        if html is None:
            snap = _wayback_snapshot_url(url)
            if not snap:
                continue
            try:
                html = _curl_bytes(snap, timeout=90).decode("utf-8", errors="replace")
            except Exception:
                continue
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        for m in row_re.finditer(text):
            ref_month = MONTHS[m.group(1).lower()]
            ref_year = int(m.group(2))
            rel_date = datetime.strptime(m.group(3).replace(".", ""), "%b %d, %Y")
            try:
                ts = _parse_et(rel_date.year, rel_date.month, rel_date.day, m.group(4))
            except ValueError:
                continue
            out.append({
                "release_key": key,
                "release_name": name,
                "ref_year": ref_year,
                "ref_month": ref_month,
                "released_at": ts,
                "schedule_source": "bls_official",
            })
    return out


def build_bls_official_release_calendar(
    start_year: int = 2020,
    end_year: int | None = None,
) -> pd.DataFrame:
    ey = int(end_year or date.today().year)
    rows: list[dict] = []
    for y in range(int(start_year), ey + 1):
        for m in range(1, 13):
            if y == ey and m > date.today().month + 2:
                break
            part = fetch_bls_month_schedule(y, m)
            rows.extend(part)
            print(f"[bls-sched] {y}-{m:02d}: {len(part)} official rows", flush=True)
    rows.extend(fetch_bls_forward_release_tables())
    if not rows:
        return pd.DataFrame(columns=[
            "release_key", "release_name", "ref_year", "ref_month",
            "released_at", "schedule_source",
        ])
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["release_key", "ref_year", "ref_month"], keep="last",
    ).sort_values(["released_at"]).reset_index(drop=True)
    return df


# --- BLS 官方日程硬编码补充(人工核验, 优先级高于一切自动源) ---
# 覆盖两类问题: ①直播/Wayback 均未覆盖的缺口; ②自动解析错误行(下月预告块误标、
# 2025 停摆后原定日期未更新)。日期按 BLS 已发布新闻稿/前瞻表/停摆修订日程核验。
# (release_key, ref_year, ref_month, 发布日期, 均为 08:30 ET)
# 注意: --refresh-bls-schedule 仅在 BLS 直播/Wayback 可达时使用;
# 不可达时 refresh 会用空表覆盖好缓存, 而本表仅在加载时合并, 无法救回丢失行。
_BLS_SCHEDULE_SUPPLEMENT: list[tuple[str, int, int, str]] = [
    ("employment_situation", 2023, 12, "2024-01-05"),
    ("cpi", 2023, 12, "2024-01-11"),
    ("employment_situation", 2024, 3, "2024-04-05"),
    ("cpi", 2024, 3, "2024-04-10"),
    # 2025 年停摆(10/1-11/12)修订日程: bls.gov/bls/2025-lapse-revised-release-dates.htm
    ("employment_situation", 2025, 6, "2025-07-03"),   # 独立日前移(原定 07-04 周五)
    ("cpi", 2025, 6, "2025-07-15"),
    ("cpi", 2025, 9, "2025-10-24"),                   # 停摆延迟(原定 10-15)
    ("employment_situation", 2025, 9, "2025-11-20"),   # 停摆延迟 6 周(原定 10-03)
    ("employment_situation", 2025, 10, "2025-12-16"),  # 与 11 月合并发布
    ("employment_situation", 2025, 11, "2025-12-16"),
    ("cpi", 2025, 11, "2025-12-18"),                  # 停摆延迟(原定 12-10)
    ("employment_situation", 2026, 1, "2026-02-11"),   # 停摆修订(原定 02-06)
    ("cpi", 2026, 1, "2026-02-13"),                   # 停摆修订(原定 02-11)
    ("employment_situation", 2026, 3, "2026-04-03"),   # 首周五(覆盖预告块误标 03-03)
    ("cpi", 2026, 3, "2026-04-10"),
    ("employment_situation", 2026, 4, "2026-05-01"),
    ("cpi", 2026, 4, "2026-05-12"),
]

# 已确认取消发布(2025 停摆期间未采集): 从日程表删除对应自动解析行
_BLS_SCHEDULE_CANCELLED: list[tuple[str, int, int]] = [
    ("cpi", 2025, 10),
]

_SUPPLEMENT_NAMES = {
    "employment_situation": "Employment Situation",
    "cpi": "Consumer Price Index",
}


def _supplement_schedule_rows() -> list[dict]:
    rows: list[dict] = []
    for key, ref_y, ref_m, date_s in _BLS_SCHEDULE_SUPPLEMENT:
        y, m, d = (int(x) for x in date_s.split("-"))
        rows.append({
            "release_key": key,
            "release_name": _SUPPLEMENT_NAMES[key],
            "ref_year": ref_y,
            "ref_month": ref_m,
            "released_at": _parse_et(y, m, d, "08:30 AM"),
            "schedule_source": "bls_official",
        })
    return rows


def _merge_supplement(df: pd.DataFrame) -> pd.DataFrame:
    """把硬编码补充行并入官方日程。人工核验优先级最高:
    与补充表同 (key, ref) 的自动解析行一律被覆盖; 取消发布的行一律删除。"""
    sup = pd.DataFrame(_supplement_schedule_rows())
    if df is None or df.empty:
        return sup
    cancelled = set(_BLS_SCHEDULE_CANCELLED)
    override = {
        (str(r["release_key"]), int(r["ref_year"]), int(r["ref_month"]))
        for r in sup.to_dict("records")
    }
    def _drop(r) -> bool:
        key = (str(r["release_key"]), int(r["ref_year"]), int(r["ref_month"]))
        return key in cancelled or key in override
    keep = df.loc[[not _drop(r) for r in df.to_dict("records")]]
    out = pd.concat([keep, sup], ignore_index=True)
    return out.sort_values("released_at").reset_index(drop=True)


def load_or_build_bls_schedule(store_dir: Path, *, refresh: bool = False) -> pd.DataFrame:
    path = Path(store_dir) / "bls_official_releases.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path, engine="pyarrow")
        return _merge_supplement(df)
    df = build_bls_official_release_calendar()
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(df):
        df.to_parquet(path, engine="pyarrow", index=False)
    return _merge_supplement(df)


def enrich_schedule_from_ff_hist(schedule: pd.DataFrame, ff_events: list[dict]) -> pd.DataFrame:
    """用 FF 历史中 USD 高影响 NFP/CPI 时刻填补官方日程空洞(标注 schedule_source=forexfactory)。"""
    rows = [] if schedule is None or schedule.empty else schedule.to_dict("records")
    have = {
        (r["release_key"], int(r["ref_year"]), int(r["ref_month"]))
        for r in rows
    }
    for ev in ff_events:
        if str(ev.get("country")) != "US":
            continue
        name = str(ev.get("name") or "")
        low = name.lower()
        # ADP 非农 ≠ BLS 就业报告: 不排除会被 "non-farm employment change" 子串误命中,
        # 导致就业日程被 ADP 时刻(早 2 天, 13:15 UTC)抢占
        if low.startswith("adp"):
            continue
        ts = pd.Timestamp(ev["released_at"])
        # 参考月 ≈ 发布时间所在月的上一个月
        ref_y, ref_m = _add_month(int(ts.year), int(ts.month), -1)
        if "non-farm employment change" in low or low == "nonfarm payrolls":
            key, rname = "employment_situation", "Employment Situation"
        elif "cpi y/y" in low or low == "cpi yoy":
            key, rname = "cpi", "Consumer Price Index"
        else:
            continue
        tup = (key, ref_y, ref_m)
        if tup in have:
            continue
        have.add(tup)
        rows.append({
            "release_key": key,
            "release_name": rname,
            "ref_year": ref_y,
            "ref_month": ref_m,
            "released_at": ts,
            "schedule_source": "forexfactory",
        })
    if not rows:
        return pd.DataFrame(columns=[
            "release_key", "release_name", "ref_year", "ref_month",
            "released_at", "schedule_source",
        ])
    df = pd.DataFrame(rows)
    # 官方优先于 FF
    df["_pri"] = df["schedule_source"].map({"bls_official": 0, "forexfactory": 1}).fillna(2)
    df = df.sort_values(["release_key", "ref_year", "ref_month", "_pri"])
    df = df.drop_duplicates(subset=["release_key", "ref_year", "ref_month"], keep="first")
    return df.drop(columns=["_pri"]).sort_values("released_at").reset_index(drop=True)


def _add_month(y: int, m: int, delta: int = 1) -> tuple[int, int]:
    m2 = m + delta
    y2 = y + (m2 - 1) // 12
    m2 = (m2 - 1) % 12 + 1
    return y2, m2


def lookup_official_release(
    schedule: pd.DataFrame,
    release_key: str,
    ref_year: int,
    ref_month: int,
) -> pd.Timestamp | None:
    if schedule is None or len(schedule) == 0:
        return None
    hit = schedule.loc[
        (schedule["release_key"] == release_key)
        & (schedule["ref_year"] == int(ref_year))
        & (schedule["ref_month"] == int(ref_month))
    ]
    if hit.empty:
        return None
    return pd.Timestamp(hit.iloc[-1]["released_at"])
