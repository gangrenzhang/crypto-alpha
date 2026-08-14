"""宏观日历严谨性: BLS 日程解析、跨源去重、首印过滤、央行时刻、沉淀合并。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_alpha.data.macro_calendar_bls_schedule import (
    _merge_supplement,
    _supplement_schedule_rows,
    enrich_schedule_from_ff_hist,
    parse_bls_month_schedule_html,
)
from crypto_alpha.data.macro_calendar_merge import (
    canonical_macro_name,
    dedupe_cross_source_events,
    filter_events_for_features,
)
from crypto_alpha.data.macro_calendar_sources import (
    build_non_us_central_bank_events,
    merge_carryover_events,
    snap_ff_events_to_official_schedule,
)


SAMPLE_BLS_CELL = """
<table><tr>
<td>10Employment SituationDecember 201908:30 AM</td>
<td>14Consumer Price IndexDecember 201908:30 AM</td>
</tr></table>
"""

# 页面底部"下月预告"块: ref 年月 == 页面年月, 发布日属于下个月
SAMPLE_BLS_PREVIEW_CELL = """
<table><tr>
<td>6Employment SituationMay 202508:30 AM</td>
<td>3Employment SituationJune 202508:30 AM</td>
</tr></table>
"""


def test_parse_bls_month_schedule_html():
    rows = parse_bls_month_schedule_html(SAMPLE_BLS_CELL, calendar_year=2020, calendar_month=1)
    assert len(rows) == 2
    keys = {r["release_key"] for r in rows}
    assert keys == {"employment_situation", "cpi"}
    emp = [r for r in rows if r["release_key"] == "employment_situation"][0]
    assert emp["ref_year"] == 2019 and emp["ref_month"] == 12
    assert emp["schedule_source"] == "bls_official"
    ts = pd.Timestamp(emp["released_at"])
    assert ts.day == 10 and ts.hour in (13, 14)  # 08:30 ET → UTC


def test_parse_bls_rejects_next_month_preview_block():
    """2025-06 页面预告块里的 ref June(07-03 发布)不得错标为 2025-06-03。"""
    rows = parse_bls_month_schedule_html(SAMPLE_BLS_PREVIEW_CELL, calendar_year=2025, calendar_month=6)
    assert len(rows) == 1
    r = rows[0]
    assert (r["ref_year"], r["ref_month"]) == (2025, 5)
    assert pd.Timestamp(r["released_at"]).strftime("%Y-%m-%d") == "2025-06-06"


def test_canonical_macro_name():
    assert canonical_macro_name("Non-Farm Employment Change") == "Nonfarm Payrolls"
    assert canonical_macro_name("CPI y/y") == "CPI YoY"
    assert canonical_macro_name("CPI YoY") == "CPI YoY"
    assert canonical_macro_name("ADP Non-Farm Employment Change") == "ADP Nonfarm Employment Change"
    assert canonical_macro_name("ADP Non-Farm Employment Change") != "Nonfarm Payrolls"
    # FF 与官方/手工日程的央行命名对齐
    assert canonical_macro_name("Federal Funds Rate") == "FOMC Rate Decision"
    assert canonical_macro_name("FOMC Statement") == "FOMC Rate Decision"
    assert canonical_macro_name("FOMC Press Conference") != "FOMC Rate Decision"
    assert canonical_macro_name("Main Refinancing Rate") == "ECB Rate Decision"
    assert canonical_macro_name("Deposit Facility Rate") == "ECB Rate Decision"
    assert canonical_macro_name("Official Bank Rate") == "BOE Rate Decision"
    assert canonical_macro_name("BOJ Monetary Policy Statement") == "BOJ Rate Decision"
    assert canonical_macro_name("BOJ Policy Rate") == "BOJ Rate Decision"
    # 纪要命名不得被 meeting 规则抢先误标为 Rate Decision
    assert canonical_macro_name("FOMC Meeting Minutes") == "FOMC Minutes"
    assert canonical_macro_name("FOMC Minutes") == "FOMC Minutes"


def test_dedupe_prefers_first_print_over_current_vintage():
    """同事件组内存在 first_print 数值行时, current_vintage 不得胜出。"""
    ts = pd.Timestamp("2020-02-07 13:30:00Z")
    df = pd.DataFrame([
        {
            "name": "Non-Farm Employment Change", "country": "US", "category": "employment",
            "importance": 5, "scheduled_at": ts, "released_at": ts,
            "previous": 100.0, "forecast": 150.0, "actual": 225.0,
            "unit": "k", "source": "forexfactory_hist",
            "print_kind": "first_print", "schedule_source": "forexfactory",
        },
        {
            "name": "Nonfarm Payrolls", "country": "US", "category": "employment",
            "importance": 5, "scheduled_at": ts, "released_at": ts,
            "previous": 100.0, "forecast": 100.0, "actual": 225.0,
            "unit": "k", "source": "bls",
            "print_kind": "current_vintage", "schedule_source": "bls_official",
        },
    ])
    out = dedupe_cross_source_events(df)
    assert len(out) == 1
    assert out.iloc[0]["source"] == "forexfactory_hist"
    assert out.iloc[0]["print_kind"] == "first_print"


def test_dedupe_alfred_beats_ff_among_first_prints():
    """组内有多行 first_print 时按得分竞争: ALFRED > FF(actual/日程), forecast 用调查源。"""
    ts = pd.Timestamp("2021-03-05 13:30:00Z")
    df = pd.DataFrame([
        {
            "name": "Nonfarm Payrolls", "country": "US", "category": "employment",
            "importance": 5, "scheduled_at": ts, "released_at": ts,
            "previous": 100.0, "forecast": 150.0, "actual": 379.0,
            "unit": "k", "print_kind": "first_print",
            "source": "forexfactory_hist", "schedule_source": "forexfactory",
        },
        {
            "name": "Nonfarm Payrolls", "country": "US", "category": "employment",
            "importance": 5, "scheduled_at": ts, "released_at": ts,
            # ALFRED 不得用 previous 冒充 survey forecast
            "previous": 100.0, "forecast": np.nan, "actual": 379.0,
            "unit": "k", "print_kind": "first_print",
            "source": "alfred_bls", "schedule_source": "bls_official",
        },
    ])
    out = dedupe_cross_source_events(df)
    assert len(out) == 1
    assert out.iloc[0]["source"] == "alfred_bls"
    assert float(out.iloc[0]["forecast"]) == 150.0


def test_dedupe_strips_alfred_naive_forecast_without_ff():
    """仅有 ALFRED 且 forecast≡previous → 去重后 forecast 清空(surprise 不可用)。"""
    ts = pd.Timestamp("2021-03-05 13:30:00Z")
    df = pd.DataFrame([{
        "name": "Nonfarm Payrolls", "country": "US", "category": "employment",
        "importance": 5, "scheduled_at": ts, "released_at": ts,
        "previous": 100.0, "forecast": 100.0, "actual": 379.0,
        "unit": "k", "print_kind": "first_print",
        "source": "alfred_bls", "schedule_source": "bls_official",
    }])
    out = dedupe_cross_source_events(df)
    assert len(out) == 1
    assert not np.isfinite(float(out.iloc[0]["forecast"]))


def test_dedupe_keeps_current_vintage_when_no_first_print():
    """组内无 first_print 时 current_vintage 正常保留(回归保护)。"""
    ts = pd.Timestamp("2025-08-01 12:30:00Z")
    df = pd.DataFrame([
        {
            "name": "Nonfarm Payrolls", "country": "US", "category": "employment",
            "importance": 5, "scheduled_at": ts, "released_at": ts,
            "previous": 100.0, "forecast": 100.0, "actual": 73.0,
            "unit": "k", "source": "bls",
            "print_kind": "current_vintage", "schedule_source": "bls_official",
        },
    ])
    out = dedupe_cross_source_events(df)
    assert len(out) == 1
    assert out.iloc[0]["source"] == "bls"


def test_bls_supplement_rows_and_merge():
    rows = _supplement_schedule_rows()
    assert len(rows) == 17
    by_key = {(r["release_key"], r["ref_year"], r["ref_month"]): r for r in rows}
    emp = by_key[("employment_situation", 2023, 12)]
    ts = pd.Timestamp(emp["released_at"])
    assert ts.strftime("%Y-%m-%d") == "2024-01-05"
    assert ts.hour == 13 and ts.minute == 30  # 08:30 ET 冬令时 → 13:30 UTC
    cpi26 = by_key[("cpi", 2026, 4)]
    ts26 = pd.Timestamp(cpi26["released_at"])
    assert ts26.strftime("%Y-%m-%d") == "2026-05-12"
    assert ts26.hour == 12 and ts26.minute == 30  # 08:30 ET 夏令时 → 12:30 UTC
    # 2025 停摆修订日期
    assert pd.Timestamp(by_key[("employment_situation", 2025, 9)]["released_at"]).strftime("%Y-%m-%d") == "2025-11-20"
    assert pd.Timestamp(by_key[("cpi", 2025, 11)]["released_at"]).strftime("%Y-%m-%d") == "2025-12-18"
    assert all(r["schedule_source"] == "bls_official" for r in rows)
    # 合并: 同 key/ref 的自动解析错误行被覆盖; 取消发布行被删除
    existing = pd.DataFrame([
        {
            "release_key": "cpi", "release_name": "Consumer Price Index",
            "ref_year": 2026, "ref_month": 1,
            "released_at": pd.Timestamp("2026-02-11 13:30:00Z"),  # 原定日期(错)
            "schedule_source": "bls_official",
        },
        {
            "release_key": "cpi", "release_name": "Consumer Price Index",
            "ref_year": 2025, "ref_month": 10,  # 停摆取消发布
            "released_at": pd.Timestamp("2025-11-13 13:30:00Z"),
            "schedule_source": "bls_official",
        },
        {
            "release_key": "cpi", "release_name": "Consumer Price Index",
            "ref_year": 2025, "ref_month": 8,  # 正确行不受影响
            "released_at": pd.Timestamp("2025-09-11 12:30:00Z"),
            "schedule_source": "bls_official",
        },
    ])
    merged = _merge_supplement(existing)
    assert len(merged) == 17 + 1  # 17 补充 + 1 保留(2025-08)
    hit = merged.loc[
        (merged["release_key"] == "cpi")
        & (merged["ref_year"] == 2026) & (merged["ref_month"] == 1)
    ]
    assert len(hit) == 1
    assert pd.Timestamp(hit.iloc[0]["released_at"]).strftime("%Y-%m-%d") == "2026-02-13"
    gone = merged.loc[
        (merged["release_key"] == "cpi")
        & (merged["ref_year"] == 2025) & (merged["ref_month"] == 10)
    ]
    assert gone.empty
    kept = merged.loc[
        (merged["release_key"] == "cpi")
        & (merged["ref_year"] == 2025) & (merged["ref_month"] == 8)
    ]
    assert len(kept) == 1


def test_enrich_schedule_excludes_adp():
    """ADP 行不得抢占就业日程(enrich 后 ref 月仍无行, 等待官方/补充表)。"""
    ff_events = [
        {
            "name": "ADP Non-Farm Employment Change", "country": "US",
            "released_at": pd.Timestamp("2020-01-08 13:15:00Z"),
        },
        {
            "name": "Non-Farm Employment Change", "country": "US",
            "released_at": pd.Timestamp("2020-01-10 13:30:00Z"),
        },
    ]
    out = enrich_schedule_from_ff_hist(pd.DataFrame(), ff_events)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["release_key"] == "employment_situation"
    assert (int(r["ref_year"]), int(r["ref_month"])) == (2019, 12)
    assert pd.Timestamp(r["released_at"]).strftime("%Y-%m-%d %H:%M") == "2020-01-10 13:30"


def test_snap_ff_events_to_official_schedule():
    """FF 归档 DST 误差 ±1h → 吸附到官方时刻; 超窗/无官方行不动。"""
    schedule = pd.DataFrame([{
        "release_key": "cpi", "release_name": "Consumer Price Index",
        "ref_year": 2021, "ref_month": 2,
        "released_at": pd.Timestamp("2021-03-10 13:30:00Z"),
        "schedule_source": "bls_official",
    }])
    ff_events = [
        {  # DST 误差行: 12:30 → 应吸附 13:30
            "name": "CPI y/y", "country": "US",
            "scheduled_at": pd.Timestamp("2021-03-10 12:30:00Z"),
            "released_at": pd.Timestamp("2021-03-10 12:30:00Z"),
            "event_id": "US|CPI y/y|20210310T123000Z",
        },
        {  # ADP 不处理
            "name": "ADP Non-Farm Employment Change", "country": "US",
            "scheduled_at": pd.Timestamp("2021-03-03 13:15:00Z"),
            "released_at": pd.Timestamp("2021-03-03 13:15:00Z"),
            "event_id": "US|ADP|x",
        },
        {  # 非 US 不处理
            "name": "CPI y/y", "country": "GB",
            "scheduled_at": pd.Timestamp("2021-03-24 07:00:00Z"),
            "released_at": pd.Timestamp("2021-03-24 07:00:00Z"),
            "event_id": "GB|CPI|x",
        },
    ]
    n = snap_ff_events_to_official_schedule(ff_events, schedule)
    assert n == 1
    assert pd.Timestamp(ff_events[0]["released_at"]).strftime("%H:%M") == "13:30"
    assert ff_events[0]["event_id"] == "US|CPI y/y|20210310T133000Z"
    assert pd.Timestamp(ff_events[1]["released_at"]).strftime("%H:%M") == "13:15"
    assert pd.Timestamp(ff_events[2]["released_at"]).strftime("%H:%M") == "07:00"


def test_non_us_cb_2026_schedule_and_ecb_time():
    evs = build_non_us_central_bank_events(start="2026-01-01")
    assert len(evs) == 24  # ECB/BOE/BOJ 各 8 场
    ecb = [e for e in evs if e["country"] == "EU"]
    assert len(ecb) == 8
    feb = [e for e in ecb if pd.Timestamp(e["released_at"]).month == 2][0]
    ts = pd.Timestamp(feb["released_at"])
    assert ts.strftime("%Y-%m-%d %H:%M") == "2026-02-05 13:15"  # 14:15 CET 冬令时
    jun = [e for e in ecb if pd.Timestamp(e["released_at"]).month == 6][0]
    ts_jun = pd.Timestamp(jun["released_at"])
    assert ts_jun.strftime("%Y-%m-%d %H:%M") == "2026-06-11 12:15"  # 14:15 CEST 夏令时
    boj = [e for e in evs if e["country"] == "JP"]
    assert pd.Timestamp(boj[0]["released_at"]).strftime("%Y-%m-%d") == "2026-01-23"


def test_merge_carryover_events_preserves_ff_week():
    ts_new = pd.Timestamp("2026-07-27 12:30:00Z")
    ts_old = pd.Timestamp("2026-07-20 12:30:00Z")
    new_df = pd.DataFrame([{
        "name": "CPI y/y", "country": "US", "category": "inflation",
        "importance": 5, "scheduled_at": ts_new, "released_at": ts_new,
        "previous": 2.7, "forecast": 2.6, "actual": 2.7,
        "unit": "%", "source": "forexfactory_week",
        "print_kind": "first_print", "schedule_source": "forexfactory",
    }])
    prev_df = pd.DataFrame([
        {
            "name": "CPI y/y", "country": "US", "category": "inflation",
            "importance": 5, "scheduled_at": ts_old, "released_at": ts_old,
            "previous": 2.4, "forecast": 2.5, "actual": 2.7,
            "unit": "%", "source": "forexfactory_week",
            "print_kind": "first_print", "schedule_source": "forexfactory",
        },
        {
            "name": "FOMC Rate Decision", "country": "US", "category": "central_bank",
            "importance": 5, "scheduled_at": ts_old, "released_at": ts_old,
            "previous": np.nan, "forecast": np.nan, "actual": np.nan,
            "unit": "", "source": "federalreserve",
            "print_kind": "n/a", "schedule_source": "federalreserve",
        },
    ])
    out = merge_carryover_events(new_df, prev_df)
    # 新行 + 历史 ff_week 行保留; federalreserve 行不沉淀
    assert len(out) == 2
    assert set(out["source"]) == {"forexfactory_week"}
    # 幂等: 再合并一次结果不变
    out2 = merge_carryover_events(out, out)
    assert len(out2) == 2


def test_dedupe_merges_ff_federal_funds_rate_with_fed_fomc():
    """FF Federal Funds Rate(带数值) 与 Fed FOMC Rate Decision 同刻 → 合并为一行, FF 胜出。"""
    ts = pd.Timestamp("2023-07-26 18:00:00Z")
    df = pd.DataFrame([
        {
            "name": "FOMC Rate Decision", "country": "US", "category": "central_bank",
            "importance": 5, "scheduled_at": ts, "released_at": ts,
            "previous": np.nan, "forecast": np.nan, "actual": np.nan,
            "unit": "", "source": "federalreserve",
            "print_kind": "n/a", "schedule_source": "federalreserve",
        },
        {
            "name": "Federal Funds Rate", "country": "US", "category": "central_bank",
            "importance": 5, "scheduled_at": ts, "released_at": ts,
            "previous": 5.25, "forecast": 5.5, "actual": 5.5,
            "unit": "%", "source": "forexfactory_hist",
            "print_kind": "first_print", "schedule_source": "forexfactory",
        },
    ])
    out = dedupe_cross_source_events(df)
    assert len(out) == 1
    assert out.iloc[0]["source"] == "forexfactory_hist"
    assert float(out.iloc[0]["actual"]) == 5.5


def test_filter_prefer_first_print():
    ts = pd.Timestamp("2021-03-05 13:30:00Z")
    df = pd.DataFrame([
        {
            "event_id": "US|Nonfarm Payrolls|20210305T133000Z|fp",
            "name": "Nonfarm Payrolls", "country": "US", "category": "employment",
            "importance": 5, "scheduled_at": ts, "released_at": ts,
            "previous": 1.0, "forecast": 1.5, "actual": 2.0,
            "unit": "k", "source": "forexfactory_hist",
            "print_kind": "first_print", "schedule_source": "forexfactory",
        },
        {
            "event_id": "US|Nonfarm Payrolls|20210305T133000Z|cv",
            "name": "Nonfarm Payrolls", "country": "US", "category": "employment",
            "importance": 5, "scheduled_at": ts, "released_at": ts,
            "previous": 1.0, "forecast": 1.0, "actual": 99.0,
            "unit": "k", "source": "bls",
            "print_kind": "current_vintage", "schedule_source": "bls_official",
        },
    ])
    out = filter_events_for_features(df, prefer_first_print=True)
    assert len(out) == 1
    assert float(out.iloc[0]["actual"]) == 2.0
    assert out.iloc[0]["print_kind"] == "first_print"


def test_filter_bans_current_vintage_surprise():
    """仅有 current_vintage 时保留行(注意力), 但清空数值 → surprise 不可用。"""
    ts = pd.Timestamp("2025-08-01 12:30:00Z")
    df = pd.DataFrame([{
        "event_id": "US|Nonfarm Payrolls|cv",
        "name": "Nonfarm Payrolls", "country": "US", "category": "employment",
        "importance": 5, "scheduled_at": ts, "released_at": ts,
        "previous": 100.0, "forecast": 100.0, "actual": 73.0,
        "unit": "k", "source": "bls",
        "print_kind": "current_vintage", "schedule_source": "bls_official",
    }])
    out = filter_events_for_features(df, prefer_first_print=True)
    assert len(out) == 1
    assert out.iloc[0]["importance"] == 5
    assert not np.isfinite(float(out.iloc[0]["actual"]))
    assert not np.isfinite(float(out.iloc[0]["forecast"]))
