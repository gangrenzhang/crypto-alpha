"""宏观日历严谨性: BLS 日程解析、跨源去重、首印过滤。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_alpha.data.macro_calendar_bls_schedule import parse_bls_month_schedule_html
from crypto_alpha.data.macro_calendar_merge import (
    canonical_macro_name,
    dedupe_cross_source_events,
    filter_events_for_features,
)


SAMPLE_BLS_CELL = """
<table><tr>
<td>10Employment SituationDecember 201908:30 AM</td>
<td>14Consumer Price IndexJanuary 202008:30 AM</td>
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


def test_canonical_macro_name():
    assert canonical_macro_name("Non-Farm Employment Change") == "Nonfarm Payrolls"
    assert canonical_macro_name("CPI y/y") == "CPI YoY"
    assert canonical_macro_name("CPI YoY") == "CPI YoY"
    assert canonical_macro_name("ADP Non-Farm Employment Change") == "ADP Nonfarm Employment Change"
    assert canonical_macro_name("ADP Non-Farm Employment Change") != "Nonfarm Payrolls"


def test_dedupe_prefers_bls_over_ff():
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
    assert out.iloc[0]["source"] == "bls"


def test_filter_prefer_first_print():
    ts = pd.Timestamp("2021-03-05 13:30:00Z")
    df = pd.DataFrame([
        {
            "event_id": "US|Nonfarm Payrolls|20210305T133000Z|fp",
            "name": "Nonfarm Payrolls", "country": "US", "category": "employment",
            "importance": 5, "scheduled_at": ts, "released_at": ts,
            "previous": 1.0, "forecast": 1.0, "actual": 2.0,
            "unit": "k", "source": "alfred_bls",
            "print_kind": "first_print", "schedule_source": "bls_official",
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
