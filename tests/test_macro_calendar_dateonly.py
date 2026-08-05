"""日历源缺盘中时刻时的保守定时: 绝不可让 actual 早于真实公布可见。

历史缺陷: FF 派生源把「只有日期」的行编成本地午夜 / 中午 CST, 换算成 UTC 后
分别落到**前一天傍晚** / **当日 04:00Z**, 都早于欧美真实公布时刻, 等于把
actual/surprise 提前泄漏给特征。
"""
from __future__ import annotations

import pandas as pd
import pytest

from crypto_alpha.data.macro_calendar import (
    DATE_ONLY_SCHEDULE_SOURCE,
    date_only_release_ts,
)
from crypto_alpha.data.macro_calendar_global_ff import _parse_ff_datetime, ff_csv_to_events
from crypto_alpha.data.macro_calendar_merge import _event_score


def test_date_only_ts_is_utc_end_of_day():
    ts = date_only_release_ts("2024-01-11")
    assert ts == pd.Timestamp("2024-01-11 23:59:59", tz="UTC")
    # 本地日 D 的任何时区实例都不晚于 D 23:59:59Z
    assert ts >= pd.Timestamp("2024-01-11 23:59:59", tz="Etc/GMT-14")


def test_date_only_ts_ignores_incoming_tz():
    """带 tz 的输入按其本地日期取日终, 不得因换算漂到前一天。"""
    ts = date_only_release_ts(pd.Timestamp("2024-01-11 00:00:00", tz="Asia/Tehran"))
    assert ts == pd.Timestamp("2024-01-11 23:59:59", tz="UTC")


def test_date_only_ts_invalid_returns_none():
    assert date_only_release_ts(None) is None
    assert date_only_release_ts(pd.NaT) is None


@pytest.mark.parametrize("time_s", ["All Day", "Tentative", "", "nan"])
def test_ff_all_day_goes_to_utc_end_of_day(time_s):
    ts, date_only = _parse_ff_datetime("2023-04-28", time_s)
    assert date_only is True
    assert ts == pd.Timestamp("2023-04-28 23:59:59", tz="UTC")
    # 旧行为(中午 CST = 04:00Z)早于当天美盘公布, 必须已被淘汰
    assert ts > pd.Timestamp("2023-04-28 04:00:00", tz="UTC")


def test_ff_explicit_time_still_shanghai_wall_clock():
    """有明确时刻的行仍按 Asia/Shanghai 还原(21:30 CST = 13:30Z = 08:30 ET)。"""
    ts, date_only = _parse_ff_datetime("2023-01-12", "9:30pm")
    assert date_only is False
    assert ts == pd.Timestamp("2023-01-12 13:30:00", tz="UTC")


def test_ff_csv_marks_date_only_schedule_source(tmp_path):
    csv = tmp_path / "ff.csv"
    csv.write_text(
        "Date,Time,Currency,Impact,Event,Actual,Forecast,Previous\n"
        "2023-04-28,All Day,EUR,High,German Prelim CPI m/m,0.4%,0.6%,0.8%\n"
        "2023-04-28,9:30pm,USD,High,Core PCE Price Index m/m,0.3%,0.3%,0.3%\n",
        encoding="utf-8",
    )
    rows = ff_csv_to_events(csv, min_importance=3)
    by_name = {r["name"]: r for r in rows}

    allday = by_name["German Prelim CPI m/m"]
    assert allday["schedule_source"] == DATE_ONLY_SCHEDULE_SOURCE
    assert allday["released_at"] == pd.Timestamp("2023-04-28 23:59:59", tz="UTC")
    assert allday["scheduled_at"] == allday["released_at"]

    timed = by_name["Core PCE Price Index m/m"]
    assert timed["schedule_source"] == "forexfactory"
    assert timed["released_at"] == pd.Timestamp("2023-04-28 13:30:00", tz="UTC")


def test_date_only_loses_to_precise_source_in_dedupe():
    """同事件若另有精确时刻源, 推出来的日终行必须让位。"""
    precise = {
        "source": "forexfactory_hist", "schedule_source": "forexfactory",
        "print_kind": "first_print", "importance": 5,
        "previous": 0.1, "forecast": 0.2, "actual": 0.3,
    }
    guessed = {**precise, "schedule_source": DATE_ONLY_SCHEDULE_SOURCE}
    assert _event_score(precise) > _event_score(guessed)
