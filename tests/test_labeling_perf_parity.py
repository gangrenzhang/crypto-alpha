"""标注热点加速的语义对拍: 新实现必须与 pandas 慢路径逐事件一致。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _apply_pt_sl_legacy(close, high, low, events, pt_sl):
    """历史 pandas 切片实现(仅测试对拍用)。"""
    pt_mult, sl_mult = pt_sl
    idx = events.index
    t1_list, pt_list, sl_list = [], [], []
    for t0 in idx:
        t1 = events.at[t0, "t1"]
        if pd.isna(t1):
            t1_list.append(pd.NaT)
            pt_list.append(pd.NaT)
            sl_list.append(pd.NaT)
            continue
        side = float(events.at[t0, "side"])
        trgt = float(events.at[t0, "trgt"])
        entry_px = float(close.loc[t0])
        atr_abs = max(trgt, 0.0) * entry_px
        tp_price = entry_px + side * pt_mult * atr_abs
        sl_price = entry_px - side * sl_mult * atr_abs
        path_high = high.loc[t0:t1].iloc[1:]
        path_low = low.loc[t0:t1].iloc[1:]
        if side > 0:
            pt_touch = path_high[path_high >= tp_price].index.min()
            sl_touch = path_low[path_low <= sl_price].index.min()
        else:
            pt_touch = path_low[path_low <= tp_price].index.min()
            sl_touch = path_high[path_high >= sl_price].index.min()
        t1_list.append(t1)
        pt_list.append(pt_touch)
        sl_list.append(sl_touch)
    return pd.DataFrame({"t1": t1_list, "pt": pt_list, "sl": sl_list}, index=idx)


def _num_concurrent_legacy(bar_index, t1):
    t1 = t1.dropna()
    count = pd.Series(0.0, index=bar_index, dtype=float)
    for t0, t1_ in t1.items():
        count.loc[t0:t1_] += 1.0
    return count


def _average_uniqueness_legacy(bar_index, t1):
    conc = _num_concurrent_legacy(bar_index, t1).replace(0, np.nan)
    out = {}
    for t0, t1_ in t1.dropna().items():
        seg = 1.0 / conc.loc[t0:t1_]
        out[t0] = float(seg.mean())
    return pd.Series(out).reindex(t1.index)


def _make_panel(n_bars=2000, n_events=400, seed=0, vertical=48):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_bars, freq="30min", tz="UTC")
    close = pd.Series(30000 * np.exp(np.cumsum(rng.normal(0, 0.001, n_bars))), index=idx)
    high = close * (1.0 + rng.uniform(0, 0.003, n_bars))
    low = close * (1.0 - rng.uniform(0, 0.003, n_bars))
    # 制造同 bar 可同时触碰的尖刺, 覆盖悲观/空头分支
    spike = rng.choice(np.arange(50, n_bars - vertical - 1), size=30, replace=False)
    high.iloc[spike] = close.iloc[spike] * 1.05
    low.iloc[spike] = close.iloc[spike] * 0.95

    from crypto_alpha.labeling.triple_barrier import get_vertical_barriers

    locs = np.sort(rng.choice(np.arange(20, n_bars - vertical - 1), size=n_events, replace=False))
    t_events = idx[locs]
    vb = get_vertical_barriers(close, t_events, vertical)
    events = pd.DataFrame(
        {
            "t1": vb,
            "trgt": rng.uniform(0.005, 0.02, len(t_events)),
            "side": rng.choice([-1.0, 1.0], size=len(t_events)),
        },
        index=t_events,
    ).dropna(subset=["t1"])
    return close, high, low, events


def _assert_touch_frame_equal(a: pd.DataFrame, b: pd.DataFrame):
    assert list(a.columns) == list(b.columns)
    assert a.index.equals(b.index)
    for col in ("t1", "pt", "sl"):
        aa = pd.to_datetime(a[col], utc=True)
        bb = pd.to_datetime(b[col], utc=True)
        both_na = aa.isna() & bb.isna()
        both_eq = (aa == bb) | both_na
        assert bool(both_eq.all()), f"mismatch in {col}: {(~both_eq).sum()} rows"


def test_apply_pt_sl_matches_legacy_pandas_path():
    from crypto_alpha.labeling.triple_barrier import apply_pt_sl_on_t1

    close, high, low, events = _make_panel()
    pt_sl = (1.5, 1.5)
    got = apply_pt_sl_on_t1(close, high, low, events, pt_sl)
    exp = _apply_pt_sl_legacy(close, high, low, events, pt_sl)
    _assert_touch_frame_equal(got, exp)


def test_apply_pt_sl_empty_and_nat_events():
    from crypto_alpha.labeling.triple_barrier import apply_pt_sl_on_t1

    idx = pd.date_range("2021-01-01", periods=10, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100, 110, 10), index=idx)
    high, low = close * 1.01, close * 0.99
    empty = pd.DataFrame(columns=["t1", "trgt", "side"])
    out = apply_pt_sl_on_t1(close, high, low, empty, (1.0, 1.0))
    assert len(out) == 0

    ev = pd.DataFrame(
        {"t1": [pd.NaT], "trgt": [0.01], "side": [1.0]},
        index=pd.DatetimeIndex([idx[1]]),
    )
    out = apply_pt_sl_on_t1(close, high, low, ev, (1.0, 1.0))
    assert pd.isna(out["pt"].iloc[0]) and pd.isna(out["sl"].iloc[0])


def test_num_concurrent_and_uniqueness_match_legacy():
    from crypto_alpha.labeling.sample_weights import (
        average_uniqueness,
        num_concurrent_events,
    )

    close, _h, _l, events = _make_panel(n_bars=3000, n_events=500, seed=1)
    t1 = events["t1"]
    got_c = num_concurrent_events(close.index, t1)
    exp_c = _num_concurrent_legacy(close.index, t1)
    assert np.allclose(got_c.to_numpy(), exp_c.to_numpy(), equal_nan=True)

    got_u = average_uniqueness(close.index, t1)
    exp_u = _average_uniqueness_legacy(close.index, t1)
    assert np.allclose(
        got_u.to_numpy(dtype=float), exp_u.to_numpy(dtype=float), equal_nan=True, rtol=0, atol=1e-12,
    )


def test_barrier_log_returns_no_side_arg_and_symmetric():
    from crypto_alpha.labeling.triple_barrier import _barrier_log_returns

    pt, sl = _barrier_log_returns(1.5, 1.5, 0.01)
    assert pt == float(np.log1p(0.015))
    assert sl == float(np.log1p(-0.015))


def test_concurrent_skips_inverted_interval():
    """畸形 t1 < t0 不得污染差分数组(等价于空 loc 切片)。"""
    from crypto_alpha.labeling.sample_weights import num_concurrent_events

    idx = pd.date_range("2022-01-01", periods=10, freq="1h", tz="UTC")
    t1 = pd.Series([idx[1]], index=[idx[5]])  # end before start
    got = num_concurrent_events(idx, t1)
    assert float(got.sum()) == 0.0
    assert np.allclose(got.to_numpy(), 0.0)


def test_get_bins_unchanged_vs_legacy_touches():
    """加速触碰后 get_bins 与慢路径触碰喂入同一 get_bins 结果一致。"""
    from crypto_alpha.labeling.triple_barrier import apply_pt_sl_on_t1, get_bins

    close, high, low, events = _make_panel(seed=2)
    pt_sl = (1.2, 1.0)
    fast = apply_pt_sl_on_t1(close, high, low, events, pt_sl)
    slow = _apply_pt_sl_legacy(close, high, low, events, pt_sl)
    ev_f = events.copy()
    ev_s = events.copy()
    ev_f["pt_touch"], ev_f["sl_touch"] = fast["pt"], fast["sl"]
    ev_s["pt_touch"], ev_s["sl_touch"] = slow["pt"], slow["sl"]
    bf = get_bins(ev_f, close, pt_sl)
    bs = get_bins(ev_s, close, pt_sl)
    assert bf.index.equals(bs.index)
    assert np.allclose(bf["ret"].to_numpy(), bs["ret"].to_numpy())
    assert np.array_equal(bf["bin"].to_numpy(), bs["bin"].to_numpy())
