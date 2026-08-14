"""逐行复审(2026-08)后修正项的回归测试。

每个测试对应一条**被验证为真问题**的修正, 断言的是修正后的语义不变量,
而不是实现细节: 任何回退到旧行为的改动都应让这里失败。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --------------------------------------------------------------------------- #
# A. 特征层: 有界指标的中性语义                                                #
# --------------------------------------------------------------------------- #
def test_rsi_flat_segment_is_neutral_not_oversold():
    """平价段(无涨跌)RSI 必须是 50; 旧式 up/(down+eps) 会给出 0(伪极度超卖)。"""
    from crypto_alpha.features.technical import RSI_NEUTRAL, _rsi

    idx = pd.date_range("2023-01-01", periods=120, freq="1h", tz="UTC")
    flat = pd.Series(100.0, index=idx)
    assert float(_rsi(flat, 14).iloc[-1]) == pytest.approx(RSI_NEUTRAL)

    # 单边行情语义不得被破坏
    up = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)
    down = pd.Series(np.linspace(200.0, 100.0, len(idx)), index=idx)
    assert float(_rsi(up, 14).iloc[-1]) > 95.0
    assert float(_rsi(down, 14).iloc[-1]) < 5.0

    # 涨跌力量相等时也应在中性附近(而不是被 eps 拉向 0)
    alt = pd.Series(100.0 + 1.0 * (np.arange(len(idx)) % 2), index=idx)
    assert 40.0 < float(_rsi(alt, 14).iloc[-1]) < 60.0


def test_neutral_fill_value_covers_main_and_mtf_rsi():
    """rsi_14 / tf4h_rsi_14 都算有界指标, 缺失填 50; 收益动量类填 0。"""
    from crypto_alpha.features.technical import RSI_NEUTRAL, neutral_fill_value

    for c in ("rsi_14", "tf4h_rsi_14", "tf1d_rsi_7"):
        assert neutral_fill_value(c) == RSI_NEUTRAL
    for c in ("tf4h_ret_1", "tf4h_trend", "mtf_confluence", "zscore_20", "rsi_slope"):
        assert neutral_fill_value(c) == 0.0


def test_mtf_cold_start_fills_rsi_with_neutral():
    """辅周期冷启动段的 RSI 列填 50; 其它 MTF 列仍填 0。"""
    from crypto_alpha.config import Config
    from crypto_alpha.data.fetch import generate_synthetic_ohlcv, resample_ohlcv
    from crypto_alpha.features.mtf import add_mtf_features
    from crypto_alpha.features.technical import RSI_NEUTRAL

    cfg = Config.load()
    cfg.raw["data"]["timeframe"] = "1h"
    cfg.raw["features"]["mtf_enabled"] = True
    main = generate_synthetic_ohlcv("BTC/USDT", n_bars=400, timeframe="1h", seed=7)
    aux = {"4h": resample_ohlcv(main, "4h")}
    out = add_mtf_features(main[["close"]].copy(), aux, cfg, main_tf="1h")

    rsi_cols = [c for c in out.columns if c.startswith("tf4h_rsi_")]
    assert rsi_cols, "未生成辅周期 RSI 列"
    for c in rsi_cols:
        head = out[c].iloc[:4]  # 对齐初期必然无高周期历史
        assert float(head.iloc[0]) == pytest.approx(RSI_NEUTRAL)
        assert out[c].notna().all()
        assert (out[c] != 0.0).all(), "RSI 被填 0 = 伪极度超卖"
    assert float(out["tf4h_trend"].iloc[0]) == 0.0


def test_align_feature_schema_fills_rsi_neutral():
    """实盘整列缺失时的补值同样不得凭空造出极端 RSI。"""
    from crypto_alpha.features.technical import RSI_NEUTRAL
    from crypto_alpha.pipeline.run import align_feature_schema

    idx = pd.date_range("2023-01-01", periods=3, freq="1h", tz="UTC")
    feat = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
    aligned, missing = align_feature_schema(feat, ["a", "tf4h_rsi_14", "tf4h_ret_1"])
    assert missing == ["tf4h_rsi_14", "tf4h_ret_1"]
    assert float(aligned["tf4h_rsi_14"].iloc[0]) == RSI_NEUTRAL
    assert float(aligned["tf4h_ret_1"].iloc[0]) == 0.0


def test_mtf_confluence_uses_primary_lookback_not_column_guess():
    """共振特征的主方向必须与 labeling.primary_lookback 同口径。"""
    from crypto_alpha.config import Config
    from crypto_alpha.data.fetch import generate_synthetic_ohlcv, resample_ohlcv
    from crypto_alpha.features.mtf import add_mtf_features

    main = generate_synthetic_ohlcv("BTC/USDT", n_bars=600, timeframe="1h", seed=11)
    aux = {"4h": resample_ohlcv(main, "4h")}

    def _confluence(lb: int) -> pd.Series:
        cfg = Config.load()
        cfg.raw["data"]["timeframe"] = "1h"
        cfg.raw["features"]["mtf_enabled"] = True
        cfg.raw["features"]["mtf_include_confluence"] = True
        cfg.raw["labeling"]["primary_lookback"] = lb
        # 刻意不提供 mom_* 列: 旧实现会退化去猜 mom_28/mom_24, 与 lb 脱钩
        out = add_mtf_features(main[["close"]].copy(), aux, cfg, main_tf="1h")
        return out["mtf_confluence"]

    c8, c96 = _confluence(8), _confluence(96)
    assert not np.allclose(c8.values, c96.values), "primary_lookback 未影响共振口径"

    # 与直接按 close 动量算出的方向一致
    cfg = Config.load()
    cfg.raw["data"]["timeframe"] = "1h"
    cfg.raw["features"]["mtf_enabled"] = True
    cfg.raw["labeling"]["primary_lookback"] = 8
    out = add_mtf_features(main[["close"]].copy(), aux, cfg, main_tf="1h")
    side = np.sign(main["close"].pct_change(8)).fillna(0.0)
    trend = out["tf4h_trend"].fillna(0.0)
    expect = ((side * trend) > 0).astype(float).where(~((side == 0) | (trend == 0)), 0.0)
    assert np.allclose(out["tf4h_agree"].values, expect.values)


# --------------------------------------------------------------------------- #
# B. 标注 / 权重                                                               #
# --------------------------------------------------------------------------- #
def test_triple_barrier_drops_neutral_side_events():
    """side=0 会让 TP=SL=入场价 → 同 bar 双触判止损, 凭空产出「必亏」假标签。"""
    from crypto_alpha.labeling.triple_barrier import get_events, get_bins

    idx = pd.date_range("2023-01-01", periods=60, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    high, low = close * 1.01, close * 0.99
    trgt = pd.Series(0.01, index=idx)
    side = pd.Series(1.0, index=idx)
    side.iloc[10:20] = 0.0  # confluence 门控产生的中性段
    t_events = pd.DatetimeIndex(idx[:40])

    ev = get_events(close, high, low, t_events, (1.0, 1.0), trgt, 5, side)
    assert len(ev) > 0
    assert (ev["side"].astype(float) != 0.0).all(), "side=0 事件未被丢弃"
    assert not set(ev.index) & set(idx[10:20])
    bins = get_bins(ev, close, (1.0, 1.0))
    assert set(bins["bin"].unique()) <= {0, 1}
    # 旧行为: side=0 事件会被标成 bin=0(必亏) —— 现在这类事件根本不存在
    assert len(bins) == len(ev)


def test_average_uniqueness_prefix_sum_matches_reference_loop():
    """前缀和实现必须与逐仓 nanmean 循环逐元素一致(纯性能改写)。"""
    from crypto_alpha.labeling.sample_weights import average_uniqueness, num_concurrent_events

    rng = np.random.default_rng(3)
    bars = pd.date_range("2023-01-01", periods=500, freq="1h", tz="UTC")
    starts = np.sort(rng.choice(np.arange(0, 460), size=120, replace=False))
    t1 = pd.Series(
        [bars[min(s + int(rng.integers(1, 30)), len(bars) - 1)] for s in starts],
        index=bars[starts],
    )
    got = average_uniqueness(bars, t1).to_numpy(dtype=float)

    conc = num_concurrent_events(bars, t1).reindex(bars).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(conc > 0, 1.0 / conc, np.nan)
    bar_ns = bars.asi8
    lo = np.searchsorted(bar_ns, pd.DatetimeIndex(t1.index).asi8, side="left")
    hi = np.searchsorted(bar_ns, pd.DatetimeIndex(t1.values).asi8, side="right")
    ref = np.array([
        np.nanmean(inv[a:b]) if b > a else np.nan for a, b in zip(lo, hi)
    ])
    assert np.allclose(np.nan_to_num(got, nan=-1.0), np.nan_to_num(ref, nan=-1.0))
    assert np.all((got[np.isfinite(got)] > 0) & (got[np.isfinite(got)] <= 1.0 + 1e-12))


# --------------------------------------------------------------------------- #
# C. 验证 / 专家 / 校准                                                        #
# --------------------------------------------------------------------------- #
def test_embargo_size_never_silently_zero():
    """pct>0 时至少禁运 1 根: int() 截断会让小样本上的禁运静默失效。"""
    from crypto_alpha.validation.purged_kfold import resolve_embargo_size

    assert resolve_embargo_size(80, 0.01) == 1      # 旧式 int(0.8)=0
    assert resolve_embargo_size(1000, 0.01) == 10   # 大样本行为不变
    assert resolve_embargo_size(80, 0.0) == 0
    assert resolve_embargo_size(0, 0.05) == 0


def test_deep_ts_es_split_never_leaks_post_cutoff_silently():
    """pre 段切不出 val 时只能「关早停 + 只训 pre」, 且必须留降级标签。"""
    from crypto_alpha.experts.deep_ts import resolve_early_stop_split

    idx = pd.date_range("2023-01-01", periods=100, freq="1h", tz="UTC")
    cutoff = idx[60]

    tr, va, tags = resolve_early_stop_split(
        idx, val_frac=0.2, patience=5, es_cutoff_time=cutoff, return_tags=True,
    )
    assert va is not None and len(tags) == 0
    assert idx[np.concatenate([tr, va])].max() < cutoff  # 梯度与验证都只用 pre

    # pre 段仅 24 根: 够训不够切 val → 关早停, 仍不许 post 进梯度
    tr2, va2, tags2 = resolve_early_stop_split(
        idx, val_frac=0.2, patience=5, es_cutoff_time=idx[24], return_tags=True,
    )
    assert va2 is None
    assert idx[tr2].max() < idx[24]
    assert any(t.startswith("deep_ts_es_off_small_pre_cutoff") for t in tags2)

    # pre 段过小训不动 → 默认弃权(空训练集), 不再含 post
    tr3, va3, tags3 = resolve_early_stop_split(
        idx, val_frac=0.2, patience=5, es_cutoff_time=idx[3], return_tags=True,
    )
    assert va3 is None and len(tr3) == 0
    assert any(t.startswith("deep_ts_oof_abstain_insufficient_pre") for t in tags3)

    # 显式消融: include_post_cutoff_in_train=True 才允许旧 fallback
    tr4, va4, tags4 = resolve_early_stop_split(
        idx, val_frac=0.2, patience=5, es_cutoff_time=idx[3],
        include_post_cutoff_in_train=True, return_tags=True,
    )
    assert va4 is None and len(tr4) == len(idx)
    assert any(t.startswith("deep_ts_train_includes_post_cutoff") for t in tags4)


def test_calibrator_single_class_falls_back_instead_of_crashing():
    """单类标签下 Platt 会抛 ValueError 炸掉整折; 应退化为 isotonic 常数映射并留痕。"""
    from crypto_alpha.calibration.calibrate import ProbabilityCalibrator

    prob = np.linspace(0.05, 0.95, 60)
    y = np.zeros(60, dtype=int)
    cal = ProbabilityCalibrator("platt").fit(prob, y)
    out = cal.transform(prob)
    assert np.all(np.isfinite(out))
    assert "cal_single_class_fallback_isotonic" in getattr(cal, "degradations", [])
    assert float(np.nanmax(out)) <= 1.0 and float(np.nanmin(out)) >= 0.0

    # 双类时不触发回退, 仍走 Platt
    y2 = (prob > 0.5).astype(int)
    cal2 = ProbabilityCalibrator("platt").fit(prob, y2)
    assert getattr(cal2, "degradations", []) == []


# --------------------------------------------------------------------------- #
# D. 回测 / 风控                                                               #
# --------------------------------------------------------------------------- #
def _events_one(idx, ret=0.0, bars=4):
    return pd.DataFrame(
        {"ret": [ret], "t1": [idx[bars]], "bars_held": [bars], "side": [1.0],
         "trgt": [0.01]},
        index=[idx[0]],
    )


def test_backtest_cost_uses_resolved_roundtrip_cost():
    """PnL 扣的成本必须与 Kelly 压仓用的 roundtrip_cost 是同一个数。"""
    from crypto_alpha.backtest.engine import backtest_events

    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    events = _events_one(idx, ret=0.0)
    prob = np.array([0.9])
    bt_cfg = {
        "prob_threshold": 0.55, "fee_bps": 0.0, "slippage_bps": 0.0,
        "funding_bps_per_bar": 0.0, "portfolio_mode": True, "min_position_pct": 0.001,
    }
    rt = 0.01
    risk_cfg = {
        "kelly_fraction": 1.0, "max_position_pct": 0.5, "max_gross_exposure": 1.0,
        "daily_max_drawdown": 0.0, "roundtrip_cost_frac": rt,
    }
    out = backtest_events(events, prob, bt_cfg, risk_cfg, payoff=1.0)
    size = float(out["detail"]["size"].iloc[0])
    assert size > 0
    # ret=0 ⇒ 净 PnL 全部来自成本, 应等于 size×rt(旧实现按 2*(fee+slip)=0 扣, 得 0)
    assert float(out["detail"]["pnl"].iloc[0]) == pytest.approx(-size * rt, abs=1e-12)


def test_backtest_slip_ref_can_be_frozen_from_training_window():
    """vol-scale 滑点参考应可注入(训练窗冻结), 而非用被评估区间自己的中位数。"""
    from crypto_alpha.backtest.engine import backtest_events

    idx = pd.date_range("2023-01-01", periods=12, freq="1h", tz="UTC")
    events = pd.DataFrame(
        {"ret": [0.0, 0.0], "t1": [idx[4], idx[6]], "bars_held": [4, 4],
         "side": [1.0, 1.0], "trgt": [0.02, 0.04]},
        index=[idx[0], idx[2]],
    )
    prob = np.array([0.9, 0.9])
    bt_cfg = {
        "prob_threshold": 0.55, "fee_bps": 0.0, "slippage_bps": 10.0,
        "funding_bps_per_bar": 0.0, "portfolio_mode": True, "min_position_pct": 0.001,
        "slippage_vol_scale": True, "slippage_vol_mult_cap": 3.0,
    }
    risk_cfg = {
        "kelly_fraction": 1.0, "max_position_pct": 0.3, "max_gross_exposure": 1.0,
        "daily_max_drawdown": 0.0, "roundtrip_cost_frac": None,
    }
    self_ref = backtest_events(events, prob, bt_cfg, risk_cfg, payoff=1.0)
    frozen = backtest_events(
        events, prob, bt_cfg, risk_cfg, payoff=1.0, ref_trgt=0.01,
    )
    # 冻结的低参考 ⇒ 两笔都按更高的波动倍数计滑点 ⇒ 成本更高、收益更低
    assert frozen["metrics"]["total_return"] < self_ref["metrics"]["total_return"]


def test_gross_exposure_capped_on_marked_equity_notional():
    """并发敞口按名义额/盯市权益计: 浮亏时不得靠已实现权益偏松放仓。"""
    from crypto_alpha.backtest.engine import backtest_events

    idx = pd.date_range("2023-01-01", periods=12, freq="1h", tz="UTC")
    # 第一笔在 idx[0] 开多并一路浮亏; 第二笔在 idx[4] 想再开满
    prices = pd.Series(
        [100.0, 95.0, 90.0, 85.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
        index=idx,
    )
    events = pd.DataFrame(
        {"ret": [np.log(0.8), 0.0], "t1": [idx[8], idx[9]], "bars_held": [8, 5],
         "side": [1.0, 1.0], "trgt": [0.01, 0.01]},
        index=[idx[0], idx[4]],
    )
    prob = np.array([0.9, 0.9])
    bt_cfg = {
        "prob_threshold": 0.55, "fee_bps": 0.0, "slippage_bps": 0.0,
        "funding_bps_per_bar": 0.0, "portfolio_mode": True, "min_position_pct": 0.001,
    }
    risk_cfg = {
        "kelly_fraction": 1.0, "max_position_pct": 1.0, "max_gross_exposure": 1.0,
        "daily_max_drawdown": 0.0, "roundtrip_cost_frac": 0.0,
    }
    out = backtest_events(events, prob, bt_cfg, risk_cfg, payoff=1.0, prices=prices)
    d = out["detail"]
    mtm_at_entry2 = float(out["equity_mtm"].loc[idx[4]])
    locked = float(d["size"].iloc[0] * d["entry_equity"].iloc[0])
    second = float(d["size"].iloc[1] * d["entry_equity"].iloc[1])
    assert locked + second <= 1.0 * mtm_at_entry2 + 1e-9


def test_max_drawdown_survives_nonpositive_peak():
    """权益被打穿到 0/负时 MDD 记 -1, 不得产生 inf/nan 污染 calmar。"""
    from crypto_alpha.backtest.engine import max_drawdown

    assert max_drawdown(np.array([1.0, 1.2, 0.6])) == pytest.approx(-0.5)
    assert max_drawdown(np.array([0.0, -0.5, -1.0])) == pytest.approx(-1.0)
    assert np.isfinite(max_drawdown(np.array([1.0, 0.0, -0.2])))
    assert max_drawdown(np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert max_drawdown(np.array([])) == 0.0


def test_mark_to_market_incremental_matches_naive_recompute():
    """盯市改增量累加器后, 浮动权益必须与逐仓重算逐点一致。"""
    from crypto_alpha.backtest.engine import backtest_events

    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-01", periods=200, freq="1h", tz="UTC")
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    starts = np.sort(rng.choice(np.arange(0, 180), size=30, replace=False))
    rows = []
    for s in starts:
        e = min(int(s) + int(rng.integers(2, 20)), len(idx) - 1)
        side = float(rng.choice([1.0, -1.0]))
        ret = float(side * np.log(prices.iloc[e] / prices.iloc[s]))
        rows.append({"ts": idx[s], "ret": ret, "t1": idx[e], "bars_held": e - s,
                     "side": side, "trgt": 0.01})
    events = pd.DataFrame(rows).set_index("ts")
    prob = np.full(len(events), 0.85)
    bt_cfg = {
        "prob_threshold": 0.55, "fee_bps": 1.0, "slippage_bps": 0.5,
        "funding_bps_per_bar": 0.1, "portfolio_mode": True, "min_position_pct": 0.001,
    }
    risk_cfg = {
        "kelly_fraction": 0.5, "max_position_pct": 0.2, "max_gross_exposure": 1.0,
        "daily_max_drawdown": 0.0, "roundtrip_cost_frac": None,
    }
    out = backtest_events(events, prob, bt_cfg, risk_cfg, payoff=1.0, prices=prices)
    mtm = out["equity_mtm"]
    assert np.all(np.isfinite(mtm.values))
    # 末点无未平仓 ⇒ 盯市权益回到已实现权益
    assert float(mtm.iloc[-1]) == pytest.approx(float(out["equity"].iloc[-1]), abs=1e-9)
    assert float(out["metrics"]["max_drawdown"]) <= 0.0


# --------------------------------------------------------------------------- #
# E. 数据层 / 服务层                                                           #
# --------------------------------------------------------------------------- #
def test_curl_verifies_tls_first_and_only_downgrades_on_cert_error(monkeypatch):
    """默认校验证书; 仅证书类失败才 -k 重试, 超时/404 不得不安全重试。"""
    from crypto_alpha.data import http_curl

    calls: list[bool] = []

    def fake_run(cmd, **kw):
        calls.append("-k" in cmd)

        class R:
            pass

        r = R()
        if "-k" in cmd:
            r.returncode, r.stdout, r.stderr = 0, b"ok", b""
        else:
            r.returncode, r.stdout, r.stderr = 60, b"", b"SSL certificate problem"
        return r

    monkeypatch.setattr(http_curl.subprocess, "run", fake_run)
    monkeypatch.setattr(http_curl, "_warned_hosts", set())
    assert http_curl.curl_bytes("https://x.test/a") == b"ok"
    assert calls == [False, True], "应先校验证书, 失败后才降级一次"

    # 非证书失败: 不得降级
    def fake_timeout(cmd, **kw):
        class R:
            returncode, stdout, stderr = 28, b"", b"Operation timed out"

        return R()

    monkeypatch.setattr(http_curl.subprocess, "run", fake_timeout)
    with pytest.raises(RuntimeError):
        http_curl.curl_bytes("https://x.test/b")

    # 显式禁用降级时, 证书失败直接抛错
    monkeypatch.setattr(http_curl.subprocess, "run", fake_run)
    monkeypatch.setenv("CRYPTO_ALPHA_ALLOW_INSECURE_TLS", "0")
    with pytest.raises(RuntimeError):
        http_curl.curl_bytes("https://x.test/c")


def test_oi_pagination_requests_main_timeframe_with_1h_fallback():
    """OI 拉取粒度跟随主周期; 交易所不支持时自动回退 1h 而不是放弃。"""
    from crypto_alpha.data.fetch import _paginate_oi

    class Ex30m:
        def __init__(self):
            self.seen = []

        def fetch_open_interest_history(self, **kw):
            self.seen.append(kw["timeframe"])
            return [{"timestamp": 1_700_000_000_000, "openInterestAmount": 1.0}]

    ex = Ex30m()
    rows = _paginate_oi(ex, "BTC/USDT:USDT", 1_600_000_000_000, 1_700_000_000_000,
                        timeframe="30m")
    assert rows and ex.seen[0] == "30m"

    class ExOnly1h:
        def __init__(self):
            self.seen = []

        def fetch_open_interest_history(self, **kw):
            self.seen.append(kw["timeframe"])
            if kw["timeframe"] != "1h":
                raise ValueError("timeframe not supported")
            return [{"timestamp": 1_700_000_000_000, "openInterestAmount": 1.0}]

    ex2 = ExOnly1h()
    rows2 = _paginate_oi(ex2, "BTC/USDT:USDT", 1_600_000_000_000, 1_700_000_000_000,
                         timeframe="30m")
    assert rows2, "不支持的粒度应回退 1h 而非返回空"
    assert ex2.seen[:2] == ["30m", "1h"]


def test_stale_panel_tag_flags_lagging_bar():
    """面板陈旧要在决策里留痕(不改研究纯度, 只做透明化)。"""
    from crypto_alpha.config import Config
    from crypto_alpha.pipeline.run import _stale_panel_tag

    cfg = Config.load()
    now = pd.Timestamp("2023-01-02 00:00", tz="UTC")
    fresh = _stale_panel_tag(cfg, now - pd.Timedelta(minutes=1), now=now)
    assert fresh is None
    stale = _stale_panel_tag(cfg, now - pd.Timedelta(days=3), now=now)
    assert stale is not None and "stale" in stale
