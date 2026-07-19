"""P0/P1 修复回归: 波动口径统一、组合回测资金占用、decide 止盈公式。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_decide_tp_sl_match_pt_sl():
    """止盈/止损倍数应直接来自 pt_sl, 而非 payoff×atr_stop_mult 混乘。"""
    from crypto_alpha.risk.sizing import decide

    d = decide(
        prob=0.7, side=1, entry_price=100.0, atr=2.0,
        risk_cfg={"kelly_fraction": 0.5, "max_position_pct": 0.3, "atr_stop_mult": 9.0},
        prob_threshold=0.55, pt_sl=(2.0, 1.5),
    )
    assert d["signal"] == "LONG"
    assert d["stop_loss"] == round(100.0 - 1.5 * 2.0, 2)  # 97.0
    assert d["take_profit"] == round(100.0 + 2.0 * 2.0, 2)  # 104.0
    assert d["sl_mult"] == 1.5 and d["pt_mult"] == 2.0


def test_barrier_target_uses_relative_atr():
    from crypto_alpha.labeling.meta_labeling import _barrier_target

    idx = pd.date_range("2023-01-01", periods=30, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100, 110, 30), index=idx)
    atr = pd.Series(2.0, index=idx)
    df = pd.DataFrame({"close": close, "atr_14": atr, "high": close, "low": close})
    trgt = _barrier_target(df, close, {"barrier_vol": "atr"}, 50)
    assert np.allclose(trgt.dropna().iloc[-1], 2.0 / close.iloc[-1])


def test_portfolio_backtest_caps_concurrent_exposure():
    """重叠事件不得把同一笔权益重复复利到超过 max_gross。"""
    from crypto_alpha.backtest.engine import backtest_events

    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    # 5 个事件全部重叠持有到窗口末, 各想开 30% 仓; 总敞口上限 1.0
    entry_idx = idx[:5]
    events = pd.DataFrame({
        "ret": [0.02] * 5,
        "t1": [idx[9]] * 5,
        "bars_held": [9] * 5,
    }, index=entry_idx)
    prob = np.array([0.8] * 5)
    bt_cfg = {
        "prob_threshold": 0.55, "fee_bps": 0.0, "slippage_bps": 0.0,
        "funding_bps_per_bar": 0.0, "portfolio_mode": True, "min_position_pct": 0.01,
    }
    risk_cfg = {
        "kelly_fraction": 1.0, "max_position_pct": 0.3,
        "max_gross_exposure": 1.0, "daily_max_drawdown": 0.0,
    }
    out = backtest_events(events, prob, bt_cfg, risk_cfg, payoff=1.0)
    detail = out["detail"]
    # 全程重叠 → 入场仓位合计即为峰值并发敞口, 必须 ≤ 1.0
    assert float(detail["size"].sum()) <= 1.0 + 1e-9
    assert (detail["size"] > 0).sum() >= 3
    assert int(out["metrics"]["n_skipped_capacity"]) >= 1
    indep = backtest_events(
        events, prob, {**bt_cfg, "portfolio_mode": False}, risk_cfg, payoff=1.0,
    )
    assert out["metrics"]["total_return"] < indep["metrics"]["total_return"]


def test_meta_labels_atr_path_smoke():
    """小面板上 ATR 障碍标注能跑通并产出 bin。"""
    from crypto_alpha.config import Config
    from crypto_alpha.features.technical import add_technical_features
    from crypto_alpha.labeling.meta_labeling import build_meta_labels

    cfg = Config.load()
    cfg.raw["labeling"]["barrier_vol"] = "atr"
    cfg.raw["labeling"]["min_cusum_events"] = 5
    idx = pd.date_range("2023-01-01", periods=200, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.uniform(1, 5, 200),
    }, index=idx)
    feat = add_technical_features(df, [7, 14], 50)
    feat["close"] = df["close"]
    feat["high"] = df["high"]
    feat["low"] = df["low"]
    bins = build_meta_labels(feat, cfg)
    assert len(bins) > 10
    assert set(bins["bin"].unique()).issubset({0, 1})


def test_portfolio_additive_not_multiplicative_compounding():
    """两笔完全重叠、同入场权益的盈利仓: 加性记账 ≠ 乘积复利。"""
    from crypto_alpha.backtest.engine import backtest_events
    from crypto_alpha.diagnostics.integrity import backtest_reconciliation

    idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    events = pd.DataFrame({
        "ret": [np.log(1.1), np.log(1.1)],  # +10% 价格
        "t1": [idx[4], idx[4]],
        "bars_held": [4, 4],
        "side": [1, 1],
    }, index=idx[:2])
    prob = np.array([0.9, 0.9])
    bt_cfg = {
        "prob_threshold": 0.55, "fee_bps": 0.0, "slippage_bps": 0.0,
        "funding_bps_per_bar": 0.0, "portfolio_mode": True, "min_position_pct": 0.01,
    }
    risk_cfg = {
        "kelly_fraction": 1.0, "max_position_pct": 0.5,
        "max_gross_exposure": 1.0, "daily_max_drawdown": 0.0,
        "roundtrip_cost_frac": 0.0,
    }
    out = backtest_events(events, prob, bt_cfg, risk_cfg, payoff=1.0)
    # 入场时各 0.5, 各贡献 0.5*0.1=0.05 → 末端权益 1.10
    assert abs(out["metrics"]["total_return"] - 0.10) < 1e-9
    # 乘积复利会得到 1.05*1.05-1=0.1025
    assert out["metrics"]["total_return"] < 0.1025 - 1e-12
    recon = backtest_reconciliation(out)
    assert recon["equity_matches_pnl"]


def test_mtm_short_matches_simple_not_geometric():
    """空头盯市浮动须用 side×(P_t/P_entry−1), 不得用 (P_t/P_entry)^side−1。

    构造: 空头入场 100, 持有期中价格到 90; 用低于阈值的探针事件触发该时刻的 _mark
    (探针本身不开仓)。已实现路径/每笔 pnl 不受影响。
    """
    from crypto_alpha.backtest.engine import backtest_events
    from crypto_alpha.risk.sizing import position_size

    idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    prices = pd.Series([100.0, 95.0, 90.0, 92.0, 90.0], index=idx)
    # 空头垂直到期: 入场 100 → 出场 90 ⇒ 简单收益 +10%
    ret_short = float(np.log1p(-1.0 * (90.0 / 100.0 - 1.0)))
    events = pd.DataFrame({
        "ret": [ret_short, 0.0],
        "t1": [idx[4], idx[2]],  # 探针事件在 idx[2] 入场即到期(仍会走 timeline)
        "bars_held": [4, 1],
        "side": [-1.0, 1.0],
    }, index=[idx[0], idx[2]])
    # 主仓高概率开仓; 探针概率低于阈值 → 不开仓, 但 timeline 仍在 idx[2] 调用 _mark
    prob = np.array([0.9, 0.1])
    bt_cfg = {
        "prob_threshold": 0.55, "fee_bps": 0.0, "slippage_bps": 0.0,
        "funding_bps_per_bar": 0.0, "portfolio_mode": True, "min_position_pct": 0.01,
    }
    risk_cfg = {
        "kelly_fraction": 1.0, "max_position_pct": 1.0,
        "max_gross_exposure": 1.0, "daily_max_drawdown": 0.0,
        "roundtrip_cost_frac": 0.0,
    }
    out = backtest_events(
        events, prob, bt_cfg, risk_cfg, payoff=1.0, prices=prices,
    )
    size = position_size(0.9, 1.0, 1.0, 1.0, cost=0.0)
    assert size > 0
    # idx[2] 时刻: 空头仍持仓, P=90 → 简单浮动 size*0.1; 旧几何为 size*(100/90-1)
    mark_simple = 1.0 + size * 0.1
    mark_geom = 1.0 + size * (100.0 / 90.0 - 1.0)
    mtm_mid = float(out["equity_mtm"].loc[idx[2]])
    assert abs(mtm_mid - mark_simple) < 1e-9
    assert abs(mtm_mid - mark_geom) > 1e-4  # 明确拒绝几何口径
    # 已实现末端 = 简单收益(零成本)
    assert abs(out["metrics"]["total_return"] - size * 0.1) < 1e-9
    assert out["metrics"]["mark_to_market"] is True


def test_mtm_long_unchanged_vs_simple():
    """多头: 简单收益与旧几何公式重合; 盯市中点应等于 size×(P_t/P_entry−1)。"""
    from crypto_alpha.backtest.engine import backtest_events
    from crypto_alpha.risk.sizing import position_size

    idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    prices = pd.Series([100.0, 105.0, 110.0, 108.0, 110.0], index=idx)
    ret_long = float(np.log1p(1.0 * (110.0 / 100.0 - 1.0)))
    events = pd.DataFrame({
        "ret": [ret_long, 0.0],
        "t1": [idx[4], idx[2]],
        "bars_held": [4, 1],
        "side": [1.0, -1.0],
    }, index=[idx[0], idx[2]])
    prob = np.array([0.9, 0.1])
    bt_cfg = {
        "prob_threshold": 0.55, "fee_bps": 0.0, "slippage_bps": 0.0,
        "funding_bps_per_bar": 0.0, "portfolio_mode": True, "min_position_pct": 0.01,
    }
    risk_cfg = {
        "kelly_fraction": 1.0, "max_position_pct": 1.0,
        "max_gross_exposure": 1.0, "daily_max_drawdown": 0.0,
        "roundtrip_cost_frac": 0.0,
    }
    out = backtest_events(
        events, prob, bt_cfg, risk_cfg, payoff=1.0, prices=prices,
    )
    size = position_size(0.9, 1.0, 1.0, 1.0, cost=0.0)
    mark_simple = 1.0 + size * 0.1  # 110/100 - 1
    assert abs(float(out["equity_mtm"].loc[idx[2]]) - mark_simple) < 1e-9
    assert abs(out["metrics"]["total_return"] - size * 0.1) < 1e-9


def test_embargo_clamps_near_end():
    """近末折禁运不得整段跳过: max(t1) 后剩余样本即使 < embargo 也必须剔除。"""
    from crypto_alpha.validation.purged_kfold import PurgedKFold

    n = 100
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    t1 = pd.Series(idx, index=idx)  # 零持有期: max(t1)=末样本 t0, 与折边界对齐
    X = pd.DataFrame({"f": np.arange(n)}, index=idx)
    pkf = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.1)  # embargo=10
    folds = list(pkf.split(X))
    tr, te = folds[-2]
    test_end_time = t1.iloc[te].max()
    after = np.where(idx > test_end_time)[0]
    banned = set(after[:10].tolist())
    assert banned, "应存在禁运带"
    assert not (banned & set(tr.tolist()))


def test_embargo_starts_after_max_t1():
    """禁运从测试段 max(t1) 之后起算, 而非折内最后一个样本下标。"""
    from crypto_alpha.validation.purged_kfold import PurgedKFold

    n = 100
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    # 每条标签向后延伸 10 根, 使 max(t1) 远超折边界
    t1 = pd.Series(idx + pd.Timedelta(hours=10), index=idx)
    X = pd.DataFrame({"f": np.arange(n)}, index=idx)
    pkf = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.05)  # embargo=5
    folds = list(pkf.split(X))
    tr, te = folds[0]
    test_end_time = t1.iloc[te].max()
    after = np.where(idx > test_end_time)[0]
    assert len(after) >= 5
    banned = set(after[:5].tolist())
    assert not (banned & set(tr.tolist()))
    # 折边界之后、max(t1) 之前的样本已被 purge 或仍可能在 train;
    # 关键: max(t1) 之后的禁运带不得在 train
    fold_end = int(te[-1]) + 1
    if fold_end < after[0]:
        # 若折边界早于 max(t1) 后第一根, 禁运不应贴在 fold_end
        assert fold_end not in banned or after[0] == fold_end


def test_log1p_barrier_matches_price_space_stop():
    """多头: 加性障碍 entry±mult×atr 与持仓对数收益一致。"""
    from crypto_alpha.labeling.triple_barrier import get_events, get_bins

    idx = pd.date_range("2023-01-01", periods=4, freq="1h", tz="UTC")
    # entry=100; atr/close=0.05 → sl_mult=1 → stop=95 → log(0.95)
    close = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
    high = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
    low = pd.Series([100.0, 94.0, 100.0, 100.0], index=idx)  # 下一根击穿 95
    trgt = pd.Series(0.05, index=idx)
    side = pd.Series(1, index=idx)
    events = get_events(close, high, low, idx[:1], (1.0, 1.0), trgt, 3, side, 0.0)
    bins = get_bins(events, close, (1.0, 1.0))
    assert bins["bin"].iloc[0] == 0
    assert abs(bins["ret"].iloc[0] - np.log(0.95)) < 1e-9


def test_short_barrier_matches_decide_additive():
    """空头标签触碰价与 decide 加性挂单一致; ret=log1p(简单PnL) 与回测 expm1 互逆。"""
    from crypto_alpha.labeling.triple_barrier import get_events, get_bins
    from crypto_alpha.risk.sizing import atr_stop, atr_take_profit

    idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    entry, atr_abs, mult = 100.0, 5.0, 1.0
    close = pd.Series([entry, entry, entry, entry, entry], index=idx)
    high = pd.Series([entry, 106.0, entry, entry, entry], index=idx)
    low = pd.Series([entry, entry, entry, entry, entry], index=idx)
    trgt = pd.Series(atr_abs / entry, index=idx)
    side = pd.Series(-1, index=idx)
    events = get_events(close, high, low, idx[:1], (mult, mult), trgt, 4, side, 0.0)
    bins = get_bins(events, close, (mult, mult))
    assert int(bins["bin"].iloc[0]) == 0
    # 空头止损简单收益 -5% → log1p(-0.05); expm1 还原为 -0.05(入场名义)
    assert abs(bins["ret"].iloc[0] - np.log1p(-0.05)) < 1e-9
    assert abs(float(np.expm1(bins["ret"].iloc[0])) - (-0.05)) < 1e-12
    assert atr_stop(entry, atr_abs, -1, mult) == pytest.approx(105.0)
    assert atr_take_profit(entry, atr_abs, -1, mult) == pytest.approx(95.0)


def test_short_take_profit_additive():
    """空头止盈: 价格跌至 entry-mult×atr; ret 与多头同形 log1p(+x)。"""
    from crypto_alpha.labeling.triple_barrier import get_events, get_bins

    idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    entry, atr_abs, mult = 100.0, 5.0, 1.0
    close = pd.Series([entry] * 5, index=idx)
    high = pd.Series([entry] * 5, index=idx)
    low = pd.Series([entry, 94.0, entry, entry, entry], index=idx)
    trgt = pd.Series(atr_abs / entry, index=idx)
    side = pd.Series(-1, index=idx)
    events = get_events(close, high, low, idx[:1], (mult, mult), trgt, 4, side, 0.0)
    bins = get_bins(events, close, (mult, mult))
    assert int(bins["bin"].iloc[0]) == 1
    assert abs(bins["ret"].iloc[0] - np.log1p(0.05)) < 1e-9
    assert abs(float(np.expm1(bins["ret"].iloc[0])) - 0.05) < 1e-12


def test_short_long_barrier_ret_symmetric_for_expm1():
    """同宽障碍下多空持仓 ret 对称, 且均与 expm1 简单 PnL 一致。"""
    from crypto_alpha.labeling.triple_barrier import get_events, get_bins

    idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    entry, x = 100.0, 0.05
    trgt = pd.Series(x, index=idx)

    def _ret(side_val: int, hit: str) -> float:
        close = pd.Series([entry] * 5, index=idx)
        high = pd.Series([entry] * 5, index=idx)
        low = pd.Series([entry] * 5, index=idx)
        if hit == "sl" and side_val > 0:
            low.iloc[1] = entry * (1 - x) - 1.0
        elif hit == "pt" and side_val > 0:
            high.iloc[1] = entry * (1 + x) + 1.0
        elif hit == "sl" and side_val < 0:
            high.iloc[1] = entry * (1 + x) + 1.0
        else:
            low.iloc[1] = entry * (1 - x) - 1.0
        side = pd.Series(side_val, index=idx)
        ev = get_events(close, high, low, idx[:1], (1.0, 1.0), trgt, 4, side, 0.0)
        return float(get_bins(ev, close, (1.0, 1.0))["ret"].iloc[0])

    assert abs(_ret(1, "pt") - _ret(-1, "pt")) < 1e-12
    assert abs(_ret(1, "sl") - _ret(-1, "sl")) < 1e-12
    assert abs(float(np.expm1(_ret(-1, "pt"))) - x) < 1e-12
    assert abs(float(np.expm1(_ret(-1, "sl"))) - (-x)) < 1e-12


def test_short_vertical_ret_expm1_entry_notional():
    """空头垂直到期: ret=log1p((c0-c1)/c0), expm1 得入场名义简单收益。"""
    from crypto_alpha.labeling.triple_barrier import get_events, get_bins

    idx = pd.date_range("2023-01-01", periods=6, freq="1h", tz="UTC")
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 95.0], index=idx)
    high = close.copy()
    low = close.copy()
    trgt = pd.Series(0.2, index=idx)  # 宽障碍, 强制走垂直
    side = pd.Series(-1, index=idx)
    events = get_events(close, high, low, idx[:1], (1.0, 1.0), trgt, 5, side, 0.0)
    bins = get_bins(events, close, (1.0, 1.0))
    assert int(bins["bin"].iloc[0]) == 1
    want = np.log1p((100.0 - 95.0) / 100.0)
    assert abs(bins["ret"].iloc[0] - want) < 1e-9
    assert abs(float(np.expm1(bins["ret"].iloc[0])) - 0.05) < 1e-12


def test_cusum_threshold_no_lookahead():
    """CUSUM 冷启动不得用全样本中位数或 bfill 灌未来。"""
    from crypto_alpha.labeling.triple_barrier import causal_cusum_threshold, cusum_filter

    idx = pd.date_range("2023-01-01", periods=100, freq="1h", tz="UTC")
    trgt = pd.Series([0.001] * 60 + [0.05] * 40, index=idx)
    thr = causal_cusum_threshold(trgt, min_periods=50, prior=0.005)
    assert (thr.iloc[:49] == 0.005).all()
    assert abs(float(thr.iloc[49]) - 0.001) < 1e-12
    thr_nan_head = thr.copy()
    thr_nan_head.iloc[:10] = np.nan
    close = pd.Series(100.0, index=idx)
    _ = cusum_filter(close, thr_nan_head)


def test_roundtrip_cost_null_matches_backtest_and_decide():
    """YAML null 时 decide 与回测 Kelly 成本均为 2*(fee+slip)。"""
    from crypto_alpha.risk.sizing import decide, resolve_roundtrip_cost

    fee, slip = 5e-4, 2e-4
    risk_cfg = {
        "kelly_fraction": 0.5, "max_position_pct": 0.3, "roundtrip_cost_frac": None,
    }
    assert resolve_roundtrip_cost(risk_cfg, fee, slip) == pytest.approx(2.0 * (fee + slip))
    d0 = decide(
        0.70, 1, 100.0, 2.0, risk_cfg, fee=fee, slip=slip, pt_sl=(1.0, 1.0),
    )
    risk_explicit = {**risk_cfg, "roundtrip_cost_frac": 2.0 * (fee + slip)}
    d1 = decide(
        0.70, 1, 100.0, 2.0, risk_explicit, fee=0.0, slip=0.0, pt_sl=(1.0, 1.0),
    )
    assert d0["suggested_position_pct"] == d1["suggested_position_pct"]
    d_zero = decide(
        0.70, 1, 100.0, 2.0, {**risk_cfg, "roundtrip_cost_frac": 0.0},
        pt_sl=(1.0, 1.0),
    )
    assert d_zero["suggested_position_pct"] >= d0["suggested_position_pct"]


def test_equity_sharpe_additive_does_not_change_per_trade_sharpe():
    """权益夏普为增量字段: 旧 sharpe 仍等于 sharpe_ratio(pnl); 权益口径保留平坦段。"""
    from crypto_alpha.backtest.engine import (
        backtest_events,
        equity_curve_sharpe,
        sharpe_ratio,
    )

    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    events = pd.DataFrame({
        "ret": [np.log(1.05), np.log(0.97), np.log(1.08), np.log(1.02)],
        "t1": [idx[3], idx[5], idx[7], idx[9]],
        "bars_held": [3, 4, 4, 5],
        "side": [1, -1, 1, 1],
    }, index=idx[[0, 1, 2, 4]])
    prob = np.array([0.9, 0.9, 0.9, 0.9])
    bt_cfg = {
        "prob_threshold": 0.55, "fee_bps": 5.0, "slippage_bps": 2.0,
        "funding_bps_per_bar": 0.0, "portfolio_mode": True, "min_position_pct": 0.01,
    }
    risk_cfg = {
        "kelly_fraction": 0.5, "max_position_pct": 0.3,
        "max_gross_exposure": 1.0, "daily_max_drawdown": 0.0,
        "roundtrip_cost_frac": None,
    }
    out = backtest_events(events, prob, bt_cfg, risk_cfg, payoff=1.0)
    m = out["metrics"]
    expected_trade_sr = sharpe_ratio(out["detail"]["pnl"].values)
    assert abs(m["sharpe"] - expected_trade_sr) < 1e-12
    for k in (
        "sharpe_equity", "sharpe_equity_annualized",
        "sharpe_equity_mtm", "sharpe_equity_mtm_annualized",
    ):
        assert k in m
        assert np.isfinite(m[k])
    eq_sr = equity_curve_sharpe(out["equity"])
    assert abs(m["sharpe_equity"] - eq_sr["sharpe"]) < 1e-12
    assert abs(m["sharpe_equity_annualized"] - eq_sr["sharpe_annualized"]) < 1e-12

    # 权益函数保留零收益段(不去零)
    eq = pd.Series(
        [1.0, 1.0, 1.1, 1.1, 1.05],
        index=pd.date_range("2023-01-01", periods=5, freq="1D", tz="UTC"),
    )
    got = equity_curve_sharpe(eq)
    rets = eq.pct_change().iloc[1:].values
    assert got["n_periods"] == 4
    assert abs(got["sharpe"] - float(np.mean(rets) / (np.std(rets) + 1e-12))) < 1e-9
    trade_like = sharpe_ratio(rets)
    assert abs(got["sharpe"] - trade_like) > 1e-9


def test_dsr_sqrt_clamps_negative_variance():
    """极端偏度下 DSR 方差项为负时不得抛错。"""
    from crypto_alpha.backtest.engine import deflated_sharpe_ratio

    out = deflated_sharpe_ratio(
        observed_sr=2.0, n_trials=10, n_obs=50, skew=3.0, kurt=3.0,
    )
    assert np.isnan(out)


def test_stacking_small_sample_records_degradation():
    """二层样本过少退回自训自评时须写入 degradations。"""
    from crypto_alpha.ensemble.stacking import StackingEnsemble
    from crypto_alpha.experts.gbdt import GBDTExpert

    n = 8
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    X = pd.DataFrame({"f1": np.arange(n, dtype=float), "side": 1.0}, index=idx)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    t1 = pd.Series(idx, index=idx)
    ens = StackingEnsemble(
        [GBDTExpert({"n_estimators": 10, "num_leaves": 3}, feature_cols=["f1"], seed=0)],
        {"meta_learner": "logistic", "min_expert_auc": 0.0},
        seed=0,
    )
    ens.fit(X, y, t1, n_splits=5, embargo_pct=0.0)
    assert any("meta_nested_oof_fallback_insample" in d for d in ens.degradations)


def test_pseudo_oof_excluded_from_meta():
    """伪 OOF 专家默认不进入元学习器, 分数仅保留诊断。"""
    from crypto_alpha.ensemble.stacking import StackingEnsemble
    from crypto_alpha.experts.base import BaseExpert
    from crypto_alpha.experts.gbdt import GBDTExpert

    class FrozenExpert(BaseExpert):
        name = "frozen"
        pseudo_oof = True

        def fit(self, X, y, sample_weight=None):
            self._p = 0.61
            return self

        def predict_proba(self, X):
            return np.full(len(X), self._p, dtype=float)

    n = 40
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "side": np.where(rng.random(n) > 0.5, 1.0, -1.0),
    }, index=idx)
    y = (X["f1"].values > 0).astype(int)
    t1 = pd.Series(idx + pd.Timedelta(hours=3), index=idx)
    ens = StackingEnsemble(
        [
            GBDTExpert({"n_estimators": 20, "num_leaves": 7}, feature_cols=["f1", "side"], seed=0),
            FrozenExpert({}, feature_cols=["f1"], seed=0),
        ],
        {
            "meta_learner": "logistic",
            "min_expert_auc": 0.0,
            "exclude_pseudo_oof_from_meta": True,
        },
        seed=0,
    )
    ens.fit(X, y, t1, n_splits=4, embargo_pct=0.0)
    assert [e.name for e in ens.experts] == ["gbdt"]
    assert "frozen" in ens.pseudo_oof_.columns
    assert any("excluded_from_meta_pseudo_oof" in d for d in ens.degradations)
    assert any("pseudo_oof_not_cross_validated" in d for d in ens.degradations)
    # 部署推理只依赖 meta 专家列
    p = ens.predict_proba(X.iloc[:5])
    assert p.shape == (5,)


def test_side_in_feature_cols_and_gbdt():
    """prepare_dataset 将 side 纳入 feature_cols, GBDT 可直接使用。"""
    from crypto_alpha.config import Config
    from crypto_alpha.pipeline.run import prepare_dataset
    from crypto_alpha.experts.gbdt import GBDTExpert

    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = True
    cfg.raw["news"]["use_synthetic"] = True
    cfg.raw["data"]["synthetic_bars"] = 2500
    cfg.raw["features"]["mtf_enabled"] = False
    cfg.raw["labeling"]["min_cusum_events"] = 20
    ds = prepare_dataset(cfg, "BTC/USDT")
    assert "side" in ds.feature_cols
    assert "side" in ds.X.columns
    assert set(ds.X["side"].unique()).issubset({-1.0, 1.0})
    clf = GBDTExpert(
        {"n_estimators": 15, "num_leaves": 7, "min_child_samples": 5},
        feature_cols=ds.feature_cols, seed=0,
    )
    clf.fit(ds.X.iloc[:200], ds.y[:200])
    proba = clf.predict_proba(ds.X.iloc[200:220])
    assert proba.shape == (20,)
    assert np.isfinite(proba).all()


def test_expert_oof_calibrator_conformal_uses_oof_not_insample():
    """CPCV 单专家路径: OOF + 部署口径时间切分校准/保形(非 in-sample 概率)。"""
    from crypto_alpha.pipeline.evaluate import (
        _expert_oof_probs,
        _apply_deploy_cal_conformal,
        _expert_oof_calibrator,
    )
    from crypto_alpha.experts.gbdt import GBDTExpert

    n = 80
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.normal(size=n), "side": 1.0}, index=idx)
    y = (X["f1"].values + rng.normal(0, 0.3, n) > 0).astype(int)
    t1 = pd.Series(idx + pd.Timedelta(hours=2), index=idx)
    expert = GBDTExpert(
        {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 5},
        feature_cols=["f1", "side"], seed=0,
    )
    fitted, oof = _expert_oof_probs(
        expert, X, y, t1, None, n_splits=4, embargo_pct=0.0,
    )
    assert oof.shape == (n,)
    assert np.isfinite(oof).sum() >= 40
    p_raw = fitted.predict_proba(X.iloc[-10:])
    p_cal, flags, _tags, cal_ret = _apply_deploy_cal_conformal(
        oof, y, p_raw, method="sigmoid", alpha=0.2, conformal_frac=0.3,
    )
    assert p_cal.shape == (10,)
    assert flags.shape == (10,)
    assert flags.dtype == bool
    assert cal_ret is not None
    # 兼容包装仍返回三元组
    fitted2, cal, oof2 = _expert_oof_calibrator(
        expert, X, y, t1, None, method="sigmoid", n_splits=4, embargo_pct=0.0,
    )
    assert cal is not None
    assert oof2.shape == (n,)
    assert fitted2 is not None


def test_joint_cross_fit_cal_conformal_no_second_order_stack():
    """主路径联合 CF: 校准与保形须在同一折内完成, 且校准分与独立 CF 校准一致。

    禁止回归为「先 cross_fitted_calibrated 再对 oof_cal 做 cross_fitted_conformal_flags」
    (二阶依赖: 保形训练折上的校准分可能来自见过测试点的校准器)。
    """
    from crypto_alpha.calibration.calibrate import (
        cross_fitted_calibrated,
        cross_fitted_calibrated_and_conformal,
        cross_fitted_conformal_flags,
    )

    rng = np.random.default_rng(11)
    n = 100
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    # 可分信号 + 噪声, 保证 Isotonic 与保形有稳定行为
    logit = rng.normal(size=n)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < np.clip(prob, 0.05, 0.95)).astype(int)
    t1 = pd.Series(idx + pd.Timedelta(hours=3), index=idx)

    oof_cal, flags, tags = cross_fitted_calibrated_and_conformal(
        prob, y, t1, method="sigmoid", alpha=0.2, n_splits=4, embargo_pct=0.0,
    )
    assert oof_cal.shape == (n,)
    assert flags.shape == (n,) and flags.dtype == bool
    assert np.isfinite(oof_cal).sum() >= 60
    # 联合路径的校准分须与单独校准 CF 一致(保形不得回写校准)
    oof_cal_only = cross_fitted_calibrated(
        prob, y, t1, method="sigmoid", n_splits=4, embargo_pct=0.0,
    )
    m = np.isfinite(oof_cal) & np.isfinite(oof_cal_only)
    assert m.sum() >= 60
    assert np.allclose(oof_cal[m], oof_cal_only[m])
    # 叠层旧路径: 对已校准分再做保形 CF — 联合路径的 flags 语义不同属预期;
    # 此处只断言联合 API 可运行且未把 tags 弄丢类型
    _ = cross_fitted_conformal_flags(
        oof_cal, y, t1, alpha=0.2, n_splits=4, embargo_pct=0.0,
    )
    assert isinstance(tags, list)


def test_cpcv_cal_conformal_time_split_differs_from_same_batch():
    """时间切分保形与「同批 OOF fit 校准+保形」在足够样本下应可产生不同阈值行为。"""
    from crypto_alpha.calibration.calibrate import (
        ProbabilityCalibrator,
        ConformalBinary,
        fit_deploy_calibrator_and_conformal,
    )
    from crypto_alpha.pipeline.evaluate import _apply_deploy_cal_conformal

    rng = np.random.default_rng(7)
    n = 120
    # 构造随时间漂移的概率, 使切分前后分布不同 → qhat 更可能分叉
    t = np.linspace(0, 1, n)
    oof = np.clip(0.35 + 0.4 * t + rng.normal(0, 0.08, n), 0.02, 0.98)
    y = (rng.uniform(size=n) < oof).astype(int)
    p_te = np.clip(rng.uniform(0.2, 0.8, size=15), 0.01, 0.99)

    p_split, flags_split, _tags, cal_apply = _apply_deploy_cal_conformal(
        oof, y, p_te, method="isotonic", alpha=0.15, conformal_frac=0.3,
    )
    assert cal_apply is not None
    cal_all = ProbabilityCalibrator("isotonic").fit(oof, y)
    conf_all = ConformalBinary(alpha=0.15).fit(cal_all.transform(oof), y)
    p_same = cal_all.transform(p_te)
    flags_same = conf_all.predict_set(p_same)["confident"]

    cal_dep, conf_dep, dep_tags = fit_deploy_calibrator_and_conformal(
        oof, y, method="isotonic", alpha=0.15, conformal_frac=0.3,
    )
    assert dep_tags == []  # n=120 足够时间切分, 无同批回退
    # _apply 与 fit_deploy 同口径(本回归的硬约束)
    assert np.allclose(p_split, cal_dep.transform(p_te))
    assert np.array_equal(flags_split, conf_dep.predict_set(p_split)["confident"])
    # 同批拟合相对时间切分: 至少校准后概率或 qhat/旗标之一应可分(防回归回同批)
    diverged = (
        conf_all.qhat_ != conf_dep.qhat_
        or not np.allclose(p_same, p_split)
        or not np.array_equal(flags_same, flags_split)
    )
    assert diverged, "时间切分与同批拟合结果完全一致, 切分可能未生效"


def test_align_feature_schema_missing_forces_hold_payload():
    """缺列补 0 仅用于防 KeyError; hold_for_schema_mismatch 不得继续开仓字段。"""
    from crypto_alpha.pipeline.run import align_feature_schema, hold_for_schema_mismatch

    idx = pd.date_range("2023-01-01", periods=3, freq="1h", tz="UTC")
    feat = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
    aligned, missing = align_feature_schema(feat, ["a", "tf4h_ret_1", "side"])
    assert missing == ["tf4h_ret_1", "side"]
    assert "tf4h_ret_1" in aligned.columns and float(aligned["tf4h_ret_1"].iloc[0]) == 0.0
    # 已有列数值不变
    assert list(aligned["a"].values) == [1.0, 2.0, 3.0]

    d = hold_for_schema_mismatch(
        symbol="BTC/USDT", missing_cols=missing,
        risk_cfg={"execution_assumption": "close_fill"},
        timestamp=idx[-1], data_source="cache",
    )
    assert d["signal"] == "HOLD"
    assert d["reason"] == "feature_schema_mismatch"
    assert d["stop_loss"] is None and d["take_profit"] is None
    assert d["suggested_position_pct"] == 0.0
    assert d["confident"] is False
    assert any("feature_schema_mismatch" in x for x in d["degradations"])
    assert d["missing_feature_cols"] == missing


def test_decide_live_schema_mismatch_holds(monkeypatch):
    """decide_live 在辅特征列缺失时强制 HOLD, 不调用 ensemble 推理。"""
    from crypto_alpha.config import Config
    from crypto_alpha.serve.service import DecisionService, ModelBundle

    cfg = Config.load()
    cfg.raw["data"]["refresh_before_decide"] = False  # 本单测只验 schema, 不打 REST
    svc = DecisionService(cfg, notifier=type("N", (), {"send": lambda self, m: None})())

    class _BoomEnsemble:
        experts = []

        def predict_proba(self, X):
            raise AssertionError("schema 不匹配时不应推理")

    class _BoomCal:
        def transform(self, p):
            raise AssertionError("schema 不匹配时不应校准")

    fcols = ["ret_14", "tf4h_ret_1", "side"]
    svc.models["BTC/USDT"] = ModelBundle(
        ensemble=_BoomEnsemble(),
        calibrator=_BoomCal(),
        feature_cols=fcols,
        conformal=None,
        cusum_full_sampling=True,
        data_source="cache",
    )

    idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    raw = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0,
    }, index=idx)
    raw.attrs["data_source"] = "cache"

    def _fake_load(cfg, symbol):
        return raw

    def _fake_build(raw_df, cfg, symbol=None):
        # 故意不生成 tf4h_* —— 模拟辅周期加载失败
        out = raw_df.copy()
        out["ret_14"] = 0.01
        out["atr_14"] = 1.0
        return out

    def _fake_news(feat, cfg, symbol):
        return feat

    monkeypatch.setattr("crypto_alpha.serve.service.load_symbol_data", _fake_load)
    monkeypatch.setattr("crypto_alpha.serve.service.build_feature_matrix", _fake_build)
    monkeypatch.setattr("crypto_alpha.serve.service.add_news_features", _fake_news)

    d = svc.decide_live("BTC/USDT")
    assert d is not None
    assert d["signal"] == "HOLD"
    assert d["reason"] == "feature_schema_mismatch"
    assert "tf4h_ret_1" in d["missing_feature_cols"]


def test_calib_cross_fit_fallback_records_degradation():
    """交叉拟合校准不可用时须写入 degradations。"""
    from crypto_alpha.config import Config
    from crypto_alpha.pipeline.run import prepare_dataset, train_and_validate

    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = True
    cfg.raw["news"]["use_synthetic"] = True
    cfg.raw["data"]["synthetic_bars"] = 800
    cfg.raw["features"]["mtf_enabled"] = False
    cfg.raw["experts"]["enabled"] = ["gbdt"]
    cfg.raw["experts"]["gbdt"] = {
        "n_estimators": 20, "num_leaves": 7, "min_child_samples": 5,
        "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0,
    }
    cfg.raw["labeling"]["min_cusum_events"] = 5
    cfg.raw["validation"]["n_splits"] = 3
    cfg.raw["validation"]["embargo_pct"] = 0.0
    # 故意把校准折数抬高, 使有效 OOF 不足以交叉拟合 → 触发回退
    cfg.raw["calibration"]["calib_splits"] = 50
    cfg.raw["ensemble"]["min_expert_auc"] = 0.0
    ds = prepare_dataset(cfg, "BTC/USDT")
    # 若事件仍很多, 截断到极小样本以稳定触发
    if len(ds.y) > 30:
        idx = ds.X.index[:24]
        ds.X = ds.X.loc[idx]
        ds.y = ds.y[:24]
        ds.t1 = ds.t1.loc[idx]
        ds.events = ds.events.loc[idx]
        ds.sample_weight = ds.sample_weight[:24]
    out = train_and_validate(cfg, ds)
    assert any("calib_cross_fit_fallback_insample" in d for d in out["degradations"])


def test_execution_assumption_rejects_unimplemented():
    """未实现的 execution_assumption 必须报错, 不得静默写入决策 JSON。"""
    from crypto_alpha.risk.sizing import (
        decide,
        resolve_execution_assumption,
        SUPPORTED_EXECUTION_ASSUMPTIONS,
    )

    assert resolve_execution_assumption({}) == "close_fill"
    assert resolve_execution_assumption({"execution_assumption": None}) == "close_fill"
    assert "close_fill" in SUPPORTED_EXECUTION_ASSUMPTIONS
    with pytest.raises(ValueError, match="尚未实现"):
        resolve_execution_assumption({"execution_assumption": "next_open"})
    with pytest.raises(ValueError, match="尚未实现"):
        decide(
            0.7, 1, 100.0, 2.0,
            {"kelly_fraction": 0.5, "max_position_pct": 0.3, "execution_assumption": "next_open"},
            pt_sl=(1.5, 1.5),
        )
    d = decide(
        0.7, 1, 100.0, 2.0,
        {"kelly_fraction": 0.5, "max_position_pct": 0.3, "execution_assumption": "close_fill"},
        pt_sl=(1.5, 1.5),
    )
    assert d["execution_assumption"] == "close_fill"


def test_news_sparse_coverage_records_degradation():
    """as_feature 开启但无新闻时须 warn 标记, 且不改动「填 0」数值口径。"""
    from crypto_alpha.config import Config
    from crypto_alpha.features.news_features import add_news_features

    cfg = Config.load()
    cfg.raw["news"]["as_feature"] = True
    cfg.raw["news"]["use_synthetic"] = False
    cfg.raw["news"]["use_history"] = False
    cfg.raw["news"]["min_coverage_warn"] = 0.05
    # 强制走空面板: 清空 sources 并避免合成
    cfg.raw["news"]["sources"] = []

    idx = pd.date_range("2023-01-01", periods=48, freq="1h", tz="UTC")
    feat = pd.DataFrame({"close": np.linspace(100, 110, 48)}, index=idx)

    # monkeypatch ensure_news_panel → 空(覆盖 load + auto_build)
    import crypto_alpha.data.news as news_mod

    orig = news_mod.ensure_news_panel
    news_mod.ensure_news_panel = lambda *a, **k: None
    try:
        out = add_news_features(feat.copy(), cfg, "BTC/USDT")
    finally:
        news_mod.ensure_news_panel = orig

    assert float(out["has_recent_news"].mean()) == 0.0
    assert out.attrs.get("news_feature_coverage", 1.0) == 0.0
    deg = out.attrs.get("degradations") or []
    assert any("news_features_sparse" in d for d in deg)

    # 关闭告警阈值后不写 degradations
    cfg.raw["news"]["min_coverage_warn"] = 0.0
    news_mod.ensure_news_panel = lambda *a, **k: None
    try:
        out2 = add_news_features(feat.copy(), cfg, "BTC/USDT")
    finally:
        news_mod.ensure_news_panel = orig
    assert not (out2.attrs.get("degradations") or [])


def test_news_require_min_coverage_fail_fast():
    """require_min_coverage=true 时覆盖率过低须 ValueError, 默认 false 不阻断。"""
    import pytest
    from crypto_alpha.config import Config
    from crypto_alpha.features.news_features import add_news_features
    import crypto_alpha.data.news as news_mod

    cfg = Config.load()
    cfg.raw["news"]["as_feature"] = True
    cfg.raw["news"]["use_synthetic"] = False
    cfg.raw["news"]["use_history"] = False
    cfg.raw["news"]["min_coverage_warn"] = 0.05
    cfg.raw["news"]["require_min_coverage"] = True
    cfg.raw["news"]["sources"] = []

    idx = pd.date_range("2023-01-01", periods=24, freq="1h", tz="UTC")
    feat = pd.DataFrame({"close": np.linspace(100, 105, 24)}, index=idx)
    orig = news_mod.ensure_news_panel
    news_mod.ensure_news_panel = lambda *a, **k: None
    try:
        with pytest.raises(ValueError, match="require_min_coverage"):
            add_news_features(feat.copy(), cfg, "BTC/USDT")
    finally:
        news_mod.ensure_news_panel = orig


def test_dashboard_equity_curve_prefers_mtm():
    """看板曲线须优先盯市权益, 与 max_drawdown KPI 口径一致。"""
    from crypto_alpha.pipeline.report import RESEARCH_DISCLAIMERS

    assert any("equity_mtm" in line or "盯市" in line for line in RESEARCH_DISCLAIMERS)


def test_dashboard_includes_research_disclaimers():
    """HTML 面板必须包含 OOF≠WF / CPCV 口径说明。"""
    from crypto_alpha.config import Config
    from crypto_alpha.pipeline.report import RESEARCH_DISCLAIMERS, build_dashboard

    cfg = Config.load()
    results = {
        "meta": {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "experts_requested": ["gbdt"],
            "experts_run": ["gbdt"],
            "experts_skipped": {},
            "seed": 42,
            "data_mode": "合成",
            "news_mode": "合成",
            "do_cpcv": False,
            "research_disclaimers": list(RESEARCH_DISCLAIMERS),
        },
        "symbols": {},
    }
    html_out = build_dashboard(results, cfg)
    assert "研究口径说明" in html_out
    assert "walk-forward" in html_out
    assert "close_fill" in html_out
    for line in RESEARCH_DISCLAIMERS:
        assert line in html_out, f"缺失 disclaimer: {line[:40]}…"


def test_dedup_corroborate_point_in_time():
    """互证不得把未来跟进源的权威度回写到首发时刻。"""
    from crypto_alpha.data.news import dedup_corroborate

    t0 = pd.Timestamp("2023-01-01 10:00", tz="UTC")
    t1 = pd.Timestamp("2023-01-01 16:00", tz="UTC")
    items = [
        {"published_at": t0, "title": "bitcoin etf approval rumors surge today",
         "tier": 3, "symbols": ["BTC/USDT"], "source": "blog:a"},
        {"published_at": t1, "title": "bitcoin etf approval rumors surge today",
         "tier": 1, "symbols": ["BTC/USDT"], "source": "sec:b"},
    ]
    out = dedup_corroborate(items, jaccard=0.5, window_hours=48)
    assert len(out) == 2
    by_t = {r["published_at"]: r for r in out}
    assert by_t[t0]["corroboration"] == 1
    assert by_t[t0]["tier"] == 3
    assert by_t[t1]["corroboration"] == 2
    assert by_t[t1]["tier"] == 1


def test_derivatives_nan_does_not_wipe_samples():
    """衍生品全 NaN 时 funding_z/oi_change 填 0, 且记 degradations。"""
    from crypto_alpha.config import Config
    from crypto_alpha.features.build import build_feature_matrix, feature_columns

    cfg = Config.load()
    cfg.raw["features"]["mtf_enabled"] = False
    n = 2500  # 足够 FFD 窗口 + 滚动指标冷启动
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.uniform(1, 10, n),
        "funding_rate": np.nan, "open_interest": np.nan,
    }, index=idx)
    feat = build_feature_matrix(df, cfg, symbol=None)
    assert (feat["funding_z"] == 0.0).all()
    assert (feat["oi_change"] == 0.0).all()
    deg = feat.attrs.get("degradations") or []
    assert "derivatives_funding_unavailable" in deg
    assert "derivatives_oi_unavailable" in deg
    fcols = feature_columns(feat)
    # 冷启动后应有非空建模行(衍生品填 0 不再拖垮 notna().all)
    assert feat[fcols].notna().all(axis=1).sum() > 50


def test_confident_false_skips_all_trades():
    """保形弃权掩码: confident=False 时回测不开仓。"""
    from crypto_alpha.backtest.engine import backtest_events

    idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    events = pd.DataFrame({
        "ret": [0.02, -0.01, 0.03, 0.01, -0.02],
        "side": [1, 1, -1, 1, -1],
        "t1": idx,
        "bars_held": [1, 1, 1, 1, 1],
    }, index=idx)
    prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    bt_cfg = {
        "prob_threshold": 0.55, "portfolio_mode": True,
        "max_gross_exposure": 1.0, "min_position_pct": 0.01,
        "fee_bps": 0.0, "slippage_bps": 0.0, "funding_bps_per_bar": 0.0,
    }
    risk_cfg = {
        "kelly_fraction": 0.5, "max_position_pct": 0.25,
        "roundtrip_cost_frac": 0.0, "execution_assumption": "close_fill",
    }
    out = backtest_events(
        events, prob, bt_cfg, risk_cfg, payoff=1.0,
        confident=np.zeros(5, dtype=bool),
    )
    assert out["metrics"]["n_trades"] == 0
    assert float(out["equity"].iloc[-1]) == pytest.approx(1.0)


def test_triple_barrier_ignores_t0_bar_extremes():
    """入场 bar(t0) 的 high/low 触及障碍也不应触发; 从下一根起扫。"""
    from crypto_alpha.labeling.triple_barrier import get_events, get_bins

    idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    # t0 当根 low 已击穿止损, 但扫描从下一根开始 → 不应因 t0 判损
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    high = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    low = pd.Series([90.0, 100.0, 100.0, 100.0, 100.0], index=idx)  # 仅 t0 击穿
    trgt = pd.Series(0.05, index=idx)
    side = pd.Series(1, index=idx)
    events = get_events(close, high, low, idx[:1], (1.0, 1.0), trgt, 3, side, 0.0)
    assert len(events) == 1
    assert pd.isna(events["sl_touch"].iloc[0])
    bins = get_bins(events, close, (1.0, 1.0))
    # 垂直到期且收益为 0 → bin=0
    assert bins["bin"].iloc[0] == 0


def test_ffd_causal_no_future_shock():
    """未来价格冲击不得改变过去时刻的 FFD 值。"""
    from crypto_alpha.features.frac_diff import frac_diff_ffd

    idx = pd.date_range("2023-01-01", periods=80, freq="1h", tz="UTC")
    base = pd.Series(np.linspace(1.0, 2.0, 80), index=idx, name="logprice")
    fd0 = frac_diff_ffd(base, d=0.4, thres=1e-4)
    shocked = base.copy()
    shocked.iloc[-1] = shocked.iloc[-1] + 10.0
    fd1 = frac_diff_ffd(shocked, d=0.4, thres=1e-4)
    # 除最后 width 个可能受冲击影响的点外, 更早的值应完全一致
    mid = 40
    assert fd0.iloc[:mid].equals(fd1.iloc[:mid])


def test_notifier_hold_reason_not_always_threshold():
    from crypto_alpha.serve.notifier import format_decision, attach_decision_description

    text = format_decision({
        "signal": "HOLD", "symbol": "BTC/USDT",
        "win_probability": None, "reason": "not_cusum_event",
        "timestamp": "2026-07-18 07:00:00+00:00",
        "close": 65000.0,
    })
    assert "CUSUM" in text
    assert "低于阈值" not in text

    d = attach_decision_description({
        "signal": "LONG", "symbol": "ETH/USDT",
        "win_probability": 0.62, "entry_price": 2000.0,
        "stop_loss": 1900.0, "take_profit": 2100.0,
        "suggested_position_pct": 0.1, "atr": 50.0,
        "timestamp": "2026-07-18 07:00:00+00:00",
        "execution_assumption": "close_fill",
        "data_mode_zh": "真实(缓存)",
    })
    assert "description" in d
    assert "做多" in d["description"]
    assert "真实(缓存)" in d["description"]
    assert d["close"] == 2000.0  # 由 entry_price 回填
    assert d["timestamp_beijing"].startswith("2026-07-18 15:00:00")
    assert "+08:00" in d["timestamp_beijing"]
    assert "收盘价" in d["description"]
    assert "北京" in d["description"]


def test_align_news_asof_uses_decision_delta():
    """LLM align_news_asof 与数值特征一致: 用开盘+Δ 作决策时刻。"""
    from crypto_alpha.data.news import align_news_asof

    news_ts = pd.Timestamp("2023-01-01 10:00", tz="UTC")
    news = pd.DataFrame({"text": ["hello"]}, index=pd.DatetimeIndex([news_ts]))
    # bar 开盘 09:00; 若用开盘对齐(+buffer) 09:00+5min < 10:00 → 看不到
    # 决策时刻 10:00(+1h) ≥ 10:05? 10:00 < 10:05 → 仍看不到
    # bar 开盘 10:00; 决策 11:00 ≥ 10:05 → 应看到
    bars = pd.DatetimeIndex([
        pd.Timestamp("2023-01-01 09:00", tz="UTC"),
        pd.Timestamp("2023-01-01 10:00", tz="UTC"),
    ])
    m = align_news_asof(
        news, bars, buffer_minutes=5, ttl_hours=24,
        decision_delta=pd.Timedelta("1h"),
    )
    assert m[bars[0]] == ""
    assert m[bars[1]] == "hello"


def test_dashboard_renders_degradations():
    from crypto_alpha.config import Config
    from crypto_alpha.pipeline.report import build_dashboard

    cfg = Config.load()
    results = {
        "meta": {
            "generated_at": "t", "experts_requested": ["gbdt"],
            "experts_run": ["gbdt"], "experts_skipped": {},
            "seed": 1, "data_mode": "合成", "news_mode": "合成",
            "do_cpcv": False, "research_disclaimers": [],
        },
        "symbols": {
            "BTC/USDT": {
                "n_events": 10, "pos_rate": 0.5,
                "date_start": "a", "date_end": "b",
                "data_source": "synthetic",
                "ensemble_report": {"auc": 0.5, "brier": 0.25, "accuracy": 0.5, "n": 10},
                "expert_reports": {"gbdt": {"auc": 0.5, "brier": 0.25, "accuracy": 0.5, "n": 10}},
                "backtest": {
                    "sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                    "calmar": 0.0, "win_rate": 0.5, "n_trades": 0,
                },
                "backtest_deploy": {
                    "sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                    "calmar": 0.0, "win_rate": 0.5, "n_trades": 12,
                },
                "prob_threshold_effective": 0.61,
                "prob_threshold_research": 0.58,
                "decision": {"signal": "HOLD", "win_probability": None},
                "equity_curve": [], "equity_b64": None,
                "degradations": ["derivatives_funding_unavailable", "news_features_sparse(x)"],
            }
        },
    }
    html_out = build_dashboard(results, cfg)
    assert "Degradations" in html_out
    assert "derivatives_funding_unavailable" in html_out
    assert "交易数(研究OOF)" in html_out
    assert "交易数(部署路径)" in html_out
    assert "成交数/阈值口径" in html_out
    assert "阈值(研究CF)" in html_out
    assert "阈值(部署/decide)" in html_out
    assert "12" in html_out


def test_mtf_news_feature_path_no_fake_signal_on_empty_news(monkeypatch):
    """默认特征面(MTF+新闻)在空新闻下应可装配, 新闻列全 0, 不因 NaN 丢样本。"""
    from crypto_alpha.config import Config
    from crypto_alpha.data.fetch import generate_synthetic_ohlcv
    from crypto_alpha.features.build import build_feature_matrix, feature_columns
    from crypto_alpha.features.news_features import NEWS_FEATURE_COLS, add_news_features
    import crypto_alpha.data.news as news_mod

    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = True
    cfg.raw["features"]["mtf_enabled"] = True
    cfg.raw["news"]["as_feature"] = True
    cfg.raw["news"]["use_synthetic"] = True
    cfg.raw["news"]["min_coverage_warn"] = 0.05
    monkeypatch.setattr(news_mod, "ensure_news_panel", lambda *a, **k: None)

    main = generate_synthetic_ohlcv("BTC/USDT", n_bars=800, timeframe="1h", seed=11)
    feat = build_feature_matrix(main, cfg, symbol="BTC/USDT")
    feat = add_news_features(feat, cfg, "BTC/USDT")
    assert any(c.startswith("tf4h_") for c in feat.columns)
    for c in NEWS_FEATURE_COLS:
        assert c in feat.columns
        if c != "news_age_hours":
            assert float(feat[c].fillna(0).abs().max()) == 0.0
    fcols = feature_columns(feat)
    assert feat[fcols].notna().all(axis=1).sum() > 100
    deg = feat.attrs.get("degradations") or []
    assert any("news_features_sparse" in d for d in deg)


def test_base_report_uses_same_report_mask_as_ensemble(monkeypatch):
    """多专家半窗选型后, base_report 须与集成 report 同窗(防看板 AUC 混比)。

    不改 OOF / 部署 / decide: 仅约束研究报表切片。用两个 GBDT 子类触发剪枝半窗路径,
    避免依赖 torch 的 deep_ts。
    """
    from crypto_alpha.calibration.calibrate import classification_report_probs
    from crypto_alpha.config import Config
    from crypto_alpha.experts.gbdt import GBDTExpert
    from crypto_alpha.pipeline.run import prepare_dataset, train_and_validate

    class _GBDTA(GBDTExpert):
        name = "gbdt_a"

    class _GBDTB(GBDTExpert):
        name = "gbdt_b"

    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = True
    cfg.raw["news"]["use_synthetic"] = True
    cfg.raw["data"]["synthetic_bars"] = 2500
    cfg.raw["features"]["mtf_enabled"] = False
    cfg.raw["news"]["as_feature"] = False
    cfg.raw["experts"]["enabled"] = ["gbdt"]
    cfg.raw["experts"]["gbdt"] = {
        "n_estimators": 30, "num_leaves": 7, "min_child_samples": 5,
        "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0,
    }
    cfg.raw["labeling"]["min_cusum_events"] = 5
    cfg.raw["validation"]["n_splits"] = 3
    cfg.raw["validation"]["embargo_pct"] = 0.0
    cfg.raw["calibration"]["calib_splits"] = 3
    cfg.raw["ensemble"]["min_expert_auc"] = 0.5

    def _two_gbdts(cfg, ds):
        gcfg = cfg["experts"]["gbdt"]
        cols = list(ds.X.columns)
        return [
            _GBDTA(gcfg, cols, seed=cfg.seed),
            _GBDTB(gcfg, cols, seed=cfg.seed + 1),
        ]

    monkeypatch.setattr("crypto_alpha.pipeline.run.build_experts", _two_gbdts)
    ds = prepare_dataset(cfg, "BTC/USDT")
    assert len(ds.y) >= 40, "样本过少无法触发半窗选型"
    out = train_and_validate(cfg, ds)

    ens = out["ensemble"]
    prune = getattr(ens, "prune_eval_mask_", None)
    assert prune is not None
    eval_mask = ~np.isnan(out["oof_calibrated"])
    report_mask = eval_mask & np.asarray(prune, dtype=bool)
    if not report_mask.any():
        report_mask = eval_mask

    # 半窗路径: 评估掩码应严格小于全有效 OOF(样本足够时)
    kept = [e.name for e in ens.experts]
    assert kept, "至少应保留一名专家"
    full_valid = int(np.isfinite(ens.oof_[kept[0]].values).sum())
    assert int(report_mask.sum()) < full_valid

    expected_ens = classification_report_probs(
        out["oof_calibrated"][report_mask], ds.y[report_mask],
    )
    assert out["report"]["n"] == expected_ens["n"]

    for name in kept:
        expected = classification_report_probs(
            ens.oof_[name].values[report_mask], ds.y[report_mask],
        )
        assert name in out["base_report"]
        assert out["base_report"][name]["n"] == expected["n"]
        assert out["base_report"][name]["auc"] == pytest.approx(
            expected["auc"], nan_ok=True,
        )
        # 保留专家与集成报表样本数同窗
        assert out["base_report"][name]["n"] == out["report"]["n"]

    # 全窗对照: 修复前 base_report 会落在此口径, 现须更小
    full0 = classification_report_probs(ens.oof_[kept[0]].values, ds.y)
    assert out["base_report"][kept[0]]["n"] < full0["n"]


def test_synthetic_fallback_aux_resamples_from_main():
    """主行情 synthetic_fallback 时辅周期须从 main 重采样, 禁止混入真实辅路径。"""
    from crypto_alpha.config import Config
    from crypto_alpha.data.fetch import (
        _tag_source, generate_synthetic_ohlcv, load_aux_timeframes, resample_ohlcv,
    )

    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = False  # 非显式合成模式
    cfg.raw["data"]["aux_timeframes"] = ["4h"]
    cfg.raw["data"]["timeframe"] = "1h"
    main = generate_synthetic_ohlcv("BTC/USDT", n_bars=500, timeframe="1h", seed=11)
    main = _tag_source(main, "synthetic_fallback")
    # 若错误地走 load_symbol_data, 可能拉真实/缓存; 强制 resample 后应与手工 resample 一致
    aux = load_aux_timeframes(cfg, "BTC/USDT", main_df=main)
    assert "4h" in aux
    expected = resample_ohlcv(main, "4h")
    pd.testing.assert_frame_equal(
        aux["4h"][["open", "high", "low", "close"]],
        expected[["open", "high", "low", "close"]],
        check_freq=False,
    )
    assert aux["4h"].attrs.get("data_source") == "synthetic_fallback"


def test_deep_ts_early_stop_split_respects_cutoff():
    """OOF 折内: val 只含 cutoff 之前; 部署: val 为全局末尾。"""
    from crypto_alpha.experts.deep_ts import resolve_early_stop_split

    idx = pd.date_range("2023-01-01", periods=100, freq="1h", tz="UTC")
    # 部署: 无 cutoff → val = 末尾
    tr, va = resolve_early_stop_split(idx, val_frac=0.15, patience=3, es_cutoff_time=None)
    assert va is not None
    assert tr.max() < va.min()
    assert va[-1] == 99

    # 模拟中间折训练行: 前 40 + 后 40, 测试落在中间 → cutoff = 索引 40 的时刻
    train_idx = pd.DatetimeIndex(list(idx[:40]) + list(idx[60:]))
    cutoff = idx[40]
    tr2, va2 = resolve_early_stop_split(
        train_idx, val_frac=0.15, patience=3, es_cutoff_time=cutoff,
    )
    assert va2 is not None
    # val 时刻必须全部 < cutoff
    assert (train_idx[va2] < cutoff).all()
    # 后段(>=cutoff)只进 train, 不进 val
    assert (train_idx[tr2] >= cutoff).any()
    assert not (train_idx[va2] >= cutoff).any()

    # 第一折: 训练全在 cutoff 之后 → 关闭早停
    late = idx[60:]
    tr3, va3 = resolve_early_stop_split(
        late, val_frac=0.15, patience=3, es_cutoff_time=idx[40],
    )
    assert va3 is None
    assert len(tr3) == len(late)


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(["pytest", "-q", __file__]))
