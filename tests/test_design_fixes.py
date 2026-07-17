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


def test_embargo_clamps_near_end():
    """近末折禁运不得整段跳过: 测试段后剩余样本即使 < embargo 也必须剔除。"""
    from crypto_alpha.validation.purged_kfold import PurgedKFold

    n = 100
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    t1 = pd.Series(idx, index=idx)  # 零持有期, 专注禁运
    X = pd.DataFrame({"f": np.arange(n)}, index=idx)
    pkf = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.1)  # embargo=10
    folds = list(pkf.split(X))
    # 倒数第二折: te 结束后应禁运 min(10, 剩余) 根
    tr, te = folds[-2]
    end = int(te[-1]) + 1
    ban_end = min(end + 10, n)
    banned = set(range(end, ban_end))
    assert banned, "应存在禁运带"
    assert not (banned & set(tr.tolist()))


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
    """空头标签触碰价与 decide 加性挂单一致(非几何 entry/(1±x))。"""
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
    assert abs(bins["ret"].iloc[0] - (-np.log(1.05))) < 1e-9
    assert atr_stop(entry, atr_abs, -1, mult) == pytest.approx(105.0)
    assert atr_take_profit(entry, atr_abs, -1, mult) == pytest.approx(95.0)


def test_short_take_profit_additive():
    """空头止盈: 价格跌至 entry-mult×atr 即触碰。"""
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
    assert abs(bins["ret"].iloc[0] - (-np.log(0.95))) < 1e-9


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
    """CPCV 单专家路径: 保形应基于 OOF 分数, 且返回三元组含 oof。"""
    from crypto_alpha.pipeline.evaluate import _expert_oof_calibrator
    from crypto_alpha.experts.gbdt import GBDTExpert
    from crypto_alpha.calibration.calibrate import ConformalBinary

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
    fitted, cal, oof = _expert_oof_calibrator(
        expert, X, y, t1, None, method="sigmoid", n_splits=4, embargo_pct=0.0,
    )
    assert cal is not None
    assert oof.shape == (n,)
    assert np.isfinite(oof).sum() >= 40
    # 用 OOF 拟合保形不应抛错, 且与「全量 in-sample 概率」分布可区分用途
    m = ~np.isnan(oof)
    conf = ConformalBinary(alpha=0.2).fit(cal.transform(oof[m]), y[m])
    p_te = cal.transform(fitted.predict_proba(X.iloc[-10:]))
    flags = conf.predict_set(p_te)["confident"]
    assert flags.shape == (10,)


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


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(["pytest", "-q", __file__]))
