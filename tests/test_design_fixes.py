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
    p_cal, flags = _apply_deploy_cal_conformal(
        oof, y, p_raw, method="sigmoid", alpha=0.2, conformal_frac=0.3,
    )
    assert p_cal.shape == (10,)
    assert flags.shape == (10,)
    assert flags.dtype == bool
    # 兼容包装仍返回三元组
    fitted2, cal, oof2 = _expert_oof_calibrator(
        expert, X, y, t1, None, method="sigmoid", n_splits=4, embargo_pct=0.0,
    )
    assert cal is not None
    assert oof2.shape == (n,)
    assert fitted2 is not None


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

    p_split, flags_split = _apply_deploy_cal_conformal(
        oof, y, p_te, method="isotonic", alpha=0.15, conformal_frac=0.3,
    )
    cal_all = ProbabilityCalibrator("isotonic").fit(oof, y)
    conf_all = ConformalBinary(alpha=0.15).fit(cal_all.transform(oof), y)
    p_same = cal_all.transform(p_te)
    flags_same = conf_all.predict_set(p_same)["confident"]

    cal_dep, conf_dep = fit_deploy_calibrator_and_conformal(
        oof, y, method="isotonic", alpha=0.15, conformal_frac=0.3,
    )
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

    # monkeypatch load_news_panel → 空
    import crypto_alpha.data.news as news_mod

    orig = news_mod.load_news_panel
    news_mod.load_news_panel = lambda *a, **k: None
    try:
        out = add_news_features(feat.copy(), cfg, "BTC/USDT")
    finally:
        news_mod.load_news_panel = orig

    assert float(out["has_recent_news"].mean()) == 0.0
    assert out.attrs.get("news_feature_coverage", 1.0) == 0.0
    deg = out.attrs.get("degradations") or []
    assert any("news_features_sparse" in d for d in deg)

    # 关闭告警阈值后不写 degradations
    cfg.raw["news"]["min_coverage_warn"] = 0.0
    news_mod.load_news_panel = lambda *a, **k: None
    try:
        out2 = add_news_features(feat.copy(), cfg, "BTC/USDT")
    finally:
        news_mod.load_news_panel = orig
    assert not (out2.attrs.get("degradations") or [])


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
        assert line[:20] in html_out or "OOF" in html_out


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(["pytest", "-q", __file__]))
