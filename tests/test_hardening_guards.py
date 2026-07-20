"""硬化护栏回归: 环境 HOLD、实验日志、波动滑点、单专家报告窗、保形跳过、oi 墙钟。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_alpha.backtest.engine import resolve_event_slippage
from crypto_alpha.diagnostics.decision_audit import build_decision_audit
from crypto_alpha.diagnostics.env_guard import score_degradations, should_hold_for_environment
from crypto_alpha.diagnostics.experiments import (
    append_experiment,
    count_experiments,
    resolve_dsr_n_trials,
)
from crypto_alpha.ensemble.stacking import StackingEnsemble
from crypto_alpha.experts.base import BaseExpert


class _ConstExpert(BaseExpert):
    """不依赖 LightGBM 的常量概率专家(仅测剪枝掩码)。"""

    name = "const"

    def fit(self, X, y, sample_weight=None, **fit_params):
        return self

    def predict_proba(self, X):
        return np.full(len(X), 0.55)

    def clone(self):
        return _ConstExpert(self.cfg, self.feature_cols, seed=self.seed)


class _DummyCfg:
    def __init__(self, raw, artifacts_dir):
        self.raw = raw
        self.artifacts_dir = artifacts_dir

    def __getitem__(self, k):
        return self.raw[k]


def test_env_guard_stacks_and_holds(tmp_path):
    score, keys = score_degradations([
        "ohlcv_synthetic_fallback",
        "ohlcv_tip_exchange_fallback",
        "derivatives_funding_unavailable",
    ])
    assert score >= 50
    assert "ohlcv_synthetic_fallback" in keys
    hold, s, tag = should_hold_for_environment(
        ["ohlcv_synthetic_fallback", "ohlcv_tip_exchange_fallback"],
        threshold=50,
    )
    assert hold and s >= 50 and "low_confidence_environment" in (tag or "")
    hold2, _, _ = should_hold_for_environment(["derivatives_oi_unavailable"], threshold=50)
    assert not hold2
    hold3, _, _ = should_hold_for_environment(
        ["ohlcv_synthetic_fallback"], threshold=0,
    )
    assert not hold3
    # 训练期质量标签不得计入环境分(否则会永久 HOLD)
    score_train, matched = score_degradations([
        "deploy_cal_conformal_fallback_insample",
        "meta_nested_oof_fallback_insample",
        "gbdt:dropped_low_auc(0.48)",
        "conformal_cf_fold_skipped(n_folds=2)",
    ])
    assert score_train == 0 and matched == []
    hold4, _, _ = should_hold_for_environment(
        ["deploy_cal_conformal_fallback_insample", "meta_nested_oof_fallback_insample"],
        threshold=50,
    )
    assert not hold4


def test_experiment_log_raises_dsr_trials(tmp_path):
    raw = {
        "project": {"random_seed": 1},
        "data": {"symbols": ["BTC/USDT"], "timeframe": "30m", "aux_timeframes": [],
                 "use_synthetic": True},
        "labeling": {"primary_signal": "momentum", "primary_lookback": 24, "pt_sl": [1.5, 1.5],
                     "vertical_barrier_bars": 48, "barrier_vol": "atr"},
        "features": {"mtf_enabled": True, "frac_diff_d": 0.4},
        "news": {"as_feature": False},
        "experts": {"enabled": ["gbdt"]},
        "ensemble": {"meta_learner": "logistic", "min_expert_auc": 0.5},
        "backtest": {"prob_threshold": 0.55, "prob_threshold_mode": "fixed",
                     "prob_quantile": 0.98, "slippage_bps": 2.0, "slippage_vol_scale": True},
        "calibration": {"method": "isotonic", "conformal_alpha": 0.1},
        "validation": {"dsr_n_trials": 1, "log_experiments": True},
    }
    cfg = _DummyCfg(raw, tmp_path)
    assert count_experiments(tmp_path) == 0
    append_experiment(tmp_path, cfg, source="unit")
    append_experiment(tmp_path, cfg, source="unit")
    n, tags = resolve_dsr_n_trials(cfg, n_configs=1)
    assert n >= 2
    assert any("experiment_log" in t for t in tags)


def test_slippage_vol_scale_never_below_base():
    base = 2e-4
    # 低波动不降价
    assert resolve_event_slippage(base, 0.01, 0.02, {"slippage_vol_scale": True}) == base
    # 高波动放大, 有上限
    hi = resolve_event_slippage(
        base, 0.06, 0.02, {"slippage_vol_scale": True, "slippage_vol_mult_cap": 3.0},
    )
    assert hi == base * 3.0
    assert resolve_event_slippage(base, 0.06, 0.02, {"slippage_vol_scale": False}) == base


def test_single_expert_report_mask_is_time_second_half():
    rng = np.random.default_rng(0)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="30min", tz="UTC")
    X = pd.DataFrame({"f1": rng.normal(size=n), "side": np.ones(n)}, index=idx)
    y = (rng.random(n) > 0.45).astype(int)
    t1 = pd.Series(idx + pd.Timedelta(hours=6), index=idx)
    e = _ConstExpert({}, ["f1", "side"], seed=0)
    ens = StackingEnsemble([e], {"meta_learner": "logistic", "min_expert_auc": 0.5}, seed=0)
    ens.fit(X, y, t1, n_splits=3, embargo_pct=0.0)
    mask = ens.prune_eval_mask_
    assert mask is not None
    assert int(mask.sum()) < n
    # 后半窗: 第一个 True 应落在时间序后半附近
    pos = np.flatnonzero(mask)
    assert pos[0] >= n // 4


def test_conformal_insufficient_defaults_not_confident():
    from crypto_alpha.calibration.calibrate import cross_fitted_calibrated_and_conformal

    # 样本不足以做 CF 时: 默认 confident=False(不再全 True)
    n = 6
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    prob = np.linspace(0.2, 0.8, n)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    t1 = pd.Series(idx + pd.Timedelta(hours=2), index=idx)
    _, flags, tags = cross_fitted_calibrated_and_conformal(
        prob, y, t1, method="sigmoid", alpha=0.2, n_splits=4, embargo_pct=0.0,
    )
    assert flags.dtype == bool
    assert not flags.any()
    assert any("conformal_cf_insufficient_samples" in t for t in tags)


def test_oi_change_bars_wall_clock_30m():
    from crypto_alpha.config import Config
    from crypto_alpha.features.build import build_feature_matrix

    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = True
    cfg.raw["data"]["timeframe"] = "30m"
    cfg.raw["features"]["mtf_enabled"] = False
    cfg.raw["news"]["as_feature"] = False
    idx = pd.date_range("2024-01-01", periods=200, freq="30min", tz="UTC")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.001, size=200)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": 1.0,
        "open_interest": np.linspace(1e6, 1.2e6, 200),
        "funding_rate": 0.0,
    }, index=idx)
    df.attrs["data_source"] = "synthetic"
    feat = build_feature_matrix(df, cfg, symbol=None)
    assert "oi_change" in feat.columns
    # 30m → 48 bars ≈ 24h; 第 47 根仍可能为 0(pct_change 需要 48)
    assert feat["oi_change"].iloc[47] == 0.0 or np.isnan(feat["oi_change"].iloc[47])
    assert np.isfinite(feat["oi_change"].iloc[60])


def test_decision_audit_fields(tmp_path):
    from crypto_alpha.config import Config

    cfg = Config.load()
    idx = pd.date_range("2024-01-01", periods=10, freq="30min", tz="UTC")
    panel = pd.DataFrame({"close": np.arange(10, dtype=float) + 100}, index=idx)
    audit = build_decision_audit(
        cfg, panel=panel, feature_cols=["a", "b"],
        trained={"prob_threshold_effective": 0.7, "degradations": ["x"]},
    )
    assert audit["config_fingerprint"]
    assert audit["data_window"]["n_bars"] == 10
    assert audit["data_window_hash"]
    assert audit["prob_threshold_effective"] == 0.7
