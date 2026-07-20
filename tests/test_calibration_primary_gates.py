"""部署校准 auto→Platt 回退与主信号 confluence 门控。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_alpha.calibration.calibrate import (
    ProbabilityCalibrator,
    count_unique_prob_levels,
    fit_deploy_calibrator_and_conformal,
)
from crypto_alpha.labeling.meta_labeling import primary_signal


def test_count_unique_prob_levels():
    assert count_unique_prob_levels(np.array([0.1, 0.1, 0.2, np.nan])) == 2


def test_deploy_cal_auto_falls_back_on_isotonic_collapse():
    """弱可分 + 塌缩 isotonic → auto 回退 sigmoid。"""
    rng = np.random.default_rng(0)
    # 近随机分数, isotonic 易塌成少数台阶
    n = 400
    p = rng.uniform(0.45, 0.55, size=n)
    y = rng.integers(0, 2, size=n)
    cal, conf, tags = fit_deploy_calibrator_and_conformal(
        p, y, method="auto", alpha=0.2, conformal_frac=0.3,
        min_margin=0.0, min_unique_levels=20,
    )
    assert conf is not None
    assert any("deploy_cal_method" in t for t in tags)
    # 若触发 fallback, 方法应为 sigmoid
    if any("auto_fallback_sigmoid" in t for t in tags):
        assert cal.method == "sigmoid"
        out = cal.transform(p)
        assert count_unique_prob_levels(out) >= 5


def test_primary_confluence_gate_sets_side_zero():
    idx = pd.date_range("2020-01-01", periods=60, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100, 120, 60), index=idx)
    conf = pd.Series(0.0, index=idx)
    conf.iloc[30:] = 1.0
    side = primary_signal(
        close, kind="momentum", lookback=5,
        confluence=conf, min_confluence=0.5,
    )
    assert (side.iloc[:30] == 0).all()
    assert (side.iloc[30:].abs() == 1).all()


def test_explicit_isotonic_does_not_auto_switch():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.4, 0.6, size=200)
    y = (p > 0.5).astype(int)
    cal, _, tags = fit_deploy_calibrator_and_conformal(
        p, y, method="isotonic", min_unique_levels=50,
    )
    assert cal.method == "isotonic"
    # 可能打 low_unique 告警, 但不改方法
    assert not any("auto_fallback" in t for t in tags)
