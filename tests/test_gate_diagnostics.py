"""开仓门控: 阈值解析 / 保形 margin / 校准健康 / 诊断结构。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_alpha.calibration.calibrate import ConformalBinary
from crypto_alpha.diagnostics.gates import (
    assess_calibration_pass_health,
    build_threshold_reference_mask,
    freeze_threshold_on_reference,
    gate_diagnostics,
    raise_threshold_if_inflated,
    resolve_prob_threshold,
)


def test_resolve_prob_threshold_fixed_default():
    thr, tags = resolve_prob_threshold({"prob_threshold": 0.55}, None)
    assert thr == 0.55
    assert tags == []


def test_resolve_prob_threshold_max_of_uses_reference():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.4, 0.7, size=500)
    thr, tags = resolve_prob_threshold(
        {
            "prob_threshold": 0.55,
            "prob_threshold_mode": "max_of",
            "prob_quantile": 0.9,
        },
        reference_probs=ref,
    )
    q = float(np.quantile(ref, 0.9))
    assert thr == max(0.55, q)
    assert any("prob_threshold_resolved" in t for t in tags)


def test_resolve_prob_threshold_quantile_fallback_without_ref():
    thr, tags = resolve_prob_threshold(
        {"prob_threshold": 0.55, "prob_threshold_mode": "quantile", "prob_quantile": 0.99},
        reference_probs=None,
    )
    assert thr == 0.55
    assert any("fallback_fixed" in t for t in tags)


def test_conformal_min_margin_tightens_confident():
    """margin=0 时略高概率可自信; margin>0 时靠近 0.5 被弃权。"""
    # 构造使 qhat 较大、p=0.56 在无 margin 时 confident
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 10)
    p_fit = np.where(y == 1, 0.7, 0.3).astype(float)
    conf0 = ConformalBinary(alpha=0.2, min_margin=0.0).fit(p_fit, y)
    conf_m = ConformalBinary(alpha=0.2, min_margin=0.08).fit(p_fit, y)
    p_te = np.array([0.56, 0.70, 0.30])
    f0 = conf0.predict_set(p_te)["confident"]
    fm = conf_m.predict_set(p_te)["confident"]
    assert f0.dtype == bool and fm.dtype == bool
    # 有 margin 时 confident 集合 ⊆ 无 margin
    assert np.all(fm <= f0)
    # 0.56 距 0.5 仅 0.06 < 0.08 → 在有 margin 时不应自信
    assert bool(fm[0]) is False


def test_calibration_inflate_and_low_unique_tags():
    raw = np.full(100, 0.50)
    # 校准后大量抬过 0.55, 且只有少数台阶
    cal = np.array([0.56] * 80 + [0.45] * 20)
    tags = assess_calibration_pass_health(
        raw, cal, thr=0.55, pass_rate_inflate_max=1.5, min_unique_levels=10,
    )
    assert any("calibration_inflates_pass_rate" in t for t in tags)
    assert any("calibration_low_unique_levels" in t for t in tags)


def test_ref_mask_prefers_eval_minus_report():
    n = 100
    eval_m = np.ones(n, dtype=bool)
    report_m = np.zeros(n, dtype=bool)
    report_m[50:] = True
    ref, tags = build_threshold_reference_mask(eval_m, report_m, min_ref=20)
    assert int(ref.sum()) == 50
    assert tags == []
    assert not np.any(ref & report_m)


def test_ref_mask_time_half_when_report_equals_eval():
    """单专家 report≡eval 时禁止回退全 eval, 改用时间前半段。"""
    n = 80
    eval_m = np.ones(n, dtype=bool)
    report_m = eval_m.copy()
    ref, tags = build_threshold_reference_mask(eval_m, report_m, min_ref=20)
    assert int(ref.sum()) == 40
    assert any("prob_threshold_ref_fallback_time_half" in t for t in tags)
    # 前半 True、后半 False
    assert bool(ref[0]) and bool(ref[39]) and (not bool(ref[40]))


def test_raise_threshold_on_inflate_reference_only():
    raw = np.full(200, 0.50)  # 原始几乎不过 0.55
    cal = np.array([0.90] * 180 + [0.40] * 20)  # 校准后大量过线
    thr0 = 0.55
    thr1, tags = raise_threshold_if_inflated(
        thr0, raw, cal,
        {"prob_quantile": 0.98, "inflate_raise_quantile": 0.99},
        pass_rate_inflate_max=1.5, enabled=True,
    )
    assert any("calibration_inflates_pass_rate" in t for t in tags)
    assert thr1 > thr0
    assert any("prob_threshold_raised_on_inflate" in t for t in tags)


def test_dual_threshold_research_vs_deploy_scales():
    """方案B: 同一 raw 参考窗, CF 尺度与 deploy 尺度分位 thr 应可分叉; 禁止二次校准。"""
    from crypto_alpha.calibration.calibrate import ProbabilityCalibrator

    rng = np.random.default_rng(11)
    raw = np.clip(rng.uniform(0.25, 0.75, size=400), 0.01, 0.99)
    y = (raw > 0.5).astype(int)
    # 模拟「CF 校准分」与「deploy 校准器」两条不同映射
    cal_deploy = ProbabilityCalibrator("sigmoid").fit(raw[:280], y[:280])
    # CF 路径用另一段拟合的校准器变换, 制造尺度差
    cal_cf = ProbabilityCalibrator("sigmoid").fit(raw[120:], y[120:])
    cf_ref = cal_cf.transform(raw)
    deploy_ref = cal_deploy.transform(raw)
    bt = {
        "prob_threshold": 0.55,
        "prob_threshold_mode": "max_of",
        "prob_quantile": 0.9,
        "raise_thr_on_inflate": False,
    }
    thr_r, tags_r = freeze_threshold_on_reference(
        bt, raw, cf_ref, pass_rate_inflate_max=1.5, tag_prefix="research_",
    )
    thr_d, tags_d = freeze_threshold_on_reference(
        bt, raw, deploy_ref, pass_rate_inflate_max=1.5, tag_prefix="deploy_",
    )
    assert any(t.startswith("research_") for t in tags_r)
    assert any(t.startswith("deploy_") for t in tags_d)
    assert thr_r == max(0.55, float(np.quantile(cf_ref, 0.9)))
    assert thr_d == max(0.55, float(np.quantile(deploy_ref, 0.9)))
    # 二次校准(错误): deploy_cal(CF分) 不得当作 deploy 参考
    wrong = cal_deploy.transform(cf_ref)
    thr_wrong, _ = freeze_threshold_on_reference(
        bt, raw, wrong, pass_rate_inflate_max=1.5, tag_prefix="wrong_",
    )
    if not np.isclose(float(np.quantile(deploy_ref, 0.9)), float(np.quantile(wrong, 0.9))):
        assert not np.isclose(thr_d, thr_wrong)


def test_cpcv_thr_reference_uses_calibrated_scale():
    """回归: 阈值参考须与测试折同为校准尺度, 非原始 OOF。"""
    from crypto_alpha.calibration.calibrate import ProbabilityCalibrator

    rng = np.random.default_rng(3)
    oof_raw = rng.uniform(0.3, 0.7, size=300)
    y = (oof_raw > 0.5).astype(int)
    cal = ProbabilityCalibrator("sigmoid").fit(oof_raw, y)
    oof_cal = cal.transform(oof_raw)
    thr_raw, _ = resolve_prob_threshold(
        {"prob_threshold": 0.55, "prob_threshold_mode": "max_of", "prob_quantile": 0.9},
        reference_probs=oof_raw,
    )
    thr_cal, _ = resolve_prob_threshold(
        {"prob_threshold": 0.55, "prob_threshold_mode": "max_of", "prob_quantile": 0.9},
        reference_probs=oof_cal,
    )
    # 尺度不同时分位 thr 一般不同; 若偶然相同也至少保证 cal 路径可解析
    assert np.isfinite(thr_cal)
    assert thr_cal == max(0.55, float(np.quantile(oof_cal, 0.9)))
    # 文档化意图: CPCV 必须用 thr_cal 口径(本断言保证 cal≠raw 时口径分叉可测)
    if not np.isclose(float(np.quantile(oof_raw, 0.9)), float(np.quantile(oof_cal, 0.9))):
        assert not np.isclose(thr_raw, thr_cal)


def test_gate_diagnostics_structure():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    raw = np.array([0.5, 0.6, 0.4, 0.7, 0.55])
    cal = np.array([0.52, 0.62, 0.48, 0.71, 0.56])
    # cal>=0.55 的下标: 1,3,4；其中 confident 仅 1,3 → 交集 2；开仓 size>0: 1,3
    conf = np.array([False, True, False, True, False])
    detail = pd.DataFrame({"size": [0.0, 0.1, 0.0, 0.2, 0.0]}, index=idx)
    g = gate_diagnostics(idx, raw, cal, conf, detail, thr=0.55, conf_obj=None)
    assert g["gates"]["n_prob_ge_threshold"] == 3
    assert g["gates"]["n_prob_and_confident"] == 2
    assert g["gates"]["n_opened_size_gt_0"] == 2
    assert g["calibrated_proba"]["n_unique"] >= 1
