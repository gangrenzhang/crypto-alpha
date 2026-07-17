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


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(["pytest", "-q", __file__]))
