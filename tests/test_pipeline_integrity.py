"""闭环完整性闸门: 标注 oracle + CV 不变量 + 空/正对照 + 置换基线 + 回测对账。

目标: 在**不依赖显卡**的前提下, 用可判定的断言把"标注/回测/验证闭环"的逻辑错误
挡在合入之前。若某条 FAIL, 基本可定位到具体环节(标注/净化/堆叠/校准/回测)。
全部只用轻量 GBDT, CPU 秒级。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crypto_alpha.diagnostics.integrity import (  # noqa: E402
    make_predictable_core_dataset,
    make_random_walk_ohlcv,
    run_core_loop,
    core_auc,
    permutation_baseline,
    count_cv_overlaps,
    embargo_gap_ok,
    backtest_reconciliation,
    max_concurrent_gross,
    sanity_check_dataset,
    run_full_pipeline_with_prices,
    _cpu_cfg,
)


# --------------------------------------------------------------------------- #
# A. 标注 oracle: 三重障碍在人工构造价格上必须给出唯一确定结果                 #
# --------------------------------------------------------------------------- #
def _oracle_labels(close, high, low, side_val, vertical_bars=9):
    from crypto_alpha.labeling.triple_barrier import get_events, get_bins

    idx = close.index
    t_events = pd.DatetimeIndex([idx[0]])
    trgt = pd.Series(0.01, index=idx)
    side = pd.Series(side_val, index=idx)
    pt_sl = (1.0, 1.0)
    ev = get_events(close, high, low, t_events, pt_sl, trgt, vertical_bars, side, 0.0)
    return get_bins(ev, close, pt_sl)


def test_oracle_take_profit_only():
    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    close = pd.Series(100.0, index=idx)
    high = close.copy(); high.iloc[2] = 102.0  # +~2% > +1% 阈值
    low = close.copy()
    b = _oracle_labels(close, high, low, side_val=1)
    assert int(b["bin"].iloc[0]) == 1
    assert b["ret"].iloc[0] > 0


def test_oracle_stop_loss_only():
    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    close = pd.Series(100.0, index=idx)
    high = close.copy()
    low = close.copy(); low.iloc[2] = 98.0  # -~2% < -1% 阈值
    b = _oracle_labels(close, high, low, side_val=1)
    assert int(b["bin"].iloc[0]) == 0
    assert b["ret"].iloc[0] < 0


def test_oracle_same_bar_tie_is_loss():
    """同一根 bar 内止盈止损都被触及, 无法辨先后 => 保守判止损(bin=0)。"""
    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    close = pd.Series(100.0, index=idx)
    high = close.copy(); high.iloc[2] = 102.0
    low = close.copy(); low.iloc[2] = 98.0
    b = _oracle_labels(close, high, low, side_val=1)
    assert int(b["bin"].iloc[0]) == 0


def test_oracle_vertical_barrier():
    """未触碰任何水平障碍 => 垂直到期, 按到期收益定 bin。"""
    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100.0, 100.5, 10), index=idx)  # 仅 +0.5% < 1%
    b = _oracle_labels(close, close, close, side_val=1)
    assert int(b["bin"].iloc[0]) == 1  # 缓涨且方向为多 => 到期正收益
    assert int(b["bars_held"].iloc[0]) >= 1


def test_oracle_short_side_symmetry():
    """做空方向下, 缓涨到期应为亏损(bin=0)。"""
    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100.0, 100.5, 10), index=idx)
    b = _oracle_labels(close, close, close, side_val=-1)
    assert int(b["bin"].iloc[0]) == 0


# --------------------------------------------------------------------------- #
# B. 交叉验证不变量: 与数据无关, 恒真                                          #
# --------------------------------------------------------------------------- #
def test_purged_kfold_no_overlap():
    _, _, t1, _, _ = make_predictable_core_dataset(n=800, seed=1, overlap=6)
    assert count_cv_overlaps(t1, n_splits=5, embargo_pct=0.02, kind="purged") == 0


def test_cpcv_no_overlap():
    _, _, t1, _, _ = make_predictable_core_dataset(n=800, seed=1, overlap=6)
    assert count_cv_overlaps(t1, n_splits=6, embargo_pct=0.02, kind="cpcv", n_test_groups=2) == 0


def test_embargo_gap_effective():
    _, _, t1, _, _ = make_predictable_core_dataset(n=800, seed=1, overlap=6)
    assert embargo_gap_ok(t1, n_splits=5, embargo_pct=0.05)


# --------------------------------------------------------------------------- #
# C. 控制实验: 正对照 / 置换基线 / 全链路空对照                                #
# --------------------------------------------------------------------------- #
def test_positive_control_learns_signal():
    X, y, t1, _, fcols = make_predictable_core_dataset(n=1500, seed=0)
    auc = core_auc(run_core_loop(X, y, t1, fcols, seed=42), y)
    assert auc >= 0.62, f"正对照 AUC 过低({auc:.3f}), 闭环可能学不到真信号"


def test_permutation_null_collapses():
    X, y, t1, _, fcols = make_predictable_core_dataset(n=1500, seed=0)
    r = permutation_baseline(X, y, t1, fcols, seed=42)
    assert r["auc_shuffled"] <= 0.58, f"打乱标签仍可预测(AUC={r['auc_shuffled']:.3f}) => 疑似泄漏"
    assert r["gap"] >= 0.10


def test_null_random_walk_full_pipeline():
    """随机游走喂满全链路: OOF AUC 必须≈0.5, 否则标注/特征在制造假信号。"""
    from crypto_alpha.config import Config

    cfg = _cpu_cfg(Config.load(), n_splits=5)
    raw = make_random_walk_ohlcv(n=4000, seed=3)
    ds, trained = run_full_pipeline_with_prices(cfg, raw, "BTC/USDT")
    auc = float(trained["report"].get("auc", float("nan")))
    assert (not np.isfinite(auc)) or auc <= 0.60, f"空对照 AUC 偏高: {auc:.3f}"
    assert len(ds.y) > 100


# --------------------------------------------------------------------------- #
# D. 时移不变性                                                                #
# --------------------------------------------------------------------------- #
def test_time_shift_invariance():
    X, y, t1, _, fcols = make_predictable_core_dataset(n=900, seed=5)
    oof0 = run_core_loop(X, y, t1, fcols, seed=7)
    shift = pd.Timedelta(days=30)
    X2 = X.copy(); X2.index = X.index + shift
    t2 = t1 + shift; t2.index = t2.index + shift
    oof1 = run_core_loop(X2, y, t2, fcols, seed=7)
    assert np.allclose(np.nan_to_num(oof0), np.nan_to_num(oof1), atol=1e-9)


# --------------------------------------------------------------------------- #
# E. 回测对账 / 成本单调 / 并发敞口 / 口径对照                                 #
# --------------------------------------------------------------------------- #
def _bt(events, prob, fee=0.0, portfolio=True, max_gross=1.0):
    from crypto_alpha.backtest.engine import backtest_events

    bt_cfg = {"portfolio_mode": portfolio, "prob_threshold": 0.55,
              "fee_bps": fee, "slippage_bps": fee / 2.0, "min_position_pct": 0.01}
    risk_cfg = {"kelly_fraction": 0.5, "max_position_pct": 0.3,
                "max_gross_exposure": max_gross, "daily_max_drawdown": 0.0}
    return backtest_events(events, prob, bt_cfg, risk_cfg, payoff=1.0)


def test_backtest_reconciliation():
    _, y, _, events, _ = make_predictable_core_dataset(n=600, seed=2)
    prob = np.clip(0.5 + 0.4 * (2 * y - 1), 0.01, 0.99)
    recon = backtest_reconciliation(_bt(events, prob))
    assert recon["equity_matches_pnl"]
    assert recon["metric_matches_equity"]


def test_backtest_cost_monotonic():
    _, y, _, events, _ = make_predictable_core_dataset(n=600, seed=2)
    prob = np.clip(0.5 + 0.4 * (2 * y - 1), 0.01, 0.99)
    r0 = _bt(events, prob, fee=0.0)["metrics"]["total_return"]
    r1 = _bt(events, prob, fee=30.0)["metrics"]["total_return"]
    assert r1 <= r0 + 1e-9


def test_backtest_gross_exposure_cap():
    _, y, _, events, _ = make_predictable_core_dataset(n=400, seed=2, overlap=8)
    prob = np.full(len(y), 0.9)
    peak = max_concurrent_gross(_bt(events, prob, max_gross=1.0))
    assert peak <= 1.0 + 1e-9
    assert peak > 0.0


def test_portfolio_not_above_independent():
    _, y, _, events, _ = make_predictable_core_dataset(n=600, seed=2, overlap=8)
    prob = np.clip(0.5 + 0.4 * (2 * y - 1), 0.01, 0.99)
    port = _bt(events, prob, portfolio=True)["metrics"]["total_return"]
    indep = _bt(events, prob, portfolio=False)["metrics"]["total_return"]
    assert port <= indep + 1e-9


# --------------------------------------------------------------------------- #
# F. 数据集运行期护栏                                                          #
# --------------------------------------------------------------------------- #
def test_sanity_check_flags_bad_dataset():
    idx = pd.date_range("2023-01-01", periods=50, freq="1h", tz="UTC")
    X = pd.DataFrame({"f_const": 1.0, "f_ok": np.arange(50.0)}, index=idx)
    ds = SimpleNamespace(
        y=np.zeros(50, dtype=int),                       # 正类比例=0 => 失衡
        X=X, feature_cols=["f_const", "f_ok"],           # f_const 常数 => 无信息
        sample_weight=np.ones(50),
        t1=pd.Series(idx, index=idx),
    )
    warns = sanity_check_dataset(ds)
    assert any("失衡" in w for w in warns)
    assert any("常数" in w for w in warns)


def test_sanity_check_passes_clean_dataset():
    X, y, t1, _, fcols = make_predictable_core_dataset(n=500, seed=0)
    ds = SimpleNamespace(y=y, X=X, feature_cols=fcols,
                         sample_weight=np.ones(len(y)), t1=t1)
    assert sanity_check_dataset(ds) == []


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call(["pytest", "-q", __file__]))
