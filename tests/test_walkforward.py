"""Walk-forward 切分不变量与配置解析单测(CPU, 无显卡, 不拉网)。

覆盖:
  - train ∩ test = ∅
  - 训练 t1 不得越过 label_deadline(含 embargo)
  - test_end=None 吃到面板末
  - require / resolve 配置解析
  - 看板 slim 字段
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha.config import Config
from crypto_alpha.pipeline.walkforward import (
    WalkForwardSplitConfig,
    assert_walkforward_split_invariants,
    build_walkforward_masks,
    resolve_walkforward_split,
    slim_walkforward_for_dashboard,
    walkforward_public_summary,
    walkforward_section,
)


def _idx(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def test_build_masks_purge_by_t1():
    """标签越过 test_start 的事件不得进训练(即便 t0 较早)。"""
    ev = _idx(10, "2022-01-01")
    # 前 5 个事件: t1 在 1 月内; 后 5 个: t0 在 2 月
    # 中间插一个: t0=1月中, 但 t1 伸到 2 月 → 不得进 train
    t1 = list(ev[1:6]) + list(ev[6:])  # 长度 10? need align
    # rebuild carefully
    ev = pd.DatetimeIndex(
        [
            "2022-01-01", "2022-01-05", "2022-01-10", "2022-01-15", "2022-01-20",
            "2022-02-01", "2022-02-05", "2022-02-10", "2022-02-15", "2022-02-20",
        ],
        tz="UTC",
    )
    t1 = pd.Series(
        pd.to_datetime(
            [
                "2022-01-03", "2022-01-08", "2022-01-12", "2022-01-18",
                "2022-02-03",  # t0 在 1 月, t1 越过 2/1 → 排除出 train
                "2022-02-08", "2022-02-12", "2022-02-18", "2022-02-22", "2022-02-25",
            ],
            utc=True,
        ),
        index=ev,
    )
    split = WalkForwardSplitConfig(
        test_start=pd.Timestamp("2022-02-01", tz="UTC"),
        test_end=pd.Timestamp("2022-02-28", tz="UTC"),
    )
    train_m, test_m, tags = build_walkforward_masks(ev, t1, split)
    assert_walkforward_split_invariants(ev, t1, train_m, test_m, split)

    assert int(train_m.sum()) == 4  # 前 4 个; 第 5 个因 t1 泄漏被踢
    assert not train_m[4]
    assert int(test_m.sum()) == 5
    assert not np.any(train_m & test_m)


def test_embargo_tightens_label_deadline():
    ev = pd.DatetimeIndex(
        ["2022-01-01", "2022-01-10", "2022-01-20", "2022-02-05"], tz="UTC",
    )
    t1 = pd.Series(
        pd.to_datetime(
            ["2022-01-05", "2022-01-25", "2022-01-28", "2022-02-10"], utc=True,
        ),
        index=ev,
    )
    split = WalkForwardSplitConfig(
        test_start=pd.Timestamp("2022-02-01", tz="UTC"),
        test_end=None,
        embargo_bars=0,
    )
    # 无禁运: 第 2、3 个 t1 在 1/25、1/28 < 2/1 → 可进 train
    tr0, te0, _ = build_walkforward_masks(ev, t1, split, embargo_delta=pd.Timedelta(0))
    assert int(tr0.sum()) == 3

    # 禁运 10 天: label_deadline=1/22 → 仅第 1 个进 train
    tr1, te1, tags = build_walkforward_masks(
        ev, t1, split, embargo_delta=pd.Timedelta(days=10),
    )
    assert_walkforward_split_invariants(
        ev, t1, tr1, te1, split, embargo_delta=pd.Timedelta(days=10),
    )
    assert int(tr1.sum()) == 1
    assert any("walkforward_embargo" in t for t in tags)
    assert te1[3] and not tr1[3]


def test_nat_t1_rejected():
    ev = pd.DatetimeIndex(["2022-01-01", "2022-01-10"], tz="UTC")
    t1 = pd.Series([pd.Timestamp("2022-01-05", tz="UTC"), pd.NaT], index=ev)
    split = WalkForwardSplitConfig(
        test_start=pd.Timestamp("2022-02-01", tz="UTC"), test_end=None,
    )
    with pytest.raises(ValueError, match="NaT"):
        build_walkforward_masks(ev, t1, split)


def test_purged_tag_when_label_crosses_deadline():
    ev = pd.DatetimeIndex(["2022-01-01", "2022-01-20"], tz="UTC")
    t1 = pd.Series(
        pd.to_datetime(["2022-01-05", "2022-02-05"], utc=True), index=ev,
    )
    split = WalkForwardSplitConfig(
        test_start=pd.Timestamp("2022-02-01", tz="UTC"), test_end=None,
    )
    tr, te, tags = build_walkforward_masks(ev, t1, split)
    assert int(tr.sum()) == 1
    assert any("walkforward_purged_label_overlap" in t for t in tags)
    assert not tr[1] and not te[1]  # 净化带: 两边都不进


def test_invariants_fail_on_overlap():
    ev = _idx(4)
    t1 = pd.Series(ev + pd.Timedelta(days=2), index=ev)
    split = WalkForwardSplitConfig(
        test_start=ev[2], test_end=ev[3],
    )
    train_m = np.array([True, True, False, False])
    test_m = np.array([False, True, True, False])  # 故意重叠
    with pytest.raises(AssertionError, match="train ∩ test"):
        assert_walkforward_split_invariants(ev, t1, train_m, test_m, split)


def test_resolve_split_from_config():
    cfg = Config.load()
    # 不写盘: 深拷贝 raw
    import copy

    cfg.raw = copy.deepcopy(cfg.raw)
    cfg.raw.setdefault("validation", {})["walkforward"] = {
        "test_start": "2022-09-14T00:00:00Z",
        "test_end": None,
        "embargo_bars": 2,
        "min_train_events": 100,
        "min_test_events": 20,
        "initial_capital": 5000.0,
    }
    split = resolve_walkforward_split(cfg)
    assert split.test_start == pd.Timestamp("2022-09-14", tz="UTC")
    assert split.test_end is None
    assert split.embargo_bars == 2
    assert split.min_train_events == 100
    assert split.initial_capital == 5000.0

    split2 = resolve_walkforward_split(
        cfg, test_start="2023-01-01T00:00:00Z", test_end="2023-06-01T00:00:00Z",
    )
    assert split2.test_start.year == 2023
    assert split2.test_end.month == 6

    with pytest.raises(ValueError, match="test_end"):
        resolve_walkforward_split(
            cfg, test_start="2023-06-01", test_end="2023-01-01",
        )

    split3 = resolve_walkforward_split(
        cfg,
        train_start="2020-01-01T00:00:00Z",
        test_start="2024-09-14T00:00:00Z",
        test_end="2026-07-18T00:00:00Z",
    )
    assert split3.train_start == pd.Timestamp("2020-01-01", tz="UTC")
    with pytest.raises(ValueError, match="train_start"):
        resolve_walkforward_split(
            cfg, train_start="2024-09-14", test_start="2024-09-14",
        )


def test_train_start_drops_early_events():
    ev = _idx(10, "2019-12-28")  # crosses into 2020
    t1 = ev + pd.Timedelta(days=1)
    split = WalkForwardSplitConfig(
        test_start=pd.Timestamp("2020-01-05", tz="UTC"),
        test_end=pd.Timestamp("2020-01-08", tz="UTC"),
        train_start=pd.Timestamp("2020-01-01", tz="UTC"),
    )
    tr, te, tags = build_walkforward_masks(ev, t1, split)
    assert any("walkforward_train_start" in t for t in tags)
    assert not np.any(ev[tr] < split.train_start)
    assert_walkforward_split_invariants(ev, t1, tr, te, split)


def test_train_start_identity_and_boundaries():
    """n(无 start)=n(有)+dropped; t0==train_start 进训, t0==test_start 进测。"""
    ev = pd.DatetimeIndex(
        [
            "2019-12-31",
            "2020-01-01",
            "2020-01-02",
            "2024-09-13",
            "2024-09-14",
            "2024-09-15",
        ],
        tz="UTC",
    )
    t1 = ev + pd.Timedelta(hours=1)
    s0 = WalkForwardSplitConfig(
        test_start=pd.Timestamp("2024-09-14", tz="UTC"),
        test_end=pd.Timestamp("2026-07-18", tz="UTC"),
    )
    s1 = WalkForwardSplitConfig(
        test_start=s0.test_start,
        test_end=s0.test_end,
        train_start=pd.Timestamp("2020-01-01", tz="UTC"),
    )
    tr0, te0, _ = build_walkforward_masks(ev, t1, s0)
    tr1, te1, tags = build_walkforward_masks(ev, t1, s1)
    dropped = int(
        next(t for t in tags if "dropped_pre_train=" in t).split("=")[-1].rstrip(")")
    )
    assert int(tr0.sum()) == int(tr1.sum()) + dropped
    assert set(ev[tr1]).issubset(set(ev[tr0]))
    assert list(ev[tr1]) == [ev[1], ev[2], ev[3]]
    assert list(ev[te1]) == [ev[4], ev[5]]
    assert_walkforward_split_invariants(ev, t1, tr1, te1, s1)


def test_train_start_with_embargo():
    """embargo 抬高 label_deadline 时, train_start 仍只约束 t0。"""
    ev = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-10", "2020-01-20", "2020-02-01"], tz="UTC",
    )
    # 第二个事件 t1 越过 deadline(test_start - 5d)
    t1 = pd.Series(
        [
            pd.Timestamp("2020-01-02", tz="UTC"),
            pd.Timestamp("2020-01-28", tz="UTC"),  # crosses deadline 01-27
            pd.Timestamp("2020-01-21", tz="UTC"),
            pd.Timestamp("2020-02-02", tz="UTC"),
        ],
        index=ev,
    )
    split = WalkForwardSplitConfig(
        test_start=pd.Timestamp("2020-02-01", tz="UTC"),
        test_end=pd.Timestamp("2020-02-10", tz="UTC"),
        train_start=pd.Timestamp("2020-01-01", tz="UTC"),
    )
    embargo = pd.Timedelta(days=5)
    tr, te, tags = build_walkforward_masks(ev, t1, split, embargo_delta=embargo)
    assert any("walkforward_embargo" in t for t in tags)
    # 01-10 因 t1 越 deadline 不得进训; 01-01/01-20 可进训
    assert not bool(tr[1])
    assert bool(tr[0]) and bool(tr[2])
    assert bool(te[3])
    assert_walkforward_split_invariants(
        ev, t1, tr, te, split, embargo_delta=embargo,
    )


def test_combined_sample_weights_matches_pieces():
    from crypto_alpha.labeling.sample_weights import (
        combined_sample_weights,
        sample_weights_by_return,
        time_decay_weights,
    )

    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    bars = pd.date_range("2020-01-01", periods=20, freq="h", tz="UTC")
    events = pd.DataFrame(
        {
            "t1": idx + pd.Timedelta(hours=6),
            "ret": [0.01, -0.02, 0.03, -0.01, 0.02],
        },
        index=idx,
    )
    got = combined_sample_weights(events, bars)
    w_u = sample_weights_by_return(events, bars).reindex(events.index).fillna(1.0)
    w_d = time_decay_weights(events).reindex(events.index).fillna(1.0)
    exp = (w_u * w_d)
    exp = exp / (exp.mean() + 1e-12)
    np.testing.assert_allclose(got.to_numpy(), exp.to_numpy(), rtol=1e-12)


def test_recompute_sample_weight_train_local_decay():
    """重算后训练集最旧衰减应为 last_weight=0.5, 最新为 1。"""
    from crypto_alpha.labeling.sample_weights import (
        combined_sample_weights,
        time_decay_weights,
    )

    idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    bars = pd.date_range("2020-01-01", periods=48, freq="h", tz="UTC")
    events = pd.DataFrame(
        {"t1": idx + pd.Timedelta(hours=3), "ret": [0.01, 0.02, -0.01, 0.03]},
        index=idx,
    )
    # 人为混入更早「全量」事件只影响全量 decay, 不影响子集重算
    full_idx = pd.DatetimeIndex(["2019-01-01", *idx.astype(str)], tz="UTC")
    full = pd.DataFrame(
        {
            "t1": [pd.Timestamp("2019-01-01", tz="UTC") + pd.Timedelta(hours=3)]
            + list(events["t1"]),
            "ret": [0.01] + list(events["ret"]),
        },
        index=full_idx,
    )
    decay_full_slice = time_decay_weights(full).reindex(idx)
    decay_re = time_decay_weights(events)
    assert float(decay_re.iloc[0]) == pytest.approx(0.5)
    assert float(decay_re.iloc[-1]) == pytest.approx(1.0)
    assert float(decay_full_slice.iloc[0]) > 0.5  # 全量里 2020 不是最旧
    sw = combined_sample_weights(events, bars)
    assert len(sw) == 4
    assert abs(float(sw.mean()) - 1.0) < 1e-9


def test_run_all_require_walkforward_fails_when_disabled(monkeypatch):
    """发布硬前置: require_in_run_all 且未跑 WF → 训练前立刻 RuntimeError。"""
    import copy

    from crypto_alpha.pipeline import report as report_mod

    monkeypatch.setattr(
        report_mod, "probe_experts", lambda cfg, requested: (["gbdt"], {}),
    )
    cfg = Config.load()
    cfg.raw = copy.deepcopy(cfg.raw)
    cfg.raw.setdefault("validation", {}).setdefault("walkforward", {})
    cfg.raw["validation"]["walkforward"]["require_in_run_all"] = True
    cfg.raw["validation"]["walkforward"]["enabled_in_run_all"] = False
    with pytest.raises(RuntimeError, match="require_in_run_all"):
        report_mod.run_all(
            cfg, ["BTC/USDT"], ["gbdt"], do_cpcv=False, do_walkforward=False,
        )

def test_walkforward_public_summary_strips_private():
    raw = {
        "symbol": "BTC/USDT",
        "win_rate": 0.5,
        "n_opened_trades": 3,
        "_traded_detail": "DROP_ME",
        "_equity": "DROP_ME",
        "gate_diagnostics": {"gates": {"n_opened_size_gt_0": 3}},
        "test_report": {"auc": 0.55, "brier": 0.2},
        "degradations": ["x"],
        "note": "n",
        "mode": "walk_forward_train_then_test",
        "backtest_start": "2022-09-14",
        "backtest_end": None,
        "embargo_bars": 0,
        "prob_threshold_effective": 0.7,
        "n_train_events": 10,
        "n_test_events": 5,
        "n_wins": 2,
        "n_losses": 1,
        "total_return": 0.1,
        "max_drawdown": 0.05,
        "final_capital": 11000.0,
    }
    assert isinstance(walkforward_section(Config.load()), dict)
    pub = walkforward_public_summary(raw)
    assert "_traded_detail" not in pub
    assert "_equity" not in pub
    slim = slim_walkforward_for_dashboard(raw)
    assert slim["evaluation_unit"] == "walk_forward"
    assert slim["n_opened_trades"] == 3
    assert slim["test_auc"] == 0.55


def test_dashboard_mentions_walkforward_when_present():
    from crypto_alpha.pipeline.report import RESEARCH_DISCLAIMERS, build_dashboard

    assert any("Walk-forward" in x or "walk-forward" in x for x in RESEARCH_DISCLAIMERS)
    cfg = Config.load()
    results = {
        "meta": {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "experts_requested": ["gbdt"],
            "experts_run": ["gbdt"],
            "experts_skipped": {},
            "seed": 42,
            "data_mode": "合成",
            "news_mode": "关",
            "do_cpcv": False,
            "do_walkforward": True,
            "research_disclaimers": list(RESEARCH_DISCLAIMERS),
        },
        "symbols": {
            "BTC/USDT": {
                "n_events": 10,
                "pos_rate": 0.5,
                "date_start": "2020",
                "date_end": "2021",
                "data_source": "synthetic",
                "ensemble_report": {"auc": 0.5, "brier": 0.25, "accuracy": 0.5, "n": 10},
                "expert_reports": {},
                "backtest": {
                    "sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                    "calmar": 0.0, "win_rate": 0.0, "n_trades": 0,
                },
                "decision": {
                    "signal": "HOLD", "win_probability": None,
                    "suggested_position_pct": 0.0, "entry_price": 1.0,
                    "stop_loss": None, "take_profit": None,
                },
                "equity_curve": [],
                "equity_curve_kind": "realized",
                "equity_b64": "",
                "degradations": [],
                "walkforward": {
                    "ok": True,
                    "evaluation_unit": "walk_forward",
                    "n_opened_trades": 2,
                    "win_rate": 0.5,
                    "total_return": 0.01,
                    "max_drawdown": 0.02,
                    "test_auc": 0.51,
                    "prob_threshold_effective": 0.6,
                    "n_train_events": 100,
                    "n_test_events": 40,
                    "backtest_start": "2022-09-14",
                    "backtest_end": None,
                    "embargo_bars": 0,
                },
            },
        },
    }
    html = build_dashboard(results, cfg)
    assert "Walk-forward" in html
    assert "真外推基线" in html
    assert "WF开仓数" in html or "WF" in html
