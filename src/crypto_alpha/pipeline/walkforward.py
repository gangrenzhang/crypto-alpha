"""Walk-forward(真外推)基线评估。

与 ``train_and_validate`` 的 OOF / ``backtest_deploy`` 不同:
  - 训练窗: 仅用 ``t0 < test_start`` **且** ``t1 < test_start - embargo`` 的事件拟合;
    可选 ``train_start``: 额外要求 ``t0 >= train_start``(截断更早**事件**,
    不截断 OHLCV/特征回看所用的更早 K 线)
  - 阈值: 仅在**全部**训练窗有限 OOF 上按部署同形冻结 ``prob_threshold_effective``
    (单切 holdout 无「报告半窗」; 与联跑内 deploy 半窗参考不同, 但是对测试窗仍无泄漏)
  - 测试窗: 训练窗拟合后的 ``predict → deploy cal/conf → thr_eff`` 回测

**口径诚实**: 本模块是 **single-cut holdout**(一次切分: 过去训→未来测),
**不是**滚动再训练的多折 walk-forward。命名沿用仓库既有脚本;
上线前若需滚动曲线, 须另扩多锚点重训(见 ARCHITECTURE §21)。

禁止用测试窗刷阈值; 禁止把本模块结果与研究 OOF 成交数直接对比。

切分不变量(``assert_walkforward_split_invariants``):
  - train ∩ test = ∅
  - 任一训练事件的标签区间终点 t1 不得进入 label_deadline(含禁运)
  - 任一训练事件的入场 t0 不得落入测试窗
  - 若设 train_start: 任一训练事件的 t0 不得早于 train_start
  - 落在「t0 在测试起点前但 t1 越过 deadline」的事件两边都不进(净化丢弃);
    另: t0 < train_start 的合格训练候选被丢弃(计入 dropped_pre_train), 亦不进测试
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import backtest_events
from ..calibration.calibrate import (
    classification_report_probs,
    fit_deploy_calibrator_and_conformal,
)
from ..config import Config, set_global_seed
from ..diagnostics.gates import (
    assess_calibration_pass_health,
    freeze_threshold_on_reference,
    gate_diagnostics,
)
from ..ensemble import StackingEnsemble
from ..labeling.sample_weights import combined_sample_weights
from .run import Dataset, build_experts, prepare_dataset


@dataclass(frozen=True)
class WalkForwardSplitConfig:
    """WF 切分与门槛(来自 ``validation.walkforward`` + 调用方覆盖)。"""

    test_start: pd.Timestamp
    test_end: pd.Timestamp | None
    train_start: pd.Timestamp | None = None  # None=不截断训练起点(用面板最早事件)
    embargo_bars: int = 0
    min_train_events: int = 200
    min_test_events: int = 50
    initial_capital: float = 10_000.0


def _as_utc_ts(raw: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def walkforward_section(cfg: Config) -> dict:
    """读取 ``validation.walkforward``; 缺省为空 dict。"""
    try:
        v = cfg.get("validation")
    except Exception:
        return {}
    if not isinstance(v, dict):
        return {}
    sec = v.get("walkforward") or {}
    return dict(sec) if isinstance(sec, dict) else {}


def resolve_walkforward_split(
    cfg: Config,
    *,
    test_start: str | pd.Timestamp | None = None,
    test_end: str | pd.Timestamp | None = None,
    train_start: str | pd.Timestamp | None = None,
) -> WalkForwardSplitConfig:
    """解析 WF 切分配置; 调用方可覆盖起止时刻。"""
    sec = walkforward_section(cfg)
    start_raw = test_start if test_start is not None else sec.get("test_start")
    if start_raw is None or (isinstance(start_raw, str) and not str(start_raw).strip()):
        raise ValueError(
            "walk-forward 需要 test_start"
            "（config validation.walkforward.test_start 或函数参数）"
        )
    end_raw = test_end if test_end is not None else sec.get("test_end")
    train_raw = train_start if train_start is not None else sec.get("train_start")

    start = _as_utc_ts(start_raw)

    end: pd.Timestamp | None
    if end_raw is None or (isinstance(end_raw, str) and not str(end_raw).strip()):
        end = None
    else:
        end = _as_utc_ts(end_raw)
        if end < start:
            raise ValueError(f"walk-forward test_end ({end}) < test_start ({start})")

    train_s: pd.Timestamp | None
    if train_raw is None or (isinstance(train_raw, str) and not str(train_raw).strip()):
        train_s = None
    else:
        train_s = _as_utc_ts(train_raw)
        if train_s >= start:
            raise ValueError(
                f"walk-forward train_start ({train_s}) 必须早于 test_start ({start})"
            )

    return WalkForwardSplitConfig(
        test_start=start,
        test_end=end,
        train_start=train_s,
        embargo_bars=max(int(sec.get("embargo_bars", 0) or 0), 0),
        min_train_events=max(int(sec.get("min_train_events", 200) or 200), 1),
        min_test_events=max(int(sec.get("min_test_events", 50) or 50), 1),
        initial_capital=float(sec.get("initial_capital", 10_000.0) or 10_000.0),
    )


def _embargo_delta(cfg: Config, embargo_bars: int) -> pd.Timedelta:
    """主周期 × embargo_bars → 墙钟禁运长度; 0 → 0。"""
    if embargo_bars <= 0:
        return pd.Timedelta(0)
    from ..data.fetch import timeframe_delta

    tf = str(cfg["data"].get("timeframe") or "30m")
    return timeframe_delta(tf) * int(embargo_bars)


def build_walkforward_masks(
    event_index: pd.DatetimeIndex,
    t1: pd.Series,
    split: WalkForwardSplitConfig,
    *,
    embargo_delta: pd.Timedelta | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """构造 train/test 布尔掩码(与 ``events`` / ``y`` 行对齐)。

    训练:
      ``t0 < test_start`` 且 ``t1 < train_label_deadline``
      其中 ``train_label_deadline = test_start - embargo_delta``;
      若设 ``train_start``: 额外要求 ``t0 >= train_start``(丢掉更早事件, OHLCV 特征仍可用更早 K 线)。
    测试:
      ``test_start <= t0`` 且 (``test_end is None`` 或 ``t0 <= test_end``)

    注意: 测试事件的 t1 可以越过 ``test_end``(持仓可跨出回测展示窗);
    泄漏防护看的是训练标签不得伸进测试起点。
    """
    tags: list[str] = []
    ev = pd.DatetimeIndex(pd.to_datetime(event_index, utc=True))
    # 只用位置对齐的值, 避免 Series 索引与 event_index 顺序不一致时静默错切
    t1_vals = pd.to_datetime(np.asarray(t1), utc=True)
    if len(ev) != len(t1_vals):
        raise ValueError(
            f"event_index 与 t1 长度不一致: {len(ev)} vs {len(t1_vals)}"
        )
    if pd.isna(t1_vals).any():
        n_nat = int(pd.isna(t1_vals).sum())
        raise ValueError(
            f"walk-forward: t1 含 {n_nat} 个 NaT, 拒绝静默丢弃(请先检查标注完整性)"
        )

    delta = embargo_delta if embargo_delta is not None else pd.Timedelta(0)
    if delta < pd.Timedelta(0):
        raise ValueError("embargo_delta 不得为负")
    label_deadline = split.test_start - delta
    if delta > pd.Timedelta(0):
        tags.append(
            f"walkforward_embargo(delta={delta},label_deadline={label_deadline})"
        )

    train_mask = np.asarray(
        (ev < split.test_start) & (t1_vals < label_deadline), dtype=bool,
    )
    if split.train_start is not None:
        eligible = (ev < split.test_start) & (t1_vals < label_deadline)
        dropped = int((eligible & (ev < split.train_start)).sum())
        train_mask = np.asarray(train_mask & (ev >= split.train_start), dtype=bool)
        tags.append(
            f"walkforward_train_start={split.train_start.isoformat()}"
            f"(dropped_pre_train={dropped})"
        )
    if split.test_end is None:
        test_mask = np.asarray(ev >= split.test_start, dtype=bool)
        tags.append("walkforward_test_end=panel_tail")
    else:
        test_mask = np.asarray(
            (ev >= split.test_start) & (ev <= split.test_end), dtype=bool,
        )

    # 净化带: t0 在测试起点前、但 t1 越过 deadline → 两边都不进
    purged = np.asarray(
        (ev < split.test_start) & ~(t1_vals < label_deadline), dtype=bool,
    )
    n_purged = int(purged.sum())
    if n_purged:
        tags.append(f"walkforward_purged_label_overlap(n={n_purged})")

    return train_mask, test_mask, tags


def assert_walkforward_split_invariants(
    event_index: pd.DatetimeIndex,
    t1: pd.Series,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    split: WalkForwardSplitConfig,
    *,
    embargo_delta: pd.Timedelta | None = None,
) -> None:
    """Fail-fast: 切分不得有标签泄漏或集合重叠。"""
    train_m = np.asarray(train_mask, dtype=bool)
    test_m = np.asarray(test_mask, dtype=bool)
    if train_m.shape != test_m.shape:
        raise AssertionError("train_mask/test_mask shape mismatch")
    if train_m.shape[0] != len(event_index):
        raise AssertionError("mask 长度与事件数不一致")
    if np.any(train_m & test_m):
        raise AssertionError("walk-forward: train ∩ test 非空")

    ev = pd.DatetimeIndex(pd.to_datetime(event_index, utc=True))
    t1v = pd.to_datetime(np.asarray(t1), utc=True)
    delta = embargo_delta if embargo_delta is not None else pd.Timedelta(0)
    label_deadline = split.test_start - delta

    if np.any(train_m):
        if np.any(ev[train_m] >= split.test_start):
            raise AssertionError("walk-forward: 训练事件 t0 落入测试窗")
        if split.train_start is not None and np.any(ev[train_m] < split.train_start):
            raise AssertionError("walk-forward: 训练事件 t0 早于 train_start")
        if np.any(t1v[train_m] >= label_deadline):
            raise AssertionError(
                "walk-forward: 训练事件 t1 未在 label_deadline 之前结束(标签泄漏)"
            )
    if np.any(test_m):
        if np.any(ev[test_m] < split.test_start):
            raise AssertionError("walk-forward: 测试事件 t0 早于 test_start")
        if split.test_end is not None and np.any(ev[test_m] > split.test_end):
            raise AssertionError("walk-forward: 测试事件 t0 晚于 test_end")


def _jsonable(obj: Any) -> Any:
    """summary 落盘用: numpy/pandas → 原生 Python。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def run_walkforward(
    cfg: Config,
    symbol: str,
    *,
    ds: Dataset | None = None,
    test_start: str | pd.Timestamp | None = None,
    test_end: str | pd.Timestamp | None = None,
    train_start: str | pd.Timestamp | None = None,
    recompute_sample_weight: bool | None = None,
    prepare_kwargs: dict | None = None,
) -> dict:
    """对单币种跑部署同形 walk-forward, 返回可 JSON 序列化的 summary。

    Parameters
    ----------
    ds :
        可选预构建 Dataset(须与 symbol 一致)。传入则可避免重复 ``prepare_dataset``。
        **不得**传入已按决策 tip 刷过的面板充当研究 WF(应用冷缓存研究口径)。
    train_start :
        可选训练起点(含); 仅丢掉更早的**事件**, 不截断 OHLCV 特征回看。
    recompute_sample_weight :
        若 True: 在训练掩码确定后, 仅用训练事件重算 sample_weight
        (uniqueness×|ret|×time_decay, 与 prepare_dataset 同公式)。
        None: 读 ``validation.walkforward.recompute_sample_weight_on_split``(默认 false,
        保持历史「全量算权再切片」行为, 不影响未开开关的路径)。
    """
    split = resolve_walkforward_split(
        cfg, test_start=test_start, test_end=test_end, train_start=train_start,
    )
    embargo = _embargo_delta(cfg, split.embargo_bars)
    wf_sec = walkforward_section(cfg)
    if recompute_sample_weight is None:
        recompute_sample_weight = bool(
            wf_sec.get("recompute_sample_weight_on_split", False)
        )

    if ds is None:
        kw = dict(prepare_kwargs or {})
        # WF 基线禁止误触决策 tip REST
        kw.setdefault("for_decide", False)
        ds = prepare_dataset(cfg, symbol, **kw)
    elif ds.symbol != symbol:
        raise ValueError(f"Dataset.symbol={ds.symbol!r} 与请求 symbol={symbol!r} 不一致")

    panel = ds.panel
    events = ds.events
    if len(events) == 0:
        raise ValueError(f"{symbol}: 无事件, 无法 walk-forward")

    t1 = ds.t1.reindex(events.index)
    if t1.isna().any():
        raise ValueError(
            f"{symbol}: ds.t1 与 events 对齐后含 NaT "
            f"(n={int(t1.isna().sum())}); 拒绝静默丢弃"
        )
    train_mask, test_mask, split_tags = build_walkforward_masks(
        events.index, t1, split, embargo_delta=embargo,
    )
    assert_walkforward_split_invariants(
        events.index, t1, train_mask, test_mask, split, embargo_delta=embargo,
    )

    train_index = events.index[train_mask]
    test_index = events.index[test_mask]
    n_train, n_test = int(len(train_index)), int(len(test_index))
    if n_train < split.min_train_events:
        raise ValueError(
            f"{symbol}: WF 训练事件过少 ({n_train} < {split.min_train_events})"
        )
    if n_test < split.min_test_events:
        raise ValueError(
            f"{symbol}: WF 测试事件过少 ({n_test} < {split.min_test_events})"
        )

    X_tr = ds.X.loc[train_index]
    y_tr = np.asarray(ds.y)[train_mask]
    t1_tr = ds.t1.loc[train_index]
    if recompute_sample_weight:
        # 仅用训练子集重算权: 并发/衰减参照系 = 实际参与 fit 的事件
        # (默认关: 仍用 prepare_dataset 全量权再切片, 行为与历史一致)
        ev_tr = events.loc[train_index]
        for col in ("t1", "ret"):
            if col not in ev_tr.columns:
                raise ValueError(
                    f"{symbol}: recompute_sample_weight 需要 events 含 {col!r}"
                )
        w_tr = np.asarray(
            combined_sample_weights(ev_tr, panel.index).to_numpy(dtype=float),
            dtype=float,
        )
        if len(w_tr) != n_train:
            raise AssertionError("重算 sample_weight 长度与训练事件数不一致")
        split_tags.append(f"walkforward_sample_weight_recomputed(n={n_train})")
    else:
        w_tr = None if ds.sample_weight is None else np.asarray(ds.sample_weight)[train_mask]
    X_te = ds.X.loc[test_index]
    y_te = np.asarray(ds.y)[test_mask]
    events_te = events.loc[test_index]

    set_global_seed(cfg.seed)
    experts = build_experts(cfg, ds)
    ens = StackingEnsemble(experts, cfg["ensemble"], seed=cfg.seed)
    vcfg = cfg["validation"]
    ccfg = cfg["calibration"]
    conf_margin = float(ccfg.get("conformal_min_margin", 0.0) or 0.0)
    ens.fit(
        X_tr, y_tr, t1_tr, sample_weight=w_tr,
        n_splits=int(vcfg["n_splits"]), embargo_pct=float(vcfg["embargo_pct"]),
    )

    oof = ens.oof_proba()
    oof_mask = ~np.isnan(oof)
    cal, conf, deploy_tags = fit_deploy_calibrator_and_conformal(
        oof, y_tr, method=ccfg["method"],
        alpha=float(ccfg["conformal_alpha"]),
        conformal_frac=float(ccfg.get("conformal_frac", 0.3)),
        min_margin=conf_margin,
    )

    # 与 train_and_validate 的 thr_eff 同形: 仅训练窗 OOF × deploy cal
    oof_raw_ref = np.asarray(oof[oof_mask], dtype=float)
    oof_cal_ref = np.asarray(cal.transform(oof_raw_ref), dtype=float)
    inflate_max = float(ccfg.get("pass_rate_inflate_max", 1.5) or 0.0)
    thr_eff, thr_tags = freeze_threshold_on_reference(
        cfg["backtest"], oof_raw_ref, oof_cal_ref,
        pass_rate_inflate_max=inflate_max, tag_prefix="deploy_",
    )
    bt_cfg = dict(cfg["backtest"])
    bt_cfg["prob_threshold"] = float(thr_eff)

    raw_te = np.asarray(ens.predict_proba(X_te), dtype=float)
    prob_te = np.asarray(cal.transform(raw_te), dtype=float)
    conf_df = conf.predict_set(prob_te)
    confident = np.asarray(conf_df["confident"], dtype=bool)

    # 测试窗只告警, 不改 thr
    health = assess_calibration_pass_health(
        raw_te, prob_te, thr_eff,
        pass_rate_inflate_max=inflate_max,
        min_unique_levels=int(ccfg.get("min_unique_levels", 20) or 0),
    )

    payoff = float(cfg["labeling"]["pt_sl"][0]) / float(cfg["labeling"]["pt_sl"][1])
    prices = panel["close"] if "close" in panel.columns else None
    bt = backtest_events(
        events_te, prob_te, bt_cfg, cfg["risk"],
        payoff=payoff, prices=prices, confident=confident,
    )
    detail: pd.DataFrame = bt["detail"]
    equity: pd.Series = bt["equity"]
    traded = (
        detail[detail["size"] > 0].copy()
        if "size" in detail.columns
        else detail.iloc[0:0]
    )
    wins = int((traded["pnl"] > 0).sum()) if len(traded) else 0
    losses = int((traded["pnl"] <= 0).sum()) if len(traded) else 0
    win_rate = float(wins / len(traded)) if len(traded) else 0.0
    final_mult = float(equity.iloc[-1]) if len(equity) else 1.0

    gate_diag = gate_diagnostics(
        events_te.index, raw_te, prob_te, confident, detail, thr_eff, conf_obj=conf,
    )
    gate_diag["path"] = "walk_forward_deploy"
    gate_diag["threshold_tags"] = list(thr_tags)
    gate_diag["health_tags"] = list(health)
    gate_diag["split_tags"] = list(split_tags)

    report_te = classification_report_probs(prob_te, y_te)
    report_tr = (
        classification_report_probs(oof[oof_mask], y_tr[oof_mask])
        if oof_mask.any()
        else {}
    )

    degradations: list[str] = []
    for t in (
        list(ds.degradations)
        + list(deploy_tags or [])
        + list(thr_tags)
        + list(health)
        + list(split_tags)
        + list(ens.degradations or [])
    ):
        if t not in degradations:
            degradations.append(t)

    summary = {
        "symbol": symbol,
        "mode": "walk_forward_train_then_test",
        "evaluation_unit": "walk_forward",
        "split_kind": "single_cut_holdout",
        "note": (
            "真外推(single-cut holdout, 非滚动再训练): 仅训练窗拟合; "
            "测试窗 predict+deploy 门控; 阈值在全部训练 OOF 冻结, 禁止测试窗刷 thr。"
            "勿与 research OOF / train_and_validate.backtest_deploy 成交数直接对比。"
        ),
        "data_source": ds.data_source,
        "panel_bars": int(len(panel)),
        "panel_start": str(panel.index.min()) if len(panel) else None,
        "panel_end": str(panel.index.max()) if len(panel) else None,
        "train_start": str(split.train_start) if split.train_start is not None else None,
        "train_end_exclusive": str(split.test_start),
        "recompute_sample_weight_on_split": bool(recompute_sample_weight),
        "label_deadline": str(split.test_start - embargo),
        "embargo_bars": int(split.embargo_bars),
        "embargo_delta": str(embargo),
        "backtest_start": str(split.test_start),
        "backtest_end": str(split.test_end) if split.test_end is not None else None,
        "prob_threshold_effective": float(thr_eff),
        "n_train_events": n_train,
        "n_test_events": n_test,
        "n_opened_trades": int(len(traded)),
        "n_wins": wins,
        "n_losses": losses,
        "win_rate": win_rate,
        "total_return": float(bt["metrics"].get("total_return", final_mult - 1.0)),
        "max_drawdown": float(bt["metrics"].get("max_drawdown", 0.0)),
        "initial_capital": float(split.initial_capital),
        "final_capital": float(split.initial_capital) * final_mult,
        "train_oof_report": report_tr,
        "test_report": report_te,
        "backtest_metrics": bt["metrics"],
        "gate_diagnostics": gate_diag,
        "dropped_experts": list(ens.dropped_experts or []),
        "degradations": degradations,
        "split_tags": list(split_tags),
        # 供脚本落盘成交明细(非 JSON 友好对象, 调用方取出后应 pop)
        "_traded_detail": traded,
        "_equity": equity,
    }
    return summary


def walkforward_public_summary(summary: dict) -> dict:
    """去掉内部大对象, 供 JSON / 看板。"""
    skip = {"_traded_detail", "_equity"}
    return {k: _jsonable(v) for k, v in summary.items() if k not in skip}


def slim_walkforward_for_dashboard(summary: dict) -> dict:
    """看板用精简字段(真外推基线 KPI)。"""
    g = summary.get("gate_diagnostics") or {}
    gates = g.get("gates") or {}
    return {
        "evaluation_unit": "walk_forward",
        "split_kind": summary.get("split_kind", "single_cut_holdout"),
        "mode": summary.get("mode"),
        "backtest_start": summary.get("backtest_start"),
        "backtest_end": summary.get("backtest_end"),
        "train_start": summary.get("train_start"),
        "embargo_bars": summary.get("embargo_bars"),
        "prob_threshold_effective": summary.get("prob_threshold_effective"),
        "n_train_events": summary.get("n_train_events"),
        "n_test_events": summary.get("n_test_events"),
        "n_opened_trades": summary.get("n_opened_trades"),
        "n_wins": summary.get("n_wins"),
        "n_losses": summary.get("n_losses"),
        "win_rate": summary.get("win_rate"),
        "total_return": summary.get("total_return"),
        "max_drawdown": summary.get("max_drawdown"),
        "final_capital": summary.get("final_capital"),
        "test_auc": (summary.get("test_report") or {}).get("auc"),
        "test_brier": (summary.get("test_report") or {}).get("brier"),
        "n_prob_and_confident": gates.get("n_prob_and_confident"),
        "n_opened_size_gt_0": gates.get("n_opened_size_gt_0"),
        "degradations": list(summary.get("degradations") or []),
        "note": summary.get("note"),
    }
