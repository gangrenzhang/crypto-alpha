"""BTC 时间外推回测: 训练窗 → 固定回测窗(与上次约 3.8 年对齐)。

切分(与上次主路径回测开仓覆盖对齐):
  - 回测窗: [2022-09-14, 2026-07-18] (上次成交首笔~面板末端, ≈3.8 年)
  - 训练窗: 数据起点(期望 2015; Binance 现货实际约 2017) → 回测窗起点
            且要求标签 t1 在回测起点之前结束(防标签泄漏)

流程: 仅在训练事件上 fit 集成+校准/保形 → 在回测事件上 predict → backtest_events。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from crypto_alpha.backtest import backtest_events
from crypto_alpha.calibration.calibrate import (
    classification_report_probs,
    fit_deploy_calibrator_and_conformal,
)
from crypto_alpha.config import Config, set_global_seed
from crypto_alpha.pipeline import prepare_dataset
from crypto_alpha.pipeline.run import build_experts
from crypto_alpha.ensemble import StackingEnsemble

SYMBOL = "BTC/USDT"
INITIAL_CAPITAL = 10_000.0

# 与上次 btc_backtest_trades 开仓覆盖对齐的回测窗
BACKTEST_START = pd.Timestamp("2022-09-14 00:00:00+00:00")
BACKTEST_END = pd.Timestamp("2026-07-18 08:00:00+00:00")


def _gate_diagnostics(
    raw_te: np.ndarray,
    prob_te: np.ndarray,
    confident: np.ndarray,
    detail: pd.DataFrame,
    thr: float,
    conf_obj,
) -> dict:
    """测试窗开仓门控诊断: 阈值 / 保形 / 交集 / 概率台阶。"""
    raw_te = np.asarray(raw_te, dtype=float)
    prob_te = np.asarray(prob_te, dtype=float)
    confident = np.asarray(confident, dtype=bool)
    n = int(len(prob_te))
    pass_thr = prob_te >= thr
    pass_conf = confident
    pass_both = pass_thr & pass_conf
    sizes = (
        detail["size"].to_numpy(dtype=float)
        if "size" in detail.columns and len(detail) == n
        else None
    )
    opened = (sizes > 0) if sizes is not None else pass_both

    # 校准后概率台阶(解释为何大量刚好过线)
    rounded = np.round(prob_te, 6)
    uniq, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(-counts)
    top_levels = [
        {"prob": float(uniq[i]), "n": int(counts[i]), "frac": float(counts[i] / max(n, 1))}
        for i in order[:12]
    ]

    qhat = float(getattr(conf_obj, "qhat_", float("nan")))
    return {
        "prob_threshold": float(thr),
        "n_test_events": n,
        "raw_proba": {
            "mean": float(np.nanmean(raw_te)),
            "std": float(np.nanstd(raw_te)),
            "min": float(np.nanmin(raw_te)),
            "max": float(np.nanmax(raw_te)),
            "frac_ge_threshold": float(np.mean(raw_te >= thr)),
        },
        "calibrated_proba": {
            "mean": float(np.nanmean(prob_te)),
            "std": float(np.nanstd(prob_te)),
            "min": float(np.nanmin(prob_te)),
            "max": float(np.nanmax(prob_te)),
            "n_unique": int(len(uniq)),
            "top_levels": top_levels,
        },
        "gates": {
            "n_prob_ge_threshold": int(pass_thr.sum()),
            "frac_prob_ge_threshold": float(pass_thr.mean()),
            "n_confident": int(pass_conf.sum()),
            "frac_confident": float(pass_conf.mean()),
            "n_prob_and_confident": int(pass_both.sum()),
            "frac_prob_and_confident": float(pass_both.mean()),
            "n_opened_size_gt_0": int(np.asarray(opened).sum()),
            "frac_opened_size_gt_0": float(np.mean(opened)),
            "n_pass_gate_but_size_0": int((pass_both & ~np.asarray(opened)).sum())
            if sizes is not None
            else None,
        },
        "conformal_qhat": qhat,
        "note": (
            "frac_prob_and_confident ≈ 理论可开仓上限(未计资金占用); "
            "若与 n_opened 接近且 n_unique 很小, 说明 isotonic 台阶堆在阈值上方且保形几乎不弃权。"
        ),
    }


def _print_gate_diagnostics(g: dict) -> None:
    gates = g["gates"]
    cal = g["calibrated_proba"]
    print("\n========== 开仓门控诊断(测试窗) ==========", flush=True)
    print(f"阈值 thr:              {g['prob_threshold']:.2f}", flush=True)
    print(f"测试事件 n:            {g['n_test_events']}", flush=True)
    print(
        f"prob≥thr:              {gates['n_prob_ge_threshold']} "
        f"({gates['frac_prob_ge_threshold']:.2%})",
        flush=True,
    )
    print(
        f"confident:             {gates['n_confident']} "
        f"({gates['frac_confident']:.2%})",
        flush=True,
    )
    print(
        f"prob≥thr 且 confident: {gates['n_prob_and_confident']} "
        f"({gates['frac_prob_and_confident']:.2%})",
        flush=True,
    )
    print(
        f"实际开仓 size>0:       {gates['n_opened_size_gt_0']} "
        f"({gates['frac_opened_size_gt_0']:.2%})",
        flush=True,
    )
    if gates.get("n_pass_gate_but_size_0") is not None:
        print(f"过门但 size=0:         {gates['n_pass_gate_but_size_0']}", flush=True)
    print(f"保形 qhat:             {g['conformal_qhat']:.6f}", flush=True)
    print(
        f"校准后概率: mean={cal['mean']:.4f} std={cal['std']:.4f} "
        f"min={cal['min']:.4f} max={cal['max']:.4f} n_unique={cal['n_unique']}",
        flush=True,
    )
    print("校准后概率 Top 台阶:", flush=True)
    for lv in cal["top_levels"][:8]:
        print(f"  p={lv['prob']:.6f}  n={lv['n']}  ({lv['frac']:.2%})", flush=True)
    raw = g["raw_proba"]
    print(
        f"原始融合概率: mean={raw['mean']:.4f} std={raw['std']:.4f} "
        f"min={raw['min']:.4f} max={raw['max']:.4f} "
        f"frac≥thr={raw['frac_ge_threshold']:.2%}",
        flush=True,
    )


def main() -> None:
    cfg = Config.load()
    cfg.raw["data"]["symbols"] = [SYMBOL]
    cfg.raw["data"]["since"] = "2015-01-01T00:00:00Z"
    cfg.raw["data"]["refresh_before_decide"] = False
    cfg.raw["data"]["incremental_update"] = False
    cfg.raw["news"]["as_feature"] = False
    cfg.raw["news"]["auto_build_panel"] = False
    cfg.raw["news"]["use_history"] = False
    cfg.raw["news"]["use_synthetic"] = False

    print(f"===== {SYMBOL} walk-forward =====", flush=True)
    print(f"train: since→{BACKTEST_START} (t1 亦须 < 起点)", flush=True)
    print(f"test : [{BACKTEST_START}, {BACKTEST_END}]", flush=True)

    print("[step] prepare_dataset ...", flush=True)
    ds = prepare_dataset(cfg, SYMBOL)
    panel = ds.panel
    events = ds.events
    print(
        f"[data] source={ds.data_source} bars={len(panel)} "
        f"range={panel.index.min()} -> {panel.index.max()} events={len(events)}",
        flush=True,
    )

    t1 = pd.DatetimeIndex(pd.to_datetime(ds.t1, utc=True))
    ev_idx = pd.DatetimeIndex(pd.to_datetime(events.index, utc=True))

    train_mask = np.asarray((ev_idx < BACKTEST_START) & (t1 < BACKTEST_START))
    test_mask = np.asarray((ev_idx >= BACKTEST_START) & (ev_idx <= BACKTEST_END))
    train_index = events.index[train_mask]
    test_index = events.index[test_mask]

    n_train, n_test = int(len(train_index)), int(len(test_index))
    print(f"[split] train_events={n_train} test_events={n_test}", flush=True)
    if n_train < 200:
        raise SystemExit(f"训练事件过少 ({n_train}), 请先回填 2015/2017 起的行情缓存")
    if n_test < 50:
        raise SystemExit(f"回测事件过少 ({n_test})")

    X_tr, y_tr = ds.X.loc[train_index], ds.y[train_mask]
    t1_tr = ds.t1.loc[train_index]
    w_tr = None if ds.sample_weight is None else np.asarray(ds.sample_weight)[train_mask]
    X_te = ds.X.loc[test_index]
    y_te = ds.y[test_mask]
    events_te = events.loc[test_index]

    print("[step] fit ensemble on train ...", flush=True)
    set_global_seed(cfg.seed)
    experts = build_experts(cfg, ds)
    # 专家 panel 用全量(测试事件回看窗需要训练期历史); 拟合样本仅训练事件
    ens = StackingEnsemble(experts, cfg["ensemble"], seed=cfg.seed)
    vcfg = cfg["validation"]
    ens.fit(
        X_tr, y_tr, t1_tr, sample_weight=w_tr,
        n_splits=int(vcfg["n_splits"]), embargo_pct=float(vcfg["embargo_pct"]),
    )

    oof = ens.oof_proba()
    oof_mask = ~np.isnan(oof)
    ccfg = cfg["calibration"]
    cal, conf, deploy_tags = fit_deploy_calibrator_and_conformal(
        oof, y_tr, method=ccfg["method"],
        alpha=float(ccfg["conformal_alpha"]),
        conformal_frac=float(ccfg.get("conformal_frac", 0.3)),
    )
    print(f"[calibration] deploy_tags={deploy_tags}", flush=True)

    print("[step] predict + backtest on test window ...", flush=True)
    raw_te = ens.predict_proba(X_te)
    prob_te = np.asarray(cal.transform(raw_te), dtype=float)
    conf_df = conf.predict_set(prob_te)
    confident = np.asarray(conf_df["confident"], dtype=bool)

    payoff = float(cfg["labeling"]["pt_sl"][0]) / float(cfg["labeling"]["pt_sl"][1])
    prices = panel["close"] if "close" in panel.columns else None
    bt = backtest_events(
        events_te, prob_te, cfg["backtest"], cfg["risk"],
        payoff=payoff, prices=prices, confident=confident,
    )

    detail: pd.DataFrame = bt["detail"].copy()
    equity: pd.Series = bt["equity"]
    traded = detail[detail["size"] > 0].copy() if "size" in detail.columns else detail.iloc[0:0]
    wins = int((traded["pnl"] > 0).sum()) if len(traded) else 0
    losses = int((traded["pnl"] <= 0).sum()) if len(traded) else 0
    win_rate = float(wins / len(traded)) if len(traded) else 0.0
    final_mult = float(equity.iloc[-1]) if len(equity) else 1.0

    report_te = classification_report_probs(prob_te, y_te)
    report_tr = classification_report_probs(oof[oof_mask], y_tr[oof_mask]) if oof_mask.any() else {}

    out_dir = Path(cfg.artifacts_dir)
    trades_path = out_dir / "btc_walkforward_trades.csv"
    summary_path = out_dir / "btc_walkforward_summary.json"

    traded_out = traded.copy()
    traded_out.index.name = "entry_time"
    traded_out.to_csv(trades_path)

    summary = {
        "symbol": SYMBOL,
        "mode": "walk_forward_train_then_test",
        "data_source": ds.data_source,
        "panel_bars": int(len(panel)),
        "panel_start": str(panel.index.min()),
        "panel_end": str(panel.index.max()),
        "train_start": str(panel.index.min()),
        "train_end_exclusive": str(BACKTEST_START),
        "backtest_start": str(BACKTEST_START),
        "backtest_end": str(BACKTEST_END),
        "note": (
            "训练: 事件 t0 与 t1 均 < 回测起点; 回测窗与上次开仓覆盖对齐 "
            "(2022-09-14→2026-07-18)。Binance 现货 Vision 通常无 2015–2017 中前段, "
            "实际训练起点以 panel_start 为准。"
        ),
        "n_train_events": n_train,
        "n_test_events": n_test,
        "n_opened_trades": int(len(traded)),
        "n_wins": wins,
        "n_losses": losses,
        "win_rate": win_rate,
        "total_return": float(bt["metrics"].get("total_return", final_mult - 1.0)),
        "max_drawdown": float(bt["metrics"].get("max_drawdown", 0.0)),
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": INITIAL_CAPITAL * final_mult,
        "train_oof_report": report_tr,
        "test_report": report_te,
        "backtest_metrics": bt["metrics"],
        "dropped_experts": list(ens.dropped_experts or []),
        "degradations": list(ds.degradations) + list(deploy_tags or []) + list(ens.degradations or []),
        "trades_csv": str(trades_path),
        "compare_to_previous_oof_half_window": {
            "previous_win_rate": 0.64,
            "previous_n_trades": 50,
            "previous_final_capital": 10186.62,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== 结果 ==========", flush=True)
    print(f"实际数据起点:     {panel.index.min()}", flush=True)
    print(f"训练事件 / 回测事件: {n_train} / {n_test}", flush=True)
    print(f"实际开仓笔数:     {len(traded)}", flush=True)
    print(f"胜 / 负:          {wins} / {losses}", flush=True)
    print(f"开仓胜率:         {win_rate:.2%}", flush=True)
    print(f"累计收益:         {summary['total_return']:.2%}", flush=True)
    print(f"1万→终值:         {summary['final_capital']:,.2f}", flush=True)
    print(f"[ok] {trades_path}", flush=True)
    print(f"[ok] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
