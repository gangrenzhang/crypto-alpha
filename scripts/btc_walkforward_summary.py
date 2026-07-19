"""BTC 时间外推回测: 训练窗 → 固定回测窗(与上次约 3.8 年对齐)。

切分:
  - 回测窗: [2022-09-14, 2026-07-18]
  - 训练窗: 数据起点 → 回测窗起点(且 t1 亦须 < 起点)

流程与部署同形: fit → fit_deploy(cal,conf) → predict → 有效阈值 → backtest。
门控诊断复用 crypto_alpha.diagnostics.gates。
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
from crypto_alpha.diagnostics.gates import (
    assess_calibration_pass_health,
    freeze_threshold_on_reference,
    gate_diagnostics,
)
from crypto_alpha.ensemble import StackingEnsemble
from crypto_alpha.pipeline import prepare_dataset
from crypto_alpha.pipeline.run import build_experts

SYMBOL = "BTC/USDT"
INITIAL_CAPITAL = 10_000.0
BACKTEST_START = pd.Timestamp("2022-09-14 00:00:00+00:00")
BACKTEST_END = pd.Timestamp("2026-07-18 08:00:00+00:00")


def _print_gate_diagnostics(g: dict) -> None:
    gates = g["gates"]
    cal = g["calibrated_proba"]
    print("\n========== 开仓门控诊断(测试窗) ==========", flush=True)
    print(f"有效阈值 thr:          {g['prob_threshold']:.4f}", flush=True)
    print(f"测试事件 n:            {g.get('n_events', g.get('n_test_events'))}", flush=True)
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
    print(f"过门但 size=0:         {gates.get('n_pass_gate_but_size_0')}", flush=True)
    print(
        f"保形 qhat/margin:      {g.get('conformal_qhat')} / {g.get('conformal_min_margin')}",
        flush=True,
    )
    print(
        f"校准后概率: mean={cal['mean']:.4f} n_unique={cal['n_unique']} "
        f"max={cal['max']:.4f}",
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
        raise SystemExit(f"训练事件过少 ({n_train})")
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
    print(f"[calibration] deploy_tags={deploy_tags}", flush=True)

    # 与 train_and_validate 的 prob_threshold_effective 同形: deploy cal × 训练窗原始 OOF
    oof_raw_ref = np.asarray(oof[oof_mask], dtype=float)
    oof_cal_ref = np.asarray(cal.transform(oof_raw_ref), dtype=float)
    inflate_max = float(ccfg.get("pass_rate_inflate_max", 1.5) or 0.0)
    thr_eff, thr_tags = freeze_threshold_on_reference(
        cfg["backtest"], oof_raw_ref, oof_cal_ref,
        pass_rate_inflate_max=inflate_max, tag_prefix="deploy_",
    )
    print(f"[threshold] effective(deploy)={thr_eff:.4f} tags={thr_tags}", flush=True)
    bt_cfg = dict(cfg["backtest"])
    bt_cfg["prob_threshold"] = float(thr_eff)

    print("[step] predict + backtest on test window ...", flush=True)
    raw_te = ens.predict_proba(X_te)
    prob_te = np.asarray(cal.transform(raw_te), dtype=float)
    conf_df = conf.predict_set(prob_te)
    confident = np.asarray(conf_df["confident"], dtype=bool)

    # 测试窗只告警, 不改 thr(阈值已在训练参考窗冻结)
    health = assess_calibration_pass_health(
        raw_te, prob_te, thr_eff,
        pass_rate_inflate_max=inflate_max,
        min_unique_levels=int(ccfg.get("min_unique_levels", 20) or 0),
    )
    for t in health:
        print(f"[calibration] WARN: {t}", flush=True)

    payoff = float(cfg["labeling"]["pt_sl"][0]) / float(cfg["labeling"]["pt_sl"][1])
    prices = panel["close"] if "close" in panel.columns else None
    bt = backtest_events(
        events_te, prob_te, bt_cfg, cfg["risk"],
        payoff=payoff, prices=prices, confident=confident,
    )

    detail: pd.DataFrame = bt["detail"].copy()
    equity: pd.Series = bt["equity"]
    traded = detail[detail["size"] > 0].copy() if "size" in detail.columns else detail.iloc[0:0]
    wins = int((traded["pnl"] > 0).sum()) if len(traded) else 0
    losses = int((traded["pnl"] <= 0).sum()) if len(traded) else 0
    win_rate = float(wins / len(traded)) if len(traded) else 0.0
    final_mult = float(equity.iloc[-1]) if len(equity) else 1.0

    gate_diag = gate_diagnostics(
        events_te.index, raw_te, prob_te, confident, detail, thr_eff, conf_obj=conf,
    )
    gate_diag["path"] = "walk_forward_deploy"
    gate_diag["threshold_tags"] = thr_tags
    gate_diag["health_tags"] = health

    report_te = classification_report_probs(prob_te, y_te)
    report_tr = classification_report_probs(oof[oof_mask], y_tr[oof_mask]) if oof_mask.any() else {}

    out_dir = Path(cfg.artifacts_dir)
    trades_path = out_dir / "btc_walkforward_trades.csv"
    summary_path = out_dir / "btc_walkforward_summary.json"
    gate_path = out_dir / "btc_walkforward_gate_diagnostics.json"

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
        "train_end_exclusive": str(BACKTEST_START),
        "backtest_start": str(BACKTEST_START),
        "backtest_end": str(BACKTEST_END),
        "prob_threshold_effective": float(thr_eff),
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
        "gate_diagnostics": gate_diag,
        "dropped_experts": list(ens.dropped_experts or []),
        "degradations": list(ds.degradations) + list(deploy_tags or [])
        + list(thr_tags) + list(health) + list(ens.degradations or []),
        "trades_csv": str(trades_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    gate_path.write_text(json.dumps(gate_diag, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== 结果 ==========", flush=True)
    print(f"实际数据起点:     {panel.index.min()}", flush=True)
    print(f"训练/回测事件:    {n_train} / {n_test}", flush=True)
    print(f"实际开仓笔数:     {len(traded)}", flush=True)
    print(f"胜 / 负:          {wins} / {losses}", flush=True)
    print(f"开仓胜率:         {win_rate:.2%}", flush=True)
    print(f"累计收益:         {summary['total_return']:.2%}", flush=True)
    print(f"1万→终值:         {summary['final_capital']:,.2f}", flush=True)
    _print_gate_diagnostics(gate_diag)
    print(f"[ok] {trades_path}", flush=True)
    print(f"[ok] {summary_path}", flush=True)
    print(f"[ok] {gate_path}", flush=True)


if __name__ == "__main__":
    main()
