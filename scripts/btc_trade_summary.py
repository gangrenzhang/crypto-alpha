"""仅 BTC: 训练 + 收集回测开仓事件, 汇总胜率与 1 万本金终值。"""
import _bootstrap  # noqa: F401

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_alpha.config import Config
from crypto_alpha.pipeline import prepare_dataset, train_and_validate


INITIAL_CAPITAL = 10_000.0
SYMBOL = "BTC/USDT"


def main() -> None:
    cfg = Config.load()
    cfg.raw["data"]["symbols"] = [SYMBOL]
    # 只要回测开仓明细, 不必刷 tip / 出当下决策 / 拉新闻
    cfg.raw["data"]["refresh_before_decide"] = False
    cfg.raw["news"]["as_feature"] = False
    cfg.raw["news"]["auto_build_panel"] = False
    cfg.raw["news"]["use_history"] = False
    cfg.raw["news"]["use_synthetic"] = False

    print(f"===== {SYMBOL} 训练 + 回测开仓汇总 =====", flush=True)
    print("[step] prepare_dataset ...", flush=True)
    ds = prepare_dataset(cfg, SYMBOL)
    print(
        f"[data] source={ds.data_source} bars={len(ds.panel)} "
        f"events={len(ds.events)} y_pos={float(np.mean(ds.y)):.3f}"
    )
    if len(ds.panel):
        print(f"[data] panel range: {ds.panel.index.min()} -> {ds.panel.index.max()}")

    print("[step] train_and_validate ...", flush=True)
    trained = train_and_validate(cfg, ds)
    bt = trained["backtest"]
    metrics = bt["metrics"]
    detail: pd.DataFrame = bt["detail"].copy()
    equity: pd.Series = bt["equity"]

    # 开仓事件: size > 0(真正占用仓位的成交)
    traded = detail[detail["size"] > 0].copy() if "size" in detail.columns else detail.iloc[0:0]
    wins = int((traded["pnl"] > 0).sum()) if len(traded) else 0
    losses = int((traded["pnl"] <= 0).sum()) if len(traded) else 0
    win_rate = float(wins / len(traded)) if len(traded) else 0.0

    final_mult = float(equity.iloc[-1]) if len(equity) else 1.0
    final_capital = INITIAL_CAPITAL * final_mult

    # 报告窗说明: 多专家时为事件时间序后半窗(非「前半 K 线训练、后半 K 线回测」)
    prune = getattr(trained["ensemble"], "prune_eval_mask_", None)
    n_report_events = int(len(detail))
    n_all_events = int(len(ds.events))
    half_window = prune is not None and int(np.asarray(prune).sum()) < n_all_events

    out_dir = Path(cfg.artifacts_dir)
    trades_path = out_dir / "btc_backtest_trades.csv"
    summary_path = out_dir / "btc_trade_summary.json"

    export_cols = [
        c for c in [
            "side", "ret", "t1", "prob", "confident", "size", "pnl",
            "entry_equity", "bars_held", "bin", "halted", "skipped_capacity",
        ]
        if c in traded.columns
    ]
    traded_out = traded[export_cols].copy()
    traded_out.index.name = "entry_time"
    traded_out.to_csv(trades_path)

    summary = {
        "symbol": SYMBOL,
        "data_source": trained.get("data_source"),
        "panel_bars": int(len(ds.panel)),
        "all_label_events": n_all_events,
        "report_window_events": n_report_events,
        "report_is_second_half_events": bool(half_window),
        "split_note": (
            "Purged K-Fold 在全量事件上出 OOF; 默认 gbdt+deep_ts 时弱专家前半窗选型、"
            "主路径回测用后半窗事件(不是「前半 K 线训练 / 后半 K 线回测」)。"
        ),
        "n_opened_trades": int(len(traded)),
        "n_wins": wins,
        "n_losses": losses,
        "win_rate": win_rate,
        "metrics_win_rate": float(metrics.get("win_rate", 0.0)),
        "total_return": float(metrics.get("total_return", final_mult - 1.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": final_capital,
        "equity_multiplier": final_mult,
        "dropped_experts": list(trained.get("dropped_experts") or []),
        "degradations": list(trained.get("degradations") or []),
        "classification_report": trained.get("report"),
        "backtest_metrics": metrics,
        "trades_csv": str(trades_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== 结果 ==========")
    print(f"报告窗事件数(候选): {n_report_events} / 全量标签事件 {n_all_events}")
    print(f"是否后半窗回测:     {half_window}")
    print(f"实际开仓笔数:       {len(traded)}")
    print(f"胜 / 负:            {wins} / {losses}")
    print(f"开仓胜率:           {win_rate:.2%}")
    print(f"累计收益率:         {summary['total_return']:.2%}")
    print(f"最大回撤:           {summary['max_drawdown']:.2%}")
    print(f"本金 {INITIAL_CAPITAL:,.0f} → 终值 {final_capital:,.2f}")
    print(f"[ok] 开仓明细 -> {trades_path}")
    print(f"[ok] 汇总 JSON -> {summary_path}")


if __name__ == "__main__":
    main()
