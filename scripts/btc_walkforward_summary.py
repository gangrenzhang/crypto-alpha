"""BTC/通用 walk-forward 真外推: 训练窗 → 测试窗(部署同形门控)。

逻辑在 ``crypto_alpha.pipeline.walkforward``; 本脚本仅为 CLI 薄封装。
切分与阈值见 ``config.yaml`` → ``validation.walkforward``。

用法:
    python scripts/btc_walkforward_summary.py
    python scripts/btc_walkforward_summary.py --symbol ETH/USDT
    python scripts/btc_walkforward_summary.py --test-start 2022-09-14 --test-end 2026-07-18
    python scripts/btc_walkforward_summary.py --train-start 2020-01-01 --test-start 2024-09-14 --test-end 2026-07-18
    python scripts/btc_walkforward_summary.py --recompute-sample-weight --train-start 2020-01-01 --test-start 2024-09-14
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.pipeline.walkforward import (
    run_walkforward,
    walkforward_public_summary,
)


def _print_gate_diagnostics(g: dict) -> None:
    gates = g.get("gates") or {}
    cal = g.get("calibrated_proba") or {}
    print("\n========== 开仓门控诊断(测试窗) ==========", flush=True)
    print(f"有效阈值 thr:          {g.get('prob_threshold')}", flush=True)
    print(f"测试事件 n:            {g.get('n_events', g.get('n_test_events'))}", flush=True)
    print(f"prob≥thr:              {gates.get('n_prob_ge_threshold')} "
          f"({gates.get('frac_prob_ge_threshold')})", flush=True)
    print(f"confident:             {gates.get('n_confident')} "
          f"({gates.get('frac_confident')})", flush=True)
    print(f"prob≥thr 且 confident: {gates.get('n_prob_and_confident')} "
          f"({gates.get('frac_prob_and_confident')})", flush=True)
    print(f"实际开仓 size>0:       {gates.get('n_opened_size_gt_0')} "
          f"({gates.get('frac_opened_size_gt_0')})", flush=True)
    print(f"过门但 size=0:         {gates.get('n_pass_gate_but_size_0')}", flush=True)
    print(
        f"保形 qhat/margin:      {g.get('conformal_qhat')} / {g.get('conformal_min_margin')}",
        flush=True,
    )
    if cal:
        print(
            f"校准后概率: mean={cal.get('mean')} n_unique={cal.get('n_unique')} "
            f"max={cal.get('max')}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward 真外推基线")
    ap.add_argument("--symbol", default=None, help="默认 BTC/USDT 或 config 首个币种")
    ap.add_argument("--train-start", default=None, help="可选训练起点(含); 丢掉更早事件")
    ap.add_argument("--test-start", default=None, help="覆盖 validation.walkforward.test_start")
    ap.add_argument("--test-end", default=None, help="覆盖 test_end; 省略则用配置/面板末")
    ap.add_argument(
        "--recompute-sample-weight",
        action="store_true",
        help="仅用训练事件重算 sample_weight(覆盖 config 默认 false)",
    )
    args = ap.parse_args()

    cfg = Config.load()
    # 研究 WF: 冷缓存, 关 tip / 空新闻面(与历史脚本纪律一致, 避免混入决策 REST)
    cfg.raw["data"]["refresh_before_decide"] = False
    cfg.raw["data"]["incremental_update"] = False

    symbol = args.symbol or "BTC/USDT"
    if args.symbol is None and cfg["data"].get("symbols"):
        # 保持历史默认 BTC; 若配置无 BTC 则用首个
        syms = list(cfg["data"]["symbols"])
        symbol = "BTC/USDT" if "BTC/USDT" in syms else syms[0]

    print(f"===== {symbol} walk-forward =====", flush=True)
    if args.train_start:
        print(f"train_start: {args.train_start}", flush=True)
    if args.recompute_sample_weight:
        print("recompute_sample_weight: true", flush=True)
    summary = run_walkforward(
        cfg, symbol,
        train_start=args.train_start,
        test_start=args.test_start,
        test_end=args.test_end,
        recompute_sample_weight=True if args.recompute_sample_weight else None,
    )

    out_dir = Path(cfg.artifacts_dir)
    stem = symbol.replace("/", "_")
    # 兼容旧文件名(BTC 专用脚本习惯)
    if symbol == "BTC/USDT":
        trades_path = out_dir / "btc_walkforward_trades.csv"
        summary_path = out_dir / "btc_walkforward_summary.json"
        gate_path = out_dir / "btc_walkforward_gate_diagnostics.json"
    else:
        trades_path = out_dir / f"walkforward_{stem}_trades.csv"
        summary_path = out_dir / f"walkforward_{stem}.json"
        gate_path = out_dir / f"walkforward_{stem}_gate_diagnostics.json"

    traded = summary.pop("_traded_detail", None)
    summary.pop("_equity", None)
    if traded is not None and len(traded):
        traded_out = traded.copy()
        traded_out.index.name = "entry_time"
        traded_out.to_csv(trades_path)
        summary["trades_csv"] = str(trades_path)
    else:
        summary["trades_csv"] = None

    pub = walkforward_public_summary(summary)
    summary_path.write_text(
        json.dumps(pub, ensure_ascii=False, indent=2, default=float), encoding="utf-8",
    )
    gate_path.write_text(
        json.dumps(pub.get("gate_diagnostics") or {}, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    print("\n========== 结果 ==========", flush=True)
    print(f"实际数据起点:     {pub.get('panel_start')}", flush=True)
    print(f"训练起点(含):     {pub.get('train_start')}", flush=True)
    print(f"训练截止(不含):   {pub.get('train_end_exclusive')}", flush=True)
    print(f"训练权重重算:     {pub.get('recompute_sample_weight_on_split')}", flush=True)
    print(f"训练/回测事件:    {pub.get('n_train_events')} / {pub.get('n_test_events')}", flush=True)
    print(f"实际开仓笔数:     {pub.get('n_opened_trades')}", flush=True)
    print(f"胜 / 负:          {pub.get('n_wins')} / {pub.get('n_losses')}", flush=True)
    wr = pub.get("win_rate") or 0.0
    print(f"开仓胜率:         {wr:.2%}", flush=True)
    print(f"累计收益:         {pub.get('total_return'):.2%}", flush=True)
    print(f"1万→终值:         {pub.get('final_capital'):,.2f}", flush=True)
    _print_gate_diagnostics(pub.get("gate_diagnostics") or {})
    print(f"[ok] {summary_path}", flush=True)
    print(f"[ok] {gate_path}", flush=True)
    if summary.get("trades_csv"):
        print(f"[ok] {trades_path}", flush=True)


if __name__ == "__main__":
    main()
