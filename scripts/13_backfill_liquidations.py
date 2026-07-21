"""回填清算事件库, 并按主周期 K 线时间桶对齐验证。

用法:
  python scripts/13_backfill_liquidations.py
  python scripts/13_backfill_liquidations.py --symbol BTC/USDT
  python scripts/13_backfill_liquidations.py --symbol BTC/USDT --since 2024-01-01
  python scripts/13_backfill_liquidations.py --symbol BTC/USDT --import-csv path/to/events.csv
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.data.fetch import load_symbol_data, timeframe_delta
from crypto_alpha.data.liquidations import (
    attach_liquidations_to_ohlcv,
    fetch_and_store_liquidations,
    import_liquidation_events_frame,
    load_liquidation_events,
    liquidations_path,
)
from crypto_alpha.features.build import build_feature_matrix


def main() -> None:
    ap = argparse.ArgumentParser(description="回填清算事件并验证按 bar 对齐")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--since", default=None, help="覆盖 data.since")
    ap.add_argument("--until", default=None)
    ap.add_argument("--skip-fetch", action="store_true", help="只验证本地事件库对齐")
    ap.add_argument(
        "--import-csv",
        default=None,
        help="导入外部清算 CSV(列: timestamp, side|side_bucket, notional|quoteValue)",
    )
    ap.add_argument(
        "--refresh-ohlcv",
        action="store_true",
        help="对齐前增量刷新主周期 OHLCV tip(否则冷缓存末 bar 早于 tip 清算会落不到桶)",
    )
    args = ap.parse_args()

    cfg = Config.load()
    symbol = args.symbol

    if args.import_csv:
        frame = pd.read_csv(args.import_csv)
        merged = import_liquidation_events_frame(cfg, symbol, frame, exchange="csv")
        print(
            f"[import] n={len(merged)} path={liquidations_path(cfg, symbol)}",
            flush=True,
        )
    elif not args.skip_fetch:
        print(f"[fetch] {symbol} liquidations …", flush=True)
        summary = fetch_and_store_liquidations(
            cfg, symbol, since=args.since, until=args.until
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    else:
        ev = load_liquidation_events(cfg, symbol)
        print(f"[skip-fetch] store={liquidations_path(cfg, symbol)} n={len(ev)}", flush=True)

    if args.refresh_ohlcv:
        from crypto_alpha.data.fetch import refresh_market_data

        print(f"[refresh] OHLCV tip for {symbol} …", flush=True)
        raw = refresh_market_data(cfg, symbol)
    else:
        raw = load_symbol_data(cfg, symbol)
    print(
        f"[ohlcv] {raw.index[0]} -> {raw.index[-1]} n={len(raw)}",
        flush=True,
    )
    attached = attach_liquidations_to_ohlcv(
        raw, cfg, symbol, bar_delta=timeframe_delta(cfg["data"]["timeframe"])
    )
    ll, ls = attached["liq_long"], attached["liq_short"]
    finite = (ll.notna() & ls.notna()).sum()
    nz = ((ll.fillna(0) > 0) | (ls.fillna(0) > 0)).sum()
    print(
        f"[align] bars={len(attached)} finite={int(finite)} "
        f"nonzero_bars={int(nz)} "
        f"liq_long_sum={float(ll.fillna(0).sum()):.2f} "
        f"liq_short_sum={float(ls.fillna(0).sum()):.2f}",
        flush=True,
    )
    if int(nz) > 0:
        last_nz = attached.index[((ll.fillna(0) > 0) | (ls.fillna(0) > 0))][-1]
        print(f"[align] last_nonzero_bar={last_nz}", flush=True)

    feat = build_feature_matrix(raw, cfg, symbol=symbol)
    imb = feat["liq_imbalance"] if "liq_imbalance" in feat.columns else None
    print(
        f"[feat] degradations={feat.attrs.get('degradations')} "
        f"attached={feat.attrs.get('liquidations_attached_from_store')} "
        f"n_events={feat.attrs.get('liquidations_n_events')} "
        f"liq_imbalance_absmax={None if imb is None else float(imb.abs().max())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
