"""生成 BTC 回测 K 线面板(HTML): 含回测窗 30m 蜡烛 + 开仓/平仓标记。

默认读取:
  - data/raw/BTC_USDT__30m.parquet
  - artifacts/btc_backtest_trades.csv
  - artifacts/btc_trade_summary.json (可选, 用于 KPI)

产出: artifacts/btc_backtest_kline_panel.html
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config


def _to_unix(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp())


def _load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["t1"] = pd.to_datetime(df["t1"], utc=True)
    return df.sort_values("entry_time").reset_index(drop=True)


def _candles_payload(ohlcv: pd.DataFrame) -> list[dict]:
    out = []
    for ts, r in ohlcv.iterrows():
        out.append(
            {
                "time": _to_unix(ts),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
            }
        )
    return out


def _markers_payload(trades: pd.DataFrame, ohlcv: pd.DataFrame) -> list[dict]:
    """开仓/平仓标记; 价格取对应 bar 的 close(便于阅读)。"""
    idx = ohlcv.index
    markers: list[dict] = []
    for i, r in trades.iterrows():
        side = int(np.sign(r["side"]))
        win = float(r["pnl"]) > 0
        entry_ts = pd.Timestamp(r["entry_time"])
        exit_ts = pd.Timestamp(r["t1"])
        # 对齐到已有 K 线(开盘时间戳)
        if entry_ts not in idx:
            # 向后找最近一根(不应常发生)
            pos = idx.searchsorted(entry_ts)
            if pos >= len(idx):
                continue
            entry_ts = idx[pos]
        if exit_ts not in idx:
            pos = idx.searchsorted(exit_ts)
            if pos >= len(idx):
                pos = len(idx) - 1
            exit_ts = idx[min(pos, len(idx) - 1)]

        long = side > 0
        markers.append(
            {
                "time": _to_unix(entry_ts),
                "position": "belowBar" if long else "aboveBar",
                "color": "#16a34a" if long else "#dc2626",
                "shape": "arrowUp" if long else "arrowDown",
                "text": f"开{'多' if long else '空'}#{i + 1}",
                "size": 1,
            }
        )
        markers.append(
            {
                "time": _to_unix(exit_ts),
                "position": "aboveBar" if long else "belowBar",
                "color": "#0d9488" if win else "#ea580c",
                "shape": "circle",
                "text": f"平{'盈' if win else '亏'}#{i + 1}",
                "size": 1,
            }
        )
    # lightweight-charts 要求 markers 按 time 升序
    markers.sort(key=lambda m: (m["time"], 0 if "开" in m["text"] else 1))
    return markers


def _trades_table(trades: pd.DataFrame) -> list[dict]:
    rows = []
    for i, r in trades.iterrows():
        rows.append(
            {
                "id": i + 1,
                "entry": str(r["entry_time"]),
                "exit": str(r["t1"]),
                "side": "LONG" if float(r["side"]) > 0 else "SHORT",
                "prob": round(float(r["prob"]), 4),
                "size": round(float(r["size"]), 4),
                "pnl": round(float(r["pnl"]), 6),
                "win": bool(float(r["pnl"]) > 0),
                "bars_held": int(r["bars_held"]) if "bars_held" in r and pd.notna(r["bars_held"]) else None,
            }
        )
    return rows


def build_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BTC 回测 K 线面板</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {{
    --bg: #0f1419;
    --panel: #1a222d;
    --text: #e7ecf3;
    --muted: #8b98a8;
    --line: #2a3544;
    --long: #16a34a;
    --short: #dc2626;
    --win: #0d9488;
    --loss: #ea580c;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--text);
  }}
  header {{
    padding: 14px 18px; border-bottom: 1px solid var(--line);
    display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: baseline;
  }}
  header h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
  header .meta {{ color: var(--muted); font-size: 13px; }}
  .kpis {{
    display: flex; flex-wrap: wrap; gap: 10px; padding: 12px 18px;
    border-bottom: 1px solid var(--line); background: var(--panel);
  }}
  .kpi {{
    min-width: 120px; padding: 8px 12px; border: 1px solid var(--line); border-radius: 6px;
  }}
  .kpi .l {{ color: var(--muted); font-size: 11px; }}
  .kpi .v {{ font-size: 16px; font-weight: 600; margin-top: 2px; }}
  .legend {{
    display: flex; flex-wrap: wrap; gap: 14px; padding: 8px 18px; font-size: 12px; color: var(--muted);
  }}
  .legend span::before {{
    content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 6px;
    border-radius: 2px; vertical-align: -1px;
  }}
  .legend .long::before {{ background: var(--long); }}
  .legend .short::before {{ background: var(--short); }}
  .legend .win::before {{ background: var(--win); border-radius: 50%; }}
  .legend .loss::before {{ background: var(--loss); border-radius: 50%; }}
  #chart {{ height: 640px; width: 100%; }}
  .table-wrap {{ padding: 12px 18px 28px; }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 12px;
  }}
  th, td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; }}
  th {{ color: var(--muted); font-weight: 500; position: sticky; top: 0; background: var(--bg); }}
  tr:hover td {{ background: #1a222d; }}
  .tag-long {{ color: var(--long); }}
  .tag-short {{ color: var(--short); }}
  .tag-win {{ color: var(--win); }}
  .tag-loss {{ color: var(--loss); }}
  button.jump {{
    background: transparent; border: 1px solid var(--line); color: var(--text);
    border-radius: 4px; padding: 2px 8px; cursor: pointer; font-size: 11px;
  }}
  button.jump:hover {{ border-color: #5b6b7c; }}
</style>
</head>
<body>
<header>
  <h1>BTC/USDT 回测 K 线面板</h1>
  <div class="meta" id="meta"></div>
</header>
<div class="kpis" id="kpis"></div>
<div class="legend">
  <span class="long">开多 ▲</span>
  <span class="short">开空 ▼</span>
  <span class="win">平盈 ●</span>
  <span class="loss">平亏 ●</span>
  <span>滚轮缩放 · 拖拽平移 · 点击下方表格可跳转到该笔开仓</span>
</div>
<div id="chart"></div>
<div class="table-wrap">
  <h2 style="font-size:14px;margin:0 0 8px;">开仓明细</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>方向</th><th>开仓</th><th>平仓</th><th>概率</th>
        <th>仓位</th><th>持有(根)</th><th>PnL</th><th>跳转</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>
<script>
const DATA = {data_json};

const meta = document.getElementById("meta");
meta.textContent = `${{DATA.symbol}} · 30m · ${{DATA.bar_start}} → ${{DATA.bar_end}} · ${{DATA.n_bars}} 根K线 · ${{DATA.n_trades}} 笔开仓`;

const kpis = document.getElementById("kpis");
const kpiItems = [
  ["开仓胜率", (DATA.win_rate * 100).toFixed(2) + "%"],
  ["胜 / 负", DATA.n_wins + " / " + DATA.n_losses],
  ["累计收益", (DATA.total_return * 100).toFixed(2) + "%"],
  ["最大回撤", (DATA.max_drawdown * 100).toFixed(2) + "%"],
  ["1万→终值", DATA.final_capital.toFixed(2)],
];
kpis.innerHTML = kpiItems.map(([l,v]) => `<div class="kpi"><div class="l">${{l}}</div><div class="v">${{v}}</div></div>`).join("");

const chart = LightweightCharts.createChart(document.getElementById("chart"), {{
  layout: {{ background: {{ color: "#0f1419" }}, textColor: "#c5d0dc" }},
  grid: {{ vertLines: {{ color: "#1e2833" }}, horzLines: {{ color: "#1e2833" }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor: "#2a3544" }},
  timeScale: {{ borderColor: "#2a3544", timeVisible: true, secondsVisible: false }},
}});

const series = chart.addCandlestickSeries({{
  upColor: "#26a69a",
  downColor: "#ef5350",
  borderVisible: false,
  wickUpColor: "#26a69a",
  wickDownColor: "#ef5350",
}});
series.setData(DATA.candles);
series.setMarkers(DATA.markers);
chart.timeScale().fitContent();

window.addEventListener("resize", () => chart.applyOptions({{ width: document.getElementById("chart").clientWidth }}));

const tbody = document.getElementById("tbody");
tbody.innerHTML = DATA.trades.map(t => {{
  const sideCls = t.side === "LONG" ? "tag-long" : "tag-short";
  const pnlCls = t.win ? "tag-win" : "tag-loss";
  return `<tr>
    <td>${{t.id}}</td>
    <td class="${{sideCls}}">${{t.side}}</td>
    <td>${{t.entry}}</td>
    <td>${{t.exit}}</td>
    <td>${{t.prob}}</td>
    <td>${{(t.size * 100).toFixed(2)}}%</td>
    <td>${{t.bars_held ?? "-"}}</td>
    <td class="${{pnlCls}}">${{(t.pnl * 100).toFixed(3)}}%</td>
    <td><button class="jump" data-time="${{t.entry_unix}}">定位开仓</button></td>
  </tr>`;
}}).join("");

tbody.addEventListener("click", (e) => {{
  const btn = e.target.closest("button.jump");
  if (!btn) return;
  const t = Number(btn.dataset.time);
  chart.timeScale().setVisibleRange({{ from: t - 3 * 86400, to: t + 3 * 86400 }});
}});
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="生成后用默认浏览器打开")
    ap.add_argument(
        "--pad-days", type=int, default=7,
        help="回测首末开平仓外侧各多留 N 天 K 线(默认 7)",
    )
    args = ap.parse_args()

    cfg = Config.load()
    root = cfg.root
    trades_path = root / "artifacts" / "btc_backtest_trades.csv"
    summary_path = root / "artifacts" / "btc_trade_summary.json"
    ohlcv_path = root / "data" / "raw" / "BTC_USDT__30m.parquet"
    out_path = root / "artifacts" / "btc_backtest_kline_panel.html"

    if not trades_path.exists():
        raise SystemExit(f"缺少成交明细: {trades_path} (请先跑 scripts/btc_trade_summary.py)")
    if not ohlcv_path.exists():
        raise SystemExit(f"缺少行情缓存: {ohlcv_path}")

    trades = _load_trades(trades_path)
    ohlcv = pd.read_parquet(ohlcv_path)
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        # 兼容 timestamp 列
        if "timestamp" in ohlcv.columns:
            ohlcv = ohlcv.set_index("timestamp")
        ohlcv.index = pd.to_datetime(ohlcv.index, utc=True)
    else:
        if ohlcv.index.tz is None:
            ohlcv.index = ohlcv.index.tz_localize("UTC")
        else:
            ohlcv.index = ohlcv.index.tz_convert("UTC")
    ohlcv = ohlcv.sort_index()

    # 回测实际开仓覆盖的 K 线窗(+边距); 即「回测交易用到的」30m 序列
    t0 = trades["entry_time"].min() - pd.Timedelta(days=int(args.pad_days))
    t1 = trades["t1"].max() + pd.Timedelta(days=int(args.pad_days))
    window = ohlcv.loc[(ohlcv.index >= t0) & (ohlcv.index <= t1)].copy()
    if window.empty:
        raise SystemExit("选定时间窗内无 K 线, 请检查 trades 与 parquet 时间对齐")

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    trades_rows = _trades_table(trades)
    for row, (_, r) in zip(trades_rows, trades.iterrows()):
        row["entry_unix"] = _to_unix(r["entry_time"])

    payload = {
        "symbol": "BTC/USDT",
        "bar_start": str(window.index.min()),
        "bar_end": str(window.index.max()),
        "n_bars": int(len(window)),
        "n_trades": int(len(trades)),
        "n_wins": int(summary.get("n_wins", (trades["pnl"] > 0).sum())),
        "n_losses": int(summary.get("n_losses", (trades["pnl"] <= 0).sum())),
        "win_rate": float(summary.get("win_rate", (trades["pnl"] > 0).mean())),
        "total_return": float(summary.get("total_return", 0.0)),
        "max_drawdown": float(summary.get("max_drawdown", 0.0)),
        "final_capital": float(summary.get("final_capital", 10000.0)),
        "candles": _candles_payload(window),
        "markers": _markers_payload(trades, window),
        "trades": trades_rows,
        "note": summary.get(
            "split_note",
            "K 线范围为回测实际开平仓覆盖区间(+边距); 标记来自 btc_backtest_trades.csv",
        ),
    }

    out_path.write_text(build_html(payload), encoding="utf-8")
    print(f"[ok] 面板 -> {out_path}")
    print(f"     K线 {payload['n_bars']} 根 | 开仓标记 {payload['n_trades']} | 平仓标记 {payload['n_trades']}")
    if args.open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
