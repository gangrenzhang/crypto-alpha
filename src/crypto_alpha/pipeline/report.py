"""一键"全专家联跑"编排 + 自包含 HTML 结果面板。

- probe_experts: 探测每个专家的运行时可用性(依赖/GPU/权重), 不可用则优雅降级并说明原因。
- run_all: 对每个币种跑"数据->特征->标注->全专家 Stacking->校准->回测->决策", 可选 CPCV。
- build_dashboard: 把结果渲染为单文件 HTML(离线可看), 含决策卡/指标/专家对比/净值曲线/CPCV。
"""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import Config
from .run import prepare_dataset, build_experts, train_and_validate, latest_decision
from .evaluate import cpcv_report

ALL_EXPERTS = ["gbdt", "deep_ts", "tsfm", "llm"]


# --------------------------------------------------------------------------
# 专家可用性探测(优雅降级)
# --------------------------------------------------------------------------
def probe_experts(cfg: Config, requested: list[str]) -> tuple[list[str], dict[str, str]]:
    """返回 (可运行专家, {被跳过专家: 原因})。tsfm 有 naive 兜底始终可用。"""
    available: list[str] = []
    skipped: dict[str, str] = {}
    for name in requested:
        if name == "gbdt":
            try:
                import lightgbm  # noqa: F401
                available.append(name)
            except Exception as e:
                skipped[name] = f"缺 lightgbm: {e}"
        elif name == "deep_ts":
            try:
                import torch  # noqa: F401
                available.append(name)
            except Exception as e:
                skipped[name] = f"缺 torch: {e}"
        elif name == "tsfm":
            available.append(name)  # chronos/timesfm 缺失时自动回退 naive
        elif name == "llm":
            reason = _probe_llm(cfg)
            if reason is None:
                available.append(name)
            else:
                skipped[name] = reason
        else:
            skipped[name] = "未知专家"
    return available, skipped


def _probe_llm(cfg: Config) -> str | None:
    """LLM(72B QLoRA)需要 transformers + CUDA; 否则跳过(CPU 跑 72B 不现实)。"""
    from pathlib import Path

    try:
        import torch
        import transformers  # noqa: F401
    except Exception as e:
        return f"缺 transformers/torch: {e}"
    if not torch.cuda.is_available():
        return "无 CUDA GPU(72B 需 GPU)"
    adapter = cfg["experts"]["llm"].get("adapter_path")
    if adapter and not (cfg.root / adapter).exists():
        return f"未找到微调 adapter: {adapter}(先跑 train_llm_qlora)"
    return None


# --------------------------------------------------------------------------
# 编排
# --------------------------------------------------------------------------
def run_all(cfg: Config, symbols: list[str], experts: list[str],
            do_cpcv: bool = False) -> dict:
    """联跑全部(可用)专家, 汇总每币种指标/决策/净值(+可选 CPCV)。"""
    available, skipped = probe_experts(cfg, experts)
    if not available:
        raise RuntimeError("没有可运行的专家; 请检查依赖安装。")
    # 不永久污染 Config: 仅在本函数作用域内覆盖 enabled
    prev_enabled = list(cfg.raw["experts"].get("enabled") or [])
    cfg.raw["experts"]["enabled"] = available
    try:
        return _run_all_body(cfg, symbols, experts, available, skipped, do_cpcv)
    finally:
        cfg.raw["experts"]["enabled"] = prev_enabled


def _run_all_body(cfg, symbols, experts, available, skipped, do_cpcv) -> dict:
    results: dict = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experts_requested": experts,
            "experts_run": available,
            "experts_skipped": skipped,
            "seed": cfg.seed,
            "data_mode": "合成" if cfg["data"].get("use_synthetic", False) else "真实",
            "news_mode": ("历史库" if cfg["news"].get("use_history") else
                          ("合成" if cfg["news"].get("use_synthetic", False) else "实时")),
            "do_cpcv": do_cpcv,
        },
        "symbols": {},
    }

    for symbol in symbols:
        print(f"\n===== {symbol} 联跑(专家: {', '.join(available)}) =====")
        ds = prepare_dataset(cfg, symbol)
        trained = train_and_validate(cfg, ds)
        decision = latest_decision(cfg, ds, trained)

        eq = trained["backtest"]["equity"]
        entry = {
            "n_events": int(len(ds.y)),
            "pos_rate": float(np.mean(ds.y)),
            "date_start": str(ds.X.index.min()),
            "date_end": str(ds.X.index.max()),
            "ensemble_report": trained["report"],
            "expert_reports": trained["base_report"],
            "backtest": trained["backtest"]["metrics"],
            "decision": decision,
            "equity_curve": _downsample_equity(eq, n=180),
            "equity_b64": _equity_png_b64(eq, f"{symbol}  OOF equity"),
        }
        print(f"  集成 AUC={trained['report'].get('auc', float('nan')):.3f} "
              f"Sharpe={trained['backtest']['metrics']['sharpe']:.3f} "
              f"信号={decision['signal']} P={decision['win_probability']:.3f}")

        if do_cpcv:
            try:
                cp = cpcv_report(cfg, ds, build_experts)
                entry["cpcv"] = {
                    "n_paths": cp["n_paths"],
                    "mean_sharpe": cp["mean_sharpe"],
                    "std_sharpe": cp["std_sharpe"],
                    "deflated_sharpe": cp["deflated_sharpe"],
                    "pbo": cp["pbo"],
                    "config_sharpes": dict(zip(
                        cp["config_names"], [float(x) for x in cp["perf_matrix"].mean(1)])),
                }
                print(f"  CPCV DSR={cp['deflated_sharpe']:.3f} PBO={cp['pbo']:.3f} "
                      f"(路径数={cp['n_paths']})")
            except Exception as e:
                print(f"[warn] CPCV 失败: {e}")
                entry["cpcv"] = None

        results["symbols"][symbol] = entry
    return results


def _downsample_equity(equity, n: int = 180) -> dict:
    """把净值曲线降采样为 (date, value) 序列, 供 Canvas 交互折线图内联使用。"""
    if equity is None or len(equity) == 0:
        return {"dates": [], "values": []}
    step = max(1, len(equity) // n)
    eq = equity.iloc[::step]
    if eq.index[-1] != equity.index[-1]:
        eq = pd.concat([eq, equity.iloc[[-1]]])
    dates = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in eq.index]
    values = [round(float(v), 5) for v in eq.values]
    return {"dates": dates, "values": values}


def _equity_png_b64(equity, title: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3.2))
        equity.plot(ax=ax, color="#2563eb", lw=1.4)
        ax.axhline(1.0, color="#9ca3af", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"[warn] 净值绘图跳过: {e}")
        return None


# --------------------------------------------------------------------------
# HTML 结果面板
# --------------------------------------------------------------------------
_CSS = """
:root{--bg:#0f172a;--card:#1e293b;--muted:#94a3b8;--txt:#e2e8f0;--acc:#38bdf8;
--good:#22c55e;--bad:#ef4444;--warn:#f59e0b;--line:#334155;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
font-family:-apple-system,'Segoe UI',Roboto,'Microsoft YaHei',sans-serif;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:26px 0 12px;
border-left:3px solid var(--acc);padding-left:10px}
.sub{color:var(--muted);font-size:13px;margin-bottom:16px}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.badge{font-size:12px;padding:3px 10px;border-radius:999px;background:#0b3a4a;color:var(--acc)}
.badge.skip{background:#3a2a0b;color:var(--warn)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.3)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px}
.kpi{background:#0b1220;border:1px solid var(--line);border-radius:10px;padding:12px}
.kpi .v{font-size:20px;font-weight:600}.kpi .l{font-size:11px;color:var(--muted);margin-top:2px}
.decision{display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.sig{font-size:30px;font-weight:800;padding:8px 22px;border-radius:12px}
.sig.LONG{background:#052e1a;color:var(--good)}.sig.SHORT{background:#3a0d0d;color:var(--bad)}
.sig.HOLD{background:#25292e;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:500}tr:hover td{background:#0b1220}
.best{color:var(--good);font-weight:600}
.pos{color:var(--good)}.neg{color:var(--bad)}
img{width:100%;border-radius:8px;margin-top:8px;background:#fff}
.foot{color:var(--muted);font-size:12px;margin-top:30px;text-align:center}
"""


def _fmt(v, nd=3, pct=False):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v*100:.2f}%" if pct else f"{v:.{nd}f}"
    except Exception:
        return html.escape(str(v))


def _cls(v):
    try:
        return "pos" if float(v) > 0 else ("neg" if float(v) < 0 else "")
    except Exception:
        return ""


def build_dashboard(results: dict, cfg: Config) -> str:
    m = results["meta"]
    parts: list[str] = []
    parts.append(f"<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
                 f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
                 f"<title>Crypto-Alpha 全专家联跑面板</title><style>{_CSS}</style></head><body><div class='wrap'>")
    parts.append("<h1>Crypto-Alpha · 全专家联跑结果面板</h1>")
    parts.append(f"<div class='sub'>生成时间(UTC): {html.escape(m['generated_at'])} · "
                 f"数据: {m['data_mode']} · 新闻: {m['news_mode']} · seed: {m['seed']} · "
                 f"CPCV: {'开' if m['do_cpcv'] else '关'}</div>")

    parts.append("<div class='badges'>")
    for e in m["experts_run"]:
        parts.append(f"<span class='badge'>✓ {html.escape(e)}</span>")
    for e, why in m["experts_skipped"].items():
        parts.append(f"<span class='badge skip' title='{html.escape(why)}'>⤫ {html.escape(e)}(跳过)</span>")
    parts.append("</div>")

    for symbol, d in results["symbols"].items():
        parts.append(f"<h2>{html.escape(symbol)}</h2>")
        parts.append(_render_symbol(d))

    parts.append("<div class='foot'>本面板由 scripts/10_run_all.py 生成 · "
                 "指标基于无泄漏 OOF 概率 · DSR/PBO 越好越可信 · 仅供研究, 非投资建议</div>")
    parts.append("</div></body></html>")
    return "".join(parts)


def _render_symbol(d: dict) -> str:
    dec = d["decision"]
    bt = d["backtest"]
    ens = d["ensemble_report"]
    p = []

    # 决策卡
    sig = dec["signal"]
    p.append("<div class='card'><div class='decision'>")
    p.append(f"<div class='sig {sig}'>{sig}</div>")
    p.append("<div class='grid' style='flex:1'>")
    p.append(_kpi("胜率概率", _fmt(dec.get("win_probability"))))
    p.append(_kpi("建议仓位", _fmt(dec.get("suggested_position_pct"), pct=True)))
    p.append(_kpi("入场价", _fmt(dec.get("entry_price"), 2)))
    p.append(_kpi("止损", _fmt(dec.get("stop_loss"), 2)))
    p.append(_kpi("止盈", _fmt(dec.get("take_profit"), 2)))
    p.append("</div></div></div>")

    # 集成 + 回测 KPI
    p.append("<div class='card'><div class='grid'>")
    p.append(_kpi("集成 AUC", _fmt(ens.get("auc")), best=True))
    p.append(_kpi("Brier", _fmt(ens.get("brier"))))
    p.append(_kpi("准确率", _fmt(ens.get("accuracy"), pct=True)))
    p.append(_kpi("样本数", str(d["n_events"])))
    p.append(_kpi("Sharpe", _fmt(bt["sharpe"]), cls=_cls(bt["sharpe"])))
    p.append(_kpi("总收益", _fmt(bt["total_return"], pct=True), cls=_cls(bt["total_return"])))
    p.append(_kpi("最大回撤", _fmt(bt["max_drawdown"], pct=True), cls="neg"))
    p.append(_kpi("Calmar", _fmt(bt["calmar"])))
    p.append(_kpi("胜率", _fmt(bt["win_rate"], pct=True)))
    p.append(_kpi("交易数", str(bt["n_trades"])))
    p.append("</div></div>")

    # 专家对比表(高亮最优 AUC)
    reps = d["expert_reports"]
    best_auc = max((r.get("auc", float("nan")) for r in reps.values()
                    if not np.isnan(r.get("auc", float("nan")))), default=float("nan"))
    p.append("<div class='card'><table><thead><tr><th>专家</th><th>AUC</th>"
             "<th>Brier</th><th>准确率</th><th>样本</th></tr></thead><tbody>")
    for name, r in reps.items():
        auc = r.get("auc", float("nan"))
        is_best = (not np.isnan(auc)) and abs(auc - best_auc) < 1e-9
        cls = " class='best'" if is_best else ""
        p.append(f"<tr><td>{html.escape(name)}</td><td{cls}>{_fmt(auc)}</td>"
                 f"<td>{_fmt(r.get('brier'))}</td><td>{_fmt(r.get('accuracy'), pct=True)}</td>"
                 f"<td>{r.get('n','—')}</td></tr>")
    p.append(f"<tr><td><b>集成(stacking)</b></td><td class='best'>{_fmt(ens.get('auc'))}</td>"
             f"<td>{_fmt(ens.get('brier'))}</td><td>{_fmt(ens.get('accuracy'), pct=True)}</td>"
             f"<td>{ens.get('n','—')}</td></tr>")
    p.append("</tbody></table></div>")

    # CPCV
    cp = d.get("cpcv")
    if cp:
        p.append("<div class='card'><div class='grid'>")
        p.append(_kpi("回测路径数", str(cp["n_paths"])))
        p.append(_kpi("路径夏普均值", _fmt(cp["mean_sharpe"]), cls=_cls(cp["mean_sharpe"])))
        p.append(_kpi("夏普标准差", _fmt(cp["std_sharpe"])))
        p.append(_kpi("去偏夏普 DSR", _fmt(cp["deflated_sharpe"]), best=True))
        p.append(_kpi("过拟合概率 PBO", _fmt(cp["pbo"]),
                      cls=("pos" if cp["pbo"] < 0.5 else "neg")))
        p.append("</div>")
        p.append("<table><thead><tr><th>配置</th><th>平均夏普</th></tr></thead><tbody>")
        for cn, sv in cp["config_sharpes"].items():
            p.append(f"<tr><td>{html.escape(cn)}</td><td class='{_cls(sv)}'>{_fmt(sv)}</td></tr>")
        p.append("</tbody></table></div>")

    # 净值曲线
    if d.get("equity_b64"):
        p.append(f"<div class='card'><img alt='equity' "
                 f"src='data:image/png;base64,{d['equity_b64']}'></div>")
    return "".join(p)


def _kpi(label: str, value: str, cls: str = "", best: bool = False) -> str:
    vcls = ("best " if best else "") + cls
    return (f"<div class='kpi'><div class='v {vcls.strip()}'>{value}</div>"
            f"<div class='l'>{html.escape(label)}</div></div>")
