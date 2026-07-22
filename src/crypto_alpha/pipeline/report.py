"""一键"全专家联跑"编排 + 自包含 HTML 结果面板。

- probe_experts: 探测每个专家的运行时可用性(依赖/GPU/权重), 不可用则优雅降级并说明原因。
- run_all: 对每个币种跑"数据->特征->标注->全专家 Stacking->校准->回测->决策",
  可选 CPCV 与 walk-forward(single-cut 真外推基线)。
- build_dashboard: 把结果渲染为单文件 HTML(离线可看), 含决策卡/指标/专家对比/净值/
  CPCV/Walk-forward 基线卡。
"""
from __future__ import annotations

import base64
import html
import io
import json
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
    """LLM(QLoRA)需要 transformers + CUDA; 否则跳过(大参数 Instruct 模型 CPU 不现实)。"""
    from pathlib import Path

    try:
        import torch
        import transformers  # noqa: F401
    except Exception as e:
        return f"缺 transformers/torch: {e}"
    if not torch.cuda.is_available():
        return "无 CUDA GPU(LLM QLoRA 需 GPU)"
    adapter = cfg["experts"]["llm"].get("adapter_path")
    if adapter and not (cfg.root / adapter).exists():
        return f"未找到微调 adapter: {adapter}(先跑 train_llm_qlora)"
    return None


# --------------------------------------------------------------------------
# 编排
# --------------------------------------------------------------------------
def run_all(
    cfg: Config,
    symbols: list[str],
    experts: list[str],
    do_cpcv: bool = False,
    do_walkforward: bool | None = None,
) -> dict:
    """联跑全部(可用)专家, 汇总每币种指标/决策/净值(+可选 CPCV / walk-forward)。

    ``do_walkforward``:
      - ``None``: 读 ``validation.walkforward.enabled_in_run_all``
      - ``True/False``: 显式覆盖(如 ``10_run_all.py --walkforward``)
    若 ``validation.walkforward.require_in_run_all`` 为真且最终未跑 WF → RuntimeError。
    """
    available, skipped = probe_experts(cfg, experts)
    if not available:
        raise RuntimeError("没有可运行的专家; 请检查依赖安装。")
    wf_sec = ((cfg.get("validation") or {}).get("walkforward") or {})
    if do_walkforward is None:
        do_walkforward = bool(wf_sec.get("enabled_in_run_all", False))
    require_wf = bool(wf_sec.get("require_in_run_all", False))
    # 发布硬前置: 未启用 WF 时立刻失败, 避免先跑完昂贵 OOF 才报错
    if require_wf and not do_walkforward:
        raise RuntimeError(
            "validation.walkforward.require_in_run_all=true 但本次未跑 walk-forward;"
            " 请传 do_walkforward=True / --walkforward, 或关闭 require_in_run_all。"
        )
    # 不永久污染 Config: 仅在本函数作用域内覆盖 enabled
    prev_enabled = list(cfg.raw["experts"].get("enabled") or [])
    cfg.raw["experts"]["enabled"] = available
    try:
        results = _run_all_body(
            cfg, symbols, experts, available, skipped, do_cpcv, do_walkforward,
        )
    finally:
        cfg.raw["experts"]["enabled"] = prev_enabled
    if require_wf:
        missing = [
            s for s, d in results["symbols"].items()
            if not (d.get("walkforward") or {}).get("ok")
        ]
        if missing:
            raise RuntimeError(
                "walk-forward 为发布硬前置, 但以下币种未成功产出 WF 基线: "
                + ", ".join(missing)
            )
    return results


# --------------------------------------------------------------------------
# 研究口径说明(看板 / summary 共用, 防止 OOF 夏普被当成可部署证明)
# --------------------------------------------------------------------------
RESEARCH_DISCLAIMERS: list[str] = [
    "主面板默认「研究回测」基于 Purged nested OOF + 交叉拟合校准/保形，"
    "不是 walk-forward 实盘滚动再训练；上线前须另做滚动/外推评估。",
    "真外推基线：启用 --walkforward / validation.walkforward 后，面板会并列 "
    "Walk-forward KPI（仅过去窗拟合→未来窗部署门控）。成交/胜率/期望以 WF 为准，"
    "勿用研究 OOF 或 train_and_validate.backtest_deploy 拍板上线。",
    "摘要中的 backtest_deploy / gate_diagnostics_deploy 为部署出分路径"
    "（predict→fit_deploy 校准/保形），与 decide/serve 同形；"
    "在 train_and_validate 内对报告窗仍可能偏乐观，**not_for_go_live=true**——"
    "成交数勿与 research OOF 直接对比，更勿作上线拍板。"
    "面板 KPI：「交易数(研究OOF)」与「交易数(部署·偏乐观·勿拍板)」。",
    "开仓阈值双冻结：prob_threshold_research=交叉拟合参考窗尺度（仅研究回测）；"
    "prob_threshold_effective=deploy 校准器变换参考窗原始 OOF（decide/serve/部署回测/"
    "与 walk-forward 同形）。参考窗为空时用时间半段回退（禁止全评估窗刷 thr）。"
    "保形另有 conformal_min_margin（|p-0.5|）。",
    "CPCV（若开启）评估单元是相关组合(combo)，不是拼接后的完整路径；"
    "请阅读 caveats；DSR/PBO 在配置数少或 dsr_n_trials 低估时偏乐观。",
    "execution_assumption 当前仅实现 close_fill；未实现取值会在加载配置时报错。",
    "回测 Sharpe 字段为每笔 pnl_frac 口径(可丢弃零收益)；账户表现请看 "
    "sharpe_equity / sharpe_equity_mtm（及年化字段）。DSR/CPCV 仍基于每笔口径。",
    "看板净值曲线默认用盯市权益(equity_mtm)，与 KPI「最大回撤」口径一致；"
    "已实现权益见 max_drawdown_realized / sharpe_equity。",
]


def _run_all_body(
    cfg, symbols, experts, available, skipped, do_cpcv, do_walkforward,
) -> dict:
    results: dict = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experts_requested": experts,
            "experts_run": available,
            "experts_skipped": skipped,
            "seed": cfg.seed,
            # data_mode 在跑完首个 symbol 后按实际 data_source 回填, 避免配置与降级不一致
            "data_mode": "待检测",
            "news_mode": ("历史库" if cfg["news"].get("use_history") else
                          ("合成" if cfg["news"].get("use_synthetic", False) else "实时")),
            "do_cpcv": do_cpcv,
            "do_walkforward": bool(do_walkforward),
            "research_disclaimers": list(RESEARCH_DISCLAIMERS),
        },
        "symbols": {},
    }

    for symbol in symbols:
        print(f"\n===== {symbol} 联跑(专家: {', '.join(available)}) =====")
        ds = prepare_dataset(cfg, symbol)
        trained = train_and_validate(cfg, ds)
        decision = latest_decision(cfg, ds, trained)
        results["meta"]["data_mode"] = trained.get(
            "data_mode_zh",
            "合成" if cfg["data"].get("use_synthetic", False) else "真实",
        )
        if trained.get("degradations"):
            # 多币种累加, 避免后者覆盖前者
            prev = list(results["meta"].get("degradations") or [])
            for tag in trained["degradations"]:
                if tag not in prev:
                    prev.append(tag)
            results["meta"]["degradations"] = prev

        # 曲线与 KPI max_drawdown 对齐: 优先盯市权益; 无盯市时回退已实现
        bt_pack = trained["backtest"]
        eq_mtm = bt_pack.get("equity_mtm")
        eq_realized = bt_pack.get("equity")
        use_mtm = bool(bt_pack.get("metrics", {}).get("mark_to_market")) and (
            eq_mtm is not None and len(eq_mtm)
        )
        eq_plot = eq_mtm if use_mtm else eq_realized
        curve_kind = "mtm" if use_mtm else "realized"
        bt_dep = trained.get("backtest_deploy") or {}
        entry = {
            "n_events": int(len(ds.y)),
            "pos_rate": float(np.mean(ds.y)),
            "date_start": str(ds.X.index.min()),
            "date_end": str(ds.X.index.max()),
            "data_source": ds.data_source,
            "ensemble_report": trained["report"],
            "expert_reports": trained["base_report"],
            "backtest": bt_pack["metrics"],
            "backtest_deploy": bt_dep.get("metrics"),
            "prob_threshold_effective": trained.get("prob_threshold_effective"),
            "prob_threshold_research": trained.get("prob_threshold_research"),
            "gate_diagnostics_research": trained.get("gate_diagnostics_research"),
            "gate_diagnostics_deploy": trained.get("gate_diagnostics_deploy"),
            "decision": decision,
            "equity_curve": _downsample_equity(eq_plot, n=180),
            "equity_curve_kind": curve_kind,
            "equity_b64": _equity_png_b64(
                eq_plot,
                f"{symbol}  OOF equity ({'盯市' if curve_kind == 'mtm' else '已实现'})",
            ),
            "degradations": trained.get("degradations", []),
        }
        n_dep = (bt_dep.get("metrics") or {}).get("n_trades", "?")
        print(f"  集成 AUC={trained['report'].get('auc', float('nan')):.3f} "
              f"研究Sharpe={trained['backtest']['metrics']['sharpe']:.3f} "
              f"部署成交={n_dep} "
              f"thr_r={trained.get('prob_threshold_research')} "
              f"thr_d={trained.get('prob_threshold_effective')} "
              f"信号={decision['signal']} "
              f"P={_fmt_prob(decision.get('win_probability'))}")

        if do_walkforward:
            from .walkforward import (
                run_walkforward,
                slim_walkforward_for_dashboard,
                walkforward_public_summary,
            )

            print(f"  [walk-forward] {symbol} 真外推基线 ...", flush=True)
            try:
                # 复用同一冷缓存 Dataset, 禁止 for_decide tip
                wf_raw = run_walkforward(cfg, symbol, ds=ds)
                entry["walkforward"] = {
                    "ok": True,
                    **slim_walkforward_for_dashboard(wf_raw),
                }
                # 完整 summary 写入 artifacts(每币种), 便于审计
                stem = symbol.replace("/", "_")
                wf_path = cfg.artifacts_dir / f"walkforward_{stem}.json"
                wf_path.write_text(
                    json.dumps(
                        walkforward_public_summary(wf_raw),
                        ensure_ascii=False, indent=2, default=float,
                    ),
                    encoding="utf-8",
                )
                entry["walkforward"]["summary_path"] = str(wf_path)
                print(
                    f"  [walk-forward] 开仓={entry['walkforward'].get('n_opened_trades')} "
                    f"胜率={entry['walkforward'].get('win_rate')} "
                    f"收益={entry['walkforward'].get('total_return')} "
                    f"-> {wf_path}",
                    flush=True,
                )
            except Exception as e:
                entry["walkforward"] = {
                    "ok": False,
                    "error": str(e),
                    "evaluation_unit": "walk_forward",
                }
                print(f"  [walk-forward] FAIL: {e}", flush=True)

        if do_cpcv:
            try:
                cp = cpcv_report(cfg, ds, build_experts)
                entry["cpcv"] = {
                    "evaluation_unit": cp.get("evaluation_unit", "combo"),
                    "n_paths": cp["n_paths"],
                    "n_combos": cp.get("n_combos", cp["n_paths"]),
                    "n_paths_theoretical": cp.get("n_paths_theoretical"),
                    "mean_sharpe": cp["mean_sharpe"],
                    "std_sharpe": cp["std_sharpe"],
                    "deflated_sharpe": cp["deflated_sharpe"],
                    "pbo": cp["pbo"],
                    "pbo_warning": bool(cp.get("pbo_warning", False)),
                    "caveats": list(cp.get("caveats") or []),
                    "config_sharpes": dict(zip(
                        cp["config_names"], [float(x) for x in cp["perf_matrix"].mean(1)])),
                }
                print(f"  CPCV DSR={cp['deflated_sharpe']:.3f} PBO={cp['pbo']:.3f} "
                      f"(组合数={cp.get('n_combos', cp['n_paths'])}, "
                      f"unit={cp.get('evaluation_unit', 'combo')})")
            except Exception as e:
                print(f"[warn] CPCV 失败: {e}")
                entry["cpcv"] = None

        results["symbols"][symbol] = entry
    return results


def _fmt_prob(v) -> str:
    """决策概率可能为 None(如 CUSUM HOLD), 避免 format 崩溃。"""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{float(v):.3f}"
    except Exception:
        return "—"


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
.disclaimer{background:#3a2a0b;border:1px solid var(--warn);border-radius:10px;
padding:12px 14px;margin:14px 0;font-size:13px;color:#fde68a;line-height:1.55}
.disclaimer b{color:#fbbf24}.disclaimer ul{margin:8px 0 0 18px;padding:0}
.caveat{color:var(--warn);font-size:12px;margin-top:8px}
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
                 f"CPCV: {'开' if m['do_cpcv'] else '关'} · "
                 f"Walk-forward: {'开' if m.get('do_walkforward') else '关'}</div>")

    # 研究口径 disclaimer: OOF ≠ walk-forward; CPCV caveats
    disclaimers = m.get("research_disclaimers") or RESEARCH_DISCLAIMERS
    parts.append("<div class='disclaimer'><b>研究口径说明（必读）</b><ul>")
    for line in disclaimers:
        parts.append(f"<li>{html.escape(line)}</li>")
    parts.append("</ul></div>")

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
                 "主指标=Purged nested OOF（≠ walk-forward）· "
                 "真外推以 Walk-forward 卡为准 · "
                 "CPCV 为相关组合评估（见 caveats）· 仅供研究, 非投资建议</div>")
    parts.append("</div></body></html>")
    return "".join(parts)


def _render_walkforward_card(wf: dict) -> str:
    """真外推基线卡: 与 OOF KPI 并列展示, 强调不可混比。"""
    if not wf.get("ok", True) and wf.get("error"):
        return (
            "<div class='card caveat'><b>Walk-forward 真外推基线</b>："
            f"<span class='neg'>失败</span> — {html.escape(str(wf.get('error')))}"
            "</div>"
        )
    parts = [
        "<div class='card'><b>Walk-forward 真外推基线</b>"
        "<div class='caveat' style='margin:6px 0 10px'>"
        "仅过去窗拟合 → 未来窗部署门控；成交/胜率/期望以此为准，"
        "勿与上方研究 OOF / 部署路径成交数混比。"
        "</div><div class='grid'>",
    ]
    parts.append(_kpi("WF开仓数", str(wf.get("n_opened_trades", "—"))))
    parts.append(_kpi("WF胜率", _fmt(wf.get("win_rate"), pct=True)))
    parts.append(
        _kpi("WF总收益", _fmt(wf.get("total_return"), pct=True),
             cls=_cls(wf.get("total_return") or 0))
    )
    parts.append(
        _kpi("WF最大回撤", _fmt(wf.get("max_drawdown"), pct=True), cls="neg")
    )
    parts.append(_kpi("WF测试AUC", _fmt(wf.get("test_auc"))))
    parts.append(_kpi("WF阈值(部署)", _fmt(wf.get("prob_threshold_effective"))))
    parts.append(_kpi("训练事件", str(wf.get("n_train_events", "—"))))
    parts.append(_kpi("测试事件", str(wf.get("n_test_events", "—"))))
    parts.append("</div>")
    if wf.get("backtest_start"):
        end = wf.get("backtest_end") or "面板末"
        parts.append(
            f"<div class='caveat'>测试窗: {html.escape(str(wf['backtest_start']))}"
            f" → {html.escape(str(end))}"
            f" · embargo_bars={html.escape(str(wf.get('embargo_bars', 0)))}</div>"
        )
    if wf.get("summary_path"):
        parts.append(
            f"<div class='caveat'>明细: {html.escape(str(wf['summary_path']))}</div>"
        )
    parts.append("</div>")
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
    p.append(_kpi("每笔Sharpe", _fmt(bt["sharpe"]), cls=_cls(bt["sharpe"])))
    # 权益曲线夏普(增量字段; 缺省时不崩旧 summary)
    eq_ann = bt.get("sharpe_equity_annualized")
    if eq_ann is None:
        eq_ann = bt.get("sharpe_equity")
    p.append(_kpi("权益Sharpe年化", _fmt(eq_ann), cls=_cls(eq_ann if eq_ann is not None else 0.0)))
    if bt.get("mark_to_market") and bt.get("sharpe_equity_mtm_annualized") is not None:
        mtm_ann = bt["sharpe_equity_mtm_annualized"]
        p.append(_kpi("盯市权益Sharpe年化", _fmt(mtm_ann), cls=_cls(mtm_ann)))
    p.append(_kpi("总收益", _fmt(bt["total_return"], pct=True), cls=_cls(bt["total_return"])))
    p.append(_kpi("最大回撤", _fmt(bt["max_drawdown"], pct=True), cls="neg"))
    p.append(_kpi("Calmar", _fmt(bt["calmar"])))
    p.append(_kpi("胜率", _fmt(bt["win_rate"], pct=True)))
    p.append(_kpi("交易数(研究OOF)", str(bt["n_trades"])))
    bt_dep = d.get("backtest_deploy") or {}
    if bt_dep:
        p.append(_kpi("交易数(部署·偏乐观·勿拍板)", str(bt_dep.get("n_trades", "?"))))
    thr_r = d.get("prob_threshold_research")
    thr_eff = d.get("prob_threshold_effective")
    if thr_r is not None:
        p.append(_kpi("阈值(研究CF)", _fmt(thr_r)))
    if thr_eff is not None:
        p.append(_kpi("阈值(部署/decide)", _fmt(thr_eff)))
    p.append("</div></div>")
    p.append(
        "<div class='card caveat'><b>成交数/阈值口径</b>："
        "「研究OOF」用交叉拟合概率 + 阈值(研究CF)；"
        "「部署路径」与 decide/serve 同出分 + 阈值(部署/decide)；"
        "二者不可直接对比成交数。"
        "train_and_validate 报告窗部署回测仍可能偏乐观，真外推看下方 Walk-forward 卡"
        "（或 scripts/btc_walkforward_summary.py）。"
        "</div>"
    )

    # Walk-forward 真外推基线(若本次联跑启用)
    wf = d.get("walkforward")
    if wf:
        p.append(_render_walkforward_card(wf))

    # 降级 / degradations(透明度纪律)
    deg = list(d.get("degradations") or [])
    if deg:
        p.append("<div class='card caveat'><b>Degradations</b><ul>")
        for tag in deg:
            p.append(f"<li>{html.escape(str(tag))}</li>")
        p.append("</ul></div>")

    # 专家对比表(高亮最优 AUC; 伪 OOF 不参与 best)
    reps = d["expert_reports"]
    best_auc = max(
        (r.get("auc", float("nan")) for r in reps.values()
         if not r.get("pseudo_oof") and not np.isnan(r.get("auc", float("nan")))),
        default=float("nan"),
    )
    p.append("<div class='card'><table><thead><tr><th>专家</th><th>AUC</th>"
             "<th>Brier</th><th>准确率</th><th>样本</th></tr></thead><tbody>")
    for name, r in reps.items():
        auc = r.get("auc", float("nan"))
        is_pseudo = bool(r.get("pseudo_oof"))
        is_best = (not is_pseudo) and (not np.isnan(auc)) and abs(auc - best_auc) < 1e-9
        cls = " class='best'" if is_best else ""
        label = html.escape(name) + (" <i>(pseudo OOF)</i>" if is_pseudo else "")
        p.append(f"<tr><td>{label}</td><td{cls}>{_fmt(auc)}</td>"
                 f"<td>{_fmt(r.get('brier'))}</td><td>{_fmt(r.get('accuracy'), pct=True)}</td>"
                 f"<td>{r.get('n','—')}</td></tr>")
    p.append(f"<tr><td><b>集成(stacking)</b></td><td class='best'>{_fmt(ens.get('auc'))}</td>"
             f"<td>{_fmt(ens.get('brier'))}</td><td>{_fmt(ens.get('accuracy'), pct=True)}</td>"
             f"<td>{ens.get('n','—')}</td></tr>")
    p.append("</tbody></table></div>")

    # CPCV
    cp = d.get("cpcv")
    if cp:
        unit = cp.get("evaluation_unit", "combo")
        n_show = cp.get("n_combos", cp.get("n_paths", "—"))
        p.append("<div class='card'><div class='grid'>")
        p.append(_kpi(f"评估单元", html.escape(str(unit))))
        p.append(_kpi("组合数", str(n_show)))
        p.append(_kpi("组合夏普均值", _fmt(cp["mean_sharpe"]), cls=_cls(cp["mean_sharpe"])))
        p.append(_kpi("夏普标准差", _fmt(cp["std_sharpe"])))
        p.append(_kpi("去偏夏普 DSR", _fmt(cp["deflated_sharpe"]), best=True))
        p.append(_kpi("过拟合概率 PBO", _fmt(cp["pbo"]),
                      cls=("pos" if cp["pbo"] < 0.5 else "neg")))
        p.append("</div>")
        if cp.get("pbo_warning"):
            p.append("<div class='caveat'>PBO 配置数偏少，统计力不足，数值仅供参考。</div>")
        caveats = cp.get("caveats") or []
        if caveats:
            p.append("<div class='caveat'><b>CPCV caveats</b><ul>")
            for c in caveats:
                p.append(f"<li>{html.escape(str(c))}</li>")
            p.append("</ul></div>")
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
