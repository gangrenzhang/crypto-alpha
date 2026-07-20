"""阶段10: 一键"全专家(gbdt+deep_ts+tsfm+llm)联跑" + 生成自包含 HTML 结果面板。

对每个币种执行完整主干: 数据 -> 特征(含新闻) -> 三重障碍标注 -> 全专家 Stacking
集成 -> 概率校准 -> 含成本回测 -> 最新决策, 可选叠加 CPCV(DSR/PBO) / Walk-forward 真外推。
不可用的专家(缺依赖/无 GPU/无微调权重)自动跳过并在面板中标注原因。

用法(PowerShell):
    python scripts/10_run_all.py                       # 全专家(自动降级) + 生成面板
    python scripts/10_run_all.py --cpcv                # 额外跑 CPCV(较慢)
    python scripts/10_run_all.py --walkforward         # 额外跑 WF 真外推基线(推荐发布前)
    python scripts/10_run_all.py --experts gbdt tsfm   # 只跑指定专家
    python scripts/10_run_all.py --symbols BTC/USDT
    python scripts/10_run_all.py --open                # 生成后用默认浏览器打开

产物: artifacts/dashboard.html(自包含, 离线可看) + artifacts/run_all_summary.json
      (+ 可选 walkforward_<SYMBOL>.json)。
"""
import argparse
import json
import webbrowser

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.pipeline import run_all, build_dashboard, ALL_EXPERTS


def main():
    ap = argparse.ArgumentParser(description="全专家联跑 + HTML 结果面板")
    ap.add_argument("--experts", nargs="*", default=None,
                    help=f"要联跑的专家子集(默认全部: {' '.join(ALL_EXPERTS)})")
    ap.add_argument("--symbols", nargs="*", default=None, help="币种子集(默认取 config)")
    ap.add_argument("--cpcv", action="store_true", help="额外运行 CPCV 严谨评估(较慢)")
    ap.add_argument(
        "--walkforward", action="store_true",
        help="额外运行 walk-forward 真外推基线(部署同形; 成交/胜率以此为准)",
    )
    ap.add_argument("--open", action="store_true", help="生成后用浏览器打开面板")
    args = ap.parse_args()

    cfg = Config.load()
    experts = args.experts or ALL_EXPERTS
    symbols = args.symbols or cfg["data"]["symbols"]

    # --walkforward 显式打开; 未传则读 config enabled_in_run_all
    do_wf = True if args.walkforward else None
    results = run_all(
        cfg, symbols, experts, do_cpcv=args.cpcv, do_walkforward=do_wf,
    )

    skipped = results["meta"]["experts_skipped"]
    if skipped:
        print("\n[跳过的专家]")
        for name, why in skipped.items():
            print(f"  - {name}: {why}")

    html = build_dashboard(results, cfg)
    out_html = cfg.artifacts_dir / "dashboard.html"
    out_html.write_text(html, encoding="utf-8")

    # JSON 汇总(去掉 base64 图, 便于阅读/追踪)
    slim = {"meta": results["meta"], "symbols": {}}
    for s, d in results["symbols"].items():
        slim["symbols"][s] = {k: v for k, v in d.items() if k != "equity_b64"}
    out_json = cfg.artifacts_dir / "run_all_summary.json"
    out_json.write_text(json.dumps(slim, ensure_ascii=False, indent=2, default=float),
                        encoding="utf-8")

    print(f"\n[ok] 结果面板 -> {out_html}")
    print(f"[ok] JSON 汇总 -> {out_json}")
    if args.open:
        webbrowser.open(out_html.as_uri())


if __name__ == "__main__":
    main()
