"""阶段4: 训练四专家 Stacking 集成 + Purged CV 无泄漏评估 + 校准 + 含成本回测。

可选 ``--walkforward``: 额外跑真外推基线(写入 artifacts/walkforward_*.json)。
"""
import argparse
import json

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.pipeline import prepare_dataset, train_and_validate, latest_decision
from crypto_alpha.pipeline.walkforward import (
    run_walkforward,
    walkforward_public_summary,
)


def main():
    ap = argparse.ArgumentParser(description="训练 + 回测 + 可选 walk-forward")
    ap.add_argument(
        "--walkforward", action="store_true",
        help="训练后对每个币种再跑 WF 真外推基线",
    )
    args = ap.parse_args()

    cfg = Config.load()
    summary = {}
    for symbol in cfg["data"]["symbols"]:
        print(f"\n===== {symbol} =====")
        ds = prepare_dataset(cfg, symbol)
        trained = train_and_validate(cfg, ds)

        print("[集成 OOF 概率报告]", json.dumps(trained["report"], ensure_ascii=False, indent=2))
        print("[各专家 OOF 报告]")
        for name, rep in trained["base_report"].items():
            print(f"  - {name}: AUC={rep['auc']:.3f} Brier={rep['brier']:.3f} Acc={rep['accuracy']:.3f}")
        print("[回测指标]", json.dumps(trained["backtest"]["metrics"], ensure_ascii=False, indent=2))

        # 训练用冷缓存; 决策若开启 refresh_before_decide 则另组「当下 tip」面板再推理
        # (不重训; 与 06_decide / serve 的 tip 语义对齐)
        if (
            bool(cfg["data"].get("refresh_before_decide", True))
            and not cfg["data"].get("use_synthetic", False)
        ):
            ds_dec = prepare_dataset(cfg, symbol, for_decide=True)
            decision = latest_decision(cfg, ds_dec, trained)
        else:
            decision = latest_decision(cfg, ds, trained)
        print("[最新决策]", json.dumps(decision, ensure_ascii=False, indent=2))
        if decision.get("description"):
            print("[决策可读描述]\n" + decision["description"])
        stem = symbol.replace("/", "_")
        (cfg.artifacts_dir / f"decision_{stem}.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (cfg.artifacts_dir / f"decision_{stem}.txt").write_text(
            (decision.get("description") or "") + "\n", encoding="utf-8",
        )

        # 保存净值曲线
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            eq = trained["backtest"]["equity"]
            plt.figure(figsize=(10, 4))
            eq.plot()
            plt.title(f"{symbol} OOF Equity Curve")
            plt.tight_layout()
            out = cfg.artifacts_dir / f"equity_{symbol.replace('/', '_')}.png"
            plt.savefig(out)
            plt.close()
            print(f"[ok] 净值曲线 -> {out}")
        except Exception as e:
            print(f"[warn] 绘图跳过: {e}")

        entry = {
            "report": trained["report"],
            "backtest": trained["backtest"]["metrics"],
            "decision": decision,
        }

        if args.walkforward:
            print(f"[walk-forward] {symbol} ...", flush=True)
            wf = run_walkforward(cfg, symbol, ds=ds)
            wf.pop("_traded_detail", None)
            wf.pop("_equity", None)
            pub = walkforward_public_summary(wf)
            wf_path = cfg.artifacts_dir / f"walkforward_{stem}.json"
            wf_path.write_text(
                json.dumps(pub, ensure_ascii=False, indent=2, default=float),
                encoding="utf-8",
            )
            entry["walkforward"] = {
                "n_opened_trades": pub.get("n_opened_trades"),
                "win_rate": pub.get("win_rate"),
                "total_return": pub.get("total_return"),
                "max_drawdown": pub.get("max_drawdown"),
                "summary_path": str(wf_path),
            }
            print(
                f"[walk-forward] 开仓={pub.get('n_opened_trades')} "
                f"胜率={pub.get('win_rate')} 收益={pub.get('total_return')} -> {wf_path}",
                flush=True,
            )

        summary[symbol] = entry

    out = cfg.artifacts_dir / "train_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[ok] 汇总 -> {out}")


if __name__ == "__main__":
    main()
