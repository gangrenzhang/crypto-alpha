"""阶段4: 训练四专家 Stacking 集成 + Purged CV 无泄漏评估 + 校准 + 含成本回测。"""
import _bootstrap  # noqa: F401

import json

from crypto_alpha.config import Config
from crypto_alpha.pipeline import prepare_dataset, train_and_validate, latest_decision


def main():
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

        summary[symbol] = {
            "report": trained["report"],
            "backtest": trained["backtest"]["metrics"],
            "decision": decision,
        }

    out = cfg.artifacts_dir / "train_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[ok] 汇总 -> {out}")


if __name__ == "__main__":
    main()
