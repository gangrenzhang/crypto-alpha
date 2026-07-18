"""阶段6: 输出最新一根 bar 的结构化交易决策(做多/做空/观望 + 概率 + 止损止盈 + 仓位)。

这就是系统的 "使用方式": 输入=最新市场状态(自动计算), 输出=JSON 决策 + 可读文字描述。
可扩展为定时任务 + Telegram/邮件推送(见 README)。

落盘:
  artifacts/decision_<SYMBOL>.json  — 单币种完整决策(含 description)
  artifacts/decision_<SYMBOL>.txt   — 仅可读文案
  artifacts/decisions_latest.json   — 本轮全部币种
  artifacts/decisions_latest.txt    — 本轮全部可读文案汇总
"""
import _bootstrap  # noqa: F401

import json

from crypto_alpha.config import Config
from crypto_alpha.pipeline import prepare_dataset, train_and_validate, latest_decision


def main():
    cfg = Config.load()
    decisions = []
    text_blocks = []
    for symbol in cfg["data"]["symbols"]:
        # for_decide: 先增量到当下已收盘最后一根 K 线, 再训练/决策
        ds = prepare_dataset(cfg, symbol, for_decide=True)
        trained = train_and_validate(cfg, ds)
        d = latest_decision(cfg, ds, trained)
        decisions.append(d)
        desc = d.get("description") or ""
        text_blocks.append(desc)

        print(json.dumps(d, ensure_ascii=False, indent=2))
        print("\n--- 可读描述 ---\n")
        print(desc)
        print()

        stem = symbol.replace("/", "_")
        json_path = cfg.artifacts_dir / f"decision_{stem}.json"
        txt_path = cfg.artifacts_dir / f"decision_{stem}.txt"
        json_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        txt_path.write_text(desc + "\n", encoding="utf-8")
        print(f"[ok] {json_path}")
        print(f"[ok] {txt_path}")

    all_json = cfg.artifacts_dir / "decisions_latest.json"
    all_txt = cfg.artifacts_dir / "decisions_latest.txt"
    all_json.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    all_txt.write_text("\n\n---\n\n".join(text_blocks) + "\n", encoding="utf-8")
    print(f"\n[ok] 汇总 JSON -> {all_json}")
    print(f"[ok] 汇总文案 -> {all_txt}")
    return decisions


if __name__ == "__main__":
    main()
