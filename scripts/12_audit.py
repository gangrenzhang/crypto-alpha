"""阶段12: 闭环完整性在线体检(CPU 秒级, 不依赖显卡)。

在每次正式训练前后跑一遍, 用控制实验+不变量+对账快速回答:
"标注/回测/验证闭环的代码逻辑到底可不可信?"

覆盖:
1. 正对照   —— 真信号能否被学到(排除"永远随机"的假阴性);
2. 置换基线 —— 打乱标签后是否塌回随机(排除堆叠/校准/回测偷看测试集);
3. 时移不变 —— 逻辑不依赖绝对时间;
4. CV 不变量 —— 净化零重叠、禁运有间隔;
5. 回测对账 —— 权益可复算、并发敞口有上限、成本单调、组合不高估;
6. 全链路空对照 —— 随机游走喂满全链路, AUC 必须≈0.5;
7. 复现性   —— 同 seed 两次训练结果一致。

用法:
    python scripts/12_audit.py            # 跑默认体检
    python scripts/12_audit.py --json     # 额外输出 JSON 到 artifacts/audit_report.json
退出码: 有任一 FAIL 时非零, 可直接接入 CI。
"""
import _bootstrap  # noqa: F401

import argparse
import json
import sys

from crypto_alpha.config import Config
from crypto_alpha.diagnostics import audit_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="闭环完整性在线体检")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--bars", type=int, default=6000, help="空对照随机游走 bar 数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="额外写出 JSON 报告")
    args = ap.parse_args()

    cfg = Config.load()
    print("=" * 68)
    print("闭环完整性体检 (CPU / 仅 GBDT / 无显卡依赖)")
    print("=" * 68)

    results = audit_pipeline(cfg, symbol=args.symbol, n_bars=args.bars, seed=args.seed)

    n_fail = 0
    n_info = 0
    for r in results:
        icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "INFO": "[INFO]"}[r.status]
        print(f"\n{icon} {r.name}")
        if r.detail:
            print(f"       {json.dumps(r.detail, ensure_ascii=False, default=str)}")
        if r.note:
            print(f"       note: {r.note}")
        if r.status == "FAIL":
            n_fail += 1
        elif r.status == "INFO":
            n_info += 1

    n_pass = len(results) - n_fail - n_info
    print("\n" + "=" * 68)
    print(f"结果: {n_pass} PASS / {n_fail} FAIL / {n_info} INFO (共 {len(results)} 项)")
    print("=" * 68)

    if args.json:
        out = cfg.artifacts_dir / "audit_report.json"
        payload = [
            {"name": r.name, "status": r.status, "detail": r.detail, "note": r.note}
            for r in results
        ]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"[ok] JSON 报告 -> {out}")

    if n_fail:
        print("\n[FAIL] 存在闭环完整性问题, 请勿据此训练/上线, 先修复上面标 FAIL 的环节。")
        return 1
    print("\n[OK] 所有可判定项通过。闭环逻辑在这些维度上自洽。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
