"""阶段7: 实时决策服务 —— 定时出决策并通过 Telegram/控制台播报。

用法:
    python scripts/07_serve.py --once     # 跑一轮(适合 cron / Windows 任务计划)
    python scripts/07_serve.py --loop     # 常驻循环, 按 config.serve.poll_seconds 轮询

Telegram 配置:
    1) 在 config.yaml 设 serve.telegram.enabled: true
    2) 设置环境变量:
       PowerShell:  $env:TELEGRAM_BOT_TOKEN="xxxx"; $env:TELEGRAM_CHAT_ID="123456"
    未配置时自动回退为控制台打印。默认只在 LONG/SHORT 时播报(HOLD 静默)。
"""
import _bootstrap  # noqa: F401

import argparse

from crypto_alpha.config import Config
from crypto_alpha.serve import build_notifier, DecisionService


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true", help="只跑一轮后退出")
    g.add_argument("--loop", action="store_true", help="常驻循环轮询")
    args = parser.parse_args()

    cfg = Config.load()
    notifier = build_notifier(cfg)
    service = DecisionService(cfg, notifier)

    if args.loop:
        service.run_forever()
    else:
        # 默认 --once
        service.train_all()
        decisions = service.run_once()
        print(f"\n[done] 本轮产生 {len(decisions)} 条决策。")


if __name__ == "__main__":
    main()
