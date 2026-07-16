"""阶段6: 输出最新一根 bar 的结构化交易决策(做多/做空/观望 + 概率 + 止损止盈 + 仓位)。

这就是系统的 "使用方式": 输入=最新市场状态(自动计算), 输出=下面的 JSON 决策。
可扩展为定时任务 + Telegram/邮件推送(见 README)。
"""
import _bootstrap  # noqa: F401

import json

from crypto_alpha.config import Config
from crypto_alpha.pipeline import prepare_dataset, train_and_validate, latest_decision


def main():
    cfg = Config.load()
    decisions = []
    for symbol in cfg["data"]["symbols"]:
        ds = prepare_dataset(cfg, symbol)
        trained = train_and_validate(cfg, ds)
        d = latest_decision(cfg, ds, trained)
        decisions.append(d)
        print(json.dumps(d, ensure_ascii=False, indent=2))
    return decisions


if __name__ == "__main__":
    main()
