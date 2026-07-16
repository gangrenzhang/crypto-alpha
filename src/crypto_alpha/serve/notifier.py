"""播报通道: Telegram(基于标准库 urllib, 零额外依赖) 与 控制台回退。"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


def format_decision(d: dict) -> str:
    """把决策 JSON 格式化为可读播报文本。"""
    sig = d.get("signal", "HOLD")
    if sig == "HOLD":
        head = f"[观望] {d.get('symbol')}"
        return f"{head}\n概率={d.get('win_probability')}  (低于阈值, 建议不入场)\n时间: {d.get('timestamp')}"
    arrow = "做多 LONG" if sig == "LONG" else "做空 SHORT"
    return (
        f"[{arrow}] {d.get('symbol')}\n"
        f"盈利概率: {d.get('win_probability')}\n"
        f"入场价:   {d.get('entry_price')}\n"
        f"止损:     {d.get('stop_loss')}\n"
        f"止盈:     {d.get('take_profit')}\n"
        f"建议仓位: {d.get('suggested_position_pct')}\n"
        f"ATR:      {d.get('atr')}\n"
        f"时间: {d.get('timestamp')}"
    )


class ConsoleNotifier:
    """回退通道: 直接打印到控制台。"""

    enabled = True

    def send(self, text: str) -> bool:
        print("\n[NOTIFY]\n" + text + "\n")
        return True


class TelegramNotifier:
    """通过 Telegram Bot API 推送。token/chat_id 建议放环境变量, 勿硬编码。"""

    def __init__(self, bot_token: str, chat_id: str, timeout: float = 10.0):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout
        self.enabled = bool(bot_token and chat_id)

    def send(self, text: str) -> bool:
        if not self.enabled:
            print("[warn] Telegram 未配置(缺 token/chat_id), 跳过推送。")
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text}).encode()
        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                ok = json.loads(resp.read().decode()).get("ok", False)
                if not ok:
                    print("[warn] Telegram 返回非 ok。")
                return bool(ok)
        except Exception as e:  # 网络失败不应中断服务
            print(f"[warn] Telegram 推送失败: {e}")
            return False


def build_notifier(cfg) -> object:
    """按配置构造播报器; Telegram 未启用或缺凭证则回退到控制台。"""
    tg = cfg["serve"].get("telegram", {})
    if tg.get("enabled", False):
        token = os.environ.get(tg.get("bot_token_env", "TELEGRAM_BOT_TOKEN"), "")
        chat = os.environ.get(tg.get("chat_id_env", "TELEGRAM_CHAT_ID"), "")
        n = TelegramNotifier(token, chat)
        if n.enabled:
            return n
        print("[warn] Telegram 已启用但环境变量缺失, 回退到控制台播报。")
    return ConsoleNotifier()
