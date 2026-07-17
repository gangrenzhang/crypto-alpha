"""播报通道: Telegram(基于标准库 urllib, 零额外依赖) 与 控制台回退。"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


def _hold_reason_text(d: dict) -> str:
    """按决策 reason / degradations 生成 HOLD 说明(勿一律写「低于阈值」)。"""
    reason = str(d.get("reason") or "")
    deg = list(d.get("degradations") or [])
    joined = " ".join(str(x) for x in deg)
    mapping = {
        "feature_schema_mismatch": "特征 schema 与训练不一致, 强制观望",
        "not_cusum_event": "非 CUSUM 事件 bar, 与训练开仓门控对齐",
        "no_valid_feature_bar": "无完整特征 bar",
        "low_confidence_conformal": "保形预测弃权(不自信)",
        "prob_below_threshold": "盈利概率未达入场阈值",
        "kelly_non_positive_after_cost": "扣成本后 Kelly 仓位非正",
        "no_side": "无有效主信号方向",
    }
    for key, text in mapping.items():
        if key in reason or key in joined:
            return text
    if d.get("confident") is False:
        return mapping["low_confidence_conformal"]
    if reason:
        return f"原因: {reason}"
    return "观望(未满足开仓条件)"


def format_decision(d: dict) -> str:
    """把决策 JSON 格式化为可读播报文本。"""
    sig = d.get("signal", "HOLD")
    if sig == "HOLD":
        head = f"[观望] {d.get('symbol')}"
        return (
            f"{head}\n概率={d.get('win_probability')}  ({_hold_reason_text(d)})\n"
            f"时间: {d.get('timestamp')}"
        )
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
