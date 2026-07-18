"""播报通道: Telegram(基于标准库 urllib, 零额外依赖) 与 控制台回退。"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd


def format_timestamp_beijing(ts) -> str | None:
    """K 线开盘时间戳 → 北京时间字符串(Asia/Shanghai, 含 +08:00)。"""
    if ts is None:
        return None
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return str(t.tz_convert("Asia/Shanghai"))


def enrich_decision_display(d: dict) -> dict:
    """补齐决策展示字段: ``close``(末根收盘价) + ``timestamp_beijing``。

    - ``timestamp`` 仍为 UTC 开盘时刻(与面板索引一致);
    - ``timestamp_beijing`` 为同一根 K 线的北京时间;
    - 若未显式给 ``close`` 但有 ``entry_price``(close_fill 入场=收盘), 回填 close。
    """
    if not isinstance(d, dict):
        return d
    ts = d.get("timestamp")
    if ts is not None:
        bj = format_timestamp_beijing(ts)
        if bj is not None:
            d["timestamp_beijing"] = bj
    if d.get("close") is None and d.get("entry_price") is not None:
        try:
            d["close"] = float(d["entry_price"])
        except (TypeError, ValueError):
            pass
    return d


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


def _time_lines(d: dict) -> str:
    """UTC + 北京时间 + 收盘价(有则展示)。"""
    lines = [f"时间(UTC): {d.get('timestamp')}"]
    if d.get("timestamp_beijing") is not None:
        lines.append(f"时间(北京): {d.get('timestamp_beijing')}")
    if d.get("close") is not None:
        lines.append(f"收盘价: {d.get('close')}")
    return "\n".join(lines)


def format_decision(d: dict) -> str:
    """把决策 JSON 格式化为可读播报文本。"""
    sig = d.get("signal", "HOLD")
    mode = d.get("data_mode_zh") or d.get("data_source")
    mode_line = f"\n数据口径: {mode}" if mode else ""
    if sig == "HOLD":
        head = f"[观望] {d.get('symbol')}"
        return (
            f"{head}\n概率={d.get('win_probability')}  ({_hold_reason_text(d)})\n"
            f"{_time_lines(d)}"
            f"{mode_line}"
        )
    arrow = "做多 LONG" if sig == "LONG" else "做空 SHORT"
    exec_a = d.get("execution_assumption") or "close_fill"
    return (
        f"[{arrow}] {d.get('symbol')}\n"
        f"盈利概率: {d.get('win_probability')}\n"
        f"收盘价:   {d.get('close', d.get('entry_price'))}\n"
        f"入场价:   {d.get('entry_price')}\n"
        f"止损:     {d.get('stop_loss')}\n"
        f"止盈:     {d.get('take_profit')}\n"
        f"建议仓位: {d.get('suggested_position_pct')}\n"
        f"ATR:      {d.get('atr')}\n"
        f"{_time_lines(d)}\n"
        f"执行假设: {exec_a}"
        f"{mode_line}"
    )


def attach_decision_description(d: dict) -> dict:
    """就地写入展示字段 + ``description`` 可读文案, 与 JSON 一并返回/落盘。"""
    if not isinstance(d, dict):
        return d
    enrich_decision_display(d)
    d["description"] = format_decision(d)
    return d


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
