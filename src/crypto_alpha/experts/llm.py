"""专家4: 大语言模型 (默认 Qwen2.5-32B-Instruct, QLoRA 微调)。

角色: 把 "结构化指标摘要 + 新闻/事件文本" 转成自然语言提示, 让 LLM 做
事件驱动的推理判断(该方向该不该执行), 覆盖纯数值模型看不到的信息维度。

概率来源(verbalizer): 不做自由文本生成再正则解析, 而是让模型回答 "1(会盈利)"
或 "0(不会盈利)", 推理时读取答案位置上 token "1" vs "0" 的 softmax 概率, 得到
连续的 P(盈利)。这样训练目标(SFT 到 1/0)与推理口径完全一致, 且概率可被下游校准。

显存: 32B 4-bit QLoRA 约 24–48GB; 若改配置为 72B 则约 48–70GB(单卡 80GB)。
依赖(按需安装): transformers peft bitsandbytes accelerate datasets
训练脚本: scripts/train_llm_qlora.py。集成路径 fit() 只加载 adapter, 不训练。

伪 OOF: ``pseudo_oof=True`` — 不得假装折内重训。Stacking 默认
``exclude_pseudo_oof_from_meta=true`` 将其排除出元学习器, 分数仅保留诊断。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseExpert

#: 用于概率化的正/负 verbalizer token
POS_TOKEN = "1"
NEG_TOKEN = "0"

_SUMMARY_COLS = [
    "ret_14", "mom_28", "rsi_14", "macd_hist", "zscore_28", "vol_14",
    "funding_z", "oi_change", "atr_norm", "news_sentiment", "news_corroboration",
]

_SYSTEM = "你是严谨的加密货币交易风控助手, 依据市场指标与方向判断该笔交易是否会盈利。"


def build_prompt(row: pd.Series, side: int, news: str = "") -> str:
    """把一行特征 + 方向 + 可选新闻拼成用户提示(训练与推理共用)。"""
    parts = [f"{c}={row[c]:.4f}" for c in _SUMMARY_COLS if c in row and pd.notna(row[c])]
    direction = "做多(LONG)" if side > 0 else "做空(SHORT)"
    news_block = f"\n近期新闻/事件: {news}" if news else ""
    return (
        f"指标: {', '.join(parts)}{news_block}\n"
        f"拟执行方向: {direction}\n"
        f"该笔交易是否会盈利? 只回答 {POS_TOKEN}(会盈利) 或 {NEG_TOKEN}(不会盈利)。"
    )


def build_messages(row: pd.Series, side: int, news: str = "") -> list[dict]:
    """构造 chat 消息列表(system + user), 训练与推理复用同一格式避免分布漂移。"""
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": build_prompt(row, side, news)},
    ]


class LLMExpert(BaseExpert):
    name = "llm"
    needs_panel = True
    # fit 仅加载 adapter → 非折内重训; stacking 默认排除出 meta(见 ensemble 配置)
    pseudo_oof = True

    _news_df = None
    _news_buffer = 5
    _news_ttl = 24.0

    def set_news(self, news_df, buffer_minutes: int = 5, ttl_hours: float = 24.0) -> None:
        """提供新闻摘要面板(索引=发布时间), 推理时按事件时间做无泄漏 as-of 对齐。"""
        self._news_df = news_df
        self._news_buffer = int(buffer_minutes)
        self._news_ttl = float(ttl_hours)

    def clone(self) -> "LLMExpert":
        obj = super().clone()
        obj._news_df = self._news_df
        obj._news_buffer = self._news_buffer
        obj._news_ttl = self._news_ttl
        return obj

    def _load(self):
        try:
            import torch  # noqa
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
        except Exception as e:
            raise ImportError(
                "LLMExpert 需要 transformers/peft/bitsandbytes/accelerate。见 README。"
            ) from e
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        name = self.cfg.get("model_name", "Qwen/Qwen2.5-32B-Instruct")
        bnb = BitsAndBytesConfig(
            load_in_4bit=bool(self.cfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=bnb, device_map="auto"
        )
        adapter = self.cfg.get("adapter_path")
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        # verbalizer token id(取首 token, 兼容不同分词)
        self.pos_id = self.tok.encode(POS_TOKEN, add_special_tokens=False)[0]
        self.neg_id = self.tok.encode(NEG_TOKEN, add_special_tokens=False)[0]

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None):
        # 微调走独立脚本(QLoRA); 这里加载(已微调 adapter 的)模型做推理。
        self._load()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover - 需要 GPU 运行时
        import torch

        from ..data.news import align_news_asof

        panel = self._panel
        sides = X["side"].values if "side" in X.columns else np.ones(len(X))
        news_map = (
            align_news_asof(self._news_df, X.index, self._news_buffer, ttl_hours=self._news_ttl)
            if self._news_df is not None else {}
        )
        probs = np.full(len(X), 0.5, dtype=float)
        for i, ts in enumerate(X.index):
            row = panel.loc[ts] if (panel is not None and ts in panel.index) else X.iloc[i]
            news = news_map.get(ts, "")
            text = self.tok.apply_chat_template(
                build_messages(row, int(sides[i]), news), tokenize=False, add_generation_prompt=True
            )
            inputs = self.tok(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits[0, -1, :]
            pair = torch.softmax(torch.stack([logits[self.neg_id], logits[self.pos_id]]), dim=0)
            probs[i] = float(pair[1])
        return probs
