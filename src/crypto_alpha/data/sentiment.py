"""金融/加密情绪打分: 英文(CryptoBERT/FinBERT) + 中文金融模型 + 中英双模型融合 + 词典兜底。

设计(专业实现要点):
- 统一接口 SentimentScorer.score(texts) -> np.ndarray, 值域 [-1, 1](正=看多)。
- Transformer 后端用**概率加权的有符期望**(而非 argmax): score = P(看多) - P(看空)。
  通用标签映射自动识别 bull/positive/正面、bear/negative/负面、neutral/中性;
  对 LABEL_0/1 这类无语义标签, 支持显式 label_map, 并对二/三分类提供有序兜底。
- **中英双模型融合(MultilingualSentiment)**: 按文本语言路由到对应母语模型(英->CryptoBERT,
  中->中文金融模型), 混合语言文本按权重融合两模型; 各语言独立回退到词典。
- 批量推理 + 磁盘缓存(按 模型名+文本 哈希) + 设备自适应(cuda/cpu)。
- 任何缺依赖/加载失败都优雅回退到词典后端, 不阻断新闻流水线。

默认模型:
    cryptobert -> ElKulako/cryptobert                         (英文, 加密社媒/新闻)
    finbert    -> ProsusAI/finbert                            (英文, 通用金融)
    chinese    -> IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment (中文情绪)
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

_DEFAULT_MODELS = {
    "cryptobert": "ElKulako/cryptobert",
    "finbert": "ProsusAI/finbert",
    "chinese": "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
}


def detect_lang(text: str) -> str:
    """按 CJK 字符占比粗判语言: 'zh' / 'en' / 'mixed'。"""
    if not text:
        return "en"
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = cjk + latin
    if total == 0:
        return "en"
    zr = cjk / total
    if zr >= 0.6:
        return "zh"
    if zr <= 0.15:
        return "en"
    return "mixed"


class SentimentScorer(ABC):
    name = "base"

    @abstractmethod
    def score(self, texts: list[str]) -> np.ndarray:
        """返回与输入等长、值域 [-1,1] 的情绪分数(正=看多, 负=看空)。"""
        ...


# --------------------------------------------------------------------------
# 词典后端(零依赖, 作为兜底)
# --------------------------------------------------------------------------
class LexiconSentiment(SentimentScorer):
    name = "lexicon"

    def score(self, texts: list[str]) -> np.ndarray:
        from .news import _score_sentiment

        return np.array([_score_sentiment(t or "") for t in texts], dtype=float)


# --------------------------------------------------------------------------
# Transformer 后端(CryptoBERT / FinBERT)
# --------------------------------------------------------------------------
def _label_sign(label: str) -> float:
    """把模型标签映射为符号: 看多=+1, 看空=-1, 中性=0。兼容中英多种命名。"""
    s = str(label).lower()
    if any(k in s for k in ("bull", "positive", "pos", "上涨", "利多", "利好", "看涨", "正面", "积极")):
        return 1.0
    if any(k in s for k in ("bear", "negative", "neg", "下跌", "利空", "看跌", "负面", "消极")):
        return -1.0
    return 0.0


def _resolve_signs(id2label: dict, label_map: dict | None) -> np.ndarray:
    """把 id2label 解析为每类符号。优先显式 label_map; 否则自动识别;
    对全 0(无语义 LABEL_0/1) 的二/三分类提供有序兜底 [-1(,0),1]。"""
    n = len(id2label)
    if label_map:
        signs = np.array([float(label_map.get(id2label[i], _label_sign(id2label[i]))) for i in range(n)])
    else:
        signs = np.array([_label_sign(id2label[i]) for i in range(n)], dtype=float)
    if np.allclose(signs, 0.0):  # 无法识别 => 按升序假定 neg->pos
        if n == 2:
            signs = np.array([-1.0, 1.0])
        elif n == 3:
            signs = np.array([-1.0, 0.0, 1.0])
        print(f"[warn] 情绪标签无语义({id2label}); 采用有序兜底 signs={signs.tolist()}。")
    return signs


class TransformerSentiment(SentimentScorer):
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32,
                 max_length: int = 128, cache_path: str | None = None, label_map: dict | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = f"transformer:{model_name}"
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

        # 预计算每个类别索引的符号(基于 id2label + 可选显式映射 + 有序兜底)
        self.signs = _resolve_signs(self.model.config.id2label, label_map)

        # 磁盘缓存(可选)
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, float] = {}
        if self._cache_path and self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def _key(self, text: str) -> str:
        h = hashlib.sha1(f"{self.model_name}||{text}".encode("utf-8")).hexdigest()
        return h

    def _flush_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except Exception:
            pass

    def score(self, texts: list[str]) -> np.ndarray:
        import torch

        out = np.zeros(len(texts), dtype=float)
        todo_idx, todo_txt = [], []
        for i, t in enumerate(texts):
            t = t or ""
            k = self._key(t)
            if k in self._cache:
                out[i] = self._cache[k]
            else:
                todo_idx.append(i)
                todo_txt.append(t)

        for b in range(0, len(todo_txt), self.batch_size):
            batch = todo_txt[b : b + self.batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            scores = probs @ self.signs  # 概率加权的有符期望 => P(多)-P(空)
            for j, s in enumerate(scores):
                gi = todo_idx[b + j]
                out[gi] = float(s)
                self._cache[self._key(todo_txt[b + j])] = float(s)

        self._flush_cache()
        return np.clip(out, -1.0, 1.0)


# --------------------------------------------------------------------------
# 中英双模型融合: 按语言路由, 混合语言按权重融合
# --------------------------------------------------------------------------
class MultilingualSentiment(SentimentScorer):
    def __init__(self, en_scorer: SentimentScorer, zh_scorer: SentimentScorer,
                 fusion: str = "route", w_en: float = 0.5, w_zh: float = 0.5):
        self.name = f"multilingual({en_scorer.name}+{zh_scorer.name},{fusion})"
        self.en = en_scorer
        self.zh = zh_scorer
        self.fusion = fusion            # route(按语言路由) / average(两模型全量平均)
        self.w_en, self.w_zh = float(w_en), float(w_zh)

    def score(self, texts: list[str]) -> np.ndarray:
        texts = [t or "" for t in texts]
        n = len(texts)
        out = np.zeros(n, dtype=float)
        if n == 0:
            return out

        if self.fusion == "average":  # 两模型对所有文本打分后加权平均
            se, sz = self.en.score(texts), self.zh.score(texts)
            return np.clip(self.w_en * se + self.w_zh * sz, -1, 1)

        # route: 按语言分派; mixed 两模型融合
        langs = [detect_lang(t) for t in texts]
        en_idx = [i for i, l in enumerate(langs) if l in ("en", "mixed")]
        zh_idx = [i for i, l in enumerate(langs) if l in ("zh", "mixed")]
        en_map = dict(zip(en_idx, self.en.score([texts[i] for i in en_idx]))) if en_idx else {}
        zh_map = dict(zip(zh_idx, self.zh.score([texts[i] for i in zh_idx]))) if zh_idx else {}
        for i, l in enumerate(langs):
            if l == "en":
                out[i] = en_map.get(i, 0.0)
            elif l == "zh":
                out[i] = zh_map.get(i, 0.0)
            else:  # mixed: 加权融合
                out[i] = self.w_en * en_map.get(i, 0.0) + self.w_zh * zh_map.get(i, 0.0)
        return np.clip(out, -1, 1)


# --------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------
def _build_single_backend(cfg, backend: str, model_name: str | None = None,
                          label_map: dict | None = None) -> SentimentScorer:
    """构造单一后端打分器(lexicon 或某个 Transformer 模型); 失败回退词典。"""
    if backend == "lexicon":
        return LexiconSentiment()
    scfg = cfg["news"].get("sentiment", {}) or {}
    name = model_name or _DEFAULT_MODELS.get(backend)
    if not name:
        print(f"[warn] 未知情绪后端 {backend}, 回退词典。")
        return LexiconSentiment()
    cache_path = str(cfg.artifacts_dir / "sentiment_cache.json") if scfg.get("cache", True) else None
    try:
        return TransformerSentiment(
            model_name=name,
            device=scfg.get("device", "auto"),
            batch_size=int(scfg.get("batch_size", 32)),
            max_length=int(scfg.get("max_length", 128)),
            cache_path=cache_path,
            label_map=label_map,
        )
    except Exception as e:
        print(f"[warn] 情绪模型 {name} 加载失败({e}); 回退词典后端。")
        return LexiconSentiment()


def build_scorer(cfg) -> SentimentScorer:
    """按 config.news.sentiment 构造情绪打分器(支持中英双模型融合); 失败自动回退词典。"""
    scfg = cfg["news"].get("sentiment", {}) or {}
    backend = scfg.get("backend", "lexicon")

    if backend == "multilingual":
        en = _build_single_backend(cfg, scfg.get("en_backend", "cryptobert"),
                                    scfg.get("en_model_name"), scfg.get("en_label_map"))
        zh = _build_single_backend(cfg, scfg.get("zh_backend", "chinese"),
                                   scfg.get("zh_model_name"), scfg.get("zh_label_map"))
        fw = scfg.get("fusion_weights", {}) or {}
        return MultilingualSentiment(en, zh, fusion=scfg.get("fusion", "route"),
                                     w_en=fw.get("en", 0.5), w_zh=fw.get("zh", 0.5))

    return _build_single_backend(cfg, backend, scfg.get("model_name"), scfg.get("label_map"))
