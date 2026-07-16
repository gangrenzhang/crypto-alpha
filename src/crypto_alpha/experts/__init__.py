from .base import BaseExpert
from .gbdt import GBDTExpert
from .deep_ts import DeepTSExpert
from .tsfm import TSFMExpert
from .llm import LLMExpert

EXPERT_REGISTRY = {
    "gbdt": GBDTExpert,
    "deep_ts": DeepTSExpert,
    "tsfm": TSFMExpert,
    "llm": LLMExpert,
}

__all__ = ["BaseExpert", "GBDTExpert", "DeepTSExpert", "TSFMExpert", "LLMExpert", "EXPERT_REGISTRY"]
