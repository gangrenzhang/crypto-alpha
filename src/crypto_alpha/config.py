"""全局配置加载与路径管理。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    # src/crypto_alpha/config.py -> 上溯三级到仓库根
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """对 config.yaml 的轻量封装, 支持点式/字典式访问。"""

    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = field(default_factory=_project_root)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        root = _project_root()
        cfg_path = Path(path) if path else root / "config" / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # 启动期 fail-fast: 未实现的 execution_assumption 不得静默进入流水线
        from .risk.sizing import resolve_execution_assumption

        resolve_execution_assumption((raw or {}).get("risk") or {})
        # barrier_vol=rv 时标签障碍与 decide(atr_14) 口径分裂 — 仅告警, 不阻断旧实验
        bv = str(((raw or {}).get("labeling") or {}).get("barrier_vol", "atr")).lower()
        if bv == "rv":
            import warnings

            warnings.warn(
                "labeling.barrier_vol='rv': 三重障碍宽度用已实现波动, 但 "
                "latest_decision/decide 仍按 atr_14×pt_sl 挂单; 训练与实盘止损"
                "口径不一致。默认/实盘请用 barrier_vol='atr'。",
                UserWarning,
                stacklevel=2,
            )
        return cls(raw=raw, root=root)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # --- 常用路径 ---
    @property
    def data_dir(self) -> Path:
        d = self.root / self.raw["project"]["data_dir"]
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def artifacts_dir(self) -> Path:
        d = self.root / self.raw["project"]["artifacts_dir"]
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["random_seed"])


def set_global_seed(seed: int) -> None:
    """统一设定随机种子, 保证可复现。"""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
