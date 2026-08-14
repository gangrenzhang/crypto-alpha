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


# 训练依赖数据开关默认值(与 config/training_data.yaml 对齐)
TRAINING_DATA_DEFAULTS: dict[str, bool] = {
    "ohlcv": True,
    "funding": True,
    "open_interest": True,
    "mtf": True,
    "macro_calendar": True,
    "news": False,
    "liquidations": False,
}

_TRAINING_DATA_BOOL_KEYS = tuple(TRAINING_DATA_DEFAULTS.keys())


def resolve_training_data_toggles(
    raw: dict[str, Any] | None,
    *,
    root: Path | None = None,
    training_data_path: str | os.PathLike | None = None,
) -> dict[str, bool]:
    """合并训练数据开关: 默认 ← config.yaml.training_data ← training_data.yaml。

    独立文件 ``config/training_data.yaml`` 优先(便于只改 true/false)。
    """
    out = dict(TRAINING_DATA_DEFAULTS)
    raw = raw or {}
    inline = raw.get("training_data")
    if isinstance(inline, dict):
        for k in _TRAINING_DATA_BOOL_KEYS:
            if k in inline and inline[k] is not None:
                out[k] = bool(inline[k])

    root = root or _project_root()
    path = (
        Path(training_data_path)
        if training_data_path
        else root / "config" / "training_data.yaml"
    )
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            file_td = yaml.safe_load(f) or {}
        if not isinstance(file_td, dict):
            raise ValueError(f"{path}: 顶层须为 mapping(键=数据源, 值=bool)")
        for k in _TRAINING_DATA_BOOL_KEYS:
            if k in file_td and file_td[k] is not None:
                out[k] = bool(file_td[k])

    if not out["ohlcv"]:
        raise ValueError(
            "training_data.ohlcv=false 不可用: 主周期 OHLCV 是训练硬依赖"
            "（文件 data/raw/{SYMBOL}__{timeframe}.parquet）。请改回 true。"
        )
    return out


def apply_training_data_toggles(
    raw: dict[str, Any],
    *,
    root: Path | None = None,
    training_data_path: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """把 training_data 开关同步到遗留散落键, 保证旧代码路径一致生效。"""
    td = resolve_training_data_toggles(
        raw, root=root, training_data_path=training_data_path
    )
    raw = raw or {}

    # 若遗留键与开关板冲突, 明确告警(开关板胜出), 避免「改了 config.yaml 却不生效」
    conflicts: list[str] = []
    news = raw.get("news") if isinstance(raw.get("news"), dict) else {}
    macro = raw.get("macro_calendar") if isinstance(raw.get("macro_calendar"), dict) else {}
    feat = raw.get("features") if isinstance(raw.get("features"), dict) else {}
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    checks = [
        ("news.as_feature", news.get("as_feature"), td["news"]),
        ("macro_calendar.as_feature", macro.get("as_feature"), td["macro_calendar"]),
        ("features.mtf_enabled", feat.get("mtf_enabled"), td["mtf"]),
        ("data.fetch_liquidations", data.get("fetch_liquidations"), td["liquidations"]),
        ("features.use_funding", feat.get("use_funding"), td["funding"]),
        ("features.use_open_interest", feat.get("use_open_interest"), td["open_interest"]),
        ("features.use_liquidations", feat.get("use_liquidations"), td["liquidations"]),
    ]
    for label, old, new in checks:
        if old is not None and bool(old) != bool(new):
            conflicts.append(f"{label}={old!r} → training_data 覆盖为 {new!r}")
    if conflicts:
        import warnings

        warnings.warn(
            "training_data.yaml 覆盖了 config.yaml 中不一致的遗留开关: "
            + "; ".join(conflicts)
            + "。请只改 config/training_data.yaml, 或让两边保持一致。",
            UserWarning,
            stacklevel=2,
        )

    raw["training_data"] = dict(td)

    feat = raw.setdefault("features", {})
    if not isinstance(feat, dict):
        raise ValueError("config.features 须为 mapping")
    feat["mtf_enabled"] = bool(td["mtf"])
    feat["use_funding"] = bool(td["funding"])
    feat["use_open_interest"] = bool(td["open_interest"])
    feat["use_liquidations"] = bool(td["liquidations"])

    news = raw.setdefault("news", {})
    if not isinstance(news, dict):
        raise ValueError("config.news 须为 mapping")
    news["as_feature"] = bool(td["news"])

    macro = raw.setdefault("macro_calendar", {})
    if not isinstance(macro, dict):
        raise ValueError("config.macro_calendar 须为 mapping")
    macro["as_feature"] = bool(td["macro_calendar"])

    data = raw.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("config.data 须为 mapping")
    data["fetch_liquidations"] = bool(td["liquidations"])

    return raw


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
            raw = yaml.safe_load(f) or {}
        # 训练依赖开关: config/training_data.yaml(优先) + 可选 inline training_data
        td_path = cfg_path.parent / "training_data.yaml"
        apply_training_data_toggles(raw, root=root, training_data_path=td_path)
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
        # 主/辅周期合法性: 辅必须严格粗于主(方案B); 未知 timeframe fail-fast
        from .data.fetch import supported_timeframes, timeframe_delta

        data = (raw or {}).get("data") or {}
        main_tf = str(data.get("timeframe") or "1h")
        if main_tf not in supported_timeframes():
            raise ValueError(
                f"data.timeframe={main_tf!r} 不受支持; 可选 {supported_timeframes()}"
            )
        for aux_tf in list(data.get("aux_timeframes") or []):
            if not aux_tf:
                continue
            aux_tf = str(aux_tf)
            if aux_tf not in supported_timeframes():
                raise ValueError(
                    f"data.aux_timeframes 含不受支持的 {aux_tf!r}; "
                    f"可选 {supported_timeframes()}"
                )
            if timeframe_delta(aux_tf) <= timeframe_delta(main_tf):
                raise ValueError(
                    f"辅周期 {aux_tf} 必须严格粗于主周期 {main_tf}(方案B)"
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
