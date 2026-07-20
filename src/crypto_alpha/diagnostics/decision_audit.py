"""决策可复盘快照: 把「这次信号背后的数据窗/配置/阈值」写入决策 JSON。

不持久化完整模型权重(体积大); 提供 fingerprint 使事后至少能核对:
- 用的哪段行情窗
- 哪套超参指纹
- 部署阈值与 degradations
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .experiments import build_experiment_fingerprint, _fingerprint


def _series_hash(values) -> str:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # 子采样避免超长序列哈希过慢
    if len(arr) > 4096:
        idx = np.linspace(0, len(arr) - 1, 4096).astype(int)
        arr = arr[idx]
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return digest[:16]


def build_decision_audit(
    cfg,
    *,
    panel,
    feature_cols: list[str],
    trained: dict | None = None,
    degradations: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造可挂到决策 JSON 的 ``audit`` 字段。"""
    trained = trained or {}
    close = panel["close"] if panel is not None and "close" in getattr(panel, "columns", []) else None
    n_bars = int(len(panel)) if panel is not None else 0
    start = str(panel.index[0]) if n_bars else None
    end = str(panel.index[-1]) if n_bars else None
    fp_body = build_experiment_fingerprint(cfg)
    audit = {
        "config_fingerprint": _fingerprint(fp_body),
        "data_window": {"start": start, "end": end, "n_bars": n_bars},
        "data_window_hash": _series_hash(close.values) if close is not None and n_bars else None,
        "feature_cols_hash": _fingerprint({"cols": list(feature_cols)}),
        "n_feature_cols": len(feature_cols),
        "prob_threshold_effective": trained.get("prob_threshold_effective"),
        "prob_threshold_research": trained.get("prob_threshold_research"),
        "cusum_full_sampling": bool(trained.get("cusum_full_sampling", False)),
        "data_source": trained.get("data_mode") or trained.get("data_source"),
        "degradations": list(degradations or trained.get("degradations") or []),
        "experts": [
            getattr(e, "name", str(e))
            for e in (getattr(trained.get("ensemble"), "experts", None) or [])
        ],
    }
    if extra:
        audit.update(extra)
    return audit


def attach_decision_audit(decision: dict, audit: dict[str, Any]) -> dict:
    """原地写入 audit, 返回 decision。"""
    decision["audit"] = audit
    # 顶层也留轻量指纹, 便于日志 grep
    decision["config_fingerprint"] = audit.get("config_fingerprint")
    decision["data_window_hash"] = audit.get("data_window_hash")
    return decision
