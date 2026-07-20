"""研究实验日志: 为 DSR 的 dsr_n_trials 提供可审计的下限。

DSR 去偏依赖「真实试过多少次策略/超参」。人工填写几乎必然低估。
本模块维护 artifacts/experiment_log.jsonl(append-only):
- 每次正式训练/CPCV 可写入一条指纹;
- ``resolve_dsr_n_trials`` = max(配置值, 日志条数, 本轮配置数)。

冒烟/integrity 诊断应关闭 ``validation.log_experiments``, 避免污染计数。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def experiment_log_path(artifacts_dir: Path | str) -> Path:
    return Path(artifacts_dir) / "experiment_log.jsonl"


def _fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_experiment_fingerprint(cfg) -> dict[str, Any]:
    """从 Config 抽取影响研究结论的关键旋钮(非全量 yaml, 避免路径噪声)。"""
    raw = cfg.raw if hasattr(cfg, "raw") else dict(cfg)
    data = raw.get("data") or {}
    lab = raw.get("labeling") or {}
    feat = raw.get("features") or {}
    ens = raw.get("ensemble") or {}
    bt = raw.get("backtest") or {}
    cal = raw.get("calibration") or {}
    exp = raw.get("experts") or {}
    return {
        "seed": (raw.get("project") or {}).get("random_seed"),
        "symbols": list(data.get("symbols") or []),
        "timeframe": data.get("timeframe"),
        "aux_timeframes": list(data.get("aux_timeframes") or []),
        "use_synthetic": bool(data.get("use_synthetic", False)),
        "primary_signal": lab.get("primary_signal"),
        "primary_lookback": lab.get("primary_lookback"),
        "pt_sl": list(lab.get("pt_sl") or []),
        "vertical_barrier_bars": lab.get("vertical_barrier_bars"),
        "barrier_vol": lab.get("barrier_vol"),
        "mtf_enabled": bool(feat.get("mtf_enabled", True)),
        "frac_diff_d": feat.get("frac_diff_d"),
        "news_as_feature": bool((raw.get("news") or {}).get("as_feature", False)),
        "experts_enabled": list(exp.get("enabled") or []),
        "meta_learner": ens.get("meta_learner"),
        "min_expert_auc": ens.get("min_expert_auc"),
        "prob_threshold": bt.get("prob_threshold"),
        "prob_threshold_mode": bt.get("prob_threshold_mode"),
        "prob_quantile": bt.get("prob_quantile"),
        "slippage_bps": bt.get("slippage_bps"),
        "slippage_vol_scale": bt.get("slippage_vol_scale"),
        "calib_method": cal.get("method"),
        "conformal_alpha": cal.get("conformal_alpha"),
    }


def count_experiments(artifacts_dir: Path | str) -> int:
    path = experiment_log_path(artifacts_dir)
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def append_experiment(
    artifacts_dir: Path | str,
    cfg,
    *,
    source: str = "train",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """追加一条实验记录; 返回写入的记录(含 fingerprint / n_after)。"""
    path = experiment_log_path(artifacts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fp_body = build_experiment_fingerprint(cfg)
    if extra:
        fp_body = {**fp_body, **extra}
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "fingerprint": _fingerprint(fp_body),
        "payload": fp_body,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    rec["n_after"] = count_experiments(artifacts_dir)
    return rec


def resolve_dsr_n_trials(cfg, *, n_configs: int = 1) -> tuple[int, list[str]]:
    """解析 DSR 用的 n_trials 下限。

    ``max(yaml dsr_n_trials, 日志条数, n_configs)``。
    若日志抬高了人工值, 写入 tags 供 caveats。
    """
    tags: list[str] = []
    vcfg = cfg["validation"] if hasattr(cfg, "__getitem__") else (cfg.get("validation") or {})
    base = int(vcfg.get("dsr_n_trials", 50) or 50)
    n_cfg = max(int(n_configs), 1)
    logged = 0
    try:
        arts = cfg.artifacts_dir if hasattr(cfg, "artifacts_dir") else None
        if arts is not None:
            logged = count_experiments(arts)
    except Exception:
        logged = 0
    n = max(base, logged, n_cfg)
    if logged > base:
        tags.append(f"dsr_n_trials_raised_by_experiment_log({logged}>{base})")
    return int(n), tags
