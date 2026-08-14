"""固定 WF 协议的标注/专家/特征块消融(审计 P1)。

窗口与生产 walkforward 一致(默认 train_start/test_start/test_end)。
**不改部署门控**; 验收指标: test AUC、校准后 max/std、n_confident、n_opened。
禁止用 OOF 成交数拍板。

用法:
  PYTHONPATH=src python scripts/23_wf_ablation.py
  PYTHONPATH=src python scripts/23_wf_ablation.py --only baseline,pt_sl_2_1,gbdt_only
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.pipeline import prepare_dataset
from crypto_alpha.pipeline.walkforward import run_walkforward, walkforward_public_summary


# 名称 → 对 cfg.raw 的覆盖(深合并一层 dict)
ABLATIONS: dict[str, dict] = {
    "baseline": {},
    "pt_sl_2_1": {"labeling": {"pt_sl": [2.0, 1.0]}},
    "pt_sl_1_2": {"labeling": {"pt_sl": [1.0, 2.0]}},
    "vb_24": {"labeling": {"vertical_barrier_bars": 24}},
    "vb_96": {"labeling": {"vertical_barrier_bars": 96}},
    "meanrev": {"labeling": {"primary_signal": "meanrev"}},
    "confluence_05": {"labeling": {"min_confluence": 0.5}},
    "gbdt_only": {"experts": {"enabled": ["gbdt"]}},
    "macro_off": {"training_data": {"macro_calendar": False}},
    "mtf_off": {"training_data": {"mtf": False}},
    "oi_off": {"training_data": {"open_interest": False}},
}


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _apply_training_data(cfg: Config) -> None:
    from crypto_alpha.config import apply_training_data_toggles

    # 消融写了 training_data 内联段时需再同步遗留键(无独立 yaml 覆盖)
    td_path = cfg.root / "config" / "_ablation_no_file.yaml"
    # 用不存在的 path → 仅用 raw.training_data + 默认
    apply_training_data_toggles(cfg.raw, training_data_path=td_path)


def run_one(name: str, overrides: dict, symbol: str = "BTC/USDT") -> dict:
    cfg = Config.load()
    cfg.raw["data"]["symbols"] = [symbol]
    cfg.raw["data"]["refresh_before_decide"] = False
    cfg.raw["data"]["incremental_update"] = False
    _deep_update(cfg.raw, overrides)
    if "training_data" in overrides:
        _apply_training_data(cfg)

    t0 = time.time()
    print(f"\n===== ablation={name} =====", flush=True)
    ds = prepare_dataset(cfg, symbol)
    wf = run_walkforward(cfg, symbol, ds=ds)
    pub = walkforward_public_summary(wf)
    gate = pub.get("gate_diagnostics") or {}
    cal = gate.get("calibrated_proba") or {}
    conf_gates = gate.get("gates") or {}
    row = {
        "ablation": name,
        "overrides": overrides,
        "elapsed_sec": round(time.time() - t0, 1),
        "n_train_events": pub.get("n_train_events"),
        "n_test_events": pub.get("n_test_events"),
        "train_auc": (pub.get("train_oof_report") or {}).get("auc"),
        "test_auc": (pub.get("test_report") or {}).get("auc"),
        "cal_max": cal.get("max"),
        "cal_std": cal.get("std"),
        "n_confident": conf_gates.get("n_confident"),
        "n_prob_ge_threshold": conf_gates.get("n_prob_ge_threshold"),
        "n_opened_trades": pub.get("n_opened_trades"),
        "win_rate": pub.get("win_rate"),
        "total_return": pub.get("total_return"),
        "prob_threshold_effective": pub.get("prob_threshold_effective"),
        "degradations": pub.get("degradations"),
    }
    print(
        f"[{name}] test_auc={row['test_auc']:.4f} cal_max={row['cal_max']} "
        f"n_conf={row['n_confident']} opened={row['n_opened_trades']} "
        f"({row['elapsed_sec']}s)",
        flush=True,
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔消融名; 空=全部")
    ap.add_argument("--symbol", default="BTC/USDT")
    args = ap.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(ABLATIONS)
    unknown = [n for n in names if n not in ABLATIONS]
    if unknown:
        raise SystemExit(f"未知消融: {unknown}; 可选={list(ABLATIONS)}")

    rows = []
    for name in names:
        rows.append(run_one(name, ABLATIONS[name], symbol=args.symbol))

    out = Path("artifacts") / "wf_ablation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # 过闸判定: 需同时具备可分性与过闸分布, 禁止「1 笔偶然过闸」触发放宽门控
    any_passable = any(
        (r.get("cal_max") or 0) >= 0.55
        and (r.get("n_confident") or 0) > 0
        and (r.get("test_auc") or 0) >= 0.55
        and (r.get("n_opened_trades") or 0) >= 20
        for r in rows
    )
    payload = {
        "protocol": "walk_forward_single_cut",
        "gate_unchanged": True,
        "p2_gate_tune_allowed": bool(any_passable),
        "note": (
            "仅当某消融 cal_max>=0.55 且 n_confident>0 时才允许调门控; "
            "否则保持 thr/margin 不变。"
        ),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n[ok] {out}  p2_gate_tune_allowed={any_passable}", flush=True)


if __name__ == "__main__":
    main()
