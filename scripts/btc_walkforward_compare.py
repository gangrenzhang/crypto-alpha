"""BTC walk-forward 门控对比(只读对比 / 可选双档重跑)。

切分与部署出分口径走 ``crypto_alpha.pipeline.walkforward``(与 summary 脚本同源)。

用法:
  python scripts/btc_walkforward_compare.py \\
      --a artifacts/btc_walkforward_summary.json \\
      --b artifacts/btc_walkforward_summary_current.json

  python scripts/btc_walkforward_compare.py --run
  python scripts/btc_walkforward_compare.py --run --skip-legacy --a artifacts/btc_walkforward_summary.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from crypto_alpha.config import Config
from crypto_alpha.pipeline.walkforward import (
    run_walkforward,
    walkforward_public_summary,
)

SYMBOL = "BTC/USDT"

# 门控优化前的「旧行为」快照(仅本脚本用于对比, 不改全局默认)
LEGACY_GATE_OVERRIDES = {
    "backtest": {
        "prob_threshold": 0.55,
        "prob_threshold_mode": "fixed",
        "raise_thr_on_inflate": False,
        "inflate_raise_quantile": None,
        "target_trade_rate": None,
    },
    "calibration": {
        "conformal_min_margin": 0.0,
    },
}

COMPARE_KEYS = [
    ("prob_threshold_effective", "有效阈值 thr"),
    ("n_opened_trades", "开仓笔数"),
    ("n_wins", "胜"),
    ("n_losses", "负"),
    ("win_rate", "胜率"),
    ("total_return", "累计收益"),
    ("final_capital", "1万→终值"),
    ("max_drawdown", "最大回撤"),
]


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _apply_overrides(cfg: Config, overrides: dict[str, Any] | None) -> Config:
    cfg2 = copy.copy(cfg)
    cfg2.raw = copy.deepcopy(cfg.raw)
    if overrides:
        _deep_update(cfg2.raw, overrides)
    return cfg2


def _gate_snapshot(cfg: Config) -> dict:
    b = cfg["backtest"]
    c = cfg["calibration"]
    return {
        "prob_threshold": b.get("prob_threshold"),
        "prob_threshold_mode": b.get("prob_threshold_mode"),
        "prob_quantile": b.get("prob_quantile"),
        "raise_thr_on_inflate": b.get("raise_thr_on_inflate"),
        "inflate_raise_quantile": b.get("inflate_raise_quantile"),
        "conformal_min_margin": c.get("conformal_min_margin"),
        "conformal_alpha": c.get("conformal_alpha"),
        "conformal_frac": c.get("conformal_frac"),
        "pass_rate_inflate_max": c.get("pass_rate_inflate_max"),
    }


def run_walkforward_once(cfg: Config, *, label: str) -> dict:
    """与 summary 同源库函数; 去掉内部大对象后返回可 JSON 落盘的 dict。"""
    print(f"\n===== WF run: {label} =====", flush=True)
    print(f"[gates] {_gate_snapshot(cfg)}", flush=True)
    cfg.raw["data"]["refresh_before_decide"] = False
    cfg.raw["data"]["incremental_update"] = False
    raw = run_walkforward(cfg, SYMBOL)
    raw.pop("_traded_detail", None)
    raw.pop("_equity", None)
    out = walkforward_public_summary(raw)
    out["label"] = label
    out["mode"] = "walk_forward_compare"
    out["gate_config"] = _gate_snapshot(cfg)
    # 压缩 gate_diagnostics 便于对比表
    g = out.get("gate_diagnostics") or {}
    gates = g.get("gates") or {}
    out["gate_diagnostics"] = {
        "prob_threshold": g.get("prob_threshold"),
        "n_prob_ge_threshold": gates.get("n_prob_ge_threshold"),
        "n_confident": gates.get("n_confident"),
        "n_prob_and_confident": gates.get("n_prob_and_confident"),
        "n_opened_size_gt_0": gates.get("n_opened_size_gt_0"),
        "calibrated_n_unique": (g.get("calibrated_proba") or {}).get("n_unique"),
        "conformal_qhat": g.get("conformal_qhat"),
        "conformal_min_margin": g.get("conformal_min_margin"),
    }
    print(
        f"[threshold] {label} thr_eff={out.get('prob_threshold_effective')} "
        f"opened={out.get('n_opened_trades')}",
        flush=True,
    )
    return out


def _load_summary(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"找不到 summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(key: str, v: Any) -> str:
    if v is None:
        return "—"
    if key in ("win_rate", "total_return", "max_drawdown"):
        return f"{float(v):.2%}"
    if key == "final_capital":
        return f"{float(v):,.2f}"
    if key == "prob_threshold_effective":
        return f"{float(v):.4f}"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _delta(key: str, a: Any, b: Any) -> str:
    if a is None or b is None:
        return "—"
    try:
        da, db = float(a), float(b)
    except (TypeError, ValueError):
        return "—"
    d = db - da
    if key in ("win_rate", "total_return", "max_drawdown"):
        return f"{d:+.2%}"
    if key == "prob_threshold_effective":
        return f"{d:+.4f}"
    if key == "final_capital":
        return f"{d:+,.2f}"
    if float(da).is_integer() and float(db).is_integer():
        return f"{int(d):+d}"
    return f"{d:+.6g}"


def compare_summaries(a: dict, b: dict, *, name_a: str, name_b: str) -> dict:
    rows = []
    for key, title in COMPARE_KEYS:
        va, vb = a.get(key), b.get(key)
        rows.append({
            "metric": title,
            "key": key,
            name_a: va,
            name_b: vb,
            "delta_b_minus_a": _delta(key, va, vb),
        })

    ga = (a.get("gate_diagnostics") or {})
    gb = (b.get("gate_diagnostics") or {})
    gate_rows = []
    for key, title in [
        ("n_prob_ge_threshold", "prob≥thr"),
        ("n_confident", "confident"),
        ("n_prob_and_confident", "prob∧confident"),
        ("n_opened_size_gt_0", "实际开仓"),
        ("calibrated_n_unique", "校准唯一档位数"),
        ("conformal_min_margin", "conformal_min_margin"),
    ]:
        va = ga.get(key, a.get(key))
        vb = gb.get(key, b.get(key))
        if key == "n_opened_size_gt_0":
            va = ga.get(key, a.get("n_opened_trades"))
            vb = gb.get(key, b.get("n_opened_trades"))
        gate_rows.append({
            "metric": title,
            "key": key,
            name_a: va,
            name_b: vb,
            "delta_b_minus_a": _delta(key, va, vb),
        })

    return {
        "name_a": name_a,
        "name_b": name_b,
        "split": {
            "backtest_start": a.get("backtest_start"),
            "backtest_end": a.get("backtest_end"),
            "n_train_events_a": a.get("n_train_events"),
            "n_train_events_b": b.get("n_train_events"),
            "n_test_events_a": a.get("n_test_events"),
            "n_test_events_b": b.get("n_test_events"),
        },
        "gate_config_a": a.get("gate_config"),
        "gate_config_b": b.get("gate_config"),
        "metrics": rows,
        "gates": gate_rows,
    }


def _print_compare(cmp: dict) -> None:
    na, nb = cmp["name_a"], cmp["name_b"]
    print("\n========== Walk-forward 对比 ==========", flush=True)
    print(f"A = {na}", flush=True)
    print(f"B = {nb}", flush=True)
    if cmp.get("gate_config_a") or cmp.get("gate_config_b"):
        print(f"\n[gates A] {cmp.get('gate_config_a')}", flush=True)
        print(f"[gates B] {cmp.get('gate_config_b')}", flush=True)
    print(f"\n{'指标':<16} {na:>16} {nb:>16} {'B−A':>12}", flush=True)
    print("-" * 64, flush=True)
    for r in cmp["metrics"]:
        print(
            f"{r['metric']:<16} "
            f"{_fmt(r['key'], r[na]):>16} "
            f"{_fmt(r['key'], r[nb]):>16} "
            f"{r['delta_b_minus_a']:>12}",
            flush=True,
        )
    print(f"\n{'门控':<16} {na:>16} {nb:>16} {'B−A':>12}", flush=True)
    print("-" * 64, flush=True)
    for r in cmp["gates"]:
        print(
            f"{r['metric']:<16} "
            f"{_fmt(r['key'], r[na]):>16} "
            f"{_fmt(r['key'], r[nb]):>16} "
            f"{r['delta_b_minus_a']:>12}",
            flush=True,
        )
    print(
        "\n说明: 本对比为部署同形 walk-forward; "
        "legacy=fixed thr + margin0; current=当前 config.yaml 门控。",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="BTC walk-forward 门控对比")
    ap.add_argument("--a", type=Path, default=None, help="侧 A 的 summary JSON")
    ap.add_argument("--b", type=Path, default=None, help="侧 B 的 summary JSON")
    ap.add_argument(
        "--run", action="store_true",
        help="重跑 legacy vs current(当前 config), 写出带后缀的 summary 再对比",
    )
    ap.add_argument("--skip-legacy", action="store_true", help="--run 时跳过 legacy, 只用 --a 作 A")
    ap.add_argument(
        "--out", type=Path, default=None,
        help="对比结果 JSON 路径(默认 artifacts/btc_walkforward_compare.json)",
    )
    args = ap.parse_args()

    cfg = Config.load()
    out_dir = Path(cfg.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (out_dir / "btc_walkforward_compare.json")

    if args.run:
        name_a, name_b = "legacy", "current"
        if args.skip_legacy:
            if args.a is None:
                raise SystemExit("--skip-legacy 需要同时提供 --a 旧 summary")
            sum_a = _load_summary(args.a)
            name_a = args.a.stem
            if "gate_config" not in sum_a:
                sum_a["gate_config"] = {
                    "note": "loaded from file; gate_config 字段可能缺失(旧产物)",
                    "prob_threshold_effective": sum_a.get("prob_threshold_effective"),
                }
        else:
            cfg_legacy = _apply_overrides(cfg, LEGACY_GATE_OVERRIDES)
            sum_a = run_walkforward_once(cfg_legacy, label="legacy")
            path_a = out_dir / "btc_walkforward_summary_legacy.json"
            path_a.write_text(
                json.dumps(sum_a, ensure_ascii=False, indent=2, default=float),
                encoding="utf-8",
            )
            print(f"[ok] {path_a}", flush=True)

        cfg_cur = _apply_overrides(cfg, None)
        sum_b = run_walkforward_once(cfg_cur, label="current")
        path_b = out_dir / "btc_walkforward_summary_current.json"
        path_b.write_text(
            json.dumps(sum_b, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8",
        )
        print(f"[ok] {path_b}", flush=True)
    else:
        if args.a is None or args.b is None:
            raise SystemExit("请提供 --a 与 --b, 或使用 --run")
        sum_a = _load_summary(args.a)
        sum_b = _load_summary(args.b)
        name_a, name_b = args.a.stem, args.b.stem

    cmp = compare_summaries(sum_a, sum_b, name_a=name_a, name_b=name_b)
    _print_compare(cmp)
    out_path.write_text(
        json.dumps(cmp, ensure_ascii=False, indent=2, default=float), encoding="utf-8",
    )
    print(f"\n[ok] {out_path}", flush=True)


if __name__ == "__main__":
    main()
