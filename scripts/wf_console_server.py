#!/usr/bin/env python3
"""Walk-Forward 本地控制台：HTML 选窗/币种 → 子进程训练 → 结果轮询。

用法:
    python scripts/wf_console_server.py
    python scripts/wf_console_server.py --host 127.0.0.1 --port 8765

设计要点（避免打开页面即 SIGSEGV / exit 139）:
  - 请求处理用单线程 HTTPServer，避免在 HTTP 工作线程里首次 import/读 parquet
  - 启动时在主线程预热 Config + 面板元数据并缓存
  - 训练在独立子进程跑（与 CLI 同源），崩溃不会拖死 UI 进程
  - 不劫持进程级 sys.stdout
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "web" / "wf_console.html"
JOBS_ROOT = ROOT / "artifacts" / "wf_console_jobs"
PYTHON = sys.executable

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_run_lock = threading.Lock()  # 同时只跑一个子进程训练
_meta_cache: dict[str, Any] | None = None
_meta_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(obj: Any, code: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    return code, body, "application/json; charset=utf-8"


def _date_only(v: Any) -> str | None:
    if not v:
        return None
    return str(v).strip().replace("T", " ")[:10]


def _panel_info_light(cfg, symbol: str) -> dict[str, Any] | None:
    """只取起止/行数，避免全表进内存；失败返回 None。"""
    try:
        import pyarrow.parquet as pq
        from crypto_alpha.data.fetch import timeframe_delta

        tf = str(cfg["data"].get("timeframe") or "30m")
        stem = symbol.replace("/", "_")
        path = cfg.data_dir / "raw" / f"{stem}__{tf}.parquet"
        if not path.exists() and tf == "1h":
            path = cfg.data_dir / "raw" / f"{stem}.parquet"
        if not path.exists():
            return None

        pf = pq.ParquetFile(path)
        n = int(pf.metadata.num_rows) if pf.metadata is not None else 0
        # 索引通常在 pandas 写入时进列或 index；优先读 close 列的 row-group 统计不行
        # 轻量：只读一列（若存在）的首尾，用 pandas 会重；改用 pyarrow 扫 index/列
        cols = set(pf.schema_arrow.names)
        start_s = end_s = None
        # 常见：无单独 timestamp 列，index 写进 __index_level_0__ / timestamp
        for cand in ("__index_level_0__", "timestamp", "time", "datetime", "date"):
            if cand in cols:
                col = pq.read_table(path, columns=[cand]).column(0)
                if len(col) == 0:
                    break
                start_s = str(col[0].as_py())
                end_s = str(col[-1].as_py())
                break
        if start_s is None:
            # 回退：读 close 无时间则仅 bars
            start_s = end_s = None

        return {
            "bars": n,
            "start": start_s,
            "end": end_s,
            "path": str(path),
            "timeframe": tf,
            "bar_delta": str(timeframe_delta(tf)),
        }
    except Exception as e:
        return {"error": str(e), "bars": None, "start": None, "end": None}


# Setup 面板分组标题（与 config.yaml 顶层键对齐）
_CONFIG_SECTION_LABELS: dict[str, str] = {
    "project": "项目",
    "data": "数据",
    "news": "新闻",
    "features": "特征",
    "labeling": "标注",
    "validation": "验证 / Walk-Forward",
    "experts": "专家模型",
    "ensemble": "集成",
    "calibration": "校准 / 保形",
    "backtest": "回测门控",
    "serve": "实时服务",
    "risk": "风控",
}

# 默认展开的分组（WF 研究高频改动）
_CONFIG_SECTIONS_OPEN: frozenset[str] = frozenset({
    "validation", "calibration", "backtest", "risk", "labeling", "experts", "ensemble", "features",
})


def _jsonable(obj: Any) -> Any:
    """把 config 树变成可 JSON 往返的结构（保留 null）。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """src 覆盖 dst（就地）；嵌套 dict 递归合并，list/标量整段替换。"""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def build_meta() -> dict[str, Any]:
    from crypto_alpha.config import Config

    cfg = Config.load()
    symbols = list(cfg["data"].get("symbols") or ["BTC/USDT", "ETH/USDT"])
    wf = (cfg.get("validation") or {}).get("walkforward") or {}
    panels = {s: _panel_info_light(cfg, s) for s in symbols}
    # train_start 配置为 null 时不要伪造 2020，与 CLI「不截断」一致
    train_default = _date_only(wf.get("train_start"))
    return {
        "symbols": symbols,
        "timeframe": cfg["data"].get("timeframe"),
        "panels": panels,
        "defaults": {
            "symbol": "BTC/USDT" if "BTC/USDT" in symbols else symbols[0],
            "train_start": train_default or "",
            "test_start": _date_only(wf.get("test_start")) or "2022-09-14",
            "test_end": _date_only(wf.get("test_end")) or "2026-07-18",
            "recompute_sample_weight": bool(wf.get("recompute_sample_weight_on_split", False)),
        },
        "min_test_events": int(wf.get("min_test_events") or 50),
        "min_train_events": int(wf.get("min_train_events") or 200),
        "config": _jsonable(cfg.raw),
        "config_section_labels": dict(_CONFIG_SECTION_LABELS),
        "config_sections_open": sorted(_CONFIG_SECTIONS_OPEN),
        "note": (
            "single_cut_holdout walk-forward；训练窗拟合，测试窗部署门控。"
            "Setup 面板含完整 config.yaml；训练时以面板配置深合并覆盖后跑。"
        ),
    }


def get_meta(*, refresh: bool = False) -> dict[str, Any]:
    global _meta_cache
    with _meta_lock:
        if _meta_cache is None or refresh:
            _meta_cache = build_meta()
        return dict(_meta_cache)


def warmup_main_thread() -> None:
    """主线程预热重型依赖，避免首次在 HTTP/子线程里 import 触发原生库崩溃。"""
    import pandas as pd  # noqa: F401
    import pyarrow  # noqa: F401

    from crypto_alpha.config import Config

    Config.load()
    get_meta(refresh=True)


def _yearly_from_trades_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import pandas as pd

    df = pd.read_csv(path)
    if "pnl" not in df.columns:
        return []
    if "entry_time" in df.columns:
        ts = pd.to_datetime(df["entry_time"], utc=True)
    else:
        ts = pd.to_datetime(df.iloc[:, 0], utc=True)
    df = df.assign(_year=ts.dt.year)
    rows = []
    for year, g in df.groupby("_year"):
        side = g["side"] if "side" in g.columns else None
        rows.append({
            "year": int(year),
            "n": int(len(g)),
            "win_rate": float((g["pnl"] > 0).mean()),
            "pnl": float(g["pnl"].sum()),
            "n_long": int((side > 0).sum()) if side is not None else 0,
            "n_short": int((side < 0).sum()) if side is not None else 0,
        })
    return rows


def _trades_preview_csv(path: Path, limit: int = 80) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import pandas as pd

    df = pd.read_csv(path)
    if "entry_time" not in df.columns and len(df.columns):
        df = df.rename(columns={df.columns[0]: "entry_time"})
    cols = [c for c in ("entry_time", "side", "bars_held", "prob", "pnl", "ret", "size") if c in df.columns]
    out = []
    for _, row in df[cols].head(limit).iterrows():
        item = {}
        for c in cols:
            v = row[c]
            try:
                if pd.api.types.is_number(v):
                    item[c] = float(v)
                else:
                    item[c] = v if isinstance(v, str) else str(v)
            except Exception:
                item[c] = str(v)
        out.append(item)
    return out


def _append_log(job_id: str, text: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.setdefault("_log_parts", []).append(text)
        # 同步落盘，刷新页面也能看
        log_path = Path(job.get("log_path") or "")
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    log = job.get("log") or ""
    if not log and job.get("_log_parts"):
        log = "".join(job["_log_parts"])
    elif job.get("log_path"):
        try:
            p = Path(job["log_path"])
            if p.exists():
                log = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return {
        "id": job["id"],
        "status": job["status"],
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "params": job.get("params"),
        "log": log[-120_000:],
        "summary": job.get("summary"),
        "yearly": job.get("yearly") or [],
        "trades": job.get("trades") or [],
        "artifact_dir": job.get("artifact_dir"),
        "error": job.get("error"),
    }


def _run_worker_subprocess(job_id: str, params: dict[str, Any]) -> None:
    """在后台线程里等锁，再起子进程执行训练。"""
    out_dir = JOBS_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.json"
    params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = out_dir / "run.log"

    with _jobs_lock:
        job = _jobs[job_id]
        job["artifact_dir"] = str(out_dir)
        job["log_path"] = str(log_path)
        job["status"] = "queued"

    with _run_lock:
        with _jobs_lock:
            job = _jobs[job_id]
            job["status"] = "running"
            job["started_at"] = _utc_now()

        cmd = [
            PYTHON,
            str(Path(__file__).resolve()),
            "--worker",
            str(out_dir),
        ]
        _append_log(job_id, f"$ {' '.join(cmd)}\n")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                _append_log(job_id, line)
            code = proc.wait()
        except Exception as e:
            tb = traceback.format_exc()
            _append_log(job_id, tb)
            with _jobs_lock:
                job = _jobs[job_id]
                job["status"] = "error"
                job["finished_at"] = _utc_now()
                job["error"] = f"{type(e).__name__}: {e}"
                job["log"] = "".join(job.get("_log_parts") or [])
            return

        summary_path = out_dir / "summary.json"
        trades_path = out_dir / "trades.csv"
        err_path = out_dir / "error.txt"

        if code != 0 or err_path.exists():
            err_msg = err_path.read_text(encoding="utf-8") if err_path.exists() else f"worker exit {code}"
            with _jobs_lock:
                job = _jobs[job_id]
                job["status"] = "error"
                job["finished_at"] = _utc_now()
                job["error"] = err_msg.strip()[:4000]
                job["log"] = "".join(job.get("_log_parts") or [])
            return

        try:
            pub = json.loads(summary_path.read_text(encoding="utf-8"))
            yearly = _yearly_from_trades_csv(trades_path)
            trades = _trades_preview_csv(trades_path)
            with _jobs_lock:
                job = _jobs[job_id]
                job["status"] = "done"
                job["finished_at"] = _utc_now()
                job["summary"] = pub
                job["yearly"] = yearly
                job["trades"] = trades
                job["log"] = "".join(job.get("_log_parts") or [])
        except Exception as e:
            with _jobs_lock:
                job = _jobs[job_id]
                job["status"] = "error"
                job["finished_at"] = _utc_now()
                job["error"] = f"读取产物失败: {type(e).__name__}: {e}"
                job["log"] = "".join(job.get("_log_parts") or [])


def create_job(params: dict[str, Any]) -> dict[str, Any]:
    if not params.get("symbol"):
        raise ValueError("缺少 symbol")
    if not params.get("test_start"):
        raise ValueError("缺少 test_start（回测起点/切分点）")

    meta = get_meta()
    allowed = set(meta.get("symbols") or [])
    if allowed and params["symbol"] not in allowed:
        raise ValueError(f"不支持的 symbol: {params['symbol']}; 可选 {sorted(allowed)}")

    for k in ("train_start", "test_start", "test_end"):
        v = params.get(k)
        if v is not None and str(v).strip() == "":
            params[k] = None
        elif v:
            params[k] = str(v).strip()[:10]

    ts, te = params.get("test_start"), params.get("test_end")
    tr = params.get("train_start")
    if tr and ts and tr >= ts:
        raise ValueError(f"train_start ({tr}) 必须早于 test_start ({ts})")
    if ts and te and te < ts:
        raise ValueError(f"test_end ({te}) 不能早于 test_start ({ts})")

    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job = {
        "id": job_id,
        "status": "queued",
        "created_at": _utc_now(),
        "params": params,
        "log": "",
        "_log_parts": [],
        "summary": None,
        "yearly": [],
        "trades": [],
        "artifact_dir": None,
        "log_path": None,
        "error": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job

    threading.Thread(
        target=_run_worker_subprocess,
        args=(job_id, params),
        daemon=True,
        name=f"wf-job-{job_id}",
    ).start()
    return _public_job(job)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return _public_job(job)


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with _jobs_lock:
        items = sorted(_jobs.values(), key=lambda j: j.get("created_at") or "", reverse=True)
        return [_public_job(j) for j in items[:limit]]


def worker_main(out_dir: Path) -> int:
    """子进程入口：读 params.json，跑 walkforward，写 summary/trades。"""
    from crypto_alpha.config import Config
    from crypto_alpha.pipeline.walkforward import run_walkforward, walkforward_public_summary

    out_dir = Path(out_dir)
    params = json.loads((out_dir / "params.json").read_text(encoding="utf-8"))
    try:
        cfg = Config.load()
        # 面板完整配置覆盖（与 config.yaml 同构）
        overlay = params.get("config")
        if isinstance(overlay, dict) and overlay:
            deep_merge(cfg.raw, overlay)
            print("[console] 已合并面板 config 覆盖", flush=True)

        # 研究 WF：禁止 tip REST 拖慢/污染；面板里这两项改了也会被强制回写
        cfg.raw.setdefault("data", {})
        cfg.raw["data"]["refresh_before_decide"] = False
        cfg.raw["data"]["incremental_update"] = False

        symbol = params["symbol"]
        train_start = params.get("train_start") or None
        test_start = params["test_start"]
        test_end = params.get("test_end") or None
        recompute = bool(params.get("recompute_sample_weight"))

        # 切分参数与 validation.walkforward 对齐（便于产物自描述）
        wf = cfg.raw.setdefault("validation", {}).setdefault("walkforward", {})
        wf["train_start"] = f"{train_start}T00:00:00Z" if train_start else None
        wf["test_start"] = f"{test_start}T00:00:00Z"
        wf["test_end"] = f"{test_end}T00:00:00Z" if test_end else None
        if "recompute_sample_weight" in params:
            wf["recompute_sample_weight_on_split"] = recompute

        print(f"===== {symbol} walk-forward (console worker) =====", flush=True)
        print(
            f"train_start={train_start} test_start={test_start} test_end={test_end} "
            f"recompute_sample_weight={recompute}",
            flush=True,
        )
        print(
            f"experts={cfg.raw.get('experts', {}).get('enabled')} "
            f"cal={cfg.raw.get('calibration', {}).get('method')} "
            f"thr_mode={cfg.raw.get('backtest', {}).get('prob_threshold_mode')} "
            f"thr={cfg.raw.get('backtest', {}).get('prob_threshold')}",
            flush=True,
        )

        # 落盘本次实际生效配置，便于复现
        (out_dir / "effective_config.json").write_text(
            json.dumps(_jsonable(cfg.raw), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        summary = run_walkforward(
            cfg,
            symbol,
            train_start=train_start,
            test_start=test_start,
            test_end=test_end,
            recompute_sample_weight=True if recompute else None,
        )

        traded = summary.pop("_traded_detail", None)
        summary.pop("_equity", None)
        trades_path = out_dir / "trades.csv"
        if traded is not None and len(traded):
            traded_out = traded.copy()
            traded_out.index.name = "entry_time"
            traded_out.to_csv(trades_path)
            summary["trades_csv"] = str(trades_path)
        else:
            summary["trades_csv"] = None

        pub = walkforward_public_summary(summary)
        (out_dir / "summary.json").write_text(
            json.dumps(pub, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8",
        )
        (out_dir / "gate_diagnostics.json").write_text(
            json.dumps(pub.get("gate_diagnostics") or {}, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8",
        )
        print(f"[ok] {out_dir / 'summary.json'}", flush=True)
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        (out_dir / "error.txt").write_text(f"{type(e).__name__}: {e}\n\n{tb}", encoding="utf-8")
        return 1


class Handler(BaseHTTPRequestHandler):
    server_version = "CryptoAlphaWFConsole/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send(204, b"", "text/plain")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/" or path == "/index.html":
            if not HTML_PATH.exists():
                code, body, ct = _json_bytes({"error": f"找不到 {HTML_PATH}"}, 500)
                self._send(code, body, ct)
                return
            self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/meta":
            try:
                refresh = "refresh=1" in (parsed.query or "")
                code, body, ct = _json_bytes(get_meta(refresh=refresh))
            except Exception as e:
                code, body, ct = _json_bytes({"error": str(e)}, 500)
            self._send(code, body, ct)
            return

        if path == "/api/jobs":
            code, body, ct = _json_bytes({"jobs": list_jobs()})
            self._send(code, body, ct)
            return

        if path.startswith("/api/jobs/"):
            job_id = path.split("/api/jobs/", 1)[1]
            job = get_job(job_id)
            if job is None:
                code, body, ct = _json_bytes({"error": "job not found"}, 404)
            else:
                code, body, ct = _json_bytes(job)
            self._send(code, body, ct)
            return

        code, body, ct = _json_bytes({"error": "not found"}, 404)
        self._send(code, body, ct)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length") or 0)
        if length > 4_000_000:
            code, body, ct = _json_bytes({"error": "payload too large"}, 413)
            self._send(code, body, ct)
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            code, body, ct = _json_bytes({"error": "invalid JSON"}, 400)
            self._send(code, body, ct)
            return

        if path == "/api/jobs":
            try:
                job = create_job(payload if isinstance(payload, dict) else {})
                code, body, ct = _json_bytes(job, 202)
            except ValueError as e:
                code, body, ct = _json_bytes({"error": str(e)}, 400)
            except Exception as e:
                code, body, ct = _json_bytes({"error": str(e)}, 500)
            self._send(code, body, ct)
            return

        code, body, ct = _json_bytes({"error": "not found"}, 404)
        self._send(code, body, ct)


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-Forward HTML 控制台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--worker",
        default=None,
        help="内部：子进程模式，传入 job 输出目录",
    )
    args = ap.parse_args()

    if args.worker:
        raise SystemExit(worker_main(Path(args.worker)))

    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    if not HTML_PATH.exists():
        raise SystemExit(f"缺少前端文件: {HTML_PATH}")

    print("预热主线程依赖…", flush=True)
    warmup_main_thread()
    print("预热完成。", flush=True)

    # 单线程 HTTP：请求处理器不做重计算；训练在子进程
    httpd = HTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Crypto-Alpha Walk-Forward 控制台: {url}", flush=True)
    print("Ctrl+C 结束。训练在子进程中串行执行。", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
