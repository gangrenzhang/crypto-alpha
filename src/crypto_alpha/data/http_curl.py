"""共用的 curl 抓取封装: **默认校验 TLS**, 仅在证书失败时才降级并显式告警。

为什么不直接写 ``curl -k``:
宏观日历(BLS/Fed/FRED/ForexFactory)抓来的数值会直接变成模型特征与决策依据。
无条件 ``-k`` 等于对这些源接受任意证书 —— 在不可信网络下, 被中间人替换的
CPI/NFP 数值会静默进入训练与实盘决策, 而系统不会有任何告警。

策略(按顺序):
1. 正常校验证书抓取;
2. 仅当失败原因**明确是证书/TLS**(curl 退出码 35/51/58/59/60/77 或 stderr 含
   certificate/SSL 字样)时, 才用 ``-k`` 重试一次, 并打印一次性告警;
3. 其它失败(超时、404、代理不可达)按原样抛出, 不做不安全重试。

macOS + 本地代理常见的「urllib 证书链失败」是这个降级存在的原因, 但它必须是
可见的例外, 而不是默认口径。可用 ``CRYPTO_ALPHA_ALLOW_INSECURE_TLS=0`` 完全禁止降级。
"""
from __future__ import annotations

import os
import subprocess

#: curl 退出码中与 TLS/证书相关的集合
_TLS_EXIT_CODES = frozenset({35, 51, 58, 59, 60, 77, 83})
_TLS_HINTS = (
    "certificate",
    "ssl",
    "tls",
    "self-signed",
    "unable to get local issuer",
)
#: 已告警过的主机, 避免每个请求都刷屏
_warned_hosts: set[str] = set()


def _host(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc or url[:40]
    except Exception:
        return url[:40]


def _looks_like_tls_failure(returncode: int, stderr: bytes | None) -> bool:
    if int(returncode) in _TLS_EXIT_CODES:
        return True
    msg = (stderr or b"").decode("utf-8", errors="replace").lower()
    return any(h in msg for h in _TLS_HINTS)


def insecure_fallback_allowed() -> bool:
    raw = os.environ.get("CRYPTO_ALPHA_ALLOW_INSECURE_TLS", "1").strip().lower()
    return raw not in ("0", "false", "no")


def _resolve_proxy() -> str | None:
    proxy = (
        os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    )
    if proxy:
        return proxy
    try:
        from .news import _resolve_http_proxies

        proxies = _resolve_http_proxies()
        return proxies.get("https") or proxies.get("http")
    except Exception:
        return None


def curl_bytes(
    url: str,
    *,
    data: bytes | None = None,
    timeout: float = 90.0,
    user_agent: str = "Mozilla/5.0 (crypto-alpha)",
    label: str = "curl",
) -> bytes:
    """抓取 URL 并返回字节; 校验证书优先, 证书失败才 ``-k`` 重试。

    Raises
    ------
    RuntimeError
        非证书原因的失败, 或降级被禁用/降级后仍失败。
    """
    proxy = _resolve_proxy()

    def _run(insecure: bool):
        cmd = [
            "curl", "-sL", "-A", user_agent,
            "--connect-timeout", "20", "--max-time", str(int(timeout)),
        ]
        if insecure:
            cmd.append("-k")
        if proxy:
            cmd.extend(["-x", proxy])
        if data is not None:
            cmd.extend(["-H", "Content-Type: application/json", "--data-binary", "@-"])
        cmd.append(url)
        return subprocess.run(
            cmd, input=data, capture_output=True, timeout=timeout + 5, check=False,
        )

    proc = _run(insecure=False)
    if proc.returncode == 0 and proc.stdout:
        return proc.stdout

    err = (proc.stderr or b"").decode("utf-8", errors="replace")[:200]
    if not _looks_like_tls_failure(proc.returncode, proc.stderr):
        raise RuntimeError(f"{label} 非零/空响应({proc.returncode}): {url} {err}")
    if not insecure_fallback_allowed():
        raise RuntimeError(
            f"{label} TLS 校验失败且已禁用不安全降级"
            f"(CRYPTO_ALPHA_ALLOW_INSECURE_TLS=0): {url} {err}"
        )

    h = _host(url)
    if h not in _warned_hosts:
        _warned_hosts.add(h)
        print(
            f"[warn] {label}: {h} TLS 校验失败({proc.returncode}), 本次改用 -k 抓取。"
            "该源数据会进入特征/决策——请确认网络可信(或设 "
            "CRYPTO_ALPHA_ALLOW_INSECURE_TLS=0 强制失败)。",
            flush=True,
        )
    proc2 = _run(insecure=True)
    if proc2.returncode != 0 or not proc2.stdout:
        err2 = (proc2.stderr or b"").decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"{label} 非零/空响应({proc2.returncode}): {url} {err2}")
    return proc2.stdout
