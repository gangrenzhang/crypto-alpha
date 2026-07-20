#!/usr/bin/env bash
# macOS Apple Silicon: crypto-alpha 运行前 source 本脚本（或先 source pyro/.venv/bin/activate）。
# 目的：让 LightGBM / sklearn / torch 共用同一份 libomp，避免双 OpenMP 段错误 (exit 139)。
set -euo pipefail

_LIBOMP_DIR="/opt/homebrew/opt/libomp/lib"
if [[ ! -f "${_LIBOMP_DIR}/libomp.dylib" ]]; then
  echo "缺少 ${_LIBOMP_DIR}/libomp.dylib" >&2
  echo "请安装: brew install libomp" >&2
  echo "若 bottle 失败，可从 conda-forge llvm-openmp 解压 libomp.dylib 到该目录。" >&2
  return 1 2>/dev/null || exit 1
fi

case ":${DYLD_LIBRARY_PATH:-}:" in
  *":${_LIBOMP_DIR}:"*) ;;
  *) export DYLD_LIBRARY_PATH="${_LIBOMP_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" ;;
esac

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig-crypto-alpha}"
mkdir -p "$MPLCONFIGDIR"

echo "mac_env: DYLD_LIBRARY_PATH 已包含 libomp (${_LIBOMP_DIR})"
