#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIX_FILE="${SCRIPT_DIR}/get_trading_dates.nix"
PY_SCRIPT="${SCRIPT_DIR}/get_trading_dates.py"

if command -v nix-shell >/dev/null 2>&1; then
    exec nix-shell "${NIX_FILE}" --run "python3 \"${PY_SCRIPT}\" $*"
else
    exec python3 "${PY_SCRIPT}" "$@"
fi
