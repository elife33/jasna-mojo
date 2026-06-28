#!/usr/bin/env bash
set -euo pipefail

# Build and run jasna-mojo.
# Override tool locations with MOJO_BIN, PYTHON_BIN, or MOJO_PYTHON_LIBRARY.

MOJO_BIN="${MOJO_BIN:-mojo}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$MOJO_BIN" >/dev/null 2>&1; then
    echo "Error: Mojo compiler not found. Set MOJO_BIN or add mojo to PATH." >&2
    exit 1
fi

if [ -z "${MOJO_PYTHON_LIBRARY:-}" ] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    detected_lib="$("$PYTHON_BIN" - <<'PY'
import ctypes.util
import sysconfig

libdir = sysconfig.get_config_var("LIBDIR") or ""
ldlibrary = sysconfig.get_config_var("LDLIBRARY") or ""
if libdir and ldlibrary:
    print(f"{libdir}/{ldlibrary}")
else:
    print(ctypes.util.find_library("python") or "")
PY
)"
    if [ -n "$detected_lib" ] && [ -e "$detected_lib" ]; then
        export MOJO_PYTHON_LIBRARY="$detected_lib"
    fi
fi

if [ ! -f jasna_bin ]; then
    echo "Building jasna_bin..."
    "$MOJO_BIN" build jasna/__main__.mojo -o jasna_bin
fi

./jasna_bin "$@"
