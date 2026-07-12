#!/usr/bin/env bash
#
# Produce the Tauri sidecar executable that tauri.conf.json's `externalBin`
# expects at `src-tauri/binaries/faceray_backend-<target-triple>`.
#
#   scripts/build_sidecar.sh            # dev shim -> repo venv Python (fast)
#   scripts/build_sidecar.sh --release  # frozen PyInstaller binary (Task 2)
#
# Tauri requires the target-triple suffix so the correct binary is selected per
# architecture (macOS arm64 vs x86_64). The dev shim keeps the edit-run loop
# instant; the release path freezes a standalone binary for distribution.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
OUT_DIR="$REPO_ROOT/src-tauri/binaries"
OUT="$OUT_DIR/faceray_backend-$TRIPLE"
VENV_PY="$REPO_ROOT/.venv/bin/python"

mkdir -p "$OUT_DIR"

if [[ "${1:-}" == "--release" ]]; then
  # Frozen, dependency-free binary for bundling. Requires PyInstaller in the
  # active environment (see faceray/requirements-dev.txt, added in Task 2).
  "$VENV_PY" -m PyInstaller \
    --onefile \
    --name "faceray_backend-$TRIPLE" \
    --distpath "$OUT_DIR" \
    --workpath "$REPO_ROOT/build/pyinstaller" \
    --specpath "$REPO_ROOT/build/pyinstaller" \
    "$REPO_ROOT/faceray/sidecar_entry.py"
  echo "[build_sidecar] wrote frozen binary -> $OUT"
else
  # Dev shim: a tiny launcher that runs the sidecar entry from the repo venv.
  # Regenerate whenever the repo moves. Release builds overwrite this.
  cat > "$OUT" <<SHIM
#!/bin/sh
# AUTO-GENERATED dev shim (scripts/build_sidecar.sh). Do not commit.
# Execs the FaceRay Python sidecar entry from the repo virtualenv.
exec env PYTHONPATH="$REPO_ROOT" "$VENV_PY" -m faceray.sidecar_entry "\$@"
SHIM
  chmod +x "$OUT"
  echo "[build_sidecar] wrote dev shim -> $OUT"
fi
