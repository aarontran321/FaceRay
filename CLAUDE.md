# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository.

## What this project is

FaceRay is a real-time, AI-powered virtual camera. It captures a webcam feed,
extracts a dense 3D face mesh, applies geometry-based pixel manipulations
(virtual relighting, eye-contact/gaze correction, smart blur), and pipes the
result into a native OS virtual camera device so apps like Discord, Zoom, and
Meet see it as an ordinary webcam.

Full pipeline and feature details live in [README.md](README.md) — read that
first for the architecture diagram and algorithm descriptions.

## Repository layout

```
faceray/                # Python CV core (data plane) + CLI
├── core/
│   ├── tracker.py     # MediaPipe Face Landmarker (Tasks) & 3D landmark extraction
│   ├── relighter.py   # CUDA/CuPy Lambertian shading & surface normals
│   └── modifier.py    # Gaze correction & Gaussian blur masks
├── drivers/
│   └── virtual_sink.py # pyvirtualcam bridge to OS video loops
├── app.py              # CLI orchestration loop and OpenCV UI
├── requirements.txt    # core runtime (cross-platform, CPU path)
└── requirements-gpu.txt # optional CUDA/CuPy GPU acceleration
src/                    # desktop frontend (Vite + TypeScript): main.ts, ipc.ts
src-tauri/              # Tauri 2.0 desktop shell (Rust): window + process mgmt
│   ├── src/{main,lib,ipc}.rs   # entry, tauri commands, ControlState contract
│   ├── capabilities/  # scoped permissions (sidecar spawn)
│   └── tauri.conf.json # window + bundled faceray_backend sidecar
scripts/
├── capture_selfcheck.py # headless one-frame pipeline check -> montage PNG
└── build_sidecar.sh   # build target-triple Tauri sidecar (dev shim / PyInstaller)
tests/                  # pytest suite for pure relighter/modifier math
```

**Strict module boundaries — do not violate these:**
- `core/*` modules are pure math/CV engines. They never import `drivers/*` or
  touch camera/OS APIs.
- `drivers/*` never imports `core/*` — it only knows about raw frame arrays.
- `app.py` is the only place that wires `core` + `drivers` together.
- **Desktop shell (data/control plane split):** all computer vision stays in
  Python; all window/process orchestration stays in Rust (`src-tauri/`); the
  TypeScript (`src/`) is presentation only. Video frames never cross the IPC
  boundary — only the scalar `ControlState` payload does. Keep the three
  `ControlState` mirrors in sync: `src-tauri/src/ipc.rs` ↔ `src/ipc.ts` ↔
  `faceray/sidecar_entry.py` (Task 2). Control transport is **stdio** (Tauri
  manages the sidecar lifecycle and kills it on exit — no orphaned webcam
  hooks); see the desktop-app section in README.md.

## Pipeline data flow

```
[Raw Webcam Frame] → tracker.py → relighter.py → modifier.py → virtual_sink.py
```

`tracker.py` produces a `FaceLandmarks` object (468 dense + 10 iris 3D
landmarks) that flows through `relighter.py` and `modifier.py` unchanged;
each stage returns a new BGR frame. `FaceLandmarks` is the shared contract
between all three `core` modules — see its docstring in `tracker.py` before
changing its shape.

## Coding standards (established, keep following)

- Full type hints on every function/method signature.
- No placeholders, `TODO`s, or truncated snippets — every change must be a
  complete, working implementation.
- Guard clauses for hardware/runtime failure modes: missing camera, dropped
  frames, absent CUDA context, missing virtual-cam backend. Fail loud with a
  clear message, but never crash the capture loop on a single bad frame.
- GPU context integrity: `relighter.py` keeps array math in the active `xp`
  namespace (CuPy when available, NumPy fallback otherwise) and minimizes
  host↔device copies — only bring small per-triangle results back to host for
  `cv2` rasterization.
- No comments explaining *what* code does (names should do that); only
  comments capturing non-obvious *why* (a subtle invariant, a workaround).

## Environment notes

- The core pipeline **has been executed and verified** on macOS / Python 3.13
  with a `.venv` (`python -m venv .venv`): imports, unit tests, and a full
  tracker→relighter→modifier run against a real face image all pass. The live
  webcam + virtual-cam legs still depend on host hardware/permissions (see
  below).
- **MediaPipe Tasks migration:** the wheels that ship Python 3.13 support
  (mediapipe ≥ 0.10.30) dropped the legacy `mediapipe.solutions` API, so
  `tracker.py` targets the Tasks `FaceLandmarker`. It needs the
  `face_landmarker.task` model bundle, which `ensure_model()` downloads once to
  `~/.cache/faceray/` (override via `FACERAY_MODEL_PATH` / `FACERAY_CACHE_DIR`).
  The `FaceLandmarks` contract is unchanged, so `relighter`/`modifier`/`app`
  were untouched.
- On macOS, `cv2.VideoCapture` needs the host terminal to hold Camera
  permission (System Settings → Privacy & Security → Camera); non-GUI processes
  are denied silently rather than prompted.
- A virtual-camera backend is required to actually output to Discord/Zoom/Meet:
  OBS Virtual Camera (Windows/macOS) or `v4l2loopback` (Linux).
- GPU deps live in `requirements-gpu.txt` (`cupy-cuda12x`), separate from the
  core `requirements.txt` because they can't install without a CUDA runtime
  (e.g. on macOS). Everything degrades gracefully to NumPy without them.

## Milestone plan (for context on ordering)

1. **Phase 1** — core pipeline verification: `app.py` running a clean webcam
   loop through `drivers/virtual_sink.py`.
2. **Phase 2** — tracking: `core/tracker.py`.
3. **Phase 3** — GPU math and rendering: `core/relighter.py`,
   `core/modifier.py`.

All three CLI phases are implemented on `main`. **Desktop app (Tauri 2.0)** is
the current workstream, built on top of the CLI core without restructuring it:

- **D-Task 1** (done) — Tauri core: `src-tauri/` Rust wrapper, `ControlState`
  IPC contract, buildable Vite/TS shell, `build_sidecar.sh`.
- **D-Task 2** (next) — `faceray/sidecar_entry.py` (non-blocking stdin JSON
  reads, graceful shutdown on parent death) + Rust sidecar spawn/stdin plumbing.
- **D-Task 3** — TypeScript control panel (light-vector sliders, effect toggles).

## Testing

A `pytest` suite lives in `tests/` (`python -m pytest tests/ -q`). It exercises
the pure math in `core/relighter.py` (normal/shading computation) and
`core/modifier.py` (blur mask geometry, gaze bounds) with synthetic landmark
arrays from `tests/conftest.py::make_landmarks`, so it needs no real camera or
virtual-cam backend. Keep new pure-math logic covered here.

For a real capture→process check without a live window, run
`python -m scripts.capture_selfcheck --out selfcheck.png` (needs a webcam +
Camera permission); it writes a labelled before/after montage.

**Desktop shell checks** (run from repo root; toolchain: cargo 1.96, node 24,
tauri-cli 2.11):

```bash
npm run build                         # tsc + vite build (frontend)
cd src-tauri && cargo check           # Rust wrapper compiles
cd src-tauri && cargo test --lib      # ControlState wire-format contract tests
./scripts/build_sidecar.sh            # (re)generate the dev sidecar binary
```

`cargo check`/`generate_context!` validates that the `externalBin` sidecar
exists at `src-tauri/binaries/faceray_backend-<triple>`, so run
`build_sidecar.sh` before building the Rust crate on a fresh checkout.
