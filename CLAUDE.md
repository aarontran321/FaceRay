# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository.

## What this project is

FaceRay is a real-time, AI-powered virtual camera. It captures a webcam feed,
extracts a dense 3D face mesh, applies high-fidelity face-interaction filters
(eye-contact/gaze correction, face smoothing, face anonymizer blur, background
blur — no lighting), and pipes the result into a native OS virtual camera device
so apps like Discord, Zoom, and Meet see it as an ordinary webcam.

Full pipeline and feature details live in [README.md](README.md) — read that
first for the architecture diagram and algorithm descriptions.

## Repository layout

```
faceray/                # Python CV core (data plane) + CLI
├── core/
│   ├── tracker.py     # MediaPipe Face Landmarker (Tasks) & 3D landmark extraction
│   └── modifier.py    # Gaze correction, skin smoothing, face/background blur
├── drivers/
│   ├── virtual_sink.py # pyvirtualcam bridge to OS video loops
│   └── preview_server.py # loopback MJPEG preview of processed frames
├── app.py              # CLI orchestration loop and OpenCV UI
├── sidecar_entry.py    # Tauri sidecar: stdio-controlled headless pipeline
└── requirements.txt    # core runtime (cross-platform)
src/                    # desktop frontend (Vite + TypeScript): main.ts, ui.ts, ipc.ts
src-tauri/              # Tauri 2.0 desktop shell (Rust): window + process mgmt
│   ├── src/{main,lib,ipc}.rs   # entry, tauri commands, ControlState contract
│   ├── capabilities/  # scoped permissions (sidecar spawn)
│   └── tauri.conf.json # window + bundled faceray_backend sidecar
scripts/
├── capture_selfcheck.py # headless one-frame pipeline check -> montage PNG
└── build_sidecar.sh   # build target-triple Tauri sidecar (dev shim / PyInstaller)
tests/                  # pytest suite for pure modifier math + sidecar stdio
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
  `faceray/sidecar_entry.py` (`SidecarControl`). Control transport is **stdio** (Tauri
  manages the sidecar lifecycle and kills it on exit — no orphaned webcam
  hooks); see the desktop-app section in README.md.

## Pipeline data flow

```
[Raw Webcam Frame] → tracker.py → modifier.py → virtual_sink.py
```

`tracker.py` produces a `FaceLandmarks` object (468 dense + 10 iris 3D
landmarks) that flows through `modifier.py` unchanged; each stage returns a new
BGR frame. `FaceLandmarks` is the shared contract between the `core` modules —
see its docstring in `tracker.py` before changing its shape.

## Coding standards (established, keep following)

- Full type hints on every function/method signature.
- No placeholders, `TODO`s, or truncated snippets — every change must be a
  complete, working implementation.
- Guard clauses for hardware/runtime failure modes: missing camera, dropped
  frames, missing virtual-cam backend. Fail loud with a clear message, but never
  crash the capture loop on a single bad frame.
- Effect masks are confined to face geometry: gaze/smoothing/anonymiser work off
  the iris, eye-ring, mouth, and face-hull landmarks; feather mask edges so
  composites have no hard seam, and keep per-frame work bounded to the face
  bounding box where possible for real-time throughput.
- No comments explaining *what* code does (names should do that); only
  comments capturing non-obvious *why* (a subtle invariant, a workaround).

## Environment notes

- The core pipeline **has been executed and verified** on macOS / Python 3.13
  with a `.venv` (`python -m venv .venv`): imports, unit tests, and a full
  tracker→modifier run against a real face image all pass. The live webcam +
  virtual-cam legs still depend on host hardware/permissions (see below).
- **MediaPipe Tasks migration:** the wheels that ship Python 3.13 support
  (mediapipe ≥ 0.10.30) dropped the legacy `mediapipe.solutions` API, so
  `tracker.py` targets the Tasks `FaceLandmarker`. It needs the
  `face_landmarker.task` model bundle, which `ensure_model()` downloads once to
  `~/.cache/faceray/` (override via `FACERAY_MODEL_PATH` / `FACERAY_CACHE_DIR`).
- On macOS, `cv2.VideoCapture` needs the host terminal to hold Camera
  permission (System Settings → Privacy & Security → Camera); non-GUI processes
  are denied silently rather than prompted.
- A virtual-camera backend is required to actually output to Discord/Zoom/Meet:
  OBS Virtual Camera (Windows/macOS) or `v4l2loopback` (Linux).
- The pipeline is CPU-only (OpenCV) — there is no GPU/CUDA dependency. Lighting
  and its CuPy path were removed in the utility-first redesign.

## Current architecture (utility-first redesign)

The desktop app is a **Tauri 2.0** shell (Rust `src-tauri/`, TS `src/`) driving
the Python CV core as a **sidecar** over stdio. Lighting/relighting was removed
entirely; the product is four independent face-interaction features, all in
`core/modifier.py`:

1. **Gaze correction** (primary) — iris-recentre affine warp with a per-eye
   temporal EMA (sensitivity-controlled) for jitter-free eye contact.
2. **Face anonymizer** — heavy opaque Gaussian over the face hull only.
3. **Background blur** — depth-of-field Gaussian outside the hull; face crisp.
4. **Face smoothing** — bilateral filter over the skin mask (eyes/mouth carved
   out to keep lashes/lips sharp).

`ControlState` is the single control-plane contract. Keep the three mirrors in
lockstep: `src-tauri/src/ipc.rs` ↔ `src/ipc.ts` ↔ `faceray/sidecar_entry.py`
(`SidecarControl`). Fields: `gaze_enabled`, `gaze_sensitivity`,
`face_blur_enabled`, `background_blur_enabled`, `smoothing_enabled`,
`smoothing_strength`.

- **Sidecar** (`faceray/sidecar_entry.py`): non-blocking stdin JSON reads via a
  daemon thread; graceful shutdown on stdin EOF / SIGTERM; native hi-res mirrored
  capture; `--synthetic` / `--image` sources for camera-free testing.
- **Preview**: the sidecar serves processed frames as a **loopback MJPEG stream**
  (`drivers/preview_server.py`, `--preview`; URL announced in the `ready` event)
  that the webview's `<img>` pulls directly — no frame data crosses the Rust/TS
  IPC. Near-lossless (JPEG q95); the UI preview is a fixed 16:9 container.
- **Frontend** (`src/ui.ts`): a preview pane plus a grid of four feature cards
  (switch + optional slider each), dispatching debounced `ControlState` via the
  typed client. Framework-free DOM, macOS-styled.
- macOS camera access needs `NSCameraUsageDescription` (`src-tauri/Info.plist`)
  for the packaged app, or terminal Camera permission for `tauri dev`.

Next up: packaging a signed `.app` (PyInstaller sidecar via
`build_sidecar.sh --release`).

## Testing

A `pytest` suite lives in `tests/` (`python -m pytest tests/ -q`). It exercises
the pure math in `core/modifier.py` (gaze warp + EMA, skin-mask geometry, blur
masks) and the sidecar stdio round-trip, with synthetic landmark arrays from
`tests/conftest.py` (`make_landmarks` for noise, `make_face_landmarks` for a
realistic clustered layout). It needs no real camera or virtual-cam backend;
keep new pure-math logic covered here.

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
