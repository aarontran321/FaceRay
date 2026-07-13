# FaceRay

**Real-time, AI-powered virtual camera.** FaceRay captures your raw webcam
feed, extracts a dense 3D face mesh, applies geometry-based pixel manipulations
— **virtual relighting**, **eye-contact correction**, and **smart blur** — and
pipes the result straight into a native system virtual camera so Discord, Zoom,
and Google Meet see it as an ordinary webcam.

The pipeline targets sub-16 ms latency (60 FPS) by keeping frame matrices
resident on the GPU and minimizing host↔device copies.

---

## Pipeline

```
[Raw Webcam Frame]           cv2.VideoCapture
        │
        ▼
core/tracker.py      →  MediaPipe Face Landmarker (Tasks): 468 dense + 10 iris 3D landmarks
        │
        ▼
core/relighter.py    →  Surface normals → Lambertian N·L shading (CuPy/CUDA)
        │
        ▼
core/modifier.py     →  Gaze re-centering warp + Gaussian face/background blur
        │
        ▼
drivers/virtual_sink.py  →  RGB bytes → system virtual camera (pyvirtualcam)
```

## Features

- **Virtual relighting (Lambertian shading).** Landmarks are Delaunay-triangulated,
  per-facet surface normals `N` are computed, and illumination `max(0, N·L)` for a
  movable virtual light `L` is evaluated on the GPU via CuPy, then blended
  multiplicatively over the frame — a software ring/fill light.
- **Eye-contact / gaze correction.** Iris landmarks (left 468–472, right 473–477)
  are compared against each eye-socket center; a bounded affine warp nudges the
  iris back toward the aperture center to simulate looking into the lens.
- **Smart blur.** The face convex hull drives a feathered mask; a Gaussian blur is
  applied to the **face** (identity privacy) or the **background**, toggleable live.

## Project structure

```
faceray/                  # Python CV core (data plane) + CLI
├── core/
│   ├── tracker.py        # MediaPipe Face Landmarker (Tasks) & 3D landmark extraction
│   ├── relighter.py      # CUDA/CuPy Lambertian shading & surface normals
│   └── modifier.py       # Gaze correction & Gaussian blur masks
├── drivers/
│   ├── virtual_sink.py   # pyvirtualcam bridge to OS video loops
│   └── preview_server.py # loopback MJPEG preview of processed frames
├── app.py                # CLI orchestration loop and OpenCV UI
├── sidecar_entry.py      # Tauri sidecar: stdio-controlled headless pipeline
├── requirements.txt      # core runtime (cross-platform, CPU path)
├── requirements-gpu.txt  # optional CUDA/CuPy GPU acceleration
└── requirements-dev.txt  # pytest + pyinstaller (sidecar freezing)
src/                      # desktop control-panel frontend (Vite + TypeScript)
│   ├── main.ts           # bootstrap: mount panel, dispatch control, status events
│   ├── ui.ts             # control-panel widgets (sliders / switches / segmented)
│   └── ipc.ts            # typed control-plane client (mirrors Rust ControlState)
src-tauri/                # Tauri 2.0 desktop shell (Rust; window + process mgmt)
│   ├── src/{main,lib,ipc,sidecar}.rs  # entry, commands, contract, process mgmt
│   ├── capabilities/     # scoped permissions (sidecar spawn)
│   ├── Info.plist        # macOS camera-usage + local-networking entitlements
│   └── tauri.conf.json   # window + bundled faceray_backend sidecar config
scripts/
├── capture_selfcheck.py  # headless one-frame pipeline check -> montage PNG
└── build_sidecar.sh      # build the target-triple Tauri sidecar (dev shim / PyInstaller)
tests/                    # pytest suite for the pure relighter/modifier math
```

The layers are strictly decoupled: `core` engines never touch the driver, and
the driver never imports the math engines.

## Requirements

- Python 3.10–3.13 (verified on 3.13)
- A webcam
- A virtual-camera backend:
  - **Windows / macOS** — OBS Virtual Camera (ships with OBS Studio ≥ 26.1)
  - **Linux** — the `v4l2loopback` kernel module
- **Optional:** an NVIDIA GPU with a CUDA 12.x runtime for the CuPy shading path.
  Without it FaceRay transparently falls back to NumPy on the CPU.

> **MediaPipe Tasks model.** `core/tracker.py` uses the MediaPipe Face
> Landmarker (Tasks API). On first run it downloads the `face_landmarker.task`
> bundle (~3.6 MB) into `~/.cache/faceray/`. Override the location with
> `FACERAY_MODEL_PATH` (explicit file) or `FACERAY_CACHE_DIR` (directory), or
> pre-place the file for fully offline use.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r faceray/requirements.txt             # core, CPU path — all platforms
```

Optional GPU acceleration (NVIDIA + CUDA 12.x only; not available on macOS):

```bash
pip install -r faceray/requirements-gpu.txt         # adds cupy-cuda12x
```

> The GPU dependency lives in a separate file because it cannot install without
> a CUDA runtime. The relighter automatically uses the NumPy backend when CuPy
> is absent, so the core install is fully functional on its own.

## Run

```bash
python -m faceray.app                                  # 1280x720 @ 30 fps, cam 0
python -m faceray.app --width 1920 --height 1080 --fps 60 --camera 1
python -m faceray.app --no-preview                     # headless (virtual cam only)
python -m faceray.app --no-gpu                          # force the CPU shading path
```

Then select **FaceRay / OBS Virtual Camera** as your webcam inside Discord,
Zoom, or Meet.

### Hotkeys (preview window)

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `q` / `Esc` | quit | `l` | toggle relighting |
| `g` | toggle gaze correction | `b` | cycle blur (off / face / background) |
| `[` / `]` | orbit light left / right | `-` / `=` | dim / brighten light |
| `m` | mirror preview | `h` | toggle HUD |

## Desktop app (Tauri 2.0) — in progress

A native macOS desktop shell is being built on **Tauri 2.0** (Rust window +
process management, TypeScript control panel) while the Python CV core is reused
unchanged as a **Tauri sidecar**. The design enforces a strict split:

- **Data plane** — the video frame loop (`VideoCapture → faceray.core →
  pyvirtualcam`) stays entirely inside the Python sidecar. No frame bytes ever
  cross the IPC boundary, so UI activity can never stall the pipeline.
- **Control plane** — the UI sends only lightweight JSON control state (light
  vector, effect toggles) to the sidecar over **stdio**. Tauri owns the child
  process lifecycle and terminates it on app exit, preventing orphaned webcam
  hooks.

`ControlState` is defined once per layer and kept in lockstep:
[`src-tauri/src/ipc.rs`](src-tauri/src/ipc.rs) ↔ [`src/ipc.ts`](src/ipc.ts) ↔
[`faceray/sidecar_entry.py`](faceray/sidecar_entry.py) (`SidecarControl`).

```bash
# One-time: build the sidecar the desktop app spawns (dev shim -> repo venv)
./scripts/build_sidecar.sh

# Run the native window (Rust + Vite dev server); Tauri spawns the sidecar
npm install
npm run tauri dev
```

The sidecar is independently runnable and testable without a camera or the
desktop window — it accepts one JSON `ControlState` per line on stdin and emits
status events on stdout:

```bash
{ echo '{"blur_mode":"background","intensity":1.2}'; sleep 3; } \
  | python -m faceray.sidecar_entry --synthetic --no-sink --status-every 30
```

Status: **Phase 1 complete (Tasks 1–3).** Tauri core + IPC contract, the Python
sidecar (stdin control, graceful shutdown on parent death) with Rust stdio
plumbing (spawn / forward / kill), and the TypeScript control panel — light-vector
and intensity/ambient sliders, relight/gaze switches, and a blur segmented
control that dispatch debounced `ControlState` updates through the typed IPC
client. Verified via `cargo check` + `cargo test`, `vite build`, a 30-test
`pytest` suite, and interactive UI checks.

The control panel leads with **eye contact** (the primary feature), then other
effects, then lighting; the live preview sits on top in a fixed 16:9 frame that
holds its composition when the window is resized:

```
┌ FaceRay ───────────────── 30 fps · face ✓ ┐
│ ┌────── live 16:9 camera preview ────────┐ │
│ └────────────────────────────────────────┘ │
│ EYE CONTACT                                │
│   Gaze correction ●   Smoothing ──●──      │
│ EFFECTS                                     │
│   Relighting ●   Blur [Off|Face|Bg]        │
│ LIGHT                                       │
│   Direction X / Y / Z   Intensity  Ambient │
└────────────────────────────────────────────┘
```

The feed is mirrored at ingestion (natural webcam-mirror behavior) and streamed
near-losslessly (JPEG q95) so facial texture stays crisp. Gaze correction uses a
temporal EMA (the **Smoothing** slider) so the iris re-centres fluidly without
jitter as you glance between the lens and your monitor.

### Live preview

The window shows your **processed** camera feed so you can see each effect
apply in real time. To keep heavy frame data off the control-plane IPC, the
sidecar exposes the processed frames as an **MJPEG stream bound to loopback**
(`http://127.0.0.1:<port>/stream`, port announced in its `ready` event) and the
webview's `<img>` pulls that stream directly — nothing marshals through Rust or
TypeScript. Try it without the app or a camera:

```bash
{ echo '{"blur_mode":"background"}'; sleep 60; } \
  | python -m faceray.sidecar_entry --image portrait.jpg --no-sink \
        --preview --preview-port 8791
# then open http://127.0.0.1:8791/ in a browser
```

### Camera permission (macOS)

macOS gates camera access per app. The packaged app ships an
[`Info.plist`](src-tauri/Info.plist) with `NSCameraUsageDescription`, so it
prompts on first use. For `npm run tauri dev`, the **terminal** you launch from
is the responsible process — grant it access under **System Settings → Privacy
& Security → Camera**, then restart the terminal. If the camera is unavailable,
the preview shows a placeholder message instead of the window going dark.

## Testing

Run the pure-math unit suite (no camera or virtual-cam backend required):

```bash
pip install pytest
python -m pytest tests/ -q
```

To sanity-check the full capture→process path against your own webcam and get a
labelled before/after montage without opening a live window:

```bash
python -m scripts.capture_selfcheck --out selfcheck.png
```

## Development status

Built incrementally per the milestone plan:

- **Phase 1** — core pipeline: webcam → `virtual_sink`.
- **Phase 2** — tracking: `core/tracker.py`.
- **Phase 3** — GPU math & rendering: `core/relighter.py`, `core/modifier.py`.

All three phases are implemented and the core pipeline has been executed and
verified end-to-end on macOS / Python 3.13 (MediaPipe Tasks Face Landmarker,
NumPy CPU shading path) against a real face image.

## License

See [LICENSE](LICENSE).
