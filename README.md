# FaceRay

**Real-time, AI-powered virtual camera.** FaceRay captures your raw webcam
feed, extracts a dense 3D face mesh, applies high-fidelity face-interaction
filters — **eye-contact/gaze correction**, **face smoothing**, a **face
anonymizer**, and **background blur** — and pipes the result straight into a
native system virtual camera so Discord, Zoom, and Google Meet see it as an
ordinary webcam.

---

## Pipeline

```
[Raw Webcam Frame]           cv2.VideoCapture (native res, mirrored at ingestion)
        │
        ▼
core/tracker.py      →  MediaPipe Face Landmarker (Tasks): 468 dense + 10 iris 3D landmarks
        │
        ▼
core/modifier.py     →  Gaze warp · skin smoothing · face/background blur
        │
        ▼
drivers/virtual_sink.py  →  RGB bytes → system virtual camera (pyvirtualcam)
```

## Features

- **Eye-contact / gaze correction.** Iris landmarks (left 468–472, right 473–477)
  are compared against each eye-socket centre; a bounded affine warp nudges the
  iris back toward the aperture centre to simulate looking into the lens. A
  per-eye temporal EMA (sensitivity-controlled) keeps the warp fluid and
  jitter-free as you glance between the lens and your monitor.
- **Face smoothing (beauty filter).** An edge-preserving bilateral filter
  confined to the skin region — eyes and mouth carved out so lashes and lips
  stay sharp — with an intensity control.
- **Face anonymizer.** A heavy, opaque Gaussian restricted to the face hull for
  identity privacy; the background stays sharp.
- **Background blur.** A depth-of-field Gaussian outside the face hull; your
  face stays crisp.

## Project structure

```
faceray/                  # Python CV core (data plane) + CLI
├── core/
│   ├── tracker.py        # MediaPipe Face Landmarker (Tasks) & 3D landmark extraction
│   └── modifier.py       # Gaze correction, skin smoothing, face/background blur
├── drivers/
│   ├── virtual_sink.py   # pyvirtualcam bridge to OS video loops
│   └── preview_server.py # loopback MJPEG preview of processed frames
├── app.py                # CLI orchestration loop and OpenCV UI
├── sidecar_entry.py      # Tauri sidecar: stdio-controlled headless pipeline
├── requirements.txt      # core runtime (cross-platform)
└── requirements-dev.txt  # pytest + pyinstaller (sidecar freezing)
src/                      # desktop control-panel frontend (Vite + TypeScript)
│   ├── main.ts           # bootstrap: mount panel, dispatch control, status events
│   ├── ui.ts             # preview pane + feature cards (switches / sliders)
│   └── ipc.ts            # typed control-plane client (mirrors Rust ControlState)
src-tauri/                # Tauri 2.0 desktop shell (Rust; window + process mgmt)
│   ├── src/{main,lib,ipc,sidecar}.rs  # entry, commands, contract, process mgmt
│   ├── capabilities/     # scoped permissions (sidecar spawn)
│   ├── Info.plist        # macOS camera-usage + local-networking entitlements
│   └── tauri.conf.json   # window + bundled faceray_backend sidecar config
scripts/
├── capture_selfcheck.py  # headless one-frame pipeline check -> montage PNG
└── build_sidecar.sh      # build the target-triple Tauri sidecar (dev shim / PyInstaller)
tests/                    # pytest suite for the pure modifier math
```

The layers are strictly decoupled: `core` engines never touch the driver, and
the driver never imports the math engines.

## Requirements

- Python 3.10–3.13 (verified on 3.13)
- A webcam
- A virtual-camera backend:
  - **Windows / macOS** — OBS Virtual Camera (ships with OBS Studio ≥ 26.1)
  - **Linux** — the `v4l2loopback` kernel module

> **MediaPipe Tasks model.** `core/tracker.py` uses the MediaPipe Face
> Landmarker (Tasks API). On first run it downloads the `face_landmarker.task`
> bundle (~3.6 MB) into `~/.cache/faceray/`. Override the location with
> `FACERAY_MODEL_PATH` (explicit file) or `FACERAY_CACHE_DIR` (directory), or
> pre-place the file for fully offline use.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r faceray/requirements.txt             # all platforms, CPU-only
```

## Run

```bash
python -m faceray.app                                  # 1280x720 @ 30 fps, cam 0
python -m faceray.app --width 1920 --height 1080 --fps 60 --camera 1
python -m faceray.app --no-preview                     # headless (virtual cam only)
```

Then select **FaceRay / OBS Virtual Camera** as your webcam inside Discord,
Zoom, or Meet.

### Hotkeys (preview window)

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `q` / `Esc` | quit | `g` | toggle gaze correction |
| `f` | toggle face anonymizer | `b` | toggle background blur |
| `s` | toggle face smoothing | `m` | mirror preview |
| `h` | toggle HUD | | |

## Desktop app (Tauri 2.0) — in progress

A native macOS desktop shell is being built on **Tauri 2.0** (Rust window +
process management, TypeScript control panel) while the Python CV core is reused
unchanged as a **Tauri sidecar**. The design enforces a strict split:

- **Data plane** — the video frame loop (`VideoCapture → faceray.core →
  pyvirtualcam`) stays entirely inside the Python sidecar. No frame bytes ever
  cross the IPC boundary, so UI activity can never stall the pipeline.
- **Control plane** — the UI sends only lightweight JSON control state (feature
  toggles and sliders) to the sidecar over **stdio**. Tauri owns the child
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
{ echo '{"background_blur_enabled":true,"smoothing_enabled":true}'; sleep 3; } \
  | python -m faceray.sidecar_entry --synthetic --no-sink --status-every 30
```

The control panel is a fixed 16:9 preview pane on top and a grid of four feature
cards below:

```
┌ FaceRay ───────────────── 30 fps · face ✓ ┐
│ ┌────── live 16:9 camera preview ────────┐ │
│ └────────────────────────────────────────┘ │
│ ┌ Gaze correction ●─┐  ┌ Face anonymizer ○┐│
│ │ Sensitivity ──●── │  │                  ││
│ └───────────────────┘  └──────────────────┘│
│ ┌ Background blur ○─┐  ┌ Face smoothing  ○┐│
│ │                   │  │ Intensity ──●──  ││
│ └───────────────────┘  └──────────────────┘│
└────────────────────────────────────────────┘
```

The feed is mirrored at ingestion (natural webcam-mirror behavior) and streamed
near-losslessly (JPEG q95) so facial texture stays crisp. Every card dispatches
debounced `ControlState` updates through the typed IPC client.

### Live preview

The window shows your **processed** camera feed so you can see each effect
apply in real time. To keep heavy frame data off the control-plane IPC, the
sidecar exposes the processed frames as an **MJPEG stream bound to loopback**
(`http://127.0.0.1:<port>/stream`, port announced in its `ready` event) and the
webview's `<img>` pulls that stream directly — nothing marshals through Rust or
TypeScript. Try it without the app or a camera:

```bash
{ echo '{"background_blur_enabled":true,"smoothing_enabled":true}'; sleep 60; } \
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

The pipeline (tracker → face-interaction filters → virtual sink) and the Tauri
desktop shell are implemented and verified end-to-end on macOS / Python 3.13
(MediaPipe Tasks Face Landmarker) against a real face image: gaze correction,
face smoothing, face anonymizer, and background blur, all driven live from the
four-card control panel over the stdio control plane.

## License

See [LICENSE](LICENSE).
