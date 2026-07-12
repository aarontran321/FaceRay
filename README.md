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
faceray/
├── core/
│   ├── tracker.py        # MediaPipe Face Landmarker (Tasks) & 3D landmark extraction
│   ├── relighter.py      # CUDA/CuPy Lambertian shading & surface normals
│   └── modifier.py       # Gaze correction & Gaussian blur masks
├── drivers/
│   └── virtual_sink.py   # pyvirtualcam bridge to OS video loops
├── app.py                # Orchestration loop and OpenCV UI
├── requirements.txt      # core runtime (cross-platform, CPU path)
└── requirements-gpu.txt  # optional CUDA/CuPy GPU acceleration
scripts/
└── capture_selfcheck.py  # headless one-frame pipeline check -> montage PNG
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
