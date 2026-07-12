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
core/tracker.py      →  MediaPipe Face Mesh: 468 dense + 10 iris 3D landmarks
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
│   ├── tracker.py        # MediaPipe Face Mesh & 3D landmark extraction
│   ├── relighter.py      # CUDA/CuPy Lambertian shading & surface normals
│   └── modifier.py       # Gaze correction & Gaussian blur masks
├── drivers/
│   └── virtual_sink.py   # pyvirtualcam bridge to OS video loops
├── app.py                # Orchestration loop and OpenCV UI
└── requirements.txt
```

The layers are strictly decoupled: `core` engines never touch the driver, and
the driver never imports the math engines.

## Requirements

- Python 3.10+
- A webcam
- A virtual-camera backend:
  - **Windows / macOS** — OBS Virtual Camera (ships with OBS Studio ≥ 26.1)
  - **Linux** — the `v4l2loopback` kernel module
- **Optional:** an NVIDIA GPU with a CUDA 12.x runtime for the CuPy shading path.
  Without it FaceRay transparently falls back to NumPy on the CPU.

## Install

```bash
pip install -r faceray/requirements.txt
```

> `cupy-cuda12x` is optional — remove it if you have no CUDA 12 GPU; the
> relighter automatically uses the NumPy backend.

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

## Development status

Built incrementally per the milestone plan:

- **Phase 1** — core pipeline: webcam → `virtual_sink`.
- **Phase 2** — tracking: `core/tracker.py`.
- **Phase 3** — GPU math & rendering: `core/relighter.py`, `core/modifier.py`.

## License

See [LICENSE](LICENSE).
