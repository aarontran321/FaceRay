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
faceray/
├── core/
│   ├── tracker.py     # MediaPipe Face Mesh & 3D landmark extraction
│   ├── relighter.py   # CUDA/CuPy Lambertian shading & surface normals
│   └── modifier.py    # Gaze correction & Gaussian blur masks
├── drivers/
│   └── virtual_sink.py # pyvirtualcam bridge to OS video loops
├── app.py              # Orchestration loop and OpenCV UI
└── requirements.txt
```

**Strict module boundaries — do not violate these:**
- `core/*` modules are pure math/CV engines. They never import `drivers/*` or
  touch camera/OS APIs.
- `drivers/*` never imports `core/*` — it only knows about raw frame arrays.
- `app.py` is the only place that wires `core` + `drivers` together.

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

- This dev machine does **not** have a real Python interpreter installed
  (only the Windows Store stub) and does not have MediaPipe / CuPy /
  pyvirtualcam installed. Code here has been written carefully but **not
  executed** — treat "should work" claims with appropriate skepticism until
  verified on a machine with a real environment.
- A virtual-camera backend is required to actually output to Discord/Zoom/Meet:
  OBS Virtual Camera (Windows/macOS) or `v4l2loopback` (Linux).
- `cupy-cuda12x` in `requirements.txt` is optional — only needed for the GPU
  shading path; everything degrades gracefully to NumPy without it.

## Milestone plan (for context on ordering)

1. **Phase 1** — core pipeline verification: `app.py` running a clean webcam
   loop through `drivers/virtual_sink.py`.
2. **Phase 2** — tracking: `core/tracker.py`.
3. **Phase 3** — GPU math and rendering: `core/relighter.py`,
   `core/modifier.py`.

All three phases are implemented as of the current `main`. Future work should
build on top of this, not restructure it, unless explicitly requested.

## Testing

No test suite exists yet. There is no way to run/verify the app on this
machine (see Environment notes). If you add tests, prefer testing the pure
math in `core/relighter.py` (normal/shading computation) and `core/modifier.py`
(blur mask geometry) with synthetic landmark arrays, since those don't require
a real camera or virtual-cam backend.
