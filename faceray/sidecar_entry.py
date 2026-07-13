"""FaceRay Tauri sidecar entry point (control-plane consumer).

This is the desktop-app counterpart to :mod:`faceray.app`: instead of an OpenCV
window and keyboard hotkeys, it is driven by a parent process (the Tauri Rust
shell) over **stdio**.

IPC contract (asymmetric, line-delimited JSON)::

    stdin  <-  control  : one ``ControlState`` object per line (UI -> sidecar)
    stdout  ->  status  : ``{"type": ...}`` events per line (sidecar -> UI)
    stderr  ->  logs    : human-readable diagnostics (incl. MediaPipe/glog)

The **data plane** — ``VideoCapture -> faceray.core -> pyvirtualcam`` — stays
entirely inside this process; no pixel data ever crosses stdio. Only the small
scalar :class:`SidecarControl` payload does.

Lifecycle: a daemon thread reads stdin. When the parent dies its pipe closes,
stdin hits EOF, and the reader sets the shutdown event, so the capture loop tears
down within one frame — no orphaned webcam hooks. SIGINT/SIGTERM do the same.

``SidecarControl`` mirrors ``ControlState`` in ``src-tauri/src/ipc.rs`` and
``src/ipc.ts`` field-for-field; keep the three in sync.

Run (normally spawned by Tauri; these are for manual testing)::

    python -m faceray.sidecar_entry                      # webcam, virtual cam
    python -m faceray.sidecar_entry --synthetic --no-sink
    python -m faceray.sidecar_entry --image portrait.jpg --no-sink
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from faceray.core import FaceTracker, Modifier, Relighter
from faceray.core.modifier import BlurMode
from faceray.drivers import VirtualSink
from faceray.drivers.preview_server import PreviewServer
from faceray.drivers.virtual_sink import VirtualSinkError

# Gaze strength applied when gaze correction is enabled (matches app.py).
_GAZE_ON_STRENGTH: float = 0.7


@dataclass
class SidecarControl:
    """Control-plane state received from the UI. Mirrors Rust/TS ``ControlState``.

    Defaults match the ``faceray.core`` constructor defaults so the very first
    frame (before any control message arrives) looks identical to the CLI.
    """

    light_x: float = 0.4
    light_y: float = -0.3
    light_z: float = -1.0
    intensity: float = 0.6
    ambient: float = 0.55
    relight_enabled: bool = True
    gaze_enabled: bool = True
    blur_mode: BlurMode = BlurMode.OFF

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], base: Optional["SidecarControl"] = None
    ) -> "SidecarControl":
        """Build from a decoded JSON object, inheriting unset keys from ``base``.

        Tolerating partial payloads keeps the channel robust: a UI that sends a
        single changed field still produces a valid, fully-populated state.
        """
        base = base or cls()
        raw_mode = data.get("blur_mode", base.blur_mode.value)
        blur = raw_mode if isinstance(raw_mode, BlurMode) else BlurMode(str(raw_mode))
        return cls(
            light_x=float(data.get("light_x", base.light_x)),
            light_y=float(data.get("light_y", base.light_y)),
            light_z=float(data.get("light_z", base.light_z)),
            intensity=float(data.get("intensity", base.intensity)),
            ambient=float(data.get("ambient", base.ambient)),
            relight_enabled=bool(data.get("relight_enabled", base.relight_enabled)),
            gaze_enabled=bool(data.get("gaze_enabled", base.gaze_enabled)),
            blur_mode=blur,
        )

    @classmethod
    def from_json(
        cls, line: str, base: Optional["SidecarControl"] = None
    ) -> "SidecarControl":
        return cls.from_dict(json.loads(line), base)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "light_x": self.light_x,
            "light_y": self.light_y,
            "light_z": self.light_z,
            "intensity": self.intensity,
            "ambient": self.ambient,
            "relight_enabled": self.relight_enabled,
            "gaze_enabled": self.gaze_enabled,
            "blur_mode": self.blur_mode.value,
        }

    def apply(self, relighter: Relighter, modifier: Modifier) -> None:
        """Push this control state onto the live pipeline engines.

        ``relight_enabled`` is honoured by the caller's processing branch, not
        here, since it gates whether the relighter runs at all.
        """
        vec = (self.light_x, self.light_y, self.light_z)
        if any(abs(v) > 1e-9 for v in vec):  # setter rejects a zero vector
            relighter.light_direction = vec
        relighter.intensity = self.intensity  # property setter clips to [0, 2]
        relighter.ambient = float(np.clip(self.ambient, 0.0, 1.0))
        modifier.gaze_strength = _GAZE_ON_STRENGTH if self.gaze_enabled else 0.0
        modifier.blur_mode = self.blur_mode


class Sidecar:
    """Owns the capture loop, pipeline resources, and the stdio control channel."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._lock = threading.Lock()
        self._state = SidecarControl()
        self._shutdown = threading.Event()

        self._relighter = Relighter(use_gpu=not args.no_gpu)
        self._modifier = Modifier()

        self._capture: Optional[cv2.VideoCapture] = None
        self._still: Optional[np.ndarray] = None  # image / synthetic source frame
        self._tracker: Optional[FaceTracker] = None
        self._sink: Optional[VirtualSink] = None
        self._preview: Optional[PreviewServer] = None

    # -- stdio helpers ------------------------------------------------------
    def _emit(self, obj: Dict[str, Any]) -> None:
        """Write one status event as a JSON line to stdout (the UI channel)."""
        try:
            sys.stdout.write(json.dumps(obj) + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, ValueError):
            # Parent went away mid-write; treat as a shutdown signal.
            self._shutdown.set()

    def _log(self, message: str) -> None:
        print(f"[sidecar] {message}", file=sys.stderr, flush=True)

    # -- lifecycle ----------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            self._log(f"signal {signum} received; shutting down")
            self._shutdown.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):  # pragma: no cover - non-main thread / OS gap
                pass

    def _open_source(self) -> bool:
        """Open the frame source. Returns False (after emitting an error) on failure."""
        w, h = self._args.width, self._args.height
        if self._args.synthetic:
            self._still = np.full((h, w, 3), 128, dtype=np.uint8)
            return True
        if self._args.image:
            img = cv2.imread(self._args.image)
            if img is None:
                self._emit({"type": "error", "message": f"cannot read image: {self._args.image}"})
                return False
            self._still = img
            return True

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._args.camera, backend)
        if not cap.isOpened():
            self._emit({"type": "error", "message": f"cannot open camera {self._args.camera}"})
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, self._args.fps)
        self._capture = cap
        # Adopt the resolution the driver actually granted.
        self._args.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or w
        self._args.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or h
        return True

    def _error_frame(self, text: str) -> np.ndarray:
        h, w = self._args.height, self._args.width
        frame = np.full((h, w, 3), 20, dtype=np.uint8)
        cv2.putText(frame, text, (32, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    min(1.0, w / 900.0), (60, 120, 255), 2, cv2.LINE_AA)
        return frame

    def _setup(self) -> bool:
        if not self._open_source():
            # With a live preview we keep running and show why in-frame, so the
            # UI stays connected instead of the window going dead.
            if self._args.preview and not self._args.image and not self._args.synthetic:
                self._still = self._error_frame(
                    "Camera unavailable - check macOS Camera permission"
                )
                self._emit({"type": "warning", "message": "camera unavailable; showing placeholder"})
            else:
                return False

        static = self._still is not None
        self._tracker = FaceTracker(
            max_num_faces=1, refine_landmarks=True, static_image_mode=static
        )

        if not self._args.no_sink:
            try:
                self._sink = VirtualSink(
                    width=self._args.width, height=self._args.height, fps=self._args.fps
                ).open()
                self._log(f"virtual camera online: {self._sink.device_name}")
            except VirtualSinkError as exc:
                self._emit({"type": "warning", "message": str(exc)})
                self._sink = None

        if self._args.preview:
            self._preview = PreviewServer(
                port=self._args.preview_port,
                quality=self._args.preview_quality,
                max_width=self._args.preview_width,
            ).start()
            self._log(f"preview stream: {self._preview.url}")
        return True

    def _teardown(self) -> None:
        if self._preview is not None:
            self._preview.stop()
        if self._sink is not None:
            self._sink.close()
        if self._tracker is not None:
            self._tracker.close()
        if self._capture is not None:
            self._capture.release()

    # -- control channel ----------------------------------------------------
    def _read_stdin(self) -> None:
        """Daemon-thread loop: parse control lines; EOF => parent gone => stop."""
        for raw in sys.stdin:
            if self._shutdown.is_set():
                break
            line = raw.strip()
            if not line:
                continue
            try:
                with self._lock:
                    self._state = SidecarControl.from_json(line, self._state)
                    snapshot = self._state.to_dict()
                self._emit({"type": "ack", "state": snapshot})
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                self._emit({"type": "error", "message": f"bad control payload: {exc}"})
        self._shutdown.set()

    # -- per-frame processing ----------------------------------------------
    def _read_frame(self) -> Optional[np.ndarray]:
        if self._still is not None:
            return self._still.copy()
        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def _process(
        self, frame: np.ndarray, state: SidecarControl
    ) -> Tuple[np.ndarray, bool]:
        assert self._tracker is not None
        state.apply(self._relighter, self._modifier)
        landmarks = self._tracker.process(frame)
        if landmarks is None:
            return frame, False
        out = frame
        if state.relight_enabled:
            out = self._relighter.apply(out, landmarks)
        out = self._modifier.apply(out, landmarks)
        return out, True

    def _pace(self, frame_start: float) -> None:
        if self._args.fps <= 0:
            return
        period = 1.0 / self._args.fps
        remaining = period - (time.perf_counter() - frame_start)
        if remaining > 0:
            # Wait on the shutdown event so a stop request can't be delayed a
            # whole frame period.
            self._shutdown.wait(timeout=remaining)

    # -- main loop ----------------------------------------------------------
    def run(self) -> int:
        self._install_signal_handlers()
        if not self._setup():
            self._teardown()
            return 1

        reader = threading.Thread(target=self._read_stdin, name="stdin-reader", daemon=True)
        reader.start()

        with self._lock:
            initial = self._state.to_dict()
        self._emit({
            "type": "ready",
            "width": self._args.width,
            "height": self._args.height,
            "gpu": self._relighter.gpu_active,
            "sink": self._sink.device_name if self._sink is not None else None,
            "preview": self._preview.url if self._preview is not None else None,
            "state": initial,
        })

        frames = 0
        fps_ema = 0.0
        try:
            while not self._shutdown.is_set():
                t0 = time.perf_counter()
                frame = self._read_frame()
                if frame is None:
                    self._emit({"type": "warning", "message": "dropped frame"})
                    if not self._shutdown.wait(timeout=0.1):
                        continue
                    break

                with self._lock:
                    state = self._state
                processed, face = self._process(frame, state)

                if self._sink is not None:
                    try:
                        self._sink.send(processed)
                    except VirtualSinkError as exc:
                        self._emit({"type": "warning", "message": f"sink lost: {exc}"})
                        self._sink = None

                if self._preview is not None:
                    self._preview.update(processed)

                frames += 1
                dt = time.perf_counter() - t0
                if dt > 0:
                    inst = 1.0 / dt
                    fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst
                if self._args.status_every > 0 and frames % self._args.status_every == 0:
                    self._emit({"type": "status", "frame": frames,
                                "fps": round(fps_ema, 1), "face": face})

                if self._args.max_frames and frames >= self._args.max_frames:
                    break
                self._pace(t0)
        finally:
            self._teardown()

        self._emit({"type": "bye", "frames": frames})
        return 0


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="faceray-sidecar",
        description="FaceRay Tauri sidecar: stdio-controlled headless pipeline.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target frame rate.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force the CPU (NumPy) shading path.")
    parser.add_argument("--no-sink", action="store_true",
                        help="Skip the virtual camera (process frames only).")
    parser.add_argument("--preview", action="store_true",
                        help="Serve an MJPEG preview of the processed feed on loopback.")
    parser.add_argument("--preview-port", type=int, default=0,
                        help="Preview TCP port (0 = OS-assigned).")
    parser.add_argument("--preview-quality", type=int, default=70,
                        help="Preview JPEG quality (1-100).")
    parser.add_argument("--preview-width", type=int, default=640,
                        help="Max preview width in pixels (downscale above this).")
    parser.add_argument("--image", type=str, default=None,
                        help="Loop a static image instead of the webcam (testing).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Loop a synthetic grey frame; no camera needed (testing).")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = run until stopped).")
    parser.add_argument("--status-every", type=int, default=30,
                        help="Emit a status event every N frames (0 = never).")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    return Sidecar(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
