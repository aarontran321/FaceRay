"""FaceRay -- main execution script, orchestration loop, and CV2 UI.

Wires the four pipeline stages together::

    [webcam] -> tracker -> relighter -> modifier -> virtual_sink

and drives them from a single capture loop. A lightweight OpenCV preview window
shows the processed frame with an on-screen HUD and accepts hotkeys to toggle
each effect live. The same processed frame is pushed to the system virtual
camera so Discord / Zoom / Meet see FaceRay as a native device.

Run::

    python -m faceray.app                      # 1280x720 @ 30 fps, cam 0
    python -m faceray.app --width 1920 --height 1080 --fps 60 --camera 1
    python -m faceray.app --no-preview         # headless (virtual cam only)
    python -m faceray.app --no-gpu             # force the NumPy shading path

Hotkeys (preview window focused):
    q / Esc  quit                 l  toggle relighting
    g        toggle gaze fix      b  cycle blur (off/face/background)
    [  / ]   orbit light left/right     -/=  dim / brighten light
    m        mirror preview       h  toggle HUD
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from faceray.core import FaceTracker, Relighter, Modifier
from faceray.core.modifier import BlurMode
from faceray.drivers import VirtualSink
from faceray.drivers.virtual_sink import VirtualSinkError


WINDOW_NAME = "FaceRay"


@dataclass
class PipelineToggles:
    """Live on/off state for each effect, driven by the UI hotkeys."""

    relight: bool = True
    gaze: bool = True
    mirror: bool = True
    show_hud: bool = True
    blur: BlurMode = field(default=BlurMode.OFF)


class FaceRayApp:
    """Owns the capture loop and the lifetime of every pipeline resource."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._toggles = PipelineToggles()

        self._capture: Optional[cv2.VideoCapture] = None
        self._tracker: Optional[FaceTracker] = None
        self._sink: Optional[VirtualSink] = None
        self._relighter = Relighter(use_gpu=not args.no_gpu)
        self._modifier = Modifier()

        self._fps_ema: float = 0.0

    # -- Resource lifecycle -------------------------------------------------
    def _open_capture(self) -> cv2.VideoCapture:
        # CAP_DSHOW avoids slow MSMF startup on Windows; harmless elsewhere.
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._args.camera, backend)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self._args.camera}. "
                "Check that a webcam is connected and not in use by another app."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._args.height)
        cap.set(cv2.CAP_PROP_FPS, self._args.fps)
        return cap

    def _setup(self) -> None:
        self._capture = self._open_capture()

        # Adopt the resolution the driver actually granted us.
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or self._args.width
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self._args.height
        self._args.width, self._args.height = width, height

        self._tracker = FaceTracker(
            max_num_faces=1,
            refine_landmarks=True,  # required for iris / gaze correction
        )

        self._sink = VirtualSink(width=width, height=height, fps=self._args.fps)
        try:
            self._sink.open()
            print(f"[FaceRay] Virtual camera online: {self._sink.device_name}")
        except VirtualSinkError as exc:
            # A missing virtual-cam backend must not stop local preview.
            print(f"[FaceRay] WARNING: {exc}")
            print("[FaceRay] Continuing in preview-only mode.")
            self._sink = None

        mode = "GPU (CuPy)" if self._relighter.gpu_active else "CPU (NumPy)"
        print(f"[FaceRay] Relighting backend: {mode}")

    def _teardown(self) -> None:
        if self._sink is not None:
            self._sink.close()
        if self._tracker is not None:
            self._tracker.close()
        if self._capture is not None:
            self._capture.release()
        cv2.destroyAllWindows()

    # -- Per-frame processing ----------------------------------------------
    def _process(self, frame_bgr: np.ndarray) -> np.ndarray:
        assert self._tracker is not None
        landmarks = self._tracker.process(frame_bgr)
        if landmarks is None:
            return frame_bgr  # no face -> clean passthrough

        out = frame_bgr
        if self._toggles.relight:
            out = self._relighter.apply(out, landmarks)

        self._modifier.gaze_strength = 0.7 if self._toggles.gaze else 0.0
        self._modifier.blur_mode = self._toggles.blur
        out = self._modifier.apply(out, landmarks)
        return out

    # -- Main loop ----------------------------------------------------------
    def run(self) -> int:
        self._setup()
        assert self._capture is not None
        preview = not self._args.no_preview
        if preview:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            while True:
                t0 = time.perf_counter()
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    # Tolerate transient capture dropouts before giving up.
                    if not self._await_reconnect():
                        print("[FaceRay] Camera disconnected. Exiting.")
                        return 1
                    continue

                processed = self._process(frame)

                if self._sink is not None:
                    try:
                        self._sink.send(processed)
                    except VirtualSinkError as exc:
                        print(f"[FaceRay] Virtual camera lost: {exc}")
                        self._sink = None

                self._update_fps(time.perf_counter() - t0)

                if preview:
                    if not self._render_preview(processed):
                        break
                elif self._sink is None:
                    print("[FaceRay] No preview and no virtual camera. Exiting.")
                    return 1
        except KeyboardInterrupt:
            print("\n[FaceRay] Interrupted.")
        finally:
            self._teardown()
        return 0

    def _await_reconnect(self, attempts: int = 30, delay: float = 0.1) -> bool:
        """Poll a stalled capture device a few times before declaring it dead."""
        assert self._capture is not None
        for _ in range(attempts):
            time.sleep(delay)
            ok, _ = self._capture.read()
            if ok:
                return True
        return False

    # -- UI -----------------------------------------------------------------
    def _update_fps(self, elapsed: float) -> None:
        if elapsed <= 0:
            return
        inst = 1.0 / elapsed
        self._fps_ema = inst if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst

    def _render_preview(self, frame_bgr: np.ndarray) -> bool:
        """Draw HUD, show the window, and handle hotkeys. Returns False to quit."""
        view = cv2.flip(frame_bgr, 1) if self._toggles.mirror else frame_bgr
        if self._toggles.show_hud:
            view = view.copy()
            self._draw_hud(view)
        cv2.imshow(WINDOW_NAME, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            return False
        if key == ord("l"):
            self._toggles.relight = not self._toggles.relight
        elif key == ord("g"):
            self._toggles.gaze = not self._toggles.gaze
        elif key == ord("b"):
            self._toggles.blur = self._modifier.cycle_blur_mode()
        elif key == ord("m"):
            self._toggles.mirror = not self._toggles.mirror
        elif key == ord("h"):
            self._toggles.show_hud = not self._toggles.show_hud
        elif key == ord("["):
            self._relighter.orbit_light(-0.20)
        elif key == ord("]"):
            self._relighter.orbit_light(0.20)
        elif key in (ord("-"), ord("_")):
            self._relighter.intensity = self._relighter.intensity - 0.1
        elif key in (ord("="), ord("+")):
            self._relighter.intensity = self._relighter.intensity + 0.1

        # Window closed via the title-bar button.
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True

    def _draw_hud(self, view: np.ndarray) -> None:
        backend = "GPU" if self._relighter.gpu_active else "CPU"
        sink = self._sink.device_name if self._sink is not None else "preview-only"
        lines = [
            f"FaceRay  {self._fps_ema:5.1f} FPS  [{backend}]",
            f"relight:{_on(self._toggles.relight)} "
            f"gaze:{_on(self._toggles.gaze)} "
            f"blur:{self._toggles.blur.value} "
            f"light:{self._relighter.intensity:.1f}",
            f"out: {sink}",
            "q quit  l light  g gaze  b blur  [ ] orbit  -/= dim/bright",
        ]
        y = 24
        for text in lines:
            cv2.putText(view, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (60, 255, 120), 1, cv2.LINE_AA)
            y += 24


def _on(flag: bool) -> str:
    return "on" if flag else "off"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="faceray",
        description="Real-time AI virtual camera: relighting, gaze fix, smart blur.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target frame rate.")
    parser.add_argument("--no-preview", action="store_true",
                        help="Run headless; push to the virtual camera only.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force the CPU (NumPy) shading path even if CuPy is present.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        return FaceRayApp(args).run()
    except RuntimeError as exc:
        print(f"[FaceRay] Fatal: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
