"""pyvirtualcam bridge to OS video loops.

Final stage of the FaceRay pipeline. Wraps ``pyvirtualcam.Camera`` and exposes
a minimal, fault-tolerant sink that accepts BGR frames (OpenCV's native order),
converts them to the RGB byte layout the virtual device expects, and pushes
them to the system virtual camera.

Backends are OS-specific and must be installed separately:
    * Windows -- OBS Virtual Camera (ships with OBS Studio >= 26.1).
    * Linux   -- v4l2loopback kernel module.
    * macOS   -- OBS Virtual Camera.

The sink never crashes the capture loop: construction validates parameters,
:meth:`open` surfaces a clear error if no backend is present, and :meth:`send`
tolerates transient device write failures.
"""

from __future__ import annotations

from typing import Final, Optional

import numpy as np

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "pyvirtualcam is required for faceray.drivers.virtual_sink. "
        "Install it with `pip install -r faceray/requirements.txt`."
    ) from exc


class VirtualSinkError(RuntimeError):
    """Raised when the virtual camera backend cannot be opened."""


class VirtualSink:
    """Fault-tolerant bridge from BGR frames to a system virtual camera.

    Args:
        width: Output frame width in pixels.
        height: Output frame height in pixels.
        fps: Target frame rate advertised to consuming applications.
        device: Optional explicit backend device name; ``None`` lets
            pyvirtualcam auto-select the first available virtual camera.
        print_fps: Forwarded to pyvirtualcam for its internal FPS logging.
    """

    _MAX_CONSECUTIVE_FAILURES: Final[int] = 30

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float = 30.0,
        device: Optional[str] = None,
        print_fps: bool = False,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("VirtualSink width/height must be positive.")
        if fps <= 0:
            raise ValueError("VirtualSink fps must be positive.")

        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._device = device
        self._print_fps = print_fps

        self._camera: Optional["pyvirtualcam.Camera"] = None
        self._failures = 0

    @property
    def is_open(self) -> bool:
        return self._camera is not None

    @property
    def device_name(self) -> Optional[str]:
        """The backend device actually in use, once opened."""
        return self._camera.device if self._camera is not None else None

    def open(self) -> "VirtualSink":
        """Acquire the virtual camera backend.

        Raises:
            VirtualSinkError: if no virtual-camera backend is installed or the
                device cannot be acquired.
        """
        if self._camera is not None:
            return self
        try:
            self._camera = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                device=self._device,
                fmt=PixelFormat.RGB,
                print_fps=self._print_fps,
            )
        except Exception as exc:  # RuntimeError from missing backend, etc.
            raise VirtualSinkError(
                "Could not open a virtual camera. Ensure a backend is installed "
                "(OBS Virtual Camera on Windows/macOS, v4l2loopback on Linux). "
                f"Underlying error: {exc}"
            ) from exc
        self._failures = 0
        return self

    def send(self, frame_bgr: np.ndarray) -> bool:
        """Push one BGR frame to the virtual camera.

        The frame is resized to the sink resolution when needed and converted
        BGR->RGB. Returns ``True`` on success. Transient write failures are
        tolerated and logged internally; after
        :data:`_MAX_CONSECUTIVE_FAILURES` in a row the sink raises to let the
        caller tear down cleanly.

        Raises:
            VirtualSinkError: if the sink is closed, or the device has failed
                persistently.
        """
        if self._camera is None:
            raise VirtualSinkError("send() called before open().")
        if frame_bgr is None or frame_bgr.size == 0:
            return False

        rgb = self._to_rgb(frame_bgr)
        try:
            self._camera.send(rgb)
            self._camera.sleep_until_next_frame()
        except Exception:
            self._failures += 1
            if self._failures >= self._MAX_CONSECUTIVE_FAILURES:
                raise VirtualSinkError(
                    "Virtual camera stopped accepting frames "
                    f"({self._failures} consecutive failures)."
                )
            return False
        self._failures = 0
        return True

    def _to_rgb(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Resize (if needed) and convert a BGR frame to a contiguous RGB array.

        ``cv2`` is imported lazily so this driver module carries no hard
        dependency on OpenCV at import time.
        """
        import cv2

        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            frame_bgr = cv2.resize(
                frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)

    def close(self) -> None:
        """Release the virtual camera backend. Idempotent."""
        if self._camera is not None:
            try:
                self._camera.close()
            finally:
                self._camera = None

    def __enter__(self) -> "VirtualSink":
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
