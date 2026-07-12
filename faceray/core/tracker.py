"""MediaPipe Face Landmarker & 3D landmark extraction.

Stage 1 of the FaceRay pipeline. Wraps the MediaPipe *Tasks* Face Landmarker
and exposes a single, allocation-light entry point (:meth:`FaceTracker.process`)
that turns a raw BGR webcam frame into a dense 3D landmark set.

The Face Landmarker model bundle returns 478 landmarks: the 468 dense
face-mesh points plus 10 iris points (left iris 468-472, right iris 473-477).
Each landmark carries a normalized ``(x, y)`` in ``[0, 1]`` and a pseudo-metric
``z`` roughly in the same scale as ``x`` (negative = closer to the camera).

.. note::
   MediaPipe removed the legacy ``mediapipe.solutions.face_mesh`` API in the
   wheels that ship Python 3.13 support (0.10.30+), so this module targets the
   current Tasks API. Iris landmarks are always present (the Tasks model has no
   ``refine_landmarks`` toggle), so :attr:`FaceLandmarks.has_iris` is True for
   every detected face.

The tracker never raises on a missing face; callers receive ``None`` and are
expected to pass the frame through untouched.
"""

from __future__ import annotations

import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional, Union

import cv2
import numpy as np

try:  # MediaPipe is a hard runtime dependency but we guard the import so that
    import mediapipe as mp  # unit tooling / linting can load this module bare.
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode,
    )
    from mediapipe.tasks.python.vision.face_landmarker import (
        FaceLandmarker,
        FaceLandmarkerOptions,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "mediapipe is required for faceray.core.tracker. "
        "Install it with `pip install -r faceray/requirements.txt`."
    ) from exc


# --- Model bundle provisioning ---------------------------------------------
# The Tasks Face Landmarker needs an on-disk model asset. We fetch it once into
# a user cache directory (override with FACERAY_MODEL_PATH / FACERAY_CACHE_DIR)
# so the repository stays lean and offline runs reuse the cached copy.
MODEL_URL: Final[str] = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILENAME: Final[str] = "face_landmarker.task"


def _default_cache_dir() -> Path:
    override = os.environ.get("FACERAY_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "faceray"


def ensure_model(model_path: Optional[Union[str, Path]] = None) -> Path:
    """Return a local path to the Face Landmarker bundle, downloading if absent.

    Resolution order: explicit ``model_path`` arg, then ``FACERAY_MODEL_PATH``,
    then ``<cache>/face_landmarker.task``. The download streams to a temp file
    and is atomically renamed so an interrupted fetch never leaves a partial
    model in place.
    """
    if model_path is None:
        env = os.environ.get("FACERAY_MODEL_PATH")
        path = Path(env) if env else _default_cache_dir() / MODEL_FILENAME
    else:
        path = Path(model_path)

    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[FaceRay] Downloading Face Landmarker model -> {path}")
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(MODEL_URL) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        if tmp.stat().st_size == 0:
            raise RuntimeError("downloaded model was empty")
        tmp.replace(path)
    except Exception as exc:  # network failure, bad URL, disk error, ...
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not obtain the Face Landmarker model from {MODEL_URL}. "
            "Download it manually and pass its path via FACERAY_MODEL_PATH. "
            f"Underlying error: {exc}"
        ) from exc
    return path


# --- Canonical landmark indices --------------------------------------------
# Iris landmarks are always populated by the Tasks Face Landmarker model.
LEFT_IRIS: Final[tuple[int, ...]] = (468, 469, 470, 471, 472)
RIGHT_IRIS: Final[tuple[int, ...]] = (473, 474, 475, 476, 477)

# Eye-socket opening corner/lid points, used by gaze correction to locate the
# geometric centre of each eye aperture.
LEFT_EYE_RING: Final[tuple[int, ...]] = (33, 133, 159, 145, 158, 153, 160, 144)
RIGHT_EYE_RING: Final[tuple[int, ...]] = (362, 263, 386, 374, 385, 380, 387, 373)

# Approximate outer silhouette (jaw + forehead) used to build the face hull.
FACE_OVAL: Final[tuple[int, ...]] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)

TOTAL_LANDMARKS: Final[int] = 478


@dataclass(frozen=True)
class FaceLandmarks:
    """Immutable landmark payload for a single detected face.

    Attributes:
        pixels: ``(N, 2)`` float32 array of landmark positions in image pixel
            space (x = column, y = row).
        world: ``(N, 3)`` float32 array of normalized landmarks; ``x`` and
            ``y`` in ``[0, 1]``, ``z`` in MediaPipe's relative-depth units.
        frame_shape: ``(height, width)`` of the frame the landmarks came from.
    """

    pixels: np.ndarray
    world: np.ndarray
    frame_shape: tuple[int, int]

    @property
    def has_iris(self) -> bool:
        """True when refined iris landmarks (>=478 points) are present."""
        return self.pixels.shape[0] >= TOTAL_LANDMARKS

    def hull(self) -> np.ndarray:
        """Return the convex hull (int32, ``(M, 1, 2)``) of the face silhouette.

        Suitable for direct use with ``cv2.fillConvexPoly``.
        """
        oval = self.pixels[list(FACE_OVAL)].astype(np.int32)
        return cv2.convexHull(oval)

    def centroid(self, indices: tuple[int, ...]) -> np.ndarray:
        """Mean pixel position (float32 ``(2,)``) of the given landmark set."""
        return self.pixels[list(indices)].mean(axis=0).astype(np.float32)


class FaceTracker:
    """Stateful MediaPipe Face Landmarker wrapper.

    The underlying graph is stateful and, in VIDEO running mode, expects a
    single monotonic stream of frames, so a tracker instance is not thread-safe;
    use one per capture loop. The instance owns native resources and must be
    closed, either explicitly via :meth:`close` or through the context-manager
    protocol.

    Args:
        max_num_faces: Maximum faces to track (only the first is returned).
        refine_landmarks: Retained for API compatibility. The Tasks model
            always emits the 478-point refined mesh (iris included), so this
            flag no longer gates iris output.
        min_detection_confidence: Minimum confidence to start tracking a face.
        min_tracking_confidence: Minimum confidence to keep tracking a face.
        static_image_mode: When True, treat each frame independently (IMAGE
            running mode) instead of a temporal VIDEO stream.
        model_path: Explicit path to a ``face_landmarker.task`` bundle. When
            None the model is resolved/downloaded via :func:`ensure_model`.
    """

    def __init__(
        self,
        *,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if not 0.0 < min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be in (0, 1].")
        if not 0.0 < min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be in (0, 1].")

        self._static = static_image_mode
        running_mode = (
            VisionTaskRunningMode.IMAGE
            if static_image_mode
            else VisionTaskRunningMode.VIDEO
        )
        resolved_model = ensure_model(model_path)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_model)),
            running_mode=running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._closed = False

        # Monotonic millisecond clock for VIDEO mode. detect_for_video requires
        # strictly increasing timestamps; we derive them from a wall clock and
        # clamp to guarantee monotonicity even under clock jitter.
        self._t0_ns: Optional[int] = None
        self._last_ts_ms: int = -1

    def _next_timestamp_ms(self) -> int:
        now = time.monotonic_ns()
        if self._t0_ns is None:
            self._t0_ns = now
        ts = (now - self._t0_ns) // 1_000_000
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def process(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        """Extract 3D landmarks for the primary face in ``frame_bgr``.

        Args:
            frame_bgr: ``(H, W, 3)`` uint8 BGR frame from ``cv2.VideoCapture``.

        Returns:
            A :class:`FaceLandmarks` for the first detected face, or ``None``
            when the frame is empty or no face is found.
        """
        if self._closed:
            raise RuntimeError("process() called on a closed FaceTracker.")
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("FaceTracker expects an (H, W, 3) BGR frame.")

        height, width = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._static:
            result = self._landmarker.detect(mp_image)
        else:
            result = self._landmarker.detect_for_video(
                mp_image, self._next_timestamp_ms()
            )

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        count = len(landmarks)
        world = np.empty((count, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            world[i, 0] = lm.x
            world[i, 1] = lm.y
            world[i, 2] = lm.z

        pixels = np.empty((count, 2), dtype=np.float32)
        pixels[:, 0] = np.clip(world[:, 0] * width, 0.0, width - 1.0)
        pixels[:, 1] = np.clip(world[:, 1] * height, 0.0, height - 1.0)

        return FaceLandmarks(pixels=pixels, world=world, frame_shape=(height, width))

    def close(self) -> None:
        """Release the native MediaPipe graph. Idempotent."""
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self) -> "FaceTracker":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
