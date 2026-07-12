"""MediaPipe Face Mesh & 3D landmark extraction.

Stage 1 of the FaceRay pipeline. Wraps ``mediapipe.solutions.face_mesh`` and
exposes a single, allocation-light entry point (:meth:`FaceTracker.process`)
that turns a raw BGR webcam frame into a dense 3D landmark set.

With ``refine_landmarks=True`` MediaPipe returns 478 landmarks: the 468 dense
face-mesh points plus 10 iris points (left iris 468-472, right iris 473-477).
Each landmark carries a normalized ``(x, y)`` in ``[0, 1]`` and a pseudo-metric
``z`` roughly in the same scale as ``x`` (negative = closer to the camera).

The tracker never raises on a missing face; callers receive ``None`` and are
expected to pass the frame through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import cv2
import numpy as np

try:  # MediaPipe is a hard runtime dependency but we guard the import so that
    import mediapipe as mp  # unit tooling / linting can load this module bare.
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "mediapipe is required for faceray.core.tracker. "
        "Install it with `pip install -r faceray/requirements.txt`."
    ) from exc


# --- Canonical landmark indices --------------------------------------------
# Iris landmarks are only populated when ``refine_landmarks=True``.
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
    """Stateful MediaPipe Face Mesh wrapper.

    The underlying MediaPipe graph is stateful and expects a single monotonic
    stream of frames, so a tracker instance is not thread-safe; use one per
    capture loop. The instance owns native resources and must be closed, either
    explicitly via :meth:`close` or through the context-manager protocol.
    """

    def __init__(
        self,
        *,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        if not 0.0 < min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be in (0, 1].")
        if not 0.0 < min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be in (0, 1].")

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._closed = False

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

        # MediaPipe wants contiguous RGB and reads fastest from a read-only
        # buffer; flag it non-writeable to skip an internal copy.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
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
            self._mesh.close()
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
