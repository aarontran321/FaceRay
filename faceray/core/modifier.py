"""Gaze correction algorithms & Gaussian blurring masks.

Stage 3 of the FaceRay pipeline (Features B and C).

Feature B -- Eye-Contact / Gaze Correction
    Uses the refined iris landmarks (left 468-472, right 473-477) and the eye
    aperture ring landmarks to measure the pupil-to-socket displacement. When
    the user looks away from the lens, each eye region is warped by a bounded
    affine translation that nudges the iris texture back toward the geometric
    centre of the aperture, simulating direct eye contact.

Feature C -- Face Blur / Identity Privacy
    Builds a dynamic binary mask from the face convex hull and applies a
    Gaussian blur to either the face (identity protection) or its inverse
    (background privacy) depending on :attr:`blur_mode`.

Both features are optional and independently toggleable; every method is a
safe no-op when its inputs are missing or degenerate.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from faceray.core.tracker import (
    FaceLandmarks,
    LEFT_EYE_RING,
    LEFT_IRIS,
    RIGHT_EYE_RING,
    RIGHT_IRIS,
)


class BlurMode(Enum):
    """Which region the Gaussian blur is applied to."""

    OFF = "off"
    FACE = "face"          # blur the face -> identity privacy
    BACKGROUND = "background"  # blur everything but the face


class Modifier:
    """Gaze correction and Gaussian identity/background blur.

    Args:
        gaze_strength: Fraction of the measured pupil displacement corrected,
            in ``[0, 1]``. ``0`` disables correction.
        gaze_max_shift: Hard cap (in pixels) on per-eye translation to keep the
            warp stable when tracking is noisy.
        blur_mode: Initial :class:`BlurMode`.
        blur_kernel: Odd Gaussian kernel size for the blur pass.
    """

    def __init__(
        self,
        *,
        gaze_strength: float = 0.7,
        gaze_max_shift: float = 12.0,
        blur_mode: BlurMode = BlurMode.OFF,
        blur_kernel: int = 31,
    ) -> None:
        self.gaze_strength = gaze_strength
        self.gaze_max_shift = float(max(0.0, gaze_max_shift))
        self.blur_mode = blur_mode
        self.blur_kernel = blur_kernel

    # -- Configuration ------------------------------------------------------
    @property
    def gaze_strength(self) -> float:
        return self._gaze_strength

    @gaze_strength.setter
    def gaze_strength(self, value: float) -> None:
        self._gaze_strength = float(np.clip(value, 0.0, 1.0))

    @property
    def blur_kernel(self) -> int:
        return self._blur_kernel

    @blur_kernel.setter
    def blur_kernel(self, value: int) -> None:
        k = int(value)
        if k < 3:
            k = 3
        if k % 2 == 0:  # Gaussian kernels must be odd
            k += 1
        self._blur_kernel = k

    def cycle_blur_mode(self) -> BlurMode:
        """Advance the blur mode OFF -> FACE -> BACKGROUND -> OFF and return it."""
        order = (BlurMode.OFF, BlurMode.FACE, BlurMode.BACKGROUND)
        self.blur_mode = order[(order.index(self.blur_mode) + 1) % len(order)]
        return self.blur_mode

    # -- Public pipeline ----------------------------------------------------
    def apply(self, frame_bgr: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Run gaze correction then blur, honouring the current toggles."""
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr
        out = self.correct_gaze(frame_bgr, landmarks)
        out = self.apply_blur(out, landmarks)
        return out

    # -- Feature B: gaze correction -----------------------------------------
    def correct_gaze(self, frame_bgr: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Warp each eye region so the iris re-centres in its aperture."""
        if self._gaze_strength <= 0.0 or not landmarks.has_iris:
            return frame_bgr

        out = frame_bgr
        for iris_idx, ring_idx in (
            (LEFT_IRIS, LEFT_EYE_RING),
            (RIGHT_IRIS, RIGHT_EYE_RING),
        ):
            out = self._recentre_eye(out, landmarks, iris_idx, ring_idx)
        return out

    def _recentre_eye(
        self,
        frame_bgr: np.ndarray,
        landmarks: FaceLandmarks,
        iris_idx: Tuple[int, ...],
        ring_idx: Tuple[int, ...],
    ) -> np.ndarray:
        pupil = landmarks.centroid(iris_idx)
        socket = landmarks.centroid(ring_idx)
        offset = socket - pupil  # vector that moves the pupil to centre

        shift = offset * self._gaze_strength
        norm = float(np.linalg.norm(shift))
        if norm < 0.5:  # already centred -> nothing to do
            return frame_bgr
        if norm > self.gaze_max_shift:
            shift = shift * (self.gaze_max_shift / norm)

        roi = self._eye_roi(frame_bgr.shape[:2], landmarks, ring_idx)
        if roi is None:
            return frame_bgr
        x0, y0, x1, y1 = roi

        patch = frame_bgr[y0:y1, x0:x1]
        if patch.size == 0:
            return frame_bgr

        translation = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)
        warped = cv2.warpAffine(
            patch, translation, (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

        # Feather the warped iris patch back in with an elliptical mask so the
        # eyelids and surrounding skin stay put.
        blend = self._eye_blend_mask(patch.shape[:2], landmarks, ring_idx, (x0, y0))
        result = frame_bgr.copy()
        region = patch * (1.0 - blend) + warped * blend
        result[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _eye_roi(
        shape: Tuple[int, int],
        landmarks: FaceLandmarks,
        ring_idx: Tuple[int, ...],
        pad: float = 1.6,
    ) -> Optional[Tuple[int, int, int, int]]:
        h, w = shape
        ring = landmarks.pixels[list(ring_idx)]
        cx, cy = ring.mean(axis=0)
        half_w = (float(np.ptp(ring[:, 0])) * pad) / 2.0 + 4.0
        half_h = (float(np.ptp(ring[:, 1])) * pad) / 2.0 + 4.0
        x0 = int(np.clip(cx - half_w, 0, w - 1))
        y0 = int(np.clip(cy - half_h, 0, h - 1))
        x1 = int(np.clip(cx + half_w, 0, w))
        y1 = int(np.clip(cy + half_h, 0, h))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _eye_blend_mask(
        patch_shape: Tuple[int, int],
        landmarks: FaceLandmarks,
        ring_idx: Tuple[int, ...],
        origin: Tuple[int, int],
    ) -> np.ndarray:
        ph, pw = patch_shape
        mask = np.zeros((ph, pw), dtype=np.float32)
        ring = landmarks.pixels[list(ring_idx)].astype(np.float32)
        cx = ring[:, 0].mean() - origin[0]
        cy = ring[:, 1].mean() - origin[1]
        ax = max(3.0, float(np.ptp(ring[:, 0])) * 0.55)
        ay = max(3.0, float(np.ptp(ring[:, 1])) * 0.55)
        cv2.ellipse(
            mask, (int(cx), int(cy)), (int(ax), int(ay)),
            0, 0, 360, color=1.0, thickness=-1,
        )
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.5, min(ax, ay) * 0.25))
        return mask[:, :, None]

    # -- Feature C: Gaussian blur -------------------------------------------
    def apply_blur(self, frame_bgr: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Blur the face or background per :attr:`blur_mode`."""
        if self.blur_mode is BlurMode.OFF:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        hull = landmarks.hull()
        face_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillConvexPoly(face_mask, hull, 1.0, lineType=cv2.LINE_AA)
        # Soften the hull edge so the composite has no hard seam.
        face_mask = cv2.GaussianBlur(face_mask, (0, 0), sigmaX=max(4.0, min(h, w) * 0.01))

        k = self._blur_kernel
        blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)

        select = face_mask if self.blur_mode is BlurMode.FACE else (1.0 - face_mask)
        select = select[:, :, None]
        out = frame_bgr * (1.0 - select) + blurred * select
        return np.clip(out, 0, 255).astype(np.uint8)
