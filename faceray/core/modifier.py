"""Face-interaction filters: gaze correction, skin smoothing, and blur masks.

Stage 3 of the FaceRay pipeline. Every effect operates purely on the 3D face
landmarks plus the BGR frame — there is no lighting math in the pipeline.

Features (each independently toggleable, each a safe no-op when disabled or when
its inputs are degenerate):

* **Eye-contact / gaze correction** — remap the iris (landmarks 468-472 /
  473-477) toward the eye-aperture centre with a temporally smoothed affine
  warp, so glancing down at the monitor still reads as looking at the lens.
  The per-eye shift is EMA-smoothed to eradicate micro-jitter.
* **Face smoothing (beauty filter)** — an edge-preserving bilateral filter
  confined to the skin region, with the eyes and mouth carved out so lashes and
  lips stay sharp.
* **Face anonymiser** — a heavy, opaque Gaussian restricted to the face hull;
  the background stays sharp.
* **Background blur** — a depth-of-field Gaussian outside the face hull; the
  face stays crisp.
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
    MOUTH,
    RIGHT_EYE_RING,
    RIGHT_IRIS,
)

# How far the gaze anchor sits below the eye-socket centre, as a fraction of the
# eye's vertical extent, at attention == 1.0. Tuned so a mid setting reads as
# "attentively looking at the monitor" rather than a lens stare.
_ATTENTION_SCALE: float = 0.6

# Downscale target for the nearest-neighbour "low-res" pixelation (16:9).
_PIXELATE_BLOCKS: Tuple[int, int] = (160, 90)


class PresenceMode(Enum):
    """Presence / privacy state for the output frame."""

    LIVE = "live"                    # normal live processed feed
    FREEZE = "freeze"                # hold the last processed frame
    FAKE_LOWRES = "fake_lowres"      # hold a pixelated frame (fake bad network)
    STREAM_LOWRES = "stream_lowres"  # live but continuously pixelated


class Modifier:
    """Gaze correction, skin smoothing, and identity/background blur.

    Args:
        gaze_strength: Fraction of the measured pupil displacement corrected,
            in ``[0, 1]``. ``0`` disables correction. Held firm internally; a
            value below 1 leaves a little live micro-drift so eyes stay alive.
        gaze_max_shift: Hard cap (in pixels) on per-eye translation to keep the
            warp stable when tracking is noisy.
        gaze_smoothing: Temporal EMA inertia in ``[0, 0.98]`` applied to the
            per-eye shift; higher is steadier, lower is snappier.
        gaze_attention: "Attention vector" in ``[0, 1]`` — how far below the eye
            centre the gaze anchor sits, so the user reads as attentively looking
            at their monitor rather than staring into the lens.
        face_blur_enabled: Apply the heavy face-anonymiser blur.
        background_blur_enabled: Apply the background depth-of-field blur.
        smoothing_enabled: Apply the skin-smoothing beauty filter.
        smoothing_strength: Smoothing intensity in ``[0, 1]``.
        presence_mode: Initial :class:`PresenceMode`.
    """

    def __init__(
        self,
        *,
        gaze_strength: float = 0.85,
        gaze_max_shift: float = 14.0,
        gaze_smoothing: float = 0.6,
        gaze_attention: float = 0.35,
        face_blur_enabled: bool = False,
        background_blur_enabled: bool = False,
        smoothing_enabled: bool = False,
        smoothing_strength: float = 0.5,
        presence_mode: PresenceMode = PresenceMode.LIVE,
    ) -> None:
        self.gaze_strength = gaze_strength
        self.gaze_max_shift = float(max(0.0, gaze_max_shift))
        self.gaze_smoothing = gaze_smoothing
        self.gaze_attention = gaze_attention
        self.face_blur_enabled = bool(face_blur_enabled)
        self.background_blur_enabled = bool(background_blur_enabled)
        self.smoothing_enabled = bool(smoothing_enabled)
        self.smoothing_strength = smoothing_strength

        # Per-eye EMA of the recentre shift vector, keyed by iris landmark set.
        self._shift_ema: dict[Tuple[int, ...], np.ndarray] = {}

        # Presence state machine: the held frame for FREEZE / FAKE_LOWRES.
        self._presence_mode = presence_mode
        self._frozen: Optional[np.ndarray] = None

    # -- Configuration ------------------------------------------------------
    @property
    def gaze_strength(self) -> float:
        return self._gaze_strength

    @gaze_strength.setter
    def gaze_strength(self, value: float) -> None:
        self._gaze_strength = float(np.clip(value, 0.0, 1.0))

    @property
    def gaze_smoothing(self) -> float:
        """Temporal inertia of the gaze warp in ``[0, 0.98]`` (capped below 1)."""
        return self._gaze_smoothing

    @gaze_smoothing.setter
    def gaze_smoothing(self, value: float) -> None:
        self._gaze_smoothing = float(np.clip(value, 0.0, 0.98))

    @property
    def smoothing_strength(self) -> float:
        return self._smoothing_strength

    @smoothing_strength.setter
    def smoothing_strength(self, value: float) -> None:
        self._smoothing_strength = float(np.clip(value, 0.0, 1.0))

    @property
    def gaze_attention(self) -> float:
        """Attention vector in ``[0, 1]`` — anchor depth below the eye centre."""
        return self._gaze_attention

    @gaze_attention.setter
    def gaze_attention(self, value: float) -> None:
        self._gaze_attention = float(np.clip(value, 0.0, 1.0))

    @property
    def presence_mode(self) -> PresenceMode:
        return self._presence_mode

    @presence_mode.setter
    def presence_mode(self, mode: PresenceMode) -> None:
        # Switching modes drops the held frame so a fresh one is captured.
        if mode is not self._presence_mode:
            self._frozen = None
        self._presence_mode = mode

    @property
    def is_frozen(self) -> bool:
        """True when a held frame already exists for FREEZE / FAKE_LOWRES."""
        return (
            self._presence_mode in (PresenceMode.FREEZE, PresenceMode.FAKE_LOWRES)
            and self._frozen is not None
        )

    # -- Public pipeline ----------------------------------------------------
    def apply(self, frame_bgr: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Run gaze correction, skin smoothing, then blurs, honouring toggles."""
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr
        out = self.correct_gaze(frame_bgr, landmarks)
        out = self.smooth_skin(out, landmarks)
        out = self.anonymise_face(out, landmarks)
        out = self.blur_background(out, landmarks)
        return out

    # -- Gaze correction ----------------------------------------------------
    def correct_gaze(self, frame_bgr: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Warp each eye region so the iris re-centres in its aperture."""
        if self._gaze_strength <= 0.0 or not landmarks.has_iris:
            self._shift_ema.clear()  # forget stale motion so re-enabling is clean
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

        # Monitor gaze anchor: instead of the dead-centre socket, anchor the iris
        # slightly *below* centre (horizontally centred) so the user reads as
        # attentively looking at their screen, not staring into the lens. The
        # drop scales with the eye's own height and the attention vector.
        ring = landmarks.pixels[list(ring_idx)]
        eye_height = float(np.ptp(ring[:, 1]))
        anchor = socket.copy()
        anchor[1] += self._gaze_attention * _ATTENTION_SCALE * eye_height
        target = (anchor - pupil) * self._gaze_strength

        # Blend toward the target with an exponential moving average so noisy
        # frame-to-frame landmark wobble doesn't jitter the warp; the pupil
        # glides smoothly to the anchor and back, leaving a little live
        # micro-drift (gaze_strength < 1) so the eyes never look frozen.
        prev = self._shift_ema.get(iris_idx)
        if prev is None:
            smoothed = target.astype(np.float32)
        else:
            alpha = self._gaze_smoothing
            smoothed = (alpha * prev + (1.0 - alpha) * target).astype(np.float32)
        self._shift_ema[iris_idx] = smoothed

        shift = smoothed.copy()
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

    # -- Skin smoothing (beauty filter) -------------------------------------
    def smooth_skin(self, frame_bgr: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Edge-preserving bilateral smoothing confined to the skin region."""
        if not self.smoothing_enabled or self._smoothing_strength <= 0.0:
            return frame_bgr

        bbox = self._face_bbox(frame_bgr.shape[:2], landmarks)
        if bbox is None:
            return frame_bgr
        x0, y0, x1, y1 = bbox
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return frame_bgr

        s = self._smoothing_strength
        # Bilateral filter smooths flat skin while preserving structural edges.
        # Scale the diameter/sigmas with the requested intensity.
        diameter = int(round(5 + 8 * s))
        sigma = 30.0 + 90.0 * s
        filtered = cv2.bilateralFilter(roi, diameter, sigma, sigma)

        skin = self._skin_mask(frame_bgr.shape[:2], landmarks)[y0:y1, x0:x1]
        alpha = (skin * s)[:, :, None]
        blended = roi * (1.0 - alpha) + filtered * alpha

        out = frame_bgr.copy()
        out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    def _skin_mask(
        self, shape: Tuple[int, int], landmarks: FaceLandmarks
    ) -> np.ndarray:
        """Face hull with the eyes and mouth carved out, feathered."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillConvexPoly(mask, landmarks.hull(), 1.0, lineType=cv2.LINE_AA)
        self._carve(mask, landmarks, LEFT_EYE_RING, 0.9)
        self._carve(mask, landmarks, RIGHT_EYE_RING, 0.9)
        self._carve(mask, landmarks, MOUTH, 0.7)
        return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, min(h, w) * 0.006))

    @staticmethod
    def _carve(
        mask: np.ndarray,
        landmarks: FaceLandmarks,
        idx: Tuple[int, ...],
        scale: float,
    ) -> None:
        pts = landmarks.pixels[list(idx)]
        cx, cy = pts.mean(axis=0)
        ax = max(3.0, float(np.ptp(pts[:, 0])) * scale + 3.0)
        ay = max(3.0, float(np.ptp(pts[:, 1])) * scale + 3.0)
        cv2.ellipse(
            mask, (int(cx), int(cy)), (int(ax), int(ay)),
            0, 0, 360, color=0.0, thickness=-1,
        )

    # -- Blur effects -------------------------------------------------------
    def anonymise_face(self, frame_bgr: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Heavy, opaque Gaussian over the face hull; background left sharp."""
        if not self.face_blur_enabled:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        bbox = self._face_bbox((h, w), landmarks)
        face_w = (bbox[2] - bbox[0]) if bbox is not None else w
        # Kernel scales with face size so anonymisation stays opaque at any zoom.
        k = self._odd(max(31, int(face_w * 0.35)))
        blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
        mask = self._hull_mask((h, w), landmarks, feather=max(4.0, min(h, w) * 0.01))
        select = mask[:, :, None]
        out = frame_bgr * (1.0 - select) + blurred * select
        return np.clip(out, 0, 255).astype(np.uint8)

    def blur_background(self, frame_bgr: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Depth-of-field Gaussian outside the face hull; face left crisp."""
        if not self.background_blur_enabled:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        k = self._odd(max(21, int(min(h, w) * 0.04)))
        blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
        mask = self._hull_mask((h, w), landmarks, feather=max(4.0, min(h, w) * 0.01))
        select = (1.0 - mask)[:, :, None]
        out = frame_bgr * (1.0 - select) + blurred * select
        return np.clip(out, 0, 255).astype(np.uint8)

    # -- Shared mask helpers ------------------------------------------------
    @staticmethod
    def _hull_mask(
        shape: Tuple[int, int], landmarks: FaceLandmarks, feather: float
    ) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillConvexPoly(mask, landmarks.hull(), 1.0, lineType=cv2.LINE_AA)
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather)
        return mask

    @staticmethod
    def _face_bbox(
        shape: Tuple[int, int], landmarks: FaceLandmarks
    ) -> Optional[Tuple[int, int, int, int]]:
        h, w = shape
        hull = landmarks.hull().reshape(-1, 2)
        x0 = int(np.clip(hull[:, 0].min(), 0, w - 1))
        y0 = int(np.clip(hull[:, 1].min(), 0, h - 1))
        x1 = int(np.clip(hull[:, 0].max() + 1, 0, w))
        y1 = int(np.clip(hull[:, 1].max() + 1, 0, h))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _odd(value: int) -> int:
        """Nearest odd integer >= 3 (Gaussian kernels must be odd)."""
        v = max(3, int(value))
        return v if v % 2 == 1 else v + 1

    # -- Presence control ---------------------------------------------------
    @staticmethod
    def pixelate(frame_bgr: np.ndarray) -> np.ndarray:
        """Nearest-neighbour downscale→upscale for heavy pixel-block artifacts."""
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, _PIXELATE_BLOCKS, interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def present(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply the current presence mode to a (already processed) frame.

        * ``LIVE`` — pass through.
        * ``STREAM_LOWRES`` — pixelate every live frame (motion stays fluid).
        * ``FREEZE`` — capture the frame once, then loop it.
        * ``FAKE_LOWRES`` — capture a pixelated frame once, then loop it.

        Switching modes clears the held frame (see :meth:`presence_mode`), so the
        next call re-captures.
        """
        mode = self._presence_mode
        if mode is PresenceMode.LIVE:
            self._frozen = None
            return frame_bgr
        if mode is PresenceMode.STREAM_LOWRES:
            self._frozen = None
            return self.pixelate(frame_bgr)
        if mode is PresenceMode.FREEZE:
            if self._frozen is None:
                self._frozen = frame_bgr.copy()
            return self._frozen
        # FAKE_LOWRES
        if self._frozen is None:
            self._frozen = self.pixelate(frame_bgr)
        return self._frozen
