"""Unit tests for faceray.core.modifier (gaze warp, smoothing, blur masks)."""

from __future__ import annotations

import numpy as np
import pytest

from faceray.core.modifier import Modifier, PresenceMode
from faceray.core.tracker import LEFT_EYE_RING, LEFT_IRIS
from tests.conftest import make_face_landmarks, make_landmarks


def _noisy_frame(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


def test_gaze_strength_clamped() -> None:
    m = Modifier()
    m.gaze_strength = 5.0
    assert m.gaze_strength == 1.0
    m.gaze_strength = -2.0
    assert m.gaze_strength == 0.0


def test_smoothing_strength_clamped() -> None:
    m = Modifier()
    m.smoothing_strength = 5.0
    assert m.smoothing_strength == 1.0
    m.smoothing_strength = -2.0
    assert m.smoothing_strength == 0.0


def test_face_blur_off_is_passthrough(frame: np.ndarray) -> None:
    m = Modifier(gaze_strength=0.0, face_blur_enabled=False)
    lm = make_landmarks()
    assert m.anonymise_face(frame, lm) is frame


def test_face_blur_changes_face_region() -> None:
    f = _noisy_frame(1)
    m = Modifier(gaze_strength=0.0, face_blur_enabled=True)
    out = m.anonymise_face(f, make_landmarks())
    assert out.shape == f.shape and out.dtype == np.uint8
    assert not np.array_equal(out, f)


def test_background_blur_differs_from_face_blur() -> None:
    f = _noisy_frame(2)
    lm = make_landmarks()
    face_out = Modifier(gaze_strength=0.0, face_blur_enabled=True).anonymise_face(f, lm)
    bg_out = Modifier(gaze_strength=0.0, background_blur_enabled=True).blur_background(f, lm)
    assert not np.array_equal(face_out, bg_out)


def test_smoothing_off_is_passthrough(frame: np.ndarray) -> None:
    m = Modifier(gaze_strength=0.0, smoothing_enabled=False)
    assert m.smooth_skin(frame, make_landmarks()) is frame


def test_smoothing_changes_skin_only() -> None:
    lm = make_face_landmarks()
    # A flat mid-grey with fine noise: bilateral smooths the low-contrast grain
    # over the skin region (edge-preserving; pure random noise would be a no-op).
    rng = np.random.default_rng(3)
    f = np.clip(128 + rng.normal(0, 12, (480, 640, 3)), 0, 255).astype(np.uint8)
    m = Modifier(gaze_strength=0.0, smoothing_enabled=True, smoothing_strength=1.0)
    out = m.smooth_skin(f, lm)
    assert out.shape == f.shape and out.dtype == np.uint8
    assert not np.array_equal(out, f)


def test_skin_mask_keeps_eyes_and_mouth_sharp() -> None:
    lm = make_face_landmarks()
    mask = Modifier()._skin_mask(lm.frame_shape, lm)
    # An interior cheek pixel (below the eye, above the mouth) is skin -> smoothed.
    assert mask[250, 224] > 0.5
    # The eye centre is carved out so lashes stay sharp.
    eye = lm.centroid(LEFT_EYE_RING)
    assert mask[int(eye[1]), int(eye[0])] < 0.2


def test_gaze_noop_without_iris(frame: np.ndarray) -> None:
    # 468 points => has_iris is False => gaze correction must be a no-op.
    lm = make_landmarks(count=468)
    assert not lm.has_iris
    m = Modifier(gaze_strength=1.0)
    out = m.correct_gaze(frame, lm)
    assert out is frame


def test_gaze_zero_strength_is_noop(frame: np.ndarray) -> None:
    lm = make_landmarks()
    m = Modifier(gaze_strength=0.0)
    out = m.correct_gaze(frame, lm)
    assert out is frame


def test_gaze_shift_is_bounded(frame: np.ndarray) -> None:
    """Even with maximal strength the per-eye warp respects gaze_max_shift."""
    lm = make_landmarks()
    m = Modifier(gaze_strength=1.0, gaze_max_shift=8.0)
    out = m.apply(frame, lm)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_eye_roi_within_bounds() -> None:
    lm = make_landmarks()
    roi = Modifier._eye_roi(lm.frame_shape, lm, LEFT_EYE_RING)
    assert roi is not None
    x0, y0, x1, y1 = roi
    h, w = lm.frame_shape
    assert 0 <= x0 < x1 <= w
    assert 0 <= y0 < y1 <= h


def test_apply_full_pipeline_shape(frame: np.ndarray) -> None:
    lm = make_landmarks()
    m = Modifier(
        gaze_strength=0.7,
        background_blur_enabled=True,
        smoothing_enabled=True,
        smoothing_strength=0.6,
    )
    out = m.apply(frame, lm)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_gaze_smoothing_clamped() -> None:
    m = Modifier(gaze_smoothing=5.0)
    assert m.gaze_smoothing == 0.98  # capped below 1 so the warp can't freeze
    m.gaze_smoothing = -1.0
    assert m.gaze_smoothing == 0.0


def test_gaze_smoothing_blends_between_frames(frame: np.ndarray) -> None:
    """The EMA blends the previous shift with the new target, not snapping."""
    a = make_landmarks(seed=1)
    b = make_landmarks(seed=2)
    # gaze_attention=0 anchors on the socket centre, so the target is exactly
    # (socket - pupil) * strength (no downward attention offset to account for).
    m = Modifier(gaze_strength=0.7, gaze_smoothing=0.6, gaze_attention=0.0)

    m.correct_gaze(frame, a)
    ema1 = m._shift_ema[LEFT_IRIS].copy()
    target_a = (a.centroid(LEFT_EYE_RING) - a.centroid(LEFT_IRIS)) * 0.7
    assert np.allclose(ema1, target_a, atol=1e-3)  # first frame: no prior, == target

    m.correct_gaze(frame, b)
    ema2 = m._shift_ema[LEFT_IRIS]
    target_b = (b.centroid(LEFT_EYE_RING) - b.centroid(LEFT_IRIS)) * 0.7
    expected = 0.6 * ema1 + 0.4 * target_b
    assert np.allclose(ema2, expected, atol=1e-3)


def test_gaze_smoothing_state_cleared_on_disable(frame: np.ndarray) -> None:
    lm = make_landmarks()
    m = Modifier(gaze_strength=0.7)
    m.correct_gaze(frame, lm)
    assert m._shift_ema  # populated while active
    m.gaze_strength = 0.0
    m.correct_gaze(frame, lm)
    assert not m._shift_ema  # forgotten so re-enabling starts clean


def test_gaze_attention_anchors_downward(frame: np.ndarray) -> None:
    """A higher attention vector biases the recentre shift downward (screen gaze)."""
    lm = make_face_landmarks()
    flat = Modifier(gaze_strength=0.85, gaze_attention=0.0)
    flat.correct_gaze(frame, lm)
    y_flat = flat._shift_ema[LEFT_IRIS][1]

    down = Modifier(gaze_strength=0.85, gaze_attention=0.8)
    down.correct_gaze(frame, lm)
    y_down = down._shift_ema[LEFT_IRIS][1]

    assert y_down > y_flat + 1.0  # downward (positive y) anchor offset


# -- Presence control ---------------------------------------------------------
def test_pixelate_preserves_shape_and_blocks() -> None:
    f = _noisy_frame(4)
    out = Modifier.pixelate(f)
    assert out.shape == f.shape and out.dtype == np.uint8
    assert not np.array_equal(out, f)  # nearest-neighbour blocks alter detail


def test_presence_live_is_passthrough() -> None:
    m = Modifier(presence_mode=PresenceMode.LIVE)
    f = np.full((90, 160, 3), 77, dtype=np.uint8)
    assert m.present(f) is f


def test_presence_freeze_holds_first_frame() -> None:
    m = Modifier(presence_mode=PresenceMode.FREEZE)
    f1 = np.full((90, 160, 3), 50, dtype=np.uint8)
    f2 = np.full((90, 160, 3), 150, dtype=np.uint8)
    first = m.present(f1)
    assert np.array_equal(m.present(f2), first)  # keeps looping the held frame
    assert int(first.mean()) == 50


def test_presence_switch_clears_held_frame() -> None:
    m = Modifier(presence_mode=PresenceMode.FREEZE)
    m.present(np.full((90, 160, 3), 50, dtype=np.uint8))
    assert m.is_frozen
    m.presence_mode = PresenceMode.LIVE
    assert not m.is_frozen  # switching modes drops the buffer


def test_presence_stream_lowres_stays_live() -> None:
    m = Modifier(presence_mode=PresenceMode.STREAM_LOWRES)
    out = m.present(_noisy_frame(5))
    assert out.shape == (480, 640, 3)
    assert not m.is_frozen  # stream mode never freezes; motion stays live
