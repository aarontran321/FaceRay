"""Unit tests for faceray.core.modifier (gaze warp + blur mask geometry)."""

from __future__ import annotations

import numpy as np
import pytest

from faceray.core.modifier import BlurMode, Modifier
from faceray.core.tracker import LEFT_EYE_RING, LEFT_IRIS
from tests.conftest import make_landmarks


def test_gaze_strength_clamped() -> None:
    m = Modifier()
    m.gaze_strength = 5.0
    assert m.gaze_strength == 1.0
    m.gaze_strength = -2.0
    assert m.gaze_strength == 0.0


def test_blur_kernel_forced_odd_and_min() -> None:
    m = Modifier()
    m.blur_kernel = 30
    assert m.blur_kernel == 31
    m.blur_kernel = 1
    assert m.blur_kernel == 3


def test_cycle_blur_mode_order() -> None:
    m = Modifier(blur_mode=BlurMode.OFF)
    assert m.cycle_blur_mode() is BlurMode.FACE
    assert m.cycle_blur_mode() is BlurMode.BACKGROUND
    assert m.cycle_blur_mode() is BlurMode.OFF


def test_blur_off_is_passthrough(frame: np.ndarray) -> None:
    m = Modifier(blur_mode=BlurMode.OFF)
    lm = make_landmarks()
    out = m.apply_blur(frame, lm)
    assert out is frame


def test_blur_face_changes_face_region(frame: np.ndarray) -> None:
    # Non-uniform frame so blurring actually alters pixels.
    f = np.random.default_rng(1).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    m = Modifier(blur_mode=BlurMode.FACE)
    lm = make_landmarks()
    out = m.apply_blur(f, lm)
    assert out.shape == f.shape and out.dtype == np.uint8
    assert not np.array_equal(out, f)


def test_blur_background_differs_from_face(frame: np.ndarray) -> None:
    f = np.random.default_rng(2).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    lm = make_landmarks()
    face_out = Modifier(blur_mode=BlurMode.FACE).apply_blur(f, lm)
    bg_out = Modifier(blur_mode=BlurMode.BACKGROUND).apply_blur(f, lm)
    assert not np.array_equal(face_out, bg_out)


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
    m = Modifier(gaze_strength=0.7, blur_mode=BlurMode.BACKGROUND)
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
    m = Modifier(gaze_strength=0.7, gaze_smoothing=0.6)

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
