"""Unit tests for faceray.core.relighter (pure shading math)."""

from __future__ import annotations

import numpy as np
import pytest

from faceray.core.relighter import Relighter
from tests.conftest import make_landmarks


def test_light_direction_is_normalized() -> None:
    r = Relighter(light_direction=(0.0, 0.0, -2.0), use_gpu=False)
    assert r.light_direction == pytest.approx((0.0, 0.0, -1.0))


def test_zero_light_direction_rejected() -> None:
    with pytest.raises(ValueError):
        Relighter(light_direction=(0.0, 0.0, 0.0), use_gpu=False)


def test_intensity_clamped_to_range() -> None:
    r = Relighter(use_gpu=False)
    r.intensity = 5.0
    assert r.intensity == 2.0
    r.intensity = -1.0
    assert r.intensity == 0.0


def test_orbit_light_preserves_norm_and_elevation() -> None:
    r = Relighter(light_direction=(1.0, 0.5, -1.0), use_gpu=False)
    y_before = r.light_direction[1]
    r.orbit_light(0.5)
    vec = np.asarray(r.light_direction)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)
    # Rotation is about the vertical (y) axis, so elevation is unchanged.
    assert r.light_direction[1] == pytest.approx(y_before)


def test_shade_head_on_light_is_full() -> None:
    """A facet whose normal points at a head-on light is fully lit (shade≈1)."""
    r = Relighter(light_direction=(0.0, 0.0, -1.0), ambient=0.55, use_gpu=False)
    world = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    shade = r._lambertian_shade(world, tris)
    assert shade.shape == (1,)
    assert shade[0] == pytest.approx(1.0, abs=1e-5)


def test_shade_perpendicular_light_is_ambient() -> None:
    """A facet lit edge-on (N.L <= 0) falls back to the ambient floor."""
    ambient = 0.4
    r = Relighter(light_direction=(1.0, 0.0, 0.0), ambient=ambient, use_gpu=False)
    world = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    shade = r._lambertian_shade(world, tris)
    assert shade[0] == pytest.approx(ambient, abs=1e-5)


def test_shade_within_ambient_to_one() -> None:
    r = Relighter(ambient=0.55, use_gpu=False)
    lm = make_landmarks(seed=3)
    tris = r._triangulate(lm)
    assert tris is not None and len(tris) > 0
    shade = r._lambertian_shade(lm.world, tris)
    assert shade.min() >= 0.55 - 1e-5
    assert shade.max() <= 1.0 + 1e-5
    assert np.isfinite(shade).all()


def test_apply_returns_same_shape_uint8(frame: np.ndarray) -> None:
    r = Relighter(use_gpu=False)
    lm = make_landmarks()
    out = r.apply(frame, lm)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_apply_zero_intensity_is_passthrough(frame: np.ndarray) -> None:
    r = Relighter(intensity=0.0, use_gpu=False)
    lm = make_landmarks()
    out = r.apply(frame, lm)
    assert out is frame  # documented no-op fast path


def test_apply_changes_pixels_when_lit(frame: np.ndarray) -> None:
    r = Relighter(intensity=1.0, ambient=0.2, use_gpu=False)
    lm = make_landmarks()
    out = r.apply(frame, lm)
    assert not np.array_equal(out, frame)


def test_triangulation_cache_reused() -> None:
    r = Relighter(use_gpu=False)
    lm = make_landmarks()
    first = r._triangulate(lm)
    second = r._triangulate(lm)
    assert first is second  # same vertex-count key -> cached array identity
