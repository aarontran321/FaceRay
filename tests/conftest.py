"""Shared fixtures: synthetic FaceLandmarks that need no camera or MediaPipe."""

from __future__ import annotations

import numpy as np
import pytest

from faceray.core.tracker import FaceLandmarks, TOTAL_LANDMARKS


def make_landmarks(
    *,
    width: int = 640,
    height: int = 480,
    count: int = TOTAL_LANDMARKS,
    seed: int = 0,
) -> FaceLandmarks:
    """Build a deterministic, geometrically valid FaceLandmarks payload.

    Points are scattered across the central 80% of the frame so the convex
    hull, Delaunay triangulation, and per-eye ROIs are all non-degenerate.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.1, 0.9, size=count).astype(np.float32)
    ys = rng.uniform(0.1, 0.9, size=count).astype(np.float32)
    zs = rng.uniform(-0.05, 0.05, size=count).astype(np.float32)

    world = np.stack([xs, ys, zs], axis=1)
    pixels = np.empty((count, 2), dtype=np.float32)
    pixels[:, 0] = np.clip(xs * width, 0.0, width - 1.0)
    pixels[:, 1] = np.clip(ys * height, 0.0, height - 1.0)
    return FaceLandmarks(pixels=pixels, world=world, frame_shape=(height, width))


@pytest.fixture()
def landmarks() -> FaceLandmarks:
    return make_landmarks()


@pytest.fixture()
def frame() -> np.ndarray:
    """A mid-grey BGR frame so multiplicative shading changes are visible."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)
