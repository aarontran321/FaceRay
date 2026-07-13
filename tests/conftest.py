"""Shared fixtures: synthetic FaceLandmarks that need no camera or MediaPipe."""

from __future__ import annotations

import math

import numpy as np
import pytest

from faceray.core.tracker import (
    FACE_OVAL,
    FaceLandmarks,
    LEFT_EYE_RING,
    LEFT_IRIS,
    MOUTH,
    RIGHT_EYE_RING,
    RIGHT_IRIS,
    TOTAL_LANDMARKS,
)


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


def make_face_landmarks(
    *, width: int = 640, height: int = 480
) -> FaceLandmarks:
    """Build anatomically plausible landmarks (clustered eyes/mouth, oval hull).

    Unlike :func:`make_landmarks` (uniform noise), this places the key feature
    groups where a real face has them, so the skin mask, eye ROIs, and blur
    hulls are all well-formed — needed by the smoothing/mask tests.
    """
    pix = np.empty((TOTAL_LANDMARKS, 2), dtype=np.float32)
    pix[:, 0] = 0.5 * width
    pix[:, 1] = 0.5 * height

    def place(indices: tuple[int, ...], cx: float, cy: float, r: float) -> None:
        n = len(indices)
        for j, idx in enumerate(indices):
            a = 2.0 * math.pi * j / n
            pix[idx] = ((cx + r * math.cos(a)) * width, (cy + r * math.sin(a)) * height)

    place(FACE_OVAL, 0.5, 0.5, 0.30)  # large silhouette (r scaled below for y)
    for idx in FACE_OVAL:  # stretch the oval vertically
        pix[idx, 1] = (0.5 + (pix[idx, 1] / height - 0.5) * 1.25) * height
    place(LEFT_EYE_RING, 0.40, 0.42, 0.03)
    place(RIGHT_EYE_RING, 0.60, 0.42, 0.03)
    place(LEFT_IRIS, 0.405, 0.42, 0.010)
    place(RIGHT_IRIS, 0.595, 0.42, 0.010)
    place(MOUTH, 0.50, 0.63, 0.04)

    world = np.empty((TOTAL_LANDMARKS, 3), dtype=np.float32)
    world[:, 0] = pix[:, 0] / width
    world[:, 1] = pix[:, 1] / height
    world[:, 2] = 0.0
    return FaceLandmarks(pixels=pix, world=world, frame_shape=(height, width))


@pytest.fixture()
def landmarks() -> FaceLandmarks:
    return make_landmarks()


@pytest.fixture()
def frame() -> np.ndarray:
    """A mid-grey BGR frame so multiplicative shading changes are visible."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)
