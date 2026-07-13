"""FaceRay core processing engines.

This package holds the strictly-decoupled mathematical stages of the FaceRay
pipeline. Each engine consumes a frame (and, where relevant, the landmark set
produced by :class:`~faceray.core.tracker.FaceTracker`) and returns a new frame,
leaving the caller in :mod:`faceray.app` to orchestrate the ordering.

Public surface:
    FaceTracker   -- MediaPipe Face Landmarker 3D landmark extraction.
    FaceLandmarks -- Immutable container describing one detected face.
    Modifier      -- Face-interaction filters: gaze correction, skin smoothing,
                     face anonymiser blur, and background blur.
"""

from __future__ import annotations

from faceray.core.tracker import FaceLandmarks, FaceTracker
from faceray.core.modifier import Modifier

__all__ = ["FaceTracker", "FaceLandmarks", "Modifier"]
