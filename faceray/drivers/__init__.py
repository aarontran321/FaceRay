"""FaceRay virtual hardware layer.

Bridges the processed RGB frame stream to a native OS virtual camera device so
that external applications (Discord, Zoom, Meet) enumerate FaceRay as a normal
webcam. Keeps the driver concern strictly decoupled from the math engines in
:mod:`faceray.core`.
"""

from __future__ import annotations

from faceray.drivers.virtual_sink import VirtualSink

__all__ = ["VirtualSink"]
