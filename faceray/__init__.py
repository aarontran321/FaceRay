"""FaceRay -- real-time, AI-powered virtual camera.

Captures a raw webcam feed, extracts a dense 3D face mesh, applies real-time
geometry-based pixel manipulations (virtual relighting, eye-contact correction,
smart blur), and pipes the modified frame into a native system virtual camera
so external apps recognise it as a normal webcam.

The package is split into strictly-decoupled layers:
    faceray.core     -- frame processing & mathematical engines.
    faceray.drivers  -- virtual hardware layer interface.
    faceray.app      -- orchestration loop and OpenCV UI.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
