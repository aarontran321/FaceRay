"""CUDA / CuPy Lambertian shading, surface normals & virtual relighting.

Stage 2 of the FaceRay pipeline (Feature A).

Algorithm
---------
1. Triangulate the 2D face landmarks with ``cv2.Subdiv2D`` (Delaunay). The
   triangulation is cached per (topology) and only recomputed when the number
   of vertices changes, which is the dominant real-time case.
2. Lift each triangle's vertices into 3D using MediaPipe's normalized
   ``(x, y, z)`` and compute a per-triangle surface normal
   ``N = normalize((v1 - v0) x (v2 - v0))``.
3. Compute Lambertian illumination ``I = max(0, N . L)`` for a configurable
   virtual light direction ``L``. Steps 2-3 run entirely with the array
   namespace ``xp`` (CuPy when a CUDA device is present, otherwise NumPy), so
   the dot products execute on the GPU when available.
4. Rasterize the per-triangle illumination into a single-channel mask with
   ``cv2.fillConvexPoly`` and blend it multiplicatively over the frame with a
   configurable weight, simulating a movable fill/ring light.

The class degrades gracefully: if CuPy or a CUDA context is unavailable it
transparently falls back to NumPy and sets :attr:`gpu_active` to ``False``.
"""

from __future__ import annotations

from typing import Final, Optional, Tuple

import cv2
import numpy as np

from faceray.core.tracker import FaceLandmarks

# ``xp`` is the active array namespace. We probe for a *usable* CUDA device,
# not merely an importable module, because cupy imports fine on machines with
# no GPU and only fails at first allocation.
try:  # pragma: no cover - depends on local hardware
    import cupy as _cupy

    _cupy.cuda.runtime.getDeviceCount()
    _GPU_AVAILABLE: Final[bool] = True
except Exception:  # ImportError, CUDARuntimeError, driver mismatch, ...
    _cupy = None  # type: ignore[assignment]
    _GPU_AVAILABLE = False


class Relighter:
    """Virtual Lambertian relighting engine.

    Args:
        light_direction: 3-vector direction *towards* the virtual light in
            MediaPipe landmark space (``+x`` right, ``+y`` down, ``+z`` away
            from camera). Normalized on assignment.
        intensity: Blend weight in ``[0, 2]``. ``0`` is a no-op passthrough;
            ``1`` applies the full shading term.
        ambient: Floor illumination in ``[0, 1]`` so unlit facets never go
            fully black.
        use_gpu: Force-disable the GPU path when ``False``.
    """

    def __init__(
        self,
        *,
        light_direction: Tuple[float, float, float] = (0.4, -0.3, -1.0),
        intensity: float = 0.6,
        ambient: float = 0.55,
        use_gpu: bool = True,
    ) -> None:
        self._xp = _cupy if (use_gpu and _GPU_AVAILABLE) else np
        self.gpu_active: bool = self._xp is _cupy
        self.intensity = intensity
        self.ambient = float(np.clip(ambient, 0.0, 1.0))
        self.light_direction = light_direction  # normalized in setter

        # Triangulation cache keyed on vertex count -> (T, 3) int index array.
        self._tri_cache: Optional[np.ndarray] = None
        self._tri_cache_key: Optional[Tuple[int, int, int]] = None

    # -- Configuration ------------------------------------------------------
    @property
    def light_direction(self) -> Tuple[float, float, float]:
        return tuple(float(v) for v in self._light)  # type: ignore[return-value]

    @light_direction.setter
    def light_direction(self, value: Tuple[float, float, float]) -> None:
        vec = np.asarray(value, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            raise ValueError("light_direction must be a non-zero vector.")
        self._light = vec / norm

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        self._intensity = float(np.clip(value, 0.0, 2.0))

    def orbit_light(self, delta_azimuth: float) -> None:
        """Rotate the light around the vertical axis by ``delta_azimuth`` rad.

        Convenience hook for interactive UI controls in :mod:`faceray.app`.
        """
        lx, ly, lz = self.light_direction
        cos_a, sin_a = np.cos(delta_azimuth), np.sin(delta_azimuth)
        self.light_direction = (
            lx * cos_a + lz * sin_a,
            ly,
            -lx * sin_a + lz * cos_a,
        )

    # -- Core ---------------------------------------------------------------
    def apply(self, frame_bgr: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Return ``frame_bgr`` relit according to the current light.

        Falls back to returning the input frame unchanged (a safe no-op) when
        the shading term is disabled or the geometry is degenerate.
        """
        if self._intensity <= 0.0 or frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        triangles = self._triangulate(landmarks)
        if triangles is None or len(triangles) == 0:
            return frame_bgr

        shade = self._lambertian_shade(landmarks.world, triangles)  # host float32
        mask = self._rasterize_mask(frame_bgr.shape[:2], landmarks.pixels, triangles, shade)

        # Multiplicative blend around a neutral 1.0 gain, scaled by intensity.
        gain = (1.0 - self._intensity) + self._intensity * mask
        lit = frame_bgr.astype(np.float32) * gain[:, :, None]
        return np.clip(lit, 0.0, 255.0).astype(np.uint8)

    # -- Internals ----------------------------------------------------------
    def _triangulate(self, landmarks: FaceLandmarks) -> Optional[np.ndarray]:
        """Delaunay-triangulate the landmark cloud, caching by vertex count."""
        pts = landmarks.pixels
        h, w = landmarks.frame_shape
        key = (pts.shape[0], h, w)
        if key == self._tri_cache_key and self._tri_cache is not None:
            return self._tri_cache

        subdiv = cv2.Subdiv2D((0, 0, w, h))
        # Subdiv2D rejects out-of-rect points; landmarks are already clamped
        # in the tracker, but guard the boundary to be safe.
        for x, y in pts:
            subdiv.insert((float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))))

        # Map inserted vertex coordinates back to landmark indices via a
        # nearest lookup on a rounded coordinate grid.
        index_of: dict[Tuple[int, int], int] = {}
        for idx, (x, y) in enumerate(pts):
            index_of[(int(round(x)), int(round(y)))] = idx

        tris: list[Tuple[int, int, int]] = []
        for t in subdiv.getTriangleList():
            verts = ((t[0], t[1]), (t[2], t[3]), (t[4], t[5]))
            resolved = []
            for vx, vy in verts:
                key_v = (int(round(vx)), int(round(vy)))
                if key_v not in index_of:
                    break
                resolved.append(index_of[key_v])
            if len(resolved) == 3:
                tris.append((resolved[0], resolved[1], resolved[2]))

        if not tris:
            return None
        self._tri_cache = np.asarray(tris, dtype=np.int32)
        self._tri_cache_key = key
        return self._tri_cache

    def _lambertian_shade(self, world: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """Per-triangle ``ambient + (1-ambient) * max(0, N.L)`` on GPU/CPU.

        Returns a host (NumPy) float32 array of length ``len(triangles)`` in
        ``[0, 1]`` because the subsequent rasterization runs on the CPU.
        """
        xp = self._xp
        verts = xp.asarray(world, dtype=xp.float32)
        tri = xp.asarray(triangles, dtype=xp.int32)
        light = xp.asarray(self._light, dtype=xp.float32)

        v0 = verts[tri[:, 0]]
        v1 = verts[tri[:, 1]]
        v2 = verts[tri[:, 2]]

        normals = xp.cross(v1 - v0, v2 - v0)
        lengths = xp.linalg.norm(normals, axis=1, keepdims=True)
        lengths = xp.where(lengths < 1e-8, xp.float32(1.0), lengths)
        normals = normals / lengths

        # Camera looks down -z in MediaPipe space; flip normals that face away
        # so the visible side of every facet is shaded consistently.
        facing = normals[:, 2] > 0
        normals[facing] *= -1.0

        ndotl = xp.clip(normals @ light, 0.0, 1.0)
        shade = xp.float32(self.ambient) + xp.float32(1.0 - self.ambient) * ndotl

        if xp is _cupy:  # bring the small (T,) vector back to host for cv2
            return _cupy.asnumpy(shade).astype(np.float32)
        return shade.astype(np.float32)

    @staticmethod
    def _rasterize_mask(
        shape: Tuple[int, int],
        pixels: np.ndarray,
        triangles: np.ndarray,
        shade: np.ndarray,
    ) -> np.ndarray:
        """Flat-fill each triangle with its shade to build a lighting mask."""
        h, w = shape
        mask = np.ones((h, w), dtype=np.float32)
        pix = pixels.astype(np.int32)
        # Paint darkest facets last so overlaps bias towards shadow, avoiding
        # bright bleed at silhouette edges.
        order = np.argsort(-shade)
        for i in order:
            a, b, c = triangles[i]
            poly = pix[[a, b, c]].reshape(-1, 1, 2)
            cv2.fillConvexPoly(mask, poly, float(shade[i]), lineType=cv2.LINE_AA)

        # Soften facet seams into a smooth low-frequency light field.
        return cv2.GaussianBlur(mask, (0, 0), sigmaX=5.0)
