"""Headless pipeline self-check: capture real frames, render every effect.

Grabs a few frames from the webcam, runs each through the full FaceRay core
pipeline (tracker -> relighter -> modifier) in isolation per effect, and writes
a labelled montage PNG so the result can be eyeballed without a live GUI or a
virtual-camera backend.

    python -m scripts.capture_selfcheck --camera 0 --out selfcheck.png

Exit code 0 on success (a face was detected and the montage was written),
1 if the camera could not be opened or produced no frames, 2 if frames were
captured but no face was detected.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import cv2
import numpy as np

from faceray.core import FaceTracker, Relighter, Modifier
from faceray.core.modifier import BlurMode


def _open_camera(index: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _grab_frame(cap: cv2.VideoCapture, warmup: int, timeout_s: float) -> Optional[np.ndarray]:
    deadline = time.time() + timeout_s
    frame = None
    reads = 0
    while time.time() < deadline and reads < warmup:
        ok, f = cap.read()
        if ok and f is not None and f.size > 0:
            frame = f
            reads += 1
    return frame


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (60, 255, 120), 1, cv2.LINE_AA)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="capture_selfcheck")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", type=str, default="selfcheck.png")
    p.add_argument("--warmup", type=int, default=15,
                   help="Frames to pull so exposure/auto-focus settle.")
    p.add_argument("--timeout", type=float, default=8.0)
    args = p.parse_args(argv)

    cap = _open_camera(args.camera, args.width, args.height)
    if cap is None:
        print(f"[selfcheck] Could not open camera {args.camera}.", file=sys.stderr)
        return 1
    try:
        frame = _grab_frame(cap, args.warmup, args.timeout)
    finally:
        cap.release()

    if frame is None:
        print("[selfcheck] Camera opened but produced no frames "
              "(check camera permissions).", file=sys.stderr)
        return 1

    h, w = frame.shape[:2]
    print(f"[selfcheck] Captured a {w}x{h} frame.")

    relighter = Relighter(intensity=0.8, use_gpu=False)
    modifier = Modifier(gaze_strength=0.9)

    with FaceTracker(max_num_faces=1, refine_landmarks=True) as tracker:
        landmarks = tracker.process(frame)

    if landmarks is None:
        print("[selfcheck] No face detected — center your face and rerun.",
              file=sys.stderr)
        cv2.imwrite(args.out, _label(frame, "raw (no face detected)"))
        return 2

    print(f"[selfcheck] Face detected: {landmarks.pixels.shape[0]} landmarks, "
          f"iris={'yes' if landmarks.has_iris else 'no'}.")

    relit = relighter.apply(frame, landmarks)
    gaze = modifier.correct_gaze(frame, landmarks)
    blur_face = Modifier(blur_mode=BlurMode.FACE).apply_blur(frame, landmarks)
    blur_bg = Modifier(blur_mode=BlurMode.BACKGROUND).apply_blur(frame, landmarks)

    tiles = [
        _label(frame, "raw"),
        _label(relit, "relight"),
        _label(gaze, "gaze correction"),
        _label(blur_bg, "background blur"),
    ]
    top = np.hstack(tiles[:2])
    bottom = np.hstack([tiles[2], _label(blur_face, "face blur")])
    montage = np.vstack([top, bottom])
    cv2.imwrite(args.out, montage)
    print(f"[selfcheck] Wrote montage -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
