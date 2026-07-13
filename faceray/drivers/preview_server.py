"""Loopback MJPEG preview server.

A driver-side output that serves the latest processed BGR frame as an MJPEG
stream on localhost, so a local UI (the Tauri webview) can display the live
pipeline result **without any frame data crossing the control-plane IPC** — the
webview's ``<img>`` pulls the stream directly over the loopback socket.

Like :mod:`faceray.drivers.virtual_sink`, this driver knows only about raw BGR
frame arrays and never imports :mod:`faceray.core`.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

_INDEX_HTML = (
    b"<!doctype html><meta charset=utf-8><title>FaceRay preview</title>"
    b"<body style='margin:0;background:#111'>"
    b"<img src='/stream' style='width:100%;height:100vh;object-fit:contain'>"
    b"</body>"
)


class PreviewServer:
    """Serve processed frames as an MJPEG stream bound to loopback.

    Args:
        host: Interface to bind; loopback only by default.
        port: TCP port, or ``0`` to let the OS pick a free one (read it back
            from :attr:`port` after :meth:`start`).
        quality: JPEG quality (1-100) for the encoded stream. Defaults to a
            near-lossless 95 to preserve crisp facial/skin texture.
        max_width: Downscale frames wider than this before encoding. Defaults
            to 1280 so the preview stays sharp; ``0`` disables downscaling.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        quality: int = 95,
        max_width: int = 1280,
    ) -> None:
        self._host = host
        self._req_port = int(port)
        self._quality = int(quality)
        self._max_width = int(max_width)

        self._latest: Optional[bytes] = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stopping = False

        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1] if self._httpd is not None else self._req_port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}/stream"

    def start(self) -> "PreviewServer":
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: object) -> None:  # silence access logs
                pass

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                if self.path.startswith("/stream"):
                    server._serve_stream(self)
                else:
                    server._serve_index(self)

        self._httpd = ThreadingHTTPServer((self._host, self._req_port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="preview-server", daemon=True
        )
        self._thread.start()
        return self

    def update(self, frame_bgr: np.ndarray) -> None:
        """Publish a new frame to all connected stream clients."""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        frame = frame_bgr
        if self._max_width and frame.shape[1] > self._max_width:
            scale = self._max_width / float(frame.shape[1])
            frame = cv2.resize(
                frame,
                (self._max_width, max(1, int(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
        )
        if not ok:
            return
        with self._cond:
            self._latest = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def stop(self) -> None:
        self._stopping = True
        with self._cond:
            self._cond.notify_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # -- request handling ---------------------------------------------------
    def _serve_index(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(_INDEX_HTML)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(_INDEX_HTML)

    def _serve_stream(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Age", "0")
        handler.send_header("Cache-Control", "no-cache, private")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        handler.end_headers()

        last = -1
        try:
            while not self._stopping:
                with self._cond:
                    ready = self._cond.wait_for(
                        lambda: self._seq != last and self._latest is not None,
                        timeout=1.0,
                    )
                    if not ready:
                        continue
                    data = self._latest
                    last = self._seq
                assert data is not None
                handler.wfile.write(b"--frame\r\n")
                handler.wfile.write(b"Content-Type: image/jpeg\r\n")
                handler.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                handler.wfile.write(data)
                handler.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client closed the stream; the handler thread simply exits
