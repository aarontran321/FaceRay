"""Tests for faceray.sidecar_entry: control parsing and the stdio round-trip."""

from __future__ import annotations

import json
import subprocess
import sys

from faceray.core.modifier import Modifier, PresenceMode
from faceray.sidecar_entry import SidecarControl


def test_from_dict_full_payload() -> None:
    c = SidecarControl.from_dict(
        {
            "gaze_enabled": False,
            "gaze_attention": 0.4,
            "face_blur_enabled": True,
            "background_blur_enabled": True,
            "smoothing_enabled": True,
            "smoothing_strength": 0.8,
            "presence": "fake_lowres",
        }
    )
    assert c.gaze_enabled is False
    assert c.gaze_attention == 0.4
    assert c.face_blur_enabled is True
    assert c.smoothing_strength == 0.8
    assert c.presence == "fake_lowres"


def test_from_dict_partial_inherits_base() -> None:
    base = SidecarControl(smoothing_strength=0.9, face_blur_enabled=True)
    merged = SidecarControl.from_dict({"presence": "freeze"}, base=base)
    assert merged.smoothing_strength == 0.9  # inherited
    assert merged.face_blur_enabled is True  # inherited
    assert merged.presence == "freeze"  # overridden


def test_from_dict_rejects_bad_presence() -> None:
    import pytest

    with pytest.raises(ValueError):
        SidecarControl.from_dict({"presence": "sideways"})


def test_from_json_roundtrips_through_to_dict() -> None:
    original = SidecarControl(
        gaze_attention=0.55, smoothing_enabled=True, presence="stream_lowres"
    )
    clone = SidecarControl.from_json(json.dumps(original.to_dict()))
    assert clone == original


def test_apply_pushes_state_onto_modifier() -> None:
    modifier = Modifier()
    control = SidecarControl(
        gaze_enabled=False,
        gaze_attention=0.4,
        face_blur_enabled=True,
        background_blur_enabled=True,
        smoothing_enabled=True,
        smoothing_strength=0.7,
        presence="stream_lowres",
    )
    control.apply(modifier)

    assert modifier.gaze_strength == 0.0  # gaze disabled -> strength zeroed
    assert modifier.gaze_attention == 0.4
    assert modifier.face_blur_enabled is True
    assert modifier.background_blur_enabled is True
    assert modifier.smoothing_enabled is True
    assert modifier.smoothing_strength == 0.7
    assert modifier.presence_mode is PresenceMode.STREAM_LOWRES


def test_apply_gaze_enabled_holds_firm() -> None:
    modifier = Modifier()
    SidecarControl(gaze_enabled=True, gaze_attention=0.35).apply(modifier)
    assert modifier.gaze_strength > 0.0  # firm internal hold
    assert modifier.gaze_attention == 0.35


def _run_sidecar(control_lines: list[str], extra_args: list[str]) -> list[dict]:
    proc = subprocess.run(
        [sys.executable, "-m", "faceray.sidecar_entry", *extra_args],
        input="".join(line + "\n" for line in control_lines),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]


def test_stdio_roundtrip_and_graceful_shutdown() -> None:
    """Full process: emits ready, acks a control line, and exits on stdin EOF."""
    events = _run_sidecar(
        ['{"face_blur_enabled":true,"smoothing_strength":0.8}'],
        ["--synthetic", "--no-sink", "--max-frames", "3",
         "--status-every", "1", "--fps", "120"],
    )
    types = [e["type"] for e in events]
    assert types[0] == "ready"
    assert "bye" in types

    acks = [e for e in events if e["type"] == "ack"]
    assert acks, "expected at least one ack for the control line"
    assert acks[0]["state"]["face_blur_enabled"] is True
    assert acks[0]["state"]["smoothing_strength"] == 0.8


def test_stdio_reports_bad_payload() -> None:
    events = _run_sidecar(
        ["not-json-at-all"],
        ["--synthetic", "--no-sink", "--max-frames", "1"],
    )
    assert any(e["type"] == "error" for e in events)
    assert any(e["type"] == "bye" for e in events)  # channel survived the bad line
