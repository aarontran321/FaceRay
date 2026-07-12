"""Tests for faceray.sidecar_entry: control parsing and the stdio round-trip."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from faceray.core.modifier import BlurMode
from faceray.core.relighter import Relighter
from faceray.core.modifier import Modifier
from faceray.sidecar_entry import SidecarControl


def test_from_dict_full_payload() -> None:
    c = SidecarControl.from_dict(
        {
            "light_x": 0.1,
            "light_y": 0.2,
            "light_z": -0.9,
            "intensity": 1.3,
            "ambient": 0.25,
            "relight_enabled": False,
            "gaze_enabled": False,
            "blur_mode": "background",
        }
    )
    assert c.intensity == 1.3
    assert c.relight_enabled is False
    assert c.blur_mode is BlurMode.BACKGROUND


def test_from_dict_partial_inherits_base() -> None:
    base = SidecarControl(intensity=1.9, blur_mode=BlurMode.FACE)
    merged = SidecarControl.from_dict({"gaze_enabled": False}, base=base)
    assert merged.intensity == 1.9  # inherited
    assert merged.blur_mode is BlurMode.FACE  # inherited
    assert merged.gaze_enabled is False  # overridden


def test_from_json_roundtrips_through_to_dict() -> None:
    original = SidecarControl(light_x=0.2, intensity=1.1, blur_mode=BlurMode.FACE)
    clone = SidecarControl.from_json(json.dumps(original.to_dict()))
    assert clone == original


def test_invalid_blur_mode_rejected() -> None:
    with pytest.raises(ValueError):
        SidecarControl.from_dict({"blur_mode": "sideways"})


def test_apply_pushes_state_onto_engines() -> None:
    relighter = Relighter(use_gpu=False)
    modifier = Modifier()
    control = SidecarControl(
        light_x=0.0, light_y=0.0, light_z=-2.0,
        intensity=5.0, ambient=2.0, gaze_enabled=False, blur_mode=BlurMode.FACE,
    )
    control.apply(relighter, modifier)

    assert relighter.intensity == 2.0  # clipped to [0, 2]
    assert relighter.ambient == 1.0  # clipped to [0, 1]
    assert relighter.light_direction == pytest.approx((0.0, 0.0, -1.0))  # normalized
    assert modifier.gaze_strength == 0.0  # gaze disabled
    assert modifier.blur_mode is BlurMode.FACE


def test_apply_ignores_zero_light_vector() -> None:
    relighter = Relighter(light_direction=(0.4, -0.3, -1.0), use_gpu=False)
    before = relighter.light_direction
    SidecarControl(light_x=0.0, light_y=0.0, light_z=0.0).apply(relighter, Modifier())
    assert relighter.light_direction == before  # zero vector left the light untouched


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
        ['{"blur_mode":"face","intensity":1.2}'],
        ["--synthetic", "--no-sink", "--no-gpu", "--max-frames", "3",
         "--status-every", "1", "--fps", "120"],
    )
    types = [e["type"] for e in events]
    assert types[0] == "ready"
    assert "bye" in types

    acks = [e for e in events if e["type"] == "ack"]
    assert acks, "expected at least one ack for the control line"
    assert acks[0]["state"]["blur_mode"] == "face"
    assert acks[0]["state"]["intensity"] == 1.2


def test_stdio_reports_bad_payload() -> None:
    events = _run_sidecar(
        ["not-json-at-all"],
        ["--synthetic", "--no-sink", "--no-gpu", "--max-frames", "1"],
    )
    assert any(e["type"] == "error" for e in events)
    assert any(e["type"] == "bye" for e in events)  # channel survived the bad line
