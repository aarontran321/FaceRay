//! Control-plane data types shared across the IPC boundary.
//!
//! [`ControlState`] is the *entire* contract that crosses UI -> Rust -> Python.
//! It carries only lightweight scalar control state (light vector, toggles),
//! never pixel data — the video frame loop stays isolated in the Python
//! sidecar's data plane. It mirrors the `ControlState` interface in
//! `src/ipc.ts`; keep the two in sync.

use serde::{Deserialize, Serialize};

/// Which region the Gaussian blur is applied to. Serializes to the same
/// lowercase strings the Python `BlurMode` enum uses (`off` / `face` /
/// `background`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BlurMode {
    #[default]
    Off,
    Face,
    Background,
}

/// Control-plane payload sent from the UI toward the Python sidecar.
///
/// Serialized to a single line of JSON and (from Task 2) written to the
/// sidecar's stdin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ControlState {
    /// Virtual light direction in MediaPipe landmark space: `+x` right,
    /// `+y` down, `+z` away from the camera. Normalized by the Python
    /// relighter on receipt, so magnitude here is irrelevant.
    pub light_x: f32,
    pub light_y: f32,
    pub light_z: f32,
    /// Relight blend weight in `[0, 2]`.
    pub intensity: f32,
    /// Ambient floor in `[0, 1]`.
    pub ambient: f32,
    /// Effect toggles.
    pub relight_enabled: bool,
    pub gaze_enabled: bool,
    pub blur_mode: BlurMode,
}

impl Default for ControlState {
    fn default() -> Self {
        // Matches faceray.core defaults (Relighter / Modifier constructors).
        Self {
            light_x: 0.4,
            light_y: -0.3,
            light_z: -1.0,
            intensity: 0.6,
            ambient: 0.55,
            relight_enabled: true,
            gaze_enabled: true,
            blur_mode: BlurMode::Off,
        }
    }
}

impl ControlState {
    /// Serialize to the single-line JSON wire form the sidecar reads from
    /// stdin. Guaranteed newline-free so it frames cleanly as one line.
    pub fn to_wire(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_form_is_single_line_and_roundtrips() {
        let state = ControlState::default();
        let wire = state.to_wire().expect("serialize");
        assert!(!wire.contains('\n'));
        let back: ControlState = serde_json::from_str(&wire).expect("deserialize");
        assert_eq!(state, back);
    }

    #[test]
    fn blur_mode_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&BlurMode::Background).unwrap(),
            "\"background\""
        );
    }
}
