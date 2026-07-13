//! Control-plane data types shared across the IPC boundary.
//!
//! [`ControlState`] is the *entire* contract that crosses UI -> Rust -> Python.
//! It carries only lightweight scalar control state (four independent
//! face-interaction features), never pixel data — the video frame loop stays
//! isolated in the Python sidecar's data plane. It mirrors the `ControlState`
//! interface in `src/ipc.ts` and `SidecarControl` in `faceray/sidecar_entry.py`;
//! keep the three in sync.

use serde::{Deserialize, Serialize};

/// Control-plane payload sent from the UI toward the Python sidecar.
///
/// Serialized to a single line of JSON and written to the sidecar's stdin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ControlState {
    /// Eye-contact / gaze correction: on/off and its tracking sensitivity
    /// (`[0, 1]`).
    pub gaze_enabled: bool,
    pub gaze_sensitivity: f32,
    /// Face anonymiser — heavy opaque blur over the face hull only.
    pub face_blur_enabled: bool,
    /// Depth-of-field background blur; the face stays crisp.
    pub background_blur_enabled: bool,
    /// Skin smoothing (beauty filter): on/off and its intensity (`[0, 1]`).
    pub smoothing_enabled: bool,
    pub smoothing_strength: f32,
}

impl Default for ControlState {
    fn default() -> Self {
        // Matches faceray.core.Modifier defaults.
        Self {
            gaze_enabled: true,
            gaze_sensitivity: 0.7,
            face_blur_enabled: false,
            background_blur_enabled: false,
            smoothing_enabled: false,
            smoothing_strength: 0.5,
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
    fn defaults_enable_gaze_only() {
        let state = ControlState::default();
        assert!(state.gaze_enabled);
        assert!(!state.face_blur_enabled);
        assert!(!state.background_blur_enabled);
        assert!(!state.smoothing_enabled);
    }
}
