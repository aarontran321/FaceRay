//! FaceRay desktop shell library.
//!
//! Owns window lifecycle and the control-plane command surface. All computer
//! vision stays in the Python sidecar (data plane); this crate only ever
//! handles the lightweight [`ipc::ControlState`] payload.
//!
//! Phase 1 defines the command contract the frontend calls and the sidecar
//! (Task 2) will consume. `send_control` currently validates and returns the
//! JSON wire line; Task 2 additionally forwards it to the sidecar's stdin.

mod ipc;

use ipc::ControlState;

/// Return the canonical default control state so the UI initializes its
/// widgets from the same source of truth as the Rust and Python layers.
#[tauri::command]
fn default_control_state() -> ControlState {
    ControlState::default()
}

/// Accept a control-state update from the UI and produce the single-line JSON
/// wire form destined for the sidecar. Returns the wire line on success or a
/// serialization error message on failure.
#[tauri::command]
fn send_control(state: ControlState) -> Result<String, String> {
    state.to_wire().map_err(|err| err.to_string())
}

/// Application entry point, shared by the desktop binary and the mobile entry
/// point macro.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            default_control_state,
            send_control
        ])
        .run(tauri::generate_context!())
        .expect("error while running the FaceRay application");
}
