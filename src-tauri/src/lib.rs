//! FaceRay desktop shell library.
//!
//! Owns window lifecycle and the control-plane command surface. All computer
//! vision stays in the Python sidecar (data plane); this crate only ever
//! handles the lightweight [`ipc::ControlState`] payload and forwards it to the
//! sidecar over stdio.

mod ipc;
mod sidecar;

use ipc::ControlState;
use sidecar::SidecarState;

/// Return the canonical default control state so the UI initializes its
/// widgets from the same source of truth as the Rust and Python layers.
#[tauri::command]
fn default_control_state() -> ControlState {
    ControlState::default()
}

/// Serialize a control-state update to its single-line JSON wire form and write
/// it to the Python sidecar's stdin. Returns the wire line on success.
#[tauri::command]
fn send_control(
    sidecar: tauri::State<'_, SidecarState>,
    state: ControlState,
) -> Result<String, String> {
    let wire = state.to_wire().map_err(|err| err.to_string())?;
    sidecar.write_line(&wire)?;
    Ok(wire)
}

/// Application entry point, shared by the desktop binary and the mobile entry
/// point macro.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState::default())
        .invoke_handler(tauri::generate_handler![
            default_control_state,
            send_control
        ])
        .setup(|app| {
            // A failed sidecar spawn must not take down the window; the UI can
            // still render and surface the error. `send_control` will report
            // "sidecar is not running" until it comes up.
            if let Err(err) = sidecar::spawn(app.handle()) {
                eprintln!("[faceray] sidecar failed to start: {err}");
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the FaceRay application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                sidecar::kill(app_handle);
            }
        });
}
