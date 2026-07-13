//! Python sidecar process management (control plane, stdio transport).
//!
//! Spawns the bundled `faceray_backend`, forwards newline-delimited control
//! lines to its stdin, relays its status lines to the frontend as
//! `sidecar://status` events, and guarantees teardown on app exit so no
//! webcam hook is ever orphaned. No video data crosses this boundary.

use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Managed state holding the live sidecar child process, if one is running.
#[derive(Default)]
pub struct SidecarState {
    child: Mutex<Option<CommandChild>>,
}

impl SidecarState {
    /// Write one newline-terminated control line to the sidecar's stdin.
    pub fn write_line(&self, line: &str) -> Result<(), String> {
        let mut guard = self.child.lock().map_err(|err| err.to_string())?;
        let child = guard.as_mut().ok_or("sidecar is not running")?;
        let mut buf = line.as_bytes().to_vec();
        buf.push(b'\n');
        child.write(&buf).map_err(|err| err.to_string())
    }

    fn store(&self, child: CommandChild) {
        if let Ok(mut guard) = self.child.lock() {
            *guard = Some(child);
        }
    }

    fn take(&self) -> Option<CommandChild> {
        self.child.lock().ok().and_then(|mut guard| guard.take())
    }
}

/// Spawn the `faceray_backend` sidecar and relay its stdout/stderr.
///
/// stdout lines are forwarded to the webview as `sidecar://status` events;
/// stderr is logged for humans. The child handle is stored in [`SidecarState`]
/// so [`crate::send_control`] can write to its stdin.
pub fn spawn(app: &AppHandle) -> Result<(), String> {
    let command = app
        .shell()
        .sidecar("faceray_backend")
        .map_err(|err| err.to_string())?
        // Enable the loopback MJPEG preview the webview displays. Frame data
        // never crosses this IPC boundary — the webview pulls it over 127.0.0.1.
        .args(["--preview"]);
    let (mut rx, child) = command.spawn().map_err(|err| err.to_string())?;
    app.state::<SidecarState>().store(child);

    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes).trim().to_string();
                    if !line.is_empty() {
                        let _ = handle.emit("sidecar://status", line);
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        eprintln!("[sidecar] {trimmed}");
                    }
                }
                CommandEvent::Error(err) => {
                    eprintln!("[sidecar] error: {err}");
                }
                CommandEvent::Terminated(payload) => {
                    let _ = handle.emit("sidecar://terminated", payload.code);
                    break;
                }
                _ => {}
            }
        }
    });
    Ok(())
}

/// Kill the sidecar if it is still running. Idempotent; safe on repeated calls.
pub fn kill(app: &AppHandle) {
    if let Some(child) = app.state::<SidecarState>().take() {
        let _ = child.kill();
    }
}
