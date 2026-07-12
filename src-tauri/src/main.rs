// Prevents an extra console window on Windows in release builds. No effect on
// macOS/Linux, but keeps the entry point portable.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    faceray_lib::run()
}
