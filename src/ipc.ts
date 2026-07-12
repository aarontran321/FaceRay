/**
 * Typed control-plane client.
 *
 * This is the single source of truth on the TypeScript side for the JSON
 * payload that crosses UI -> Rust -> Python. It mirrors `ControlState` in
 * `src-tauri/src/ipc.rs` field-for-field; keep the two in sync. Only scalar
 * control state travels here — never pixel data.
 */

import { invoke } from "@tauri-apps/api/core";

export type BlurMode = "off" | "face" | "background";

export interface ControlState {
  /** Virtual light direction (MediaPipe landmark space): +x right, +y down,
   *  +z away from camera. Normalized by the Python relighter on receipt. */
  light_x: number;
  light_y: number;
  light_z: number;
  /** Relight blend weight in [0, 2]. */
  intensity: number;
  /** Ambient floor in [0, 1]. */
  ambient: number;
  /** Effect toggles. */
  relight_enabled: boolean;
  gaze_enabled: boolean;
  blur_mode: BlurMode;
}

/** True only when running inside the Tauri webview (not a bare browser tab). */
export function inTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/** Fetch the canonical default control state defined in Rust. */
export async function getDefaultControlState(): Promise<ControlState> {
  return invoke<ControlState>("default_control_state");
}

/**
 * Push a control-state update toward the sidecar. Returns the exact JSON wire
 * line the Rust layer produced (Phase 1: echoed; Task 2: also written to the
 * sidecar's stdin).
 */
export async function sendControl(state: ControlState): Promise<string> {
  return invoke<string>("send_control", { state });
}

/**
 * Debounce helper so continuous slider drags collapse into at most one IPC
 * call per `waitMs`, keeping the control channel quiet.
 */
export function debounce<A extends unknown[]>(
  fn: (...args: A) => void,
  waitMs: number,
): (...args: A) => void {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return (...args: A) => {
    if (timer !== undefined) clearTimeout(timer);
    timer = setTimeout(() => fn(...args), waitMs);
  };
}
