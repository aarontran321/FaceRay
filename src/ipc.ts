/**
 * Typed control-plane client.
 *
 * This is the single source of truth on the TypeScript side for the JSON
 * payload that crosses UI -> Rust -> Python. It mirrors `ControlState` in
 * `src-tauri/src/ipc.rs` field-for-field; keep the two in sync. Only scalar
 * control state travels here — never pixel data.
 */

import { invoke } from "@tauri-apps/api/core";

export interface ControlState {
  /** Eye-contact / gaze correction: on/off and tracking sensitivity [0, 1]. */
  gaze_enabled: boolean;
  gaze_sensitivity: number;
  /** Face anonymiser — heavy opaque blur over the face only. */
  face_blur_enabled: boolean;
  /** Depth-of-field background blur; the face stays crisp. */
  background_blur_enabled: boolean;
  /** Skin smoothing (beauty filter): on/off and intensity [0, 1]. */
  smoothing_enabled: boolean;
  smoothing_strength: number;
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
