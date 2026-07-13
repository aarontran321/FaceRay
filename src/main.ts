/**
 * FaceRay control-window bootstrap.
 *
 * Fetches the canonical default control state from Rust, mounts the control
 * panel, and connects each edit to the sidecar via a debounced `send_control`.
 * Live sidecar status (`sidecar://status` events) is surfaced in the title bar.
 */

import "./styles.css";
import {
  type ControlState,
  debounce,
  getDefaultControlState,
  inTauri,
  sendControl,
} from "./ipc";
import { mountControlPanel } from "./ui";
import { listen } from "@tauri-apps/api/event";

// Used only when previewing the UI in a plain browser (no Rust backend).
const FALLBACK: ControlState = {
  gaze_enabled: true,
  gaze_attention: 0.35,
  face_blur_enabled: false,
  background_blur_enabled: false,
  smoothing_enabled: false,
  smoothing_strength: 0.5,
  presence: "live",
};

interface StatusEvent {
  type: string;
  fps?: number;
  face?: boolean;
  sink?: string | null;
  preview?: string | null;
  message?: string;
}

async function boot(): Promise<void> {
  const root = document.querySelector<HTMLElement>("#app");
  if (root === null) throw new Error("missing #app root element");

  const tauri = inTauri();
  const initial = tauri ? await getDefaultControlState() : FALLBACK;

  const dispatch = debounce((state: ControlState) => {
    if (!tauri) {
      console.info("[faceray] (browser preview) control:", state);
      return;
    }
    sendControl(state).catch((err) =>
      console.error("[faceray] send_control failed:", err),
    );
  }, 60);

  const panel = mountControlPanel(root, initial, tauri ? "tauri" : "browser", dispatch);

  if (!tauri) return;

  // Seed the sidecar with the UI's starting values.
  sendControl(initial).catch(() => undefined);

  await listen<string>("sidecar://status", (event) => {
    let msg: StatusEvent;
    try {
      msg = JSON.parse(event.payload) as StatusEvent;
    } catch {
      return; // non-JSON diagnostic line
    }
    switch (msg.type) {
      case "status":
        panel.setStatus(`${msg.fps ?? "?"} fps · face ${msg.face ? "✓" : "—"}`);
        break;
      case "ready":
        panel.setPreview(msg.preview ?? null);
        panel.setStatus(msg.sink ? `sink: ${msg.sink}` : "no virtual cam");
        break;
      case "warning":
      case "error":
        panel.setStatus(msg.message ?? msg.type);
        break;
      default:
        break;
    }
  });
}

boot().catch((err) => {
  console.error("[faceray] boot failed:", err);
});
