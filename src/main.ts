/**
 * FaceRay control-window bootstrap.
 *
 * Phase 1 renders the app shell and confirms the IPC round-trip to Rust. The
 * interactive light/effect controls are wired onto this shell in Task 3.
 */

import "./styles.css";
import {
  type ControlState,
  getDefaultControlState,
  inTauri,
  sendControl,
} from "./ipc";

const DEFAULT_FALLBACK: ControlState = {
  light_x: 0.4,
  light_y: -0.3,
  light_z: -1.0,
  intensity: 0.6,
  ambient: 0.55,
  relight_enabled: true,
  gaze_enabled: true,
  blur_mode: "off",
};

function render(root: HTMLElement, state: ControlState, bridge: string): void {
  root.innerHTML = `
    <header class="titlebar" data-tauri-drag-region>
      <span class="brand">FaceRay</span>
      <span class="bridge bridge--${bridge}">${bridge}</span>
    </header>
    <main class="panel">
      <p class="hint">Control plane online. Effect controls arrive in Task 3.</p>
      <pre class="state">${JSON.stringify(state, null, 2)}</pre>
    </main>
  `;
}

async function boot(): Promise<void> {
  const root = document.querySelector<HTMLElement>("#app");
  if (root === null) throw new Error("missing #app root element");

  if (!inTauri()) {
    // Running under plain `vite` (browser preview): no Rust backend to call.
    render(root, DEFAULT_FALLBACK, "browser");
    return;
  }

  const state = await getDefaultControlState();
  render(root, state, "tauri");

  // Confirm the UI -> Rust round-trip end-to-end during scaffolding.
  const wire = await sendControl(state);
  console.info("[faceray] send_control round-trip:", wire);
}

boot().catch((err) => {
  console.error("[faceray] boot failed:", err);
});
