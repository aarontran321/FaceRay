/**
 * Control-panel widgets (framework-free DOM).
 *
 * Builds the light/effect controls, owns the working `ControlState`, and calls
 * `onChange` with a fresh copy after every edit. Presentation only — no IPC
 * here; `main.ts` connects `onChange` to the typed control client.
 */

import type { BlurMode, ControlState } from "./ipc";

export interface PanelHandle {
  /** Update the small live-status readout in the title bar. */
  setStatus(text: string): void;
  /** Point the preview at the sidecar's MJPEG stream URL, or clear it. */
  setPreview(url: string | null): void;
}

type Bridge = "tauri" | "browser";

function makeEl<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (className !== undefined) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function group(title: string): HTMLElement {
  const section = makeEl("section", "group");
  section.append(makeEl("h2", "group-title", title));
  return section;
}

function sliderRow(
  label: string,
  min: number,
  max: number,
  step: number,
  initial: number,
  onInput: (value: number) => void,
): HTMLElement {
  const row = makeEl("div", "row");
  const head = makeEl("div", "row-head");
  const value = makeEl("span", "row-value", initial.toFixed(2));
  head.append(makeEl("label", "row-label", label), value);

  const input = makeEl("input", "slider");
  input.type = "range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(initial);
  input.addEventListener("input", () => {
    const v = Number(input.value);
    value.textContent = v.toFixed(2);
    onInput(v);
  });

  row.append(head, input);
  return row;
}

function toggleRow(
  label: string,
  initial: boolean,
  onChange: (value: boolean) => void,
): HTMLElement {
  const row = makeEl("div", "row row--inline");
  const sw = makeEl("button", "switch");
  sw.type = "button";
  sw.setAttribute("role", "switch");

  let on = initial;
  const paint = () => {
    sw.setAttribute("aria-checked", String(on));
    sw.classList.toggle("switch--on", on);
  };
  paint();
  sw.addEventListener("click", () => {
    on = !on;
    paint();
    onChange(on);
  });

  row.append(makeEl("label", "row-label", label), sw);
  return row;
}

function segmentedRow(
  label: string,
  options: readonly string[],
  initial: string,
  onChange: (value: string) => void,
): HTMLElement {
  const row = makeEl("div", "row");
  const head = makeEl("div", "row-head");
  head.append(makeEl("label", "row-label", label));

  const seg = makeEl("div", "segmented");
  const buttons: HTMLButtonElement[] = [];
  const select = (value: string) => {
    for (const b of buttons) {
      b.classList.toggle("segmented__btn--on", b.dataset.value === value);
    }
  };
  for (const opt of options) {
    const btn = makeEl("button", "segmented__btn", opt);
    btn.type = "button";
    btn.dataset.value = opt;
    btn.addEventListener("click", () => {
      select(opt);
      onChange(opt);
    });
    buttons.push(btn);
    seg.append(btn);
  }
  select(initial);

  row.append(head, seg);
  return row;
}

const BLUR_MODES = ["off", "face", "background"] as const;

/**
 * Render the control panel into `root` and wire every widget to mutate a
 * working ControlState and notify `onChange`.
 */
export function mountControlPanel(
  root: HTMLElement,
  initial: ControlState,
  bridge: Bridge,
  onChange: (state: ControlState) => void,
): PanelHandle {
  const state: ControlState = { ...initial };
  const emit = () => onChange({ ...state });

  root.innerHTML = "";

  const titlebar = makeEl("header", "titlebar");
  titlebar.setAttribute("data-tauri-drag-region", "");
  const statusEl = makeEl(
    "span",
    "status-line",
    bridge === "tauri" ? "starting…" : "browser preview",
  );
  titlebar.append(makeEl("span", "brand", "FaceRay"), statusEl);

  const panel = makeEl("main", "panel");

  const preview = makeEl("figure", "preview");
  const previewImg = makeEl("img", "preview__img");
  previewImg.alt = "Live processed camera preview";
  const previewNote = makeEl(
    "figcaption",
    "preview__note",
    bridge === "tauri" ? "waiting for camera…" : "preview runs in the desktop app",
  );
  previewImg.addEventListener("error", () => {
    previewImg.classList.remove("preview__img--on");
    previewNote.hidden = false;
    previewNote.textContent = "preview unavailable";
  });
  preview.append(previewImg, previewNote);

  const light = group("Light");
  light.append(
    sliderRow("Direction X", -1, 1, 0.05, state.light_x, (v) => {
      state.light_x = v;
      emit();
    }),
    sliderRow("Direction Y", -1, 1, 0.05, state.light_y, (v) => {
      state.light_y = v;
      emit();
    }),
    sliderRow("Direction Z", -1, 1, 0.05, state.light_z, (v) => {
      state.light_z = v;
      emit();
    }),
    sliderRow("Intensity", 0, 2, 0.05, state.intensity, (v) => {
      state.intensity = v;
      emit();
    }),
    sliderRow("Ambient", 0, 1, 0.05, state.ambient, (v) => {
      state.ambient = v;
      emit();
    }),
  );

  const effects = group("Effects");
  effects.append(
    toggleRow("Relighting", state.relight_enabled, (v) => {
      state.relight_enabled = v;
      emit();
    }),
    toggleRow("Gaze correction", state.gaze_enabled, (v) => {
      state.gaze_enabled = v;
      emit();
    }),
    segmentedRow("Blur", BLUR_MODES, state.blur_mode, (v) => {
      state.blur_mode = v as BlurMode;
      emit();
    }),
  );

  panel.append(preview, light, effects);
  root.append(titlebar, panel);

  return {
    setStatus(text: string) {
      statusEl.textContent = text;
    },
    setPreview(url: string | null) {
      if (url === null) {
        previewImg.removeAttribute("src");
        previewImg.classList.remove("preview__img--on");
        previewNote.hidden = false;
        return;
      }
      previewImg.src = url;
      previewImg.classList.add("preview__img--on");
      previewNote.hidden = true;
    },
  };
}
