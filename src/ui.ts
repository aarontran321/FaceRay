/**
 * Control-panel UI (framework-free DOM).
 *
 * Two vertical partitions: a fixed 16:9 preview pane on top, and a grid of four
 * feature cards below (Gaze, Face anonymizer, Background blur, Face smoothing).
 * Presentation only — no IPC here; `main.ts` connects `onChange` to the typed
 * control client. The panel owns a working `ControlState` and calls `onChange`
 * with a fresh copy after every edit.
 */

import type { ControlState } from "./ipc";

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

function makeSwitch(initial: boolean, onChange: (value: boolean) => void): HTMLButtonElement {
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
  return sw;
}

function sliderControl(
  label: string,
  min: number,
  max: number,
  step: number,
  initial: number,
  onInput: (value: number) => void,
): HTMLElement {
  const row = makeEl("div", "slider");
  const head = makeEl("div", "slider__head");
  const value = makeEl("span", "slider__value", initial.toFixed(2));
  head.append(makeEl("span", "slider__label", label), value);

  const input = makeEl("input", "slider__input");
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

function featureCard(
  title: string,
  note: string,
  toggle: HTMLElement,
  slider?: HTMLElement,
): HTMLElement {
  const card = makeEl("div", "card");
  const head = makeEl("div", "card__head");
  head.append(makeEl("h3", "card__title", title), toggle);
  card.append(head, makeEl("p", "card__note", note));
  if (slider !== undefined) card.append(slider);
  return card;
}

export function mountControlPanel(
  root: HTMLElement,
  initial: ControlState,
  bridge: Bridge,
  onChange: (state: ControlState) => void,
): PanelHandle {
  const state: ControlState = { ...initial };
  const emit = () => onChange({ ...state });

  root.innerHTML = "";

  // -- Title bar ----------------------------------------------------------
  const titlebar = makeEl("header", "titlebar");
  titlebar.setAttribute("data-tauri-drag-region", "");
  const statusEl = makeEl(
    "span",
    "status-line",
    bridge === "tauri" ? "starting…" : "browser preview",
  );
  titlebar.append(makeEl("span", "brand", "FaceRay"), statusEl);

  // -- Preview pane (fixed 16:9) ------------------------------------------
  const preview = makeEl("figure", "preview");
  const previewFrame = makeEl("div", "preview__frame");
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
  previewFrame.append(previewImg, previewNote);
  preview.append(previewFrame);

  // -- Control panel: four feature cards ----------------------------------
  const cards = makeEl("section", "cards");

  const gazeCard = featureCard(
    "Gaze correction",
    "Remap your eyes back toward the lens for natural eye contact.",
    makeSwitch(state.gaze_enabled, (v) => {
      state.gaze_enabled = v;
      emit();
    }),
    sliderControl("Sensitivity", 0, 1, 0.02, state.gaze_sensitivity, (v) => {
      state.gaze_sensitivity = v;
      emit();
    }),
  );

  const anonymiseCard = featureCard(
    "Face anonymizer",
    "Heavy privacy blur over your face only; background stays sharp.",
    makeSwitch(state.face_blur_enabled, (v) => {
      state.face_blur_enabled = v;
      emit();
    }),
  );

  const backgroundCard = featureCard(
    "Background blur",
    "Depth-of-field blur behind you; your face stays crisp.",
    makeSwitch(state.background_blur_enabled, (v) => {
      state.background_blur_enabled = v;
      emit();
    }),
  );

  const smoothingCard = featureCard(
    "Face smoothing",
    "Real-time skin smoothing that keeps eyes and lips sharp.",
    makeSwitch(state.smoothing_enabled, (v) => {
      state.smoothing_enabled = v;
      emit();
    }),
    sliderControl("Intensity", 0, 1, 0.02, state.smoothing_strength, (v) => {
      state.smoothing_strength = v;
      emit();
    }),
  );

  cards.append(gazeCard, anonymiseCard, backgroundCard, smoothingCard);

  const panel = makeEl("main", "panel");
  panel.append(preview, cards);
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
