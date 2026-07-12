import { defineConfig } from "vite";

// Tauri drives Vite; keep the dev server on a fixed port it can point `devUrl`
// at, and never let Vite clear the terminal so Rust build output stays visible.
const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // The Rust crate has its own rebuild loop; don't trigger Vite HMR on it.
      ignored: ["**/src-tauri/**"],
    },
  },
});
