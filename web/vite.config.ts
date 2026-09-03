import { defineConfig } from "vite";

export default defineConfig({
  // Set BASE_PATH=/<repo>/ when deploying to a GitHub Pages project site;
  // asset URLs (including the vendored MediaPipe files) are resolved
  // against it via import.meta.env.BASE_URL.
  base: process.env.BASE_PATH ?? "/",
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 2000, // three + rapier + mediapipe are heavy
  },
});
