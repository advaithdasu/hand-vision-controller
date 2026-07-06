import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 2000, // three + rapier + mediapipe are heavy
  },
});
