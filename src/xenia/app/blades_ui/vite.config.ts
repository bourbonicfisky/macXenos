import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The app serves dist/ from its own URL scheme (blades_overlay_mac.mm), so
// absolute paths like /fonts/... resolve against the bundle's blades/ folder.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsInlineLimit: 0,
    // WebKit on the oldest supported macOS (15.0, Safari 18).
    target: "safari18"
  }
});
