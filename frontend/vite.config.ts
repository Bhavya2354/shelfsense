import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  if (!env.VITE_API_BASE_URL) {
    throw new Error("VITE_API_BASE_URL is required (see frontend/.env.example)");
  }
  return {
    base: env.VITE_BASE_PATH || "/",
    plugins: [react(), tailwindcss()],
    build: {
      target: "es2022",
      rollupOptions: {
        output: {
          // Keep the charting library in its own long-cacheable chunk.
          manualChunks: (id: string) =>
            /node_modules[\/](echarts|zrender)[\/]/.test(id) ? "charts" : undefined,
        },
      },
    },
  };
});
