import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies the API to the Python service, so the app talks to a
// single origin in dev and in production (where `fh serve` serves web/dist and
// the API from one process). FH_API_URL retargets the backend.
const API = process.env.FH_API_URL || 'http://127.0.0.1:8080'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // Sandboxes and preview proxies reach the dev server through their own
    // hostnames; Vite blocks unknown hosts by default.
    allowedHosts: true,
    proxy: {
      '/api': {
        target: API,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        // SSE: no buffering, no timeout — a research run streams for minutes.
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => proxyReq.setHeader('Accept-Encoding', 'identity'))
        },
      },
    },
  },
  build: { outDir: 'dist', sourcemap: false },
})
