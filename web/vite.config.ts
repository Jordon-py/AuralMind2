import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Purpose: Vite config for the AuralMind2 React workspace.
 * Data shapes: dev server proxy routes browser /api calls to the existing
 * Flask dashboard while production builds emit static files to dist/.
 * Syntax: npm run dev, npm run build, npm run preview.
 * Important functions: defineConfig at line 11.
 * Possible bugs: deployed static hosts need VITE_AURALMIND_API_BASE if the API
 * is not same-origin.
 * Enhance next: add a typed env schema; add Vitest when component tests land.
 */

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },
})
