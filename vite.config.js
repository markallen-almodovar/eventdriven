import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,   // fall to next port if 5173 is taken instead of crashing
    proxy: {
      // Proxy /api/ml → FastAPI on 8000  (avoids CORS issues for ML calls)
      '/api/ml': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/ml/, ''),
      },
      // Proxy /api/express → Express on 5000
      '/api/express': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/express/, ''),
      },
    },
  },
})
