import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['.ngrok-free.dev'],  
    // Proxy forwards /auth and /chat requests to FastAPI during development.
    // This means in React you call fetch('/auth/login') instead of
    // fetch('http://localhost:8000/auth/login') — no hardcoded backend URL.
    proxy: {
      '/auth': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
    },
  },
})
