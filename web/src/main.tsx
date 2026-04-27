/**
 * Purpose: React entrypoint for the AuralMind2 web workspace.
 * Data shapes: mounts the App component into the single #root DOM node.
 * Syntax: loaded by index.html through Vite.
 * Important functions: createRoot render call at line 12.
 * Possible bugs: #root must exist in index.html or the app cannot mount.
 * Enhance next: add error boundary and performance reporting hooks.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
