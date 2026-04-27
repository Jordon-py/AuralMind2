/**
 * Purpose: Playwright smoke configuration for the AuralMind2 React workspace.
 * Data shapes: starts the Vite dev server and runs Chromium checks against the
 * single-page product UI.
 * Syntax: npm run test:e2e.
 * Important functions: defineConfig at line 14.
 * Possible bugs: CI must run `npx playwright install chromium` before tests.
 * Enhance next: add visual regression snapshots; add a live Flask API project.
 */

import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: true,
    timeout: 20_000,
  },
  use: {
    baseURL: 'http://127.0.0.1:5173',
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 } },
    },
  ],
})
