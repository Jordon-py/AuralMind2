/**
 * Purpose: Browser smoke tests for the premium AuralMind2 workspace.
 * Data shapes: uses an in-memory WAV-like file object so the test does not
 * commit audio fixtures.
 * Syntax: npm run test:e2e.
 * Important functions: desktop workflow test at line 20, mobile layout test at
 * line 42.
 * Possible bugs: if the Flask API is running, the tiny in-memory file may fail
 * server-side audio parsing before preview fallback starts.
 * Enhance next: add a real curated fixture; add assertions for live API polling.
 */

import { expect, test } from '@playwright/test'

const tinyWav = Buffer.from([
  0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
  0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
  0x40, 0x1f, 0x00, 0x00, 0x80, 0x3e, 0x00, 0x00, 0x02, 0x00, 0x10, 0x00,
  0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00,
])

test('premium workspace renders and preview workflow completes', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('link', { name: /AuralMind2 home/i })).toBeVisible()
  await expect(page.getByRole('heading', { name: /Master, monitor, and deliver/i })).toBeVisible()

  const noOverflow = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)
  expect(noOverflow).toBe(true)

  await page.getByRole('radiogroup', { name: 'Stem mode' }).getByRole('radio', { name: 'auto' }).click()
  await expect(page.getByRole('radio', { name: 'auto' })).toHaveAttribute('aria-checked', 'true')

  await page.locator('input[type="file"]').setInputFiles({
    name: 'qa-audio.wav',
    mimeType: 'audio/wav',
    buffer: tinyWav,
  })
  await expect(page.getByLabel('Session setup').getByText('qa-audio.wav')).toBeVisible()

  await page.getByRole('button', { name: /Start master/i }).click()
  await expect(page.getByText('Preview mode')).toBeVisible({ timeout: 5000 })
  await expect(page.getByText(/masters\/qa-audio_premium_master.wav/i)).toBeVisible({
    timeout: 12000,
  })
})

test('mobile layout does not create horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')

  await expect(page.getByRole('link', { name: /AuralMind2 home/i })).toBeVisible()
  await expect(page.getByRole('heading', { name: /Master, monitor, and deliver/i })).toBeVisible()

  const noOverflow = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)
  expect(noOverflow).toBe(true)
})
