/**
 * Purpose: Static product copy and sample data for the premium mastering UI.
 * Data shapes: arrays of PresetOption, DeliveryOption, and QueueItem seed the
 * first-load empty/demo surface before a live Flask session exists.
 * Syntax: import PRESETS, DELIVERY_OPTIONS, and INITIAL_QUEUE.
 * Important functions: none; data constants only.
 * Possible bugs: sample queue can be confused with live history if labels drift.
 * Enhance next: load queue history from an API; localize labels for ChatGPT Apps.
 */

import type { DeliveryOption, PresetOption, QueueItem } from './masteringTypes'

export const PRESETS: PresetOption[] = [
  {
    id: 'competitive_trap',
    label: 'Trap Competitive',
    tone: '808 focus, vocal lift, controlled width',
    target: '-12.2 LUFS',
  },
  {
    id: 'hi_fi_streaming',
    label: 'Hi-Fi Streaming',
    tone: 'clean top, open midrange, softer density',
    target: '-14 LUFS',
  },
  {
    id: 'radio_loud',
    label: 'Radio Loud',
    tone: 'forward level, tight transient control',
    target: '-11 LUFS',
  },
  {
    id: 'cinematic',
    label: 'Deep & Wide',
    tone: 'wide stage, dark polish, low-end depth',
    target: '-13.5 LUFS',
  },
]

export const DELIVERY_OPTIONS: DeliveryOption[] = ['24-bit PCM', '32-bit float']

export const INITIAL_QUEUE: QueueItem[] = [
  {
    id: 'facetime',
    title: 'FaceTime (6)',
    preset: 'Trap Competitive',
    status: 'done',
    progress: 100,
    output: '24-bit + 32-bit ready',
    metrics: {
      lufs: -15.01,
      true_peak: -0.97,
      crest_db: 13.7,
      stereo_corr: 0.92,
    },
  },
  {
    id: 'explicit-batch',
    title: 'Explicit trap batch',
    preset: 'Trap Competitive',
    status: 'done',
    progress: 100,
    output: '10 masters complete',
    metrics: {
      lufs: -14.8,
      true_peak: -1,
      crest_db: 12.9,
      stereo_corr: 0.88,
    },
  },
]

export const EMPTY_METRICS = {
  lufs: -23,
  true_peak: -1,
  crest_db: 0,
  stereo_corr: 0,
}
