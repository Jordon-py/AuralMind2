/**
 * Purpose: Shared UI-only types for the AuralMind2 mastering workspace.
 * Data shapes: PresetOption, QueueItem, DeliveryOption, and RunMode drive the
 * local React state rendered by MasteringWorkspace.
 * Syntax: import types from features/mastering/masteringTypes.
 * Important functions: none; this file is type declarations only.
 * Possible bugs: backend enum drift can make preset ids stale.
 * Enhance next: hydrate preset options from /bootstrap; add saved session ids.
 */

import type { Metrics, SessionStatus } from '../../lib/api'

export type PresetOption = {
  id: string
  label: string
  tone: string
  target: string
}

export type DeliveryOption = '24-bit PCM' | '32-bit float'

export type RunMode = 'idle' | 'loading' | 'running' | 'done' | 'error'

export type QueueItem = {
  id: string
  title: string
  preset: string
  status: RunMode
  progress: number
  output?: string
  metrics: Metrics
  session?: SessionStatus
}
