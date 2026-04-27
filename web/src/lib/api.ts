/**
 * Purpose: Typed browser client for the existing Flask mastering dashboard API.
 * Data shapes: UploadResponse, SessionResponse, SessionStatus, Metrics, and
 * SpectrogramPayload mirror mastering_ui.py JSON routes.
 * Syntax: call createMasteringSession(file, request) or pollSession(sessionId).
 * Important functions: apiFetch at line 62, createMasteringSession at line 101,
 * pollSession at line 141.
 * Possible bugs: cloud deployments need VITE_AURALMIND_API_BASE or same-origin
 * proxying; otherwise browser CORS rules will block API calls.
 * Enhance next: add AbortController cancellation; add typed backend health route.
 */

export type Metrics = {
  lufs: number
  true_peak: number
  crest_db: number
  stereo_corr: number
}

export type SessionStatus = {
  session_id: string
  song_name: string
  audio_path: string
  workflow: string
  preset: string | null
  job_id: string | null
  status: 'ready' | 'queued' | 'running' | 'done' | 'error' | string
  progress: number
  is_mastering: boolean
  metrics: Metrics
  duration: number
  error: string | null
  output_file: string | null
}

export type SpectrogramPayload = {
  frequencies: number[]
  times: number[]
  magnitude: number[][]
  min_db: number
  max_db: number
}

type UploadResponse = {
  success: boolean
  filename: string
  audio_path: string
  song_name: string
  error?: string
}

type SessionResponse = {
  success: boolean
  session_id: string
  audio_path: string
  song_name: string
  duration: number
  metrics: Metrics
  error?: string
}

type StartResponse = SessionStatus & {
  success: boolean
}

export type StartMasteringRequest = {
  preset: string
  workflow: string
  stem_mode: string
}

const API_BASE = import.meta.env.VITE_AURALMIND_API_BASE?.replace(/\/$/, '') ?? ''

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...options?.headers,
    },
  })

  const contentType = response.headers.get('content-type') ?? ''
  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text()

  if (!response.ok) {
    const message =
      typeof payload === 'object' && payload && 'error' in payload
        ? String(payload.error)
        : `Request failed with ${response.status}`
    throw new Error(message)
  }

  return payload as T
}

async function uploadAudio(file: File): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  return apiFetch<UploadResponse>('/api/upload', {
    method: 'POST',
    body: form,
  })
}

async function createSession(audioPath: string, songName: string): Promise<SessionResponse> {
  return apiFetch<SessionResponse>('/api/session/new', {
    method: 'POST',
    body: JSON.stringify({
      audio_path: audioPath,
      song_name: songName,
    }),
  })
}

async function startSession(
  sessionId: string,
  request: StartMasteringRequest,
): Promise<StartResponse> {
  return apiFetch<StartResponse>(`/api/session/${sessionId}/start`, {
    method: 'POST',
    body: JSON.stringify({
      preset: request.preset,
      workflow: request.workflow,
      stem_mode: request.stem_mode,
    }),
  })
}

export async function createMasteringSession(
  file: File,
  request: StartMasteringRequest,
): Promise<SessionStatus> {
  const upload = await uploadAudio(file)
  const session = await createSession(upload.audio_path, upload.song_name)
  return startSession(session.session_id, request)
}

export async function pollSession(sessionId: string): Promise<SessionStatus> {
  return apiFetch<SessionStatus>(`/api/session/${sessionId}/status`)
}

export async function fetchSpectrogram(sessionId: string): Promise<SpectrogramPayload> {
  return apiFetch<SpectrogramPayload>(`/api/session/${sessionId}/spectrogram`)
}
