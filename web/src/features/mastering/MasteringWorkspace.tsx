/**
 * Purpose: Premium product workspace for AuralMind2 mastering sessions.
 * Data shapes: local RunMode and QueueItem state plus Flask SessionStatus DTOs
 * from src/lib/api.ts.
 * Syntax: rendered by App as <MasteringWorkspace />.
 * Important functions: MasteringWorkspace at line 73, startMastering at line 139,
 * simulatePreviewRun at line 179.
 * Possible bugs: preview mode is local-only and must not be mistaken for a real
 * rendered master when the Flask API is offline.
 * Enhance next: add authenticated session history; add direct MCP tool discovery.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Circle,
  FileAudio,
  Gauge,
  Loader2,
  Play,
  Radio,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Waves,
} from 'lucide-react'
import {
  createMasteringSession,
  fetchSpectrogram,
  pollSession,
  type Metrics,
  type SessionStatus,
  type SpectrogramPayload,
} from '../../lib/api'
import {
  DELIVERY_OPTIONS,
  EMPTY_METRICS,
  INITIAL_QUEUE,
  PRESETS,
} from './masteringData'
import type { DeliveryOption, QueueItem, RunMode } from './masteringTypes'

const waveformBars = Array.from({ length: 84 }, (_, index) => {
  const base = Math.sin(index * 0.45) * 18 + Math.cos(index * 0.19) * 10
  return Math.max(12, Math.min(74, 38 + base + (index % 7) * 2))
})

const spectrumBars = Array.from({ length: 36 }, (_, index) => {
  const rolloff = 64 - index * 1.25
  const motion = Math.sin(index * 0.64) * 12
  return Math.max(8, Math.min(78, rolloff + motion))
})

function formatMetric(value: number, suffix: string) {
  if (Number.isNaN(value)) return `-- ${suffix}`
  return suffix
    ? `${value.toFixed(value > -2 && value < 2 ? 2 : 1)} ${suffix}`
    : value.toFixed(1)
}

function statusLabel(status: RunMode) {
  if (status === 'loading') return 'Preparing'
  if (status === 'running') return 'Rendering'
  if (status === 'done') return 'Ready'
  if (status === 'error') return 'Needs attention'
  return 'Idle'
}

function currentStatusFromBackend(status: SessionStatus): RunMode {
  if (status.status === 'done') return 'done'
  if (status.status === 'error') return 'error'
  if (status.status === 'queued' || status.status === 'running') return 'running'
  return 'idle'
}

export function MasteringWorkspace() {
  const [selectedPreset, setSelectedPreset] = useState(PRESETS[0].id)
  const [stemMode, setStemMode] = useState('off')
  const [deliveries, setDeliveries] = useState<DeliveryOption[]>(DELIVERY_OPTIONS)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [queue, setQueue] = useState<QueueItem[]>(INITIAL_QUEUE)
  const [activeSession, setActiveSession] = useState<SessionStatus | null>(null)
  const [runMode, setRunMode] = useState<RunMode>('idle')
  const [apiNotice, setApiNotice] = useState('')
  const [spectrogram, setSpectrogram] = useState<SpectrogramPayload | null>(null)
  const [progress, setProgress] = useState(0)
  const pollTimer = useRef<number | null>(null)

  const preset = useMemo(
    () => PRESETS.find((item) => item.id === selectedPreset) ?? PRESETS[0],
    [selectedPreset],
  )

  const liveMetrics: Metrics = activeSession?.metrics ?? queue[0]?.metrics ?? EMPTY_METRICS
  const activeTitle = activeSession?.song_name ?? selectedFile?.name ?? 'No source selected'
  const activeOutput = activeSession?.output_file ?? queue[0]?.output ?? ''
  const canStart = runMode !== 'loading' && runMode !== 'running'

  useEffect(() => {
    return () => {
      if (pollTimer.current) window.clearInterval(pollTimer.current)
    }
  }, [])

  const upsertQueueFromStatus = useCallback(
    (status: SessionStatus) => {
      const nextItem: QueueItem = {
        id: status.session_id,
        title: status.song_name,
        preset: preset.label,
        status: currentStatusFromBackend(status),
        progress: status.progress,
        output: status.output_file ?? undefined,
        metrics: status.metrics,
        session: status,
      }
      setQueue((current) => [nextItem, ...current.filter((item) => item.id !== nextItem.id)])
    },
    [preset.label],
  )

  useEffect(() => {
    if (!activeSession?.session_id || runMode !== 'running') return

    pollTimer.current = window.setInterval(async () => {
      try {
        const status = await pollSession(activeSession.session_id)
        setActiveSession(status)
        setProgress((current) => Math.max(status.progress, current))
        setRunMode(currentStatusFromBackend(status))

        if (status.status === 'done' || status.status === 'error') {
          if (pollTimer.current) window.clearInterval(pollTimer.current)
          upsertQueueFromStatus(status)
        }
      } catch (error) {
        if (pollTimer.current) window.clearInterval(pollTimer.current)
        setRunMode('error')
        setApiNotice(error instanceof Error ? error.message : 'Unable to poll session.')
      }
    }, 2200)

    return () => {
      if (pollTimer.current) window.clearInterval(pollTimer.current)
    }
  }, [activeSession?.session_id, runMode, upsertQueueFromStatus])

  function toggleDelivery(option: DeliveryOption) {
    setDeliveries((current) =>
      current.includes(option)
        ? current.filter((item) => item !== option)
        : [...current, option],
    )
  }

  async function startMastering() {
    if (!selectedFile) {
      setRunMode('error')
      setApiNotice('Choose a source file before starting a live render.')
      return
    }

    setRunMode('loading')
    setApiNotice('')
    setProgress(8)
    setSpectrogram(null)

    try {
      const session = await createMasteringSession(selectedFile, {
        preset: selectedPreset,
        workflow: selectedPreset === 'competitive_trap' ? 'premium_trap' : 'standard',
        stem_mode: stemMode,
      })
      setActiveSession(session)
      setRunMode(currentStatusFromBackend(session))
      setProgress(session.progress || 12)
      const spectrum = await fetchSpectrogram(session.session_id)
      setSpectrogram(spectrum)
      upsertQueueFromStatus(session)
    } catch (error) {
      setApiNotice(
        error instanceof Error
          ? `Live backend unavailable: ${error.message}`
          : 'Live backend unavailable.',
      )
      simulatePreviewRun(selectedFile.name)
    }
  }

  function simulatePreviewRun(fileName = 'Preview source') {
    if (pollTimer.current) window.clearInterval(pollTimer.current)

    const sessionId = `preview-${Date.now()}`
    const previewSession: SessionStatus = {
      session_id: sessionId,
      song_name: fileName.replace(/\.[^.]+$/, ''),
      audio_path: fileName,
      workflow: selectedPreset === 'competitive_trap' ? 'premium_trap' : 'standard',
      preset: selectedPreset,
      job_id: 'preview',
      status: 'running',
      progress: 14,
      is_mastering: true,
      metrics: {
        lufs: -18.4,
        true_peak: -1.18,
        crest_db: 12.1,
        stereo_corr: 0.82,
      },
      duration: 0,
      error: null,
      output_file: null,
    }

    setActiveSession(previewSession)
    setRunMode('running')
    setProgress(14)

    let nextProgress = 14
    pollTimer.current = window.setInterval(() => {
      nextProgress = Math.min(100, nextProgress + 9)
      const done = nextProgress >= 100
      const nextSession: SessionStatus = {
        ...previewSession,
        status: done ? 'done' : 'running',
        progress: nextProgress,
        is_mastering: !done,
        output_file: done ? `masters/${previewSession.song_name}_premium_master.wav` : null,
        metrics: {
          lufs: -18.4 + nextProgress * 0.034,
          true_peak: -1.18 + nextProgress * 0.002,
          crest_db: 12.1 + Math.sin(nextProgress / 24) * 0.8,
          stereo_corr: 0.82 + nextProgress * 0.0008,
        },
      }
      setActiveSession(nextSession)
      setProgress(nextProgress)

      if (done) {
        if (pollTimer.current) window.clearInterval(pollTimer.current)
        setRunMode('done')
        upsertQueueFromStatus(nextSession)
      }
    }, 650)
  }

  return (
    <main className="app-shell">
      <header className="topbar" aria-label="Primary">
        <a className="brand" href="/" aria-label="AuralMind2 home">
          <span className="brand-mark" aria-hidden="true">
            <Waves />
          </span>
          <span>
            <strong>AuralMind2</strong>
            <small>Mastering workspace</small>
          </span>
        </a>
        <nav className="nav-links" aria-label="Workspace navigation">
          <a href="#workspace">Workspace</a>
          <a href="#queue">Queue</a>
          <a href="#delivery">Delivery</a>
        </nav>
        <div className="connection-pill" data-state={apiNotice ? 'preview' : 'ready'}>
          <Circle aria-hidden="true" />
          {apiNotice ? 'Preview mode' : 'Ready'}
        </div>
      </header>

      <section className="workspace-hero" id="workspace">
        <div>
          <p className="eyebrow">Premium audio command center</p>
          <h1>Master, monitor, and deliver with one clean workflow.</h1>
        </div>
        <div className="hero-meta" aria-label="Current mastering profile">
          <span>{preset.label}</span>
          <span>{stemMode === 'off' ? 'No stems' : stemMode}</span>
          <span>{deliveries.join(' + ')}</span>
        </div>
      </section>

      <section className="workspace-grid" aria-label="Mastering workspace">
        <aside className="panel setup-panel" aria-label="Session setup">
          <PanelTitle icon={<SlidersHorizontal />} title="Session" />

          <label className="file-drop">
            <input
              type="file"
              accept="audio/*"
              onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
            />
            <Upload aria-hidden="true" />
            <span>{selectedFile ? selectedFile.name : 'Choose source audio'}</span>
            <small>{selectedFile ? `${Math.round(selectedFile.size / 1024 / 1024)} MB` : 'WAV, MP3, FLAC, AIFF'}</small>
          </label>

          <div className="field-block">
            <span className="field-label">Mastering profile</span>
            <div className="preset-list" role="radiogroup" aria-label="Mastering profile">
              {PRESETS.map((option) => (
                <button
                  className="preset-button"
                  type="button"
                  role="radio"
                  aria-checked={selectedPreset === option.id}
                  data-selected={selectedPreset === option.id}
                  key={option.id}
                  onClick={() => setSelectedPreset(option.id)}
                >
                  <span>{option.label}</span>
                  <small>{option.tone}</small>
                </button>
              ))}
            </div>
          </div>

          <div className="field-block">
            <span className="field-label">Stem mode</span>
            <div className="segmented" role="radiogroup" aria-label="Stem mode">
              {['off', 'auto', 'on'].map((mode) => (
                <button
                  key={mode}
                  type="button"
                  role="radio"
                  aria-checked={stemMode === mode}
                  data-selected={stemMode === mode}
                  onClick={() => setStemMode(mode)}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          <button className="primary-action" type="button" onClick={startMastering} disabled={!canStart}>
            {runMode === 'loading' ? <Loader2 className="spin" aria-hidden="true" /> : <Play aria-hidden="true" />}
            {runMode === 'loading' ? 'Preparing' : 'Start master'}
          </button>

          {apiNotice && (
            <InlineNotice tone={runMode === 'error' ? 'error' : 'preview'}>{apiNotice}</InlineNotice>
          )}
        </aside>

        <section className="panel analysis-panel" aria-label="Mastering analysis">
          <div className="analysis-header">
            <PanelTitle icon={<Activity />} title="Live analysis" />
            <StatusPill status={runMode} />
          </div>

          <div className="track-title">
            <FileAudio aria-hidden="true" />
            <span>{activeTitle}</span>
          </div>

          <Waveform progress={progress} />
          <MetricGrid metrics={liveMetrics} />

          <div className="spectrum-card">
            <div>
              <h2>Spectrum</h2>
              <p>{spectrogram ? `${spectrogram.frequencies.length} bands loaded` : 'Calibrated preview'}</p>
            </div>
            <Spectrum />
          </div>
        </section>

        <aside className="panel queue-panel" id="queue" aria-label="Mastering queue">
          <PanelTitle icon={<Radio />} title="Queue" />
          <div className="queue-list">
            {queue.length === 0 ? (
              <EmptyState />
            ) : (
              queue.slice(0, 4).map((item) => <QueueRow key={item.id} item={item} />)
            )}
          </div>

          <div className="delivery-box" id="delivery">
            <PanelTitle icon={<Sparkles />} title="Delivery" />
            <div className="delivery-options" aria-label="Delivery formats">
              {DELIVERY_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  data-selected={deliveries.includes(option)}
                  aria-pressed={deliveries.includes(option)}
                  onClick={() => toggleDelivery(option)}
                >
                  <CheckCircle2 aria-hidden="true" />
                  {option}
                </button>
              ))}
            </div>
            <div className="output-line">
              <span>Output</span>
              <strong>{activeOutput || 'Waiting for completed master'}</strong>
            </div>
          </div>
        </aside>
      </section>
    </main>
  )
}

function PanelTitle({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="panel-title">
      {icon}
      <span>{title}</span>
    </div>
  )
}

function StatusPill({ status }: { status: RunMode }) {
  const Icon = status === 'error' ? AlertCircle : status === 'done' ? CheckCircle2 : Gauge
  return (
    <span className="status-pill" data-status={status}>
      <Icon aria-hidden="true" />
      {statusLabel(status)}
    </span>
  )
}

function InlineNotice({
  children,
  tone,
}: {
  children: ReactNode
  tone: 'preview' | 'error'
}) {
  return (
    <p className="inline-notice" data-tone={tone} role={tone === 'error' ? 'alert' : 'status'}>
      {tone === 'error' ? <AlertCircle aria-hidden="true" /> : <Sparkles aria-hidden="true" />}
      <span>{children}</span>
    </p>
  )
}

function Waveform({ progress }: { progress: number }) {
  return (
    <div className="waveform" aria-label={`Master progress ${Math.round(progress)} percent`}>
      <div className="waveform-progress" style={{ inlineSize: `${Math.min(100, progress)}%` }} />
      {waveformBars.map((height, index) => (
        <span
          key={`${height}-${index}`}
          style={{ blockSize: `${height}%` }}
          data-active={index < (progress / 100) * waveformBars.length}
        />
      ))}
    </div>
  )
}

function MetricGrid({ metrics }: { metrics: Metrics }) {
  const items = [
    ['LUFS', formatMetric(metrics.lufs, '')],
    ['True peak', formatMetric(metrics.true_peak, 'dBTP')],
    ['Crest', formatMetric(metrics.crest_db, 'dB')],
    ['Stereo', metrics.stereo_corr.toFixed(3)],
  ]

  return (
    <div className="metric-grid">
      {items.map(([label, value]) => (
        <div className="metric-tile" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  )
}

function Spectrum() {
  return (
    <div className="spectrum" aria-hidden="true">
      {spectrumBars.map((height, index) => (
        <span key={`${height}-${index}`} style={{ blockSize: `${height}%` }} />
      ))}
    </div>
  )
}

function QueueRow({ item }: { item: QueueItem }) {
  return (
    <article className="queue-row">
      <div>
        <strong>{item.title}</strong>
        <span>{item.preset}</span>
      </div>
      <div className="queue-progress" aria-label={`${item.progress} percent complete`}>
        <span style={{ inlineSize: `${item.progress}%` }} />
      </div>
      <StatusPill status={item.status} />
    </article>
  )
}

function EmptyState() {
  return (
    <div className="empty-state">
      <FileAudio aria-hidden="true" />
      <strong>No sessions yet</strong>
      <span>Completed masters will appear here.</span>
    </div>
  )
}
