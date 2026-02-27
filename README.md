# AuralMind Maestro - MCP Mastering Server

Production-grade AI mastering pipeline exposed as MCP tools. The server wraps
the DSP engine in `tools/auralmind_maestro.py` and provides a non-blocking
mastering workflow for LLM clients.

## Highlights

- 48 kHz mastering pipeline with float32/float64 export options.
- Async job execution via background thread pool.
- LLM-first control loop: upload -> analyze -> propose -> run -> poll -> result -> read.
- Handle-based I/O (audio_id/artifact_id) with chunked artifact reads.
- Optional Demucs stem separation when `torch` + `demucs` are available.

## Architecture

```
LLM client -> FastMCP (stdio / http)
   resources: config://system-prompt, config://mcp-docs, config://server-info
              auralmind://workflow, auralmind://metrics, auralmind://presets
              auralmind://contracts
   prompts:   generate-mastering-strategy, master_once, master_closed_loop_prompt
   tools:     upload_audio_to_session
              analyze_audio
              list_presets
              propose_master_settings
              run_master_job
              job_status
              job_result
              master_audio
              master_closed_loop
              read_artifact
              safe_read_text
              safe_write_text
   DSP engine: tools/auralmind_maestro.py
```

## Requirements

- Python >= 3.12
- numpy, scipy, soundfile
- fastmcp (server runtime)
- Optional: torch + demucs for stem separation

## Install

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv pip install -r requirements.txt
```

## Run the MCP server

```bash
python server.py
```

Or with uv:

```bash
uv run server.py
```

The server writes session files into `MAESTRO_SESSION_DIR` (default: OS temp
`maestro_sessions/`) and returns stable handles instead of filesystem paths.

For request/response examples, see `MCP.md`.

## MCP primitives (tools, resources, prompts)

Resources are read-only data (docs, presets, limits). Use them to load guidance
or static metadata.

```json
{
  "method": "resources/read",
  "params": { "uri": "auralmind://workflow" }
}
```

Prompts are reusable templates. Use them to generate a consistent mastering
strategy from metrics.

```json
{
  "method": "prompts/render",
  "params": {
    "name": "generate-mastering-strategy",
    "arguments": {
      "integrated_lufs": -13.1,
      "crest_db": 9.2,
      "platform": "spotify"
    }
  }
}
```

Tools are executable actions. Use them to upload, analyze, master, and download
artifacts.

```json
{
  "method": "tools/call",
  "params": {
    "name": "run_master_job",
    "arguments": {
      "audio_id": "aud_1234567890ab",
      "preset_name": "hi_fi_streaming",
      "target_lufs": -12.5
    }
  }
}
```

## MCP resources and prompts

### Resource: `config://system-prompt`

Returns the cognitive mastering system prompt from `resources/system_prompt.md`.

### Resource: `config://mcp-docs`

Returns the LLM-facing MCP usage guide from `resources/mcp_docs.md`.

### Resource: `config://server-info`

Returns server limits and supported bit depth as JSON.

### Resource: `auralmind://workflow`

Returns the ordered mastering flow (JSON).

### Resource: `auralmind://metrics`

Returns scoring thresholds (JSON).

### Resource: `auralmind://presets`

Returns detailed preset metadata (JSON).

### Resource: `auralmind://contracts`

Returns tool I/O contracts (JSON schema summary).

### Prompt: `generate-mastering-strategy`

Signature:

```
generate-mastering-strategy(integrated_lufs: float, crest_db: float, platform: str)
```

Returns a prompt that embeds the system prompt plus the provided metrics.

## Tools

### `upload_audio_to_session`

Upload audio to the server and return an `audio_id`.

Parameters:

- `filename` (str): original filename (used for display).
- `payload_b64` (str, preferred): audio file encoded as base64.
- `hex_payload` (str, legacy): audio file encoded as hex.

Notes:

- Max payload is 400 MB after decode.

### `analyze_audio`

Run pre-mastering analysis on the uploaded file.

Parameters:

- `audio_id` (str): from `upload_audio_to_session` or a mastered `art_` handle.

Returns:

- `integrated_lufs`, `true_peak_dbtp`, `crest_db`, `stereo_correlation`
- `duration_s`, `peak_dbfs`, `rms_dbfs`, `centroid_hz`

### `list_presets`

Return all available preset names and key parameters.

### `propose_master_settings`

Validate and clamp settings before job submission.

Parameters:

- `preset_name` (str)
- `target_lufs` (float)
- `warmth` (float, 0.0 to 1.0)
- `transient_boost_db` (float, 0.0 to 4.0)
- `enable_harshness_limiter` (bool)
- `enable_air_motion` (bool)
- `bit_depth` (str, float32/float64)

Returns:

- `settings` (safe, clamped values)

### `run_master_job`

Submit a non-blocking mastering job. Returns a `job_id`.

Parameters:

- `audio_id` (str)
- Same parameter set as `propose_master_settings`

### `job_status`

Poll for job progress.

Returns `job_id`, `status`, `progress`, `elapsed_s`, and optional `error`.

### `job_result`

Fetch output once the job is `done`.

Returns:

- `artifacts` (list of `{artifact_id, filename, media_type, size_bytes, sha256}`)
- `metrics` (final analysis metrics)
- `precision` (format string)

### `master_audio`

Run a synchronous mastering pass (same inputs as `run_master_job`).

### `master_closed_loop`

Run a deterministic 2-pass auto master for a goal/platform.

### `read_artifact`

Read artifact bytes in bounded base64 chunks.

Parameters:

- `artifact_id` (str): from `job_result` or `upload_audio_to_session`.
- `offset` (int, optional): byte offset (default 0).
- `length` (int, optional): bytes to read (default 2 MB, max 2 MB).

Returns:

- `data_b64` plus artifact metadata (`filename`, `media_type`, `size_bytes`, `sha256`)
- `offset`, `length`, `is_last`

### `safe_read_text` / `safe_write_text`

Read/write text files inside the server allowlist (session storage and `data/`).

## DSP pipeline (tools/auralmind_maestro.py)

High-level stages:

- Load audio and resample to 48 kHz
- Feature analysis and preset selection
- Dynamic masking EQ, de-ess, and harmonic glow
- Stereo enhancements (spatial realism, CGMS microshift, microdetail recovery)
- Transient sculpt + movement automation + HookLift
- Loudness governor, true-peak limiter, and export

## Presets

| Preset | Target LUFS | Use Case |
| --- | --- | --- |
| `hi_fi_streaming` | -12.8 | Streaming platforms |
| `competitive_trap` | -11.0 | Competitive loudness |
| `club` | -10.4 | Club playback |
| `club_clean` | -10.4 | Clean club master |
| `radio_loud` | -11.0 | Radio / broadcast |
| `cinematic` | -13.5 | Film / cinematic |

## CLI usage (standalone DSP)

`auralmind_match_maestro_v7_3.py` can be used directly:

```bash
python auralmind_match_maestro_v7_3.py --target input.wav --out mastered.wav --preset hi_fi_streaming --report report.md
```

Common flags:

- `--auto` (auto-select preset and safe targets)
- `--target-lufs` / `--ceiling` (override loudness and ceiling)
- `--no-limiter` / `--no-softclip`
- `--no-stems` / `--stems` and Demucs options
- `--warmth`, `--transient-boost`, `--transient-mix`

## Project structure

```
AuralMind/
  server.py
  auralmind_match_maestro_v7_3.py
  tools/
    auralmind_maestro.py
    test_pipeline.py
  resources/
    system_prompt.md
    mcp_docs.md
  MCP.md
  requirements.txt
  pyproject.toml
  README.md
```

## Verification

Syntax check:

```bash
python -m py_compile server.py tools/auralmind_maestro.py
```

Import and function availability check:

```bash
python - <<'PY'
import tools.auralmind_maestro as m
required = [
    "load_audio",
    "analyze_track_features",
    "auto_select_preset_name",
    "auto_tune_preset",
    "dynamic_masking_eq",
    "microshift_widen_side",
    "get_presets",
    "master",
    "write_audio",
    "tpdf_dither",
]
missing = [name for name in required if not hasattr(m, name)]
if missing:
    raise SystemExit(f"Missing functions: {missing}")
print("Import OK, functions present:", ", ".join(required))
PY
```
