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
LLM client -> FastMCP (stdio)
   resources: config://system-prompt
   prompts:   generate-mastering-strategy
   tools:     upload_audio_to_session
              analyze_audio
              list_presets
              propose_master_settings
              run_master_job
              job_status
              job_result
              read_artifact
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

The server writes session files into `./maestro_sessions/<session_hash>` and returns
stable handles instead of filesystem paths.

## MCP resources and prompts

### Resource: `config://system-prompt`

Returns the cognitive mastering system prompt from `resources/system_prompt.md`.

### Prompt: `generate-mastering-strategy`

Signature:

```
generate-mastering-strategy(lufs: float, crest: float, platform: str)
```

Returns a prompt that embeds the system prompt plus the provided metrics.

## Tools

All tools return a consistent envelope:

```json
{
  "ok": true,
  "result": {},
  "error": null
}
```

On failure, `ok=false` and `error` includes `code`, `message`, and optional `details`.

### `upload_audio_to_session`

Upload audio to the server and return an `audio_id`.

Parameters:

- `filename` (str): original filename (used for display).
- `payload_b64` (str, preferred): audio file encoded as base64.
- `hex_payload` (str, legacy): audio file encoded as hex.

Notes:

- The max payload is 200 MB after decode.

### `analyze_audio`

Run pre-mastering analysis on the uploaded file.

Parameters:

- `audio_id` (str): from `upload_audio_to_session`.

Returns:

- `lufs_i`, `tp_dbfs`, `peak_dbfs`, `rms_dbfs`, `crest_db`
- `corr_broadband`, `corr_low`, `sub_mono_ok`, `centroid_hz`
- `recommended_preset`, `recommended_lufs`
- `sample_rate`, `duration_s`

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
- `bit_depth` (str, float32 or float64)

Returns:

- `status` and `settings` (safe, clamped values)

### `run_master_job`

Submit a non-blocking mastering job. Returns a `job_id`.

Parameters:

- `audio_id` (str)
- Same parameter set as `propose_master_settings`

### `job_status`

Poll for job progress.

Returns `status`, `progress`, and elapsed time.

### `job_result`

Fetch output once the job is `done`.

Returns:

- `artifacts` (list of `{artifact_id, filename, media_type, size_bytes, sha256}`)
- `precision` (format string)
- Final loudness and true-peak metrics

### `read_artifact`

Read artifact bytes in bounded base64 chunks.

Parameters:

- `artifact_id` (str): from `job_result` or `upload_audio_to_session`.
- `offset` (int, optional): byte offset (default 0).
- `length` (int, optional): bytes to read (default 1 MB, max 1 MB).

Returns:

- `data_b64` plus artifact metadata (`filename`, `media_type`, `size_bytes`, `sha256`)
- `offset`, `length`, `is_last`

## DSP pipeline (tools/auralmind_maestro.py)

High-level stages:

- Load audio and resample to 48 kHz
- Feature analysis and preset selection
- Dynamic masking EQ, de-ess, and harshness limiting
- Stereo enhancements (spatial realism, CGMS microshift, air motion 3D)
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

`tools/auralmind_maestro.py` can be used directly:

```bash
python tools/auralmind_maestro.py --target input.wav --out mastered.wav --preset hi_fi_streaming --report report.md
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
  tools/
    auralmind_maestro.py
    test_pipeline.py
  resources/
    system_prompt.md
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
import auralmind_maestro as m
required = [
    "load_audio",
    "analyze_track_features",
    "auto_select_preset_name",
    "auto_tune_preset",
    "dynamic_masking_eq",
    "harshness_limiter",
    "microshift_widen_side",
    "air_motion_3d",
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
