# AuralMind Maestro - MCP Mastering Server

AuralMind Maestro is a FastMCP server that exposes an audio mastering pipeline as MCP tools.
It is designed for `transport="stdio"` and returns stable handles (`aud_*`, `job_*`, `art_*`) instead of raw file paths.

## What This Server Does

- Registers audio from the server-side `data/` folder or resumable upload.
- Analyzes loudness/crest/stereo metrics.
- Runs mastering asynchronously in background jobs.
- Returns mastered audio and JSON artifacts via chunked reads.

## Architecture (Stdio)

```text
LLM/Client -> FastMCP (stdio)
            -> resources
            -> prompts
            -> tools
            -> tools/auralmind_maestro.py (DSP engine)
```

## Requirements

- Python 3.12+
- Dependencies in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

Or:

```bash
uv pip install -r requirements.txt
```

## Run

```bash
python3 server.py
```

Or:

```bash
uv run server.py
```

If using FastMCP CLI, use `stdio` explicitly:

```bash
fastmcp run server.py --transport stdio
```

Or use the included `fastmcp.json` (defaults to `stdio`):

```bash
fastmcp run
```

`fastmcp run server.py` without `--transport stdio` defaults to Streamable HTTP, which starts Uvicorn on `127.0.0.1:8000`.

## Happy Path (Recommended)

1. `get_connect_packet` (or read `auralmind://connect-kit`)
2. `list_data_audio`
3. `register_audio_from_path`
4. `analyze_audio`
5. `propose_master_settings`
6. `run_master_job`
7. Poll `job_status`
8. `job_result`
9. `read_artifact`

## Example: Register Then Analyze

Request (`tools/call`):

```json
{
  "name": "register_audio_from_path",
  "arguments": {
    "path": "song.wav"
  }
}
```

Response (shape):

```json
{
  "audio_id": "aud_1234567890ab",
  "format": "wav",
  "size_bytes": 12345678,
  "checksum": "...sha256...",
  "registered_at": "2026-03-01T00:00:00Z"
}
```

Request (`tools/call`):

```json
{
  "name": "analyze_audio",
  "arguments": {
    "audio_id": "aud_1234567890ab"
  }
}
```

Response includes:

- `integrated_lufs`
- `true_peak_dbtp`
- `crest_db`
- `stereo_correlation`
- `duration_s`
- `peak_dbfs`, `rms_dbfs`, `centroid_hz`

## Example: Async Mastering Flow

Start job:

```json
{
  "name": "run_master_job",
  "arguments": {
    "audio_id": "aud_1234567890ab",
    "preset_name": "hi_fi_streaming",
    "target_lufs": -12.0,
    "warmth": 0.5,
    "transient_boost_db": 1.0,
    "enable_harshness_limiter": true,
    "enable_air_motion": true,
    "bit_depth": "float32"
  }
}
```

Poll status:

```json
{
  "name": "job_status",
  "arguments": {
    "job_id": "job_1234567890ab"
  }
}
```

Fetch result when `status == "done"`:

```json
{
  "name": "job_result",
  "arguments": {
    "job_id": "job_1234567890ab"
  }
}
```

Read output bytes in chunks:

```json
{
  "name": "read_artifact",
  "arguments": {
    "artifact_id": "art_1234567890ab",
    "offset": 0,
    "length": 2097152
  }
}
```

## Empty `data/` Flow (Upload Path)

If no song exists in `data/`, use resumable upload:

1. `upload_init`
2. `upload_chunk` (ordered, 0..N)
3. `upload_finalize`
4. Use returned `audio_id` with `analyze_audio` and `run_master_job`

`upload_audio_to_session` is still available for legacy clients, but resumable upload is preferred.

## Resources

- `config://system-prompt`
- `config://mcp-docs`
- `config://server-info`
- `auralmind://connect-kit`
- `auralmind://workflow`
- `auralmind://metrics`
- `auralmind://presets`
- `auralmind://contracts`

## Prompts

- `on_connect`
- `master_once`
- `master_closed_loop_prompt`
- `generate-mastering-strategy`

## Troubleshooting

- `not_found`:
  - Verify handle belongs to current session.
  - Verify file exists in `data/` before calling `register_audio_from_path`.
- `not_ready` from `job_result`:
  - Continue polling `job_status` until `done` or `error`.
- Upload size failures (`payload_too_large`, `chunk_too_large`):
  - Use resumable upload and obey `upload_chunk_max_bytes` from `config://server-info`.
- `unsupported_format`:
  - Use supported extensions (`wav`, `flac`, `ogg`, `aif`, `aiff`, `mp3`).

## Quick Validation

```bash
python3 -m py_compile server.py tools/auralmind_maestro.py
python3 - <<'PY'
import server
import tools.auralmind_maestro
print(server.capabilities().transport)
PY
```
