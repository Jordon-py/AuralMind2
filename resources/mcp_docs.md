# AuralMind Maestro MCP - LLM Guide

This server runs over `stdio` and exposes mastering workflows through MCP tools.
Use server-issued handles (`aud_*`, `job_*`, `art_*`) for all follow-up calls.

## Recommended Flow

1. `get_connect_packet` (or read `auralmind://connect-kit`)
2. `list_data_audio`
3. `register_audio_from_path`
4. `analyze_audio`
5. Optional: `generate-mastering-strategy`
6. `propose_master_settings`
7. `run_master_job`
8. Poll `job_status`
9. `job_result`
10. `read_artifact`

Optional: `master_closed_loop` for deterministic 2-pass mastering.

## End-to-End Example

```json
{
  "name": "register_audio_from_path",
  "arguments": { "path": "song.wav" }
}
```

```json
{
  "name": "analyze_audio",
  "arguments": { "audio_id": "aud_1234567890ab" }
}
```

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

```json
{
  "name": "job_status",
  "arguments": { "job_id": "job_1234567890ab" }
}
```

```json
{
  "name": "job_result",
  "arguments": { "job_id": "job_1234567890ab" }
}
```

## Resumable Upload Mini-Example

Use this if `data/` has no source file.

```json
{
  "name": "upload_init",
  "arguments": {
    "filename": "song.wav",
    "total_bytes": 12345678,
    "sha256": "<optional-sha256>"
  }
}
```

```json
{
  "name": "upload_chunk",
  "arguments": {
    "upload_id": "upl_1234567890ab",
    "index": 0,
    "chunk_b64": "<base64-chunk>"
  }
}
```

```json
{
  "name": "upload_finalize",
  "arguments": {
    "upload_id": "upl_1234567890ab"
  }
}
```

## Legacy Upload Note

`upload_audio_to_session` remains available for older clients.
For new clients, prefer `upload_init/upload_chunk/upload_finalize`.

## Resources

- `config://system-prompt`: cognitive mastering instructions.
- `config://mcp-docs`: this guide.
- `config://server-info`: limits, transport, and supported bit depth.
- `auralmind://connect-kit`: first-contact packet with examples.
- `auralmind://workflow`: ordered call sequence.
- `auralmind://metrics`: scoring thresholds.
- `auralmind://presets`: preset atlas.
- `auralmind://contracts`: tool I/O schemas.

## Prompt

`generate-mastering-strategy(integrated_lufs, crest_db, platform)`

Map output settings to `propose_master_settings` / `run_master_job` inputs:

- `preset_name`
- `target_lufs`
- `warmth`
- `transient_boost_db`
- `enable_harshness_limiter`
- `enable_air_motion`
- `bit_depth`
