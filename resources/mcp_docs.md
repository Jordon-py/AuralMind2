# AuralMind2 MCP Guide

Last updated: March 10, 2026

AuralMind2 exposes audio-mastering workflows over FastMCP. The deployment default is `streamable-http`, and the primary MCP endpoint is `/mcp`.

## First-contact workflow

1. Call `bootstrap` for a full catalog of tools, resources, prompts, and schemas.
2. Call `get_connect_packet` for a compact onboarding packet with sample calls.
3. Call `list_data_audio` to inspect server-side source files.
4. Call `register_audio_from_path` if the track already exists in `data/`.
5. Use `upload_init`, `upload_chunk`, and `upload_finalize` if the track must be uploaded.
6. Call `analyze_audio`.
7. Call `propose_master_settings` or `analyze_and_optimize_governor`.
8. Call `run_master_job`.
9. Poll `job_status`.
10. Call `job_result`, then `read_artifact`.

## Prompt surface

- `on_connect`: first-contact assistant guidance
- `master_once`: single-pass workflow planning
- `master_closed_loop_prompt`: deterministic multi-pass workflow planning
- `generate-mastering-strategy`: legacy strategy generation prompt

## Resource surface

- `config://system-prompt`
- `config://mcp-docs`
- `config://server-info`
- `auralmind://connect-kit`
- `auralmind://workflow`
- `auralmind://metrics`
- `auralmind://presets`
- `auralmind://contracts`

## Upload guidance

Use server-side registration when possible because it avoids large payload transfers:

```json
{
  "name": "register_audio_from_path",
  "arguments": {
    "path": "song.wav"
  }
}
```

Use resumable upload when the file is not already in `data/`:

```json
{
  "name": "upload_init",
  "arguments": {
    "filename": "song.wav",
    "total_bytes": 12345678,
    "sha256": "optional-lowercase-sha256"
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

`upload_audio_to_session` is still available for legacy clients, but new clients should prefer the resumable upload flow.

## Async mastering example

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
  "arguments": {
    "job_id": "job_1234567890ab"
  }
}
```

```json
{
  "name": "job_result",
  "arguments": {
    "job_id": "job_1234567890ab"
  }
}
```

## Contract guidance

- Treat `aud_*`, `job_*`, `art_*`, and `upl_*` identifiers as opaque server-issued handles.
- Do not invent handles or file paths.
- Use the JSON schemas in `auralmind://contracts` when building an agent or SDK wrapper.
- Pydantic validation is strict for contract models, so extra fields should be avoided.

## HTTP deployment notes

- Root info endpoint: `/`
- Health endpoint: `/health`
- MCP endpoint: `/mcp`

For local validation:

```text
http://127.0.0.1:8080/health
http://127.0.0.1:8080/mcp
```
