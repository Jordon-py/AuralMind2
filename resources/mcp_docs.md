# AuralMind Maestro MCP - LLM Guide

This server exposes a non-blocking mastering workflow. Use the tools in order and
reuse the returned handles instead of filesystem paths.

## Quick Flow (Recommended)

1) `upload_audio_to_session`
2) `analyze_audio`
3) Optional: `generate-mastering-strategy`
4) `propose_master_settings`
5) `run_master_job`
6) Poll `job_status`
7) `job_result`
8) `read_artifact` (audio/report download)

## Tool Envelope

All tools return:

```json
{ "ok": true, "result": {}, "error": null }
```

On failure:

```json
{ "ok": false, "result": null, "error": { "code": "...", "message": "...", "details": {} } }
```

## Resources

- `config://system-prompt` - cognitive mastering instructions (markdown).
- `config://mcp-docs` - this document (markdown).
- `config://server-info` - server limits and supported bit depth (JSON).

## Prompt

`generate-mastering-strategy(lufs, crest, platform)`

Parameters:
    - `lufs` (float): integrated loudness (LUFS)
    - `crest` (float): crest factor (dB)
    - `platform` (spotify | apple_music | youtube | soundcloud | club)

Returns a prompt that embeds the system prompt plus the provided metrics.

## Tools

### upload_audio_to_session

Parameters:
    - `filename` (str)
    - `payload_b64` (str, preferred) OR `hex_payload` (str, legacy)

Notes:

- Max payload is 400 MB after decode.

Returns:
    - `audio_id` plus metadata (`filename`, `size_bytes`, `sha256`, `media_type`)

### analyze_audio

Parameters:
    - `audio_id` (aud_...)

Returns metrics:

- `lufs_i`, `tp_dbfs`, `peak_dbfs`, `rms_dbfs`, `crest_db`
- `corr_broadband`, `corr_low`, `sub_mono_ok`
- `centroid_hz`, `recommended_preset`, `recommended_lufs`
- `sample_rate`, `duration_s`

### list_presets

Returns a `presets` map of preset names to key tuning values:

- `target_lufs`, `ceiling_dbfs`, `limiter_mode`, `governor_gr_limit_db`
- `match_strength`, `enable_harshness_limiter`

### propose_master_settings

Validates and clamps settings before job submission.

Parameters:
    - `preset_name`
    - `target_lufs`
    - `warmth` (0.0 to 1.0)
    - `transient_boost_db` (0.0 to 4.0)
    - `enable_harshness_limiter`
    - `enable_air_motion`
    - `bit_depth` (`float32` | `float64`)

Returns:
    - `settings` (safe, clamped values including `subtype`)

### run_master_job

Same parameters as `propose_master_settings`, plus:
    - `audio_id`

Returns:
    - `job_id`, `status`, `audio_id`

### job_status

Parameters:
    - `job_id`

Returns:
    - `status`, `progress`, `elapsed_s`

### job_result

Parameters:
    - `job_id`

Returns:
    - `artifacts` (audio + report)
    - `metrics` (final loudness + governor stats)
    - `precision`

### read_artifact

Parameters:
    - `artifact_id`
    - `offset` (optional, default 0)
    - `length` (optional, default 2 MB, max 2 MB)

Returns:
    - `data_b64` plus artifact metadata (`filename`, `media_type`, `size_bytes`, `sha256`)
    - `offset`, `length`, `is_last`

## Strategy-to-Settings Mapping

If you call `generate-mastering-strategy`, map its JSON fields to tool inputs:
    - `preset_name` -> `preset_name`
    - `target_lufs` -> `target_lufs`
    - `warmth` -> `warmth`
    - `transient_boost_db` -> `transient_boost_db`
    - `enable_harshness_limiter` -> `enable_harshness_limiter`
    - `enable_air_motion` -> `enable_air_motion`
    - `bit_depth` -> `bit_depth`
