# AuralMind Maestro MCP - LLM Guide

This server exposes a non-blocking mastering workflow. Register audio from the
server data directory, then reuse returned handles instead of filesystem paths.

## Quick Flow (Recommended)

1) `list_audio_assets`
2) `register_audio_from_path`
3) `analyze_audio`
4) Optional: `generate-mastering-strategy`
5) `propose_master_settings`
6) `run_master_job`
7) Poll `job_status`
8) `job_result`
9) `read_artifact` (audio/JSON download)

Optional: `master_closed_loop` for a 2-pass auto master.
Legacy: `upload_audio_to_session` for client-side uploads.

## Resources

- `config://system-prompt` - cognitive mastering instructions (markdown).
- `config://mcp-docs` - this document (markdown).
- `config://server-info` - server limits and supported bit depth (JSON).
- `auralmind://workflow` - ordered mastering workflow (JSON).
- `auralmind://metrics` - scoring thresholds (JSON).
- `auralmind://presets` - preset atlas (JSON).
- `auralmind://contracts` - tool I/O contracts (JSON).

## Prompt

`generate-mastering-strategy(integrated_lufs, crest_db, platform)`

Parameters:
    - `integrated_lufs` (float): integrated loudness (LUFS)
    - `crest_db` (float): crest factor (dB)
    - `platform` (spotify | apple_music | youtube | soundcloud | club)

Returns a prompt that embeds the system prompt plus the provided metrics.

## Tools

### list_audio_assets

Enumerates audio files in the server data directory (`data/`).

Returns a list of objects:

- `filename`
- `size_bytes`
- `format`
- `duration_seconds` (optional)

### register_audio_from_path

Registers an existing audio file by path without upload.

Parameters:
    - `path` (str): path to a file inside the data directory (absolute or relative).

Validations:
    - Allowlist to `data/` only
    - Reject `..` traversal and unsupported formats
    - Deny symlink escapes outside the data directory

Returns:
    - `audio_id`, `format`, `size_bytes`, `checksum` (sha256), `registered_at`

### upload_audio_to_session

Legacy client-side upload. Prefer `register_audio_from_path`.

Parameters:
    - `filename` (str)
    - `payload_b64` (str, preferred) OR `hex_payload` (str, legacy)

Notes:

- Max payload is 400 MB after decode.

Returns:
    - `audio_id` plus metadata (`filename`, `size_bytes`, `sha256`, `media_type`)

### analyze_audio

Parameters:
    - `audio_id` (aud_... or art_...)

Returns metrics:

- `integrated_lufs`, `true_peak_dbtp`, `crest_db`, `stereo_correlation`, `duration_s`
- `peak_dbfs`, `rms_dbfs`, `centroid_hz`

### list_presets

Returns a `presets` map of preset names to key tuning values:

- `target_lufs`, `ceiling_dbfs`, `limiter_mode`, `governor_gr_limit_db`
- `match_strength`, `enable_harshness_limiter`, `enable_air_motion`, `bit_depth`

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
    - `settings` (safe, clamped values)

### run_master_job

Same parameters as `propose_master_settings`, plus:
    - `audio_id`

Returns:
    - `job_id`, `status`, `audio_id`

### job_status

Parameters:
    - `job_id`

Returns:
    - `job_id`, `status`, `progress`, `elapsed_s`, `error` (optional)

### job_result

Parameters:
    - `job_id`

Returns:
    - `job_id`, `status`
    - `artifacts` (audio + JSON summaries)
    - `metrics` (final loudness and analysis metrics)
    - `precision`

### master_audio

Runs a synchronous, single-pass master (same inputs as `run_master_job`).

Returns:
    - `master_wav_id`, `metrics_before`, `metrics_after`, `tuning_trace_id`, `artifacts`

### master_closed_loop

Runs a 2-pass auto master for a goal/platform. Uses the original input for each pass.

Returns:
    - `best_run_id`, `artifacts`, `runner_summary_id`, `metrics_final`

### read_artifact

Parameters:
    - `artifact_id`
    - `offset` (optional, default 0)
    - `length` (optional, default 2 MB, max 2 MB)

Returns:
    - `data_b64` plus artifact metadata (`filename`, `media_type`, `size_bytes`, `sha256`)
    - `offset`, `length`, `is_last`

### safe_read_text / safe_write_text

Read/write text files within the server allowlist (session storage and `data/`).

## Strategy-to-Settings Mapping

If you call `generate-mastering-strategy`, map its JSON fields to tool inputs:
    - `preset_name` -> `preset_name`
    - `target_lufs` -> `target_lufs`
    - `warmth` -> `warmth`
    - `transient_boost_db` -> `transient_boost_db`
    - `enable_harshness_limiter` -> `enable_harshness_limiter`
    - `enable_air_motion` -> `enable_air_motion`
    - `bit_depth` -> `bit_depth`
