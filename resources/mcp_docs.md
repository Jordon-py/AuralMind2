# AuralMind2 MCP Operator Guide

Last updated: March 11, 2026

This document is written for MCP clients, orchestration agents, and LLMs driving AuralMind2.

## First principles

- Handles are opaque: `aud_*`, `art_*`, `job_*`, and `upl_*` are server-issued IDs.
- Session state is isolated. Do not assume artifacts from one client session exist in another.
- Use contracts from `auralmind://contracts` instead of guessing payload shapes.
- Prefer semantic planning first, raw overrides second.

## Fast start

1. Call `bootstrap`.
2. Read `auralmind://control-surface`.
3. Register or upload audio.
4. Call `analyze_audio`.
5. Call `plan_mastering_strategy`.
6. Execute with `run_master_job`, `master_audio`, or `master_closed_loop`.
7. Retrieve outputs with `job_result` and `read_artifact`.

## Canonical call recipes

### Discover the server

```json
{
  "name": "bootstrap",
  "arguments": {}
}
```

### Register audio from `data/`

```json
{
  "name": "register_audio_from_path",
  "arguments": {
    "path": "song.wav"
  }
}
```

### Resumable upload

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

### Analyze source audio

```json
{
  "name": "analyze_audio",
  "arguments": {
    "audio_id": "aud_1234567890ab"
  }
}
```

### Plan from natural language

```json
{
  "name": "plan_mastering_strategy",
  "arguments": {
    "audio_id": "aud_1234567890ab",
    "goal": "Wide, punchy streaming master with tight low end and smooth highs",
    "platform": "spotify",
    "control_profile": {
      "spatial_width": 0.45,
      "harshness_control": 0.35,
      "low_end_focus": 0.55
    }
  }
}
```

`plan_mastering_strategy` returns:

- source metrics
- chosen preset
- resolved `MasterSettings`
- reasoning
- warnings

### Validate or modify explicit settings

```json
{
  "name": "propose_master_settings",
  "arguments": {
    "preset_name": "competitive_trap",
    "target_lufs": -11.2,
    "warmth": 0.25,
    "transient_boost_db": 2.3,
    "enable_harshness_limiter": true,
    "enable_air_motion": true,
    "bit_depth": "float32",
    "control_profile": {
      "movement_amount": 0.45,
      "low_end_focus": 0.5
    },
    "governor_search_steps": 5,
    "governor_gr_limit_db": -2.0
  }
}
```

### Queue an async master

```json
{
  "name": "run_master_job",
  "arguments": {
    "audio_id": "aud_1234567890ab",
    "preset_name": "competitive_trap",
    "target_lufs": -11.2,
    "warmth": 0.25,
    "transient_boost_db": 2.3,
    "enable_harshness_limiter": true,
    "enable_air_motion": true,
    "bit_depth": "float32",
    "control_profile": {
      "spatial_width": 0.25,
      "movement_amount": 0.45,
      "low_end_focus": 0.5
    },
    "governor_search_steps": 5,
    "governor_gr_limit_db": -2.0,
    "stem_gains_db": {
      "vocals": 1.0,
      "bass": -0.5
    }
  }
}
```

### Poll and fetch result

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

### Read artifact bytes

```json
{
  "name": "read_artifact",
  "arguments": {
    "artifact_id": "art_1234567890ab",
    "offset": 0,
    "length": 1048576
  }
}
```

## Control profile guidance

The bounded `control_profile` is the preferred deep-control surface for LLMs.

- `spatial_width`
  Negative tightens the image, positive widens it.
- `brightness_tilt`
  Negative darkens/smooths, positive brightens/opens.
- `harshness_control`
  Negative relaxes fatigue protection, positive increases it.
- `movement_amount`
  Negative restrains motion, positive adds lift and animation.
- `low_end_focus`
  Negative lightens the low end, positive tightens and emphasizes it.

Use `auralmind://control-surface` for the exact range and precedence rules.

## Closed-loop mastering

Use `master_closed_loop` when the goal is semantic and you want the server to plan, render, score, and optionally retune automatically.

```json
{
  "name": "master_closed_loop",
  "arguments": {
    "audio_id": "aud_1234567890ab",
    "goal": "Aggressive but smooth trap master with width and vocal clarity",
    "platform": "spotify",
    "control_profile": {
      "spatial_width": 0.35,
      "harshness_control": 0.4,
      "movement_amount": 0.25
    }
  }
}
```

## Advanced AI tools

- `start_interactive_mastering`
  Render a first pass, inspect returned metrics, then finalize with `commit_interactive_mastering`.
- `semantic_a_b_mastering`
  Compare two presets in parallel.
- `analyze_and_optimize_governor`
  Recommend governor search depth and GR ceiling from crest factor.
- `ai_stem_remix`
  Use Demucs to inspect stem loudness relationships and suggest stem gain overrides.

## Troubleshooting

- `not_found`
  The referenced handle is unknown in the current session or the artifact type is wrong.
- `not_ready`
  The job is still running. Poll again.
- `demucs_unavailable`
  Stem analysis was requested but the environment does not have Demucs available.
- `unknown_preset`
  The preset name does not exist in the DSP engine.
- `invalid_upload_id`, `upload_incomplete`, `sha256_mismatch`
  Upload state is invalid; restart the upload flow.

## Strong recommendations for LLM clients

- Call `bootstrap` at the beginning of a new integration or test session.
- Use `plan_mastering_strategy` unless the user already gave exact mastering parameters.
- Use `propose_master_settings` before `run_master_job` when editing settings programmatically.
- Prefer `control_profile` over large unbounded raw-DSP surfacing.
- Read `auralmind://contracts` and `auralmind://control-surface` instead of inferring hidden rules.
