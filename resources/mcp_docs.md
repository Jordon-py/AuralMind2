# AuralMind2 MCP Operator Guide

Last updated: March 12, 2026

This document is written for MCP clients, orchestration agents, and LLMs driving AuralMind2.

## First principles

- Handles are opaque: `aud_*`, `art_*`, `job_*`, and `upl_*` are server-issued IDs.
- Session state is isolated. Do not assume artifacts from one client session exist in another.
- Use `list_session_state` to inspect the current session before assuming a remembered handle still exists.
- Use contracts from `auralmind://contracts` instead of guessing payload shapes.
- Prefer semantic planning first, raw overrides second.
- Treat mastering as a checkpointed loop: discover, diagnose, plan, execute, evaluate, intervene, finalize.

## Workflow Selector

Use this table to choose the smallest workflow branch that fits the current moment.
This is the Tool Decision Matrix for common mastering situations.

| Situation or signal | Primary tools | Why this branch exists | Default next step |
|---|---|---|---|
| New session or unknown server state | `bootstrap`, `get_connect_packet`, `list_session_state`, `auralmind://connect-kit`, `auralmind://contracts` | Discover live capabilities and current session state before building payloads | Diagnose |
| New song plus a vague goal | `register_audio_from_path` or upload flow, `analyze_audio`, `plan_mastering_strategy` | Turn measured audio plus qualitative intent into executable settings | Execute |
| Exact settings already known | `register_audio_from_path` or upload flow, `propose_master_settings`, `run_master_job` or `master_audio` | Validate explicit settings before rendering | Evaluate |
| User wants automation | `register_audio_from_path` or upload flow, `master_closed_loop` | Let the server plan, render, score, and optionally retune | Evaluate or Finalize |
| Unsure between two directions | `semantic_a_b_mastering` | Render two candidate directions instead of guessing | Evaluate |
| The current pass is close but needs final feel tweaks | `start_interactive_mastering`, `commit_interactive_mastering` | Refine a nearly-finished pass with one focused second stage | Evaluate or Finalize |
| Mix balance complaint such as buried vocals or dominant bass | `ai_stem_remix` | Derive justified `stem_gains_db` guidance from stem loudness | Plan or Execute |
| Loudness versus punch tension or crest-factor concern | `analyze_and_optimize_governor` | Tune governor depth and GR ceiling before rerendering | Plan or Execute |
| You already have a current `aud_*` or `art_*` handle | `analyze_audio`, optional `compare_audio_metrics` | Resume the workflow from the current state without re-uploading | Plan, Intervene, or Finalize |

## Default Path

This is the default path for a new song, not the only valid path:

1. Call `bootstrap`.
2. If the session may already be active, call `list_session_state`.
3. Register or upload audio.
4. Call `analyze_audio`.
5. Call `plan_mastering_strategy`.
6. Optionally call `propose_master_settings`.
7. Execute with `run_master_job`, `master_audio`, or `master_closed_loop`.
8. Evaluate with `job_result`, `analyze_audio`, `compare_audio_metrics`, and `read_artifact`.
9. If the pass is close but not done, choose one intervention branch and then re-analyze.

## Canonical call recipes

### Discover the server

```json
{
  "name": "bootstrap",
  "arguments": {}
}
```

### Register audio from `data/` or an allowed import root

```json
{
  "name": "register_audio_from_path",
  "arguments": {
    "path": "song.wav"
  }
}
```

Bare filenames resolve inside `data/`.
Absolute paths are allowed when they stay inside a configured audio source root.
Local desktop defaults include `~/Downloads`, and `config://server-info` publishes the active `register_audio_roots`.

```json
{
  "name": "register_audio_from_path",
  "arguments": {
    "path": "C:/Users/goku/Downloads/song.wav"
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

### Analyze the current source or rendered master

`analyze_audio` accepts any current `aud_*` or analyzable `art_*` handle in the session.

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

`stem_mode` is the bounded stem-processing policy:

- `off`: never run Demucs for this render.
- `auto`: premium default; only run stems when the request or source state justifies it.
- `on`: force stem processing for the render.

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

### Compare two handles before choosing the next move

```json
{
  "name": "compare_audio_metrics",
  "arguments": {
    "audio_id_a": "aud_1234567890ab",
    "audio_id_b": "art_1234567890ab"
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

## Intervention recipes

Use these branches when the current pass needs one targeted decision rather than a full restart.

### Compare two mastering directions

```json
{
  "name": "semantic_a_b_mastering",
  "arguments": {
    "audio_id": "aud_1234567890ab",
    "preset_a": "hi_fi_streaming",
    "preset_b": "cinematic"
  }
}
```

### Start an interactive finishing pass

```json
{
  "name": "start_interactive_mastering",
  "arguments": {
    "audio_id": "aud_1234567890ab",
    "preset_name": "hi_fi_streaming"
  }
}
```

```json
{
  "name": "commit_interactive_mastering",
  "arguments": {
    "session_token": "art_1234567890ab",
    "warmth": 0.35,
    "transient_boost_db": 2.2
  }
}
```

### Recommend governor tuning

```json
{
  "name": "analyze_and_optimize_governor",
  "arguments": {
    "audio_id": "aud_1234567890ab",
    "preset_name": "hi_fi_streaming"
  }
}
```

### Analyze stems for balance guidance

```json
{
  "name": "ai_stem_remix",
  "arguments": {
    "audio_id": "aud_1234567890ab"
  }
}
```

## Re-entry Points

Any current `aud_*` or `art_*` handle can be the start of a new decision cycle. You do not need to restart from upload when the user is already working from a source or rendered artifact.

Use this loop for midstream work:

`analyze -> compare if needed -> intervene -> re-analyze -> commit/finalize`

Common re-entry recipes:

1. Raw source audio:
   `register or upload -> analyze_audio -> plan_mastering_strategy -> execute`
2. Finished master artifact:
   `analyze_audio -> compare_audio_metrics if a baseline exists -> intervene or finalize`
3. Complaint after first render:
   `analyze_audio -> choose one focused intervention -> re-analyze -> finalize or plan again`
4. Comparison request:
   `semantic_a_b_mastering or compare_audio_metrics -> evaluate -> finalize or intervene`

## Control profile guidance

The bounded `control_profile` is the preferred deep-control surface for LLMs.

- `spatial_width`
  Negative tightens the image, positive widens it.
- `brightness_tilt`
  Negative darkens or smooths, positive brightens or opens.
- `harshness_control`
  Negative relaxes fatigue protection, positive increases it.
- `movement_amount`
  Negative restrains motion, positive adds lift and animation.
- `low_end_focus`
  Negative lightens the low end, positive tightens and emphasizes it.

Use `auralmind://control-surface` for the exact range and precedence rules.

## Branch guidance for advanced tools

- `semantic_a_b_mastering`
  Use when the best preset or sonic direction is uncertain.
- `start_interactive_mastering` plus `commit_interactive_mastering`
  Use when a pass is already close and only needs final warmth or transient adjustments.
- `analyze_and_optimize_governor`
  Use when loudness, crest factor, and punch are in tension.
- `ai_stem_remix`
  Use when the issue sounds like a mix-balance problem rather than a pure mastering problem.
- `apply_musical_eq`, `apply_tempo_dynamics`, `apply_harmonic_excitation`
  Use for bounded tone, groove, or color interventions on the current artifact, then re-analyze.

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
- Prefer the filename returned by `list_data_audio` when the source is already in `data/`; otherwise use an absolute path only when it is inside `register_audio_roots`.
- Use `plan_mastering_strategy` unless the user already gave exact mastering parameters.
- Use `propose_master_settings` before `run_master_job` when editing settings programmatically.
- Prefer `control_profile` over large unbounded raw-DSP surfacing.
- Re-analyze after every intervention before stacking another one.
- Use only one targeted intervention branch at a time.
- Read `auralmind://contracts` and `auralmind://control-surface` instead of inferring hidden rules.
