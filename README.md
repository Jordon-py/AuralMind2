# AuralMind2

Last updated: March 11, 2026

AuralMind2 is a FastMCP mastering server for remote MCP clients and agentic music workflows. It exposes a session-scoped audio pipeline over `streamable-http`, keeps `stdio` available for local desktop clients, and is now structured around one consistent contract for semantic planning, bounded LLM control, mastering jobs, and artifact retrieval.

## What this server exposes

- `35` tools for ingest, analysis, semantic planning, session-state discovery, mastering, artifact access, and advanced AI-assisted workflows
- `10` resources for onboarding, contracts, presets, metrics, control-surface guidance, and maintainer documentation
- `4` prompts for connect-time guidance and legacy strategy generation
- Strict Pydantic v2 contracts for every request/response shape published through `auralmind://contracts`

## Architecture

Canonical runtime files:

- `server.py`
  FastMCP server, contracts, resources, prompts, job queue, session/artifact storage, and MCP entrypoints.
- `tools/auralmind_maestro.py`
  The real DSP/mastering engine used by the server.
- `resources/mcp_docs.md`
  LLM/operator manual for driving the server as an MCP tool.
- `resources/maintainer_guide.md`
  Human maintainer guide for architecture, extension points, and design rules.
- `auralmind_match_maestro_v7_3.py`
  Legacy compatibility wrapper that delegates to `tools/auralmind_maestro.py`. It is not a second maintained engine.

## Session and artifact model

- Every MCP client session gets an isolated storage directory under `MAESTRO_SESSION_DIR` or the system temp directory.
- Uploaded or registered source audio becomes an opaque `aud_*` handle.
- `register_audio_from_path` accepts bare filenames from `data/` and absolute paths inside the audio source allowlist. Local desktop defaults include `~/Downloads`; extend with `AURALMIND_AUDIO_ROOTS`.
- Rendered outputs, metrics JSON, tuning traces, and summaries become `art_*` handles.
- Async mastering work is tracked with `job_*` handles.
- Clients should treat all handles as opaque. Do not derive paths or invent IDs.

## Recommended workflows

### 1. LLM-first semantic mastering

1. Call `bootstrap`.
2. Call `list_session_state` if the session may already have active handles.
3. Call `register_audio_from_path` with a filename in `data/` or an absolute path inside an allowed audio source root, or use the resumable upload flow.
4. Call `analyze_audio`.
5. Call `plan_mastering_strategy` with `goal`, `platform`, optional `control_profile`, and optional `stem_mode`.
6. Optionally call `propose_master_settings` to validate or adjust the returned settings.
7. Call `run_master_job` or `master_audio`.
8. Use `job_status`, `job_result`, and `read_artifact` to retrieve outputs.

### 2. Closed-loop mastering

1. Register or upload source audio.
2. Call `master_closed_loop` with `goal`, `platform`, and optional `control_profile`.
3. Fetch the winning master plus the runner summary artifact.

### 3. Deep-control mastering

Use `control_profile` when the intent is semantic but you still want bounded steerability:

- `spatial_width`
- `brightness_tilt`
- `harshness_control`
- `movement_amount`
- `low_end_focus`

If needed, add the safe overrides:

- `governor_search_steps`
- `governor_gr_limit_db`
- `stem_gains_db`
- `stem_mode`

Precedence is:

`base preset -> semantic planner -> control_profile -> explicit master fields -> safe overrides`

## Tool families

- Discovery: `bootstrap`, `capabilities`, `get_connect_packet`
- Ingest: `list_data_audio`, `register_audio_from_path`, `upload_init`, `upload_chunk`, `upload_finalize`, `upload_audio_to_session`
  `register_audio_from_path` accepts `data/` filenames plus absolute paths inside allowed audio source roots.
- Analysis and planning: `analyze_audio`, `list_presets`, `plan_mastering_strategy`, `propose_master_settings`, `compare_audio_metrics`
- Mastering execution: `run_master_job`, `job_status`, `job_result`, `master_audio`, `master_closed_loop`
- Artifact access: `read_artifact`, `delete_artifact`
- Advanced AI workflows: `start_interactive_mastering`, `commit_interactive_mastering`, `semantic_a_b_mastering`, `analyze_and_optimize_governor`, `ai_stem_remix`
- Creative DSP tools: `apply_musical_eq`, `apply_tempo_dynamics`, `apply_harmonic_excitation`

## Resources

- `auralmind://connect-kit`
- `auralmind://workflow`
- `auralmind://metrics`
- `auralmind://presets`
- `auralmind://control-surface`
- `auralmind://contracts`
- `config://system-prompt`
- `config://mcp-docs`
- `config://maintainer-guide`
- `config://server-info`

## Runtime defaults

- Transport: `streamable-http`
- Host: `0.0.0.0`
- Port: `8080`
- MCP path: `/mcp`
- Health endpoint: `/health`

Environment variables:

- `ACTIVE_TRANSPORT`
- `PORT`
- `MCP_HOST`
- `MCP_PATH`
- `MAESTRO_SESSION_DIR`
- `AURALMIND_AUDIO_ROOTS`
- `MAX_MASTER_JOBS`
- `UPLOAD_CHUNK_MAX_BYTES`

## Install

Using `pip`:

```bash
pip install -r requirements.txt
```

Using `uv`:

```bash
uv sync
```

## Run locally

HTTP mode:

```bash
python3 server.py
```

ASGI mode:

```bash
uvicorn server:app --host 0.0.0.0 --port 8080
```

Local desktop MCP mode:

```bash
export ACTIVE_TRANSPORT=stdio
python3 server.py
```

## Verification

Basic syntax check:

```bash
python3 -m py_compile server.py tools/auralmind_maestro.py
```

HTTP smoke checks:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/
```

Pytest:

```bash
python3 -m pytest -q
```

## Deployment

### Render

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- MCP endpoint: `https://<service>.onrender.com/mcp`
- Health endpoint: `https://<service>.onrender.com/health`

The repo includes a matching `render.yaml`.

### FastMCP / Horizon

- Entrypoint: `server.py:mcp`
- Transport: `streamable-http`
- Host: `0.0.0.0`
- Port: `8080`
- Path: `/mcp`

## Notes

- `server.py` is the authoritative MCP server contract.
- `tools/auralmind_maestro.py` is the authoritative mastering engine.
- The top-level legacy maestro script remains only for compatibility and should not diverge from the tool package.
