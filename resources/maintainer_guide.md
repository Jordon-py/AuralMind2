# AuralMind2 Maintainer Guide

Last updated: March 11, 2026

## Canonical source of truth

- `server.py` is the authoritative MCP server contract.
- `tools/auralmind_maestro.py` is the authoritative DSP/mastering engine.
- `auralmind_match_maestro_v7_3.py` is a compatibility wrapper only.

If behavior changes, update `server.py`, the bundled docs resources, and the tests together.

## Core architecture

### Session model

- Session identity comes from `Context.session_id`.
- Disk storage lives under `MAESTRO_SESSION_DIR` or a temp default.
- Artifact metadata is cached in memory and persisted as JSON so handles survive process restarts.

### Artifact model

- `audio`
  Uploaded or registered source audio.
- `mastered_audio`
  Rendered audio artifacts that can be re-analyzed or used in downstream tools.
- `metrics`
  JSON metric snapshots.
- `trace`
  Per-run tuning trace JSON.
- `summary`
  Closed-loop comparison summaries.

### Job model

- Async mastering uses `ThreadPoolExecutor`.
- `JobState.settings` stores the normalized `MasterSettings`.
- Workers reconstruct a `MasterRequest` from stored settings to render deterministically.

## How semantic planning works

`plan_mastering_strategy` is the contract boundary between freeform language and executable mastering.

Resolution order:

1. base preset
2. semantic planner
3. `control_profile`
4. explicit master fields
5. safe overrides

Key helpers:

- `_plan_mastering_strategy_internal`
- `_build_semantic_control_profile`
- `_build_semantic_overrides`
- `_finalize_master_settings`
- `_preset_overrides_from_settings`

Rule of thumb:

- Add new high-level LLM controls to `MasteringControlProfile` first.
- Only expose a raw override publicly if it is bounded, stable, and defensible for agent use.

## Extending tools safely

- Prefer `_get_session_info`, `_load_artifact`, `_register_existing_file`, and `_artifact_data_path` over new storage shortcuts.
- Prefer `_get_maestro()` over importing the DSP engine directly from arbitrary tool code.
- New audio-processing tools should read an artifact into the current session and write a new `mastered_audio` artifact back into the same session.
- If a tool returns an audio artifact ID, it should be analyzable by `analyze_audio`.

## Contract hygiene

- Keep tool signatures aligned with the contracts published in `auralmind://contracts`.
- If a tool advertises a Pydantic model, implement the tool with that model instead of a flat signature.
- Update these surfaces together when adding or changing behavior:
  - tool function
  - resource catalog
  - contract resource
  - README / MCP docs
  - tests

## Testing strategy

- Prefer mocking `_get_maestro()` in server tests so orchestration can be verified without Demucs, torch, or full DSP execution.
- Keep discovery/contract tests focused on `bootstrap`, resources, and public schemas.
- Use temp directories for artifact and session tests.
- Run `python3 -m py_compile server.py tools/auralmind_maestro.py` as a cheap syntax gate before heavier tests.

## Common pitfalls

- Do not introduce a second artifact storage model.
- Do not bypass `model_fields_set` semantics when resolving explicit settings from requests.
- Do not assume Demucs is installed for default tests or lightweight environments.
- Do not add public raw-DSP knobs casually; it makes the MCP surface harder for LLMs to use correctly.

## Documentation expectations

When behavior changes:

- `README.md` explains the repo and runtime model for humans.
- `resources/mcp_docs.md` explains exact usage for LLM clients.
- `resources/system_prompt.md` should describe the live operating flow, not an outdated one.
