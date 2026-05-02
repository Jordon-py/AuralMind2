# AuralMind2 Repo Info

Top-level documentation: this file is a repo dossier for fast onboarding, cleanup decisions, and safer future edits. Data shapes include MCP request/response Pydantic models, audio source paths, artifact handles, background job records, manifests, and mastered audio exports. Important functions: `server.py:3284 get_premium_trap_workflow_resource`, `server.py:3424 premium_trap_mastering_session_prompt`, `server.py:3787 run_master_job`, `server.py:3815 job_status`, `server.py:3837 job_result`, `server.py:3987 master_audio`, and `tools/auralmind_maestro.py:2804 master`. Possible bugs: generated masters and source audio can still appear in Git if already tracked; stale job manifests can claim `running` after the real process is gone; `server.py` is large enough that import-order and type drift are easy. Two extensions: split `server.py` contracts/tools/resources into modules, and move large audio artifacts to a dedicated delivery/artifact store.

## Snapshot

- Date: 2026-04-23
- Root: `C:\Users\goku\Documents\AuralMind2`
- Current identity: FastMCP audio mastering server plus repo-local batch runners.
- Active editor focus: `server.py`
- Runtime is live: local FastMCP stdio processes were detected during this pass, so cleanup avoided active runtime code and active batch output folders.
- Alfred MCP note: Christopher requested repo-info via Alfred; no callable Alfred repo-info tool was exposed in this session, so this file is the local repo-info artifact.

## Runtime Map

- `server.py`: main FastMCP composition root. It owns Pydantic contracts, resources, prompts, tool registration, session/artifact state, background job execution, and the HTTP/stdio app entrypoints.
- `tools/auralmind_maestro.py`: canonical DSP/mastering engine. The MCP server calls into this module for real audio shaping and exports.
- `resources/`: operator-facing MCP docs and prompts surfaced through config resources.
- `templates/`, `static/`, `mastering_ui.py`, `mastering_ui_bridge.py`, `run_ui.py`, `start_master_ui.py`: local browser UI path layered on top of the MCP-style mastering flow.
- `tools/run_*.py` and `tools/render_*.py`: reproducible batch runners for specific mastering deliveries.
- `tests/`: smoke and contract tests for discovery, semantic planning, and UI bridge behavior.
- `resources/premium_trap_workflow.md`: read-only MCP playbook for premium trap/rap AI mastering sessions.
- `docs/DATA-ASSET-MANIFEST.md`: checked-in reference manifest for local `data/` audio assets after removing audio blobs from Git tracking.

## Data And Artifact Boundaries

- `data/`: source audio pool and some runtime state. Treat audio files here as user assets, not disposable cache.
- `masters/`, `masters_premium_trap20_*`, `Ignorance Is Bliss/`, `Album_Ignorance_is_bliss/`: generated delivery/output roots. Keep for listening and audit, but do not treat as source code.
- `artifacts/`, `manifests/`, `server.log`, `runtime_validation.json`, `sonic_specs.json`: runtime/generated state.
- `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`: local environment/cache state.

## Current Cleanup Decisions

- Removed generated cache folders outside `.venv`.
- Moved root notebooks to `archive/notebooks_20260423/`.
- Moved older root delivery guides to `archive/legacy_docs_20260423/`.
- Moved duplicate legacy engine `auralmind_match_maestro_v7_3_expert1.py.py` to `archive/legacy_engines_20260423/`.
- Moved stale duplicate `gitignore` file to `archive/stale_config_20260423/`; `.gitignore` is now the canonical ignore file.
- Did not move `server.py`, `tools/auralmind_maestro.py`, UI bridge files, active runners, source audio, or current master output folders.

## Risk Notes

- `git status` already contains many user/runtime changes. Do not assume all diffs are from one task.
- Several audio files under `data/` are tracked. `.gitignore` prevents new accidental additions but does not untrack existing assets.
- If a batch is interrupted, validate with real process checks plus `job_status`/output files before trusting stale manifests.
- Root-level UI and AI helper scripts are still imported by tests or tools, so they were kept in place for this pass.

## Completed Safe Next Actions

- Artifact hygiene pass: source audio remains on disk but is removed from the Git index going forward; `.gitignore` blocks future audio/runtime output additions.
- Fixture allowlist: `tests/fixtures/audio/` is the only place where tiny curated test audio can be committed.
- Tracked-audio decision: `data/` audio should be local/user assets referenced by docs or manifests, not Git blobs; `docs/DATA-ASSET-MANIFEST.md` now records the current local source pool.
- Server modularization first slice: added MCP instructions, `auralmind://premium-trap-workflow`, `premium_trap_mastering_session`, and docs for future extraction order before touching job execution.

## Next Safe Actions

- Add SHA-256 hashes to `docs/DATA-ASSET-MANIFEST.md` if these assets need release-critical provenance.
- Extract Pydantic models and resource/prompt catalog builders from `server.py` only after this prompt/resource surface settles.
- Add a dedicated delivery-finish MCP tool only if exact 24-bit/32-bit release exports must become part of the public MCP contract.
