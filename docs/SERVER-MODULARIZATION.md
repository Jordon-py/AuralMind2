# Server Modularization

Top-level documentation: this file defines the first safe slices for breaking down `server.py` without changing mastering behavior. Data shapes include Pydantic request/response models, catalog entries, resources, prompts, job state, artifact state, and Maestro preset dataclasses. Important functions: `server.py:3284 get_premium_trap_workflow_resource`, `server.py:3424 premium_trap_mastering_session_prompt`, `server.py:3787 run_master_job`, and `server.py:3023 _run_master_job_worker`. Possible bugs: moving job execution too early can break resumability; moving contracts without tests can desync `bootstrap` from `auralmind://contracts`. Two extensions: extract all Pydantic schemas into `server_contracts.py`, and extract resource/prompt catalog construction into `mcp_guidance.py`.

## Completed First Slice

- Added one focused guidance resource: `auralmind://premium-trap-workflow`.
- Added one focused prompt: `premium_trap_mastering_session`.
- Added server-level FastMCP instructions for clients that honor MCP server instructions.
- Added discovery tests for the new instruction/resource/prompt surface.
- Left job execution, artifact storage, and Maestro engine integration untouched.

## Next Extraction Order

1. Pydantic request/response models.
2. Catalog builders and resource/prompt declarations.
3. Session and artifact storage helpers.
4. Job lifecycle and executor state.
5. Mastering engine orchestration.

## Guardrails

- Keep `bootstrap().tools` and `auralmind://contracts` synchronized.
- Keep resources and prompts read-only unless a tool explicitly mutates state.
- Preserve `server.py` as the composition root until tests cover extracted modules directly.
- Do not introduce a second artifact or job storage model.
