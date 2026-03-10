# AuralMind2

Last updated: March 10, 2026

AuralMind2 is a FastMCP audio-mastering server built for remote MCP clients. It now defaults to `streamable-http` on `0.0.0.0:8080`, exports an ASGI `app` for hosts such as Render, and keeps `stdio` available when you want to run it locally inside a desktop MCP client.

## What is in this server

- `33` tools for mastering, upload flows, artifact access, diagnostics, and analysis
- `8` resources for workflows, contracts, presets, metrics, and onboarding
- `4` prompts for first-contact guidance and mastering strategy generation
- Pydantic v2 models for request and response contracts, validation, and JSON schema export

## Repository layout

```text
AuralMind2/
├─ server.py                 # FastMCP server, models, tools, resources, prompts, ASGI app
├─ resources/                # System prompt and MCP usage guide exposed as resources
├─ tools/auralmind_maestro.py# DSP engine used by the MCP server
├─ tests/                    # Discovery and transport smoke tests
├─ data/                     # Server-side audio input directory
├─ render.yaml               # Render deployment definition
├─ fastmcp.json              # FastMCP deployment config for streamable HTTP
└─ requirements.txt          # Runtime dependencies
```

## Runtime defaults

- Transport: `streamable-http`
- Host: `0.0.0.0`
- Port: `8080`
- MCP path: `/mcp`
- Health endpoint: `/health`

Environment variables:

- `ACTIVE_TRANSPORT`: `streamable-http`, `sse`, or `stdio`
- `PORT`: HTTP port for deployment targets
- `MCP_HOST`: bind host for HTTP transports
- `MCP_PATH`: MCP endpoint path, defaults to `/mcp`
- `MAESTRO_SESSION_DIR`: optional override for session storage
- `MAX_MASTER_JOBS`: optional override for async job concurrency

## Install

```bash
pip install -r requirements.txt
```

If you use `uv`:

```bash
uv sync
```

## Run locally

Streamable HTTP, matching deployment:

```bash
python server.py
```

Or with `uvicorn` against the exported ASGI app:

```bash
uvicorn server:app --host 0.0.0.0 --port 8080
```

Local desktop MCP usage over stdio:

```bash
set ACTIVE_TRANSPORT=stdio
python server.py
```

FastMCP CLI:

```bash
fastmcp run server.py:mcp --transport streamable-http --host 0.0.0.0 --port 8080 --path /mcp
```

## Verify the server

Run the test suite:

```bash
python -m pytest -q
```

Smoke-test the HTTP surface:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/
```

The MCP endpoint is exposed at:

```text
http://127.0.0.1:8080/mcp
```

## Recommended client flow

1. Call `bootstrap` to discover the full contract.
2. Call `get_connect_packet` if you want a smaller, first-contact guide.
3. Use `list_data_audio` or `register_audio_from_path` for server-side files.
4. Use `upload_init` -> `upload_chunk` -> `upload_finalize` if the source audio is not in `data/`.
5. Call `analyze_audio`.
6. Call `propose_master_settings` or `analyze_and_optimize_governor`.
7. Call `run_master_job`.
8. Poll with `job_status`.
9. Fetch completed artifacts with `job_result`.
10. Download bytes with `read_artifact`.

## Pydantic model usage

The server uses Pydantic v2 as the contract layer.

- Response types returned by tools are typed `BaseModel` objects.
- Request models such as `MasterRequest`, `UploadIn`, and `ClosedLoopRequest` are used for validation and schema generation.
- `StrictBaseModel` sets `extra="forbid"` so accidental fields are rejected.
- The `auralmind://contracts` resource publishes the current JSON schemas for client builders and prompt agents.

This matters because MCP clients and orchestration agents need stable, machine-readable contracts instead of ad hoc dictionaries.

## Deployment

### Render

Render is the recommended target for this repo because AuralMind2 is a long-lived HTTP MCP server, not a frontend app.

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- Default MCP endpoint after deploy: `https://<service>.onrender.com/mcp`
- Health check endpoint: `https://<service>.onrender.com/health`

The repo already includes a matching [render.yaml](./render.yaml).

### FastMCP / Horizon

For FastMCP-hosted deployments, use:

- Entrypoint: `server.py:mcp`
- Transport: `streamable-http`
- Host: `0.0.0.0`
- Port: `8080`
- Path: `/mcp`

### Vercel

Vercel is not the preferred deployment target for this server. AuralMind2 is a stateful MCP backend with long-lived HTTP behavior and background job execution, which is a poor fit for serverless request lifecycles.

## Cleanup decisions made in this repo

The following stale artifacts were removed because they were not part of the live MCP server path and were causing confusion or broken test collection:

- old stdio launcher scripts
- broken standalone pipeline test files outside `tests/`
- unrelated filesystem MCP config files
- dead one-off Ollama code
- broken Prefect flow scaffolding that no longer matched the server entrypoint

## Useful files for development

- [server.py](./server.py)
- [resources/mcp_docs.md](./resources/mcp_docs.md)
- [render.yaml](./render.yaml)
- [fastmcp.json](./fastmcp.json)
- [tests/test_discovery_smoke.py](./tests/test_discovery_smoke.py)

## Notes

- `tools/auralmind_maestro.py` remains the mastering engine. This cleanup did not replace the DSP core.
- Session artifacts are stored outside the repo by default under the system temp directory unless `MAESTRO_SESSION_DIR` is set.
- Audio source discovery still uses the repo-local `data/` directory.
