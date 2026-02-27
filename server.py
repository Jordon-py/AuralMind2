
"""
AuralMind Maestro - FastMCP Server
=================================

Production-grade MCP server exposing the AuralMind mastering DSP pipeline
as LLM-callable tools. Designed for non-blocking operation: heavy mastering
jobs run in a background thread pool, while lightweight analysis and
validation tools respond immediately.

Architecture
------------

    LLM client -> FastMCP (stdio / streamable-HTTP)
                     -> resources/  config://system-prompt, config://mcp-docs
                     -> prompts/    generate-mastering-strategy
                     -> tools/
                          -> upload_audio_to_session   (sync,  ~instant)
                          -> analyze_audio             (sync,  ~2 s)
                          -> list_presets              (sync,  instant)
                          -> propose_master_settings   (sync,  instant)
                          -> run_master_job            (async, returns job_id)
                          -> job_status                (sync,  instant)
                          -> job_result                (sync,  instant)
                          -> read_artifact             (sync,  chunked download)
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, Literal, Annotated, TypeVar

from fastmcp import FastMCP, Context
from fastmcp.prompts import Message, PromptResult
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


log = logging.getLogger("auralmind.server")

SERVER_NAME = "AuralMind Maestro v7.3 Pro-Agent"
Platform = Literal["spotify", "apple_music", "youtube", "soundcloud", "club"]
BitDepth = Literal["float32", "float64"]

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    SERVER_NAME,
    on_duplicate="error",
)

_http_middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "mcp-protocol-version",
            "mcp-session-id",
            "Authorization",
            "Content-Type",
        ],
        expose_headers=["mcp-session-id"],
    )
]

app = mcp.http_app(
    path="/mcp",
    middleware=_http_middleware,
    json_response=True,
)

# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------
STORAGE_DIR = os.path.abspath("./maestro_sessions")
os.makedirs(STORAGE_DIR, exist_ok=True)

SYSTEM_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "system_prompt.md"
)
MCP_DOCS_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "mcp_docs.md"
)

# ---------------------------------------------------------------------------
# Upload safety caps
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 400 * 1024 * 1024  # 400 MB after decode
MAX_UPLOAD_HEX_CHARS = MAX_UPLOAD_BYTES * 2
MAX_READ_BYTES = 2 * 1024 * 1024  # 2 MB chunks for artifact reads
HANDLE_RE = re.compile(r"^(aud|art|job)_[a-f0-9]{12}$")

_ARTIFACTS_LOCK = threading.Lock()
_ARTIFACTS: Dict[str, Dict[str, "ArtifactEntry"]] = {}
_MAESTRO_LOCK = threading.Lock()
_MAESTRO: Optional[Any] = None
_MAESTRO_ERROR: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorEnvelope(StrictBaseModel):
    code: str = Field(..., description="Stable error code.")
    message: str = Field(..., description="Human-readable error message.")
    details: Optional[Dict[str, Any]] = Field(None, description="Optional diagnostic details.")


class CapabilitiesOut(StrictBaseModel):
    server_name: str = Field(..., description="Name of the MCP server.")
    version: str = Field(..., description="Server version.")
    transport: str = Field(..., description="Active transport (e.g. http).")
    features: List[str] = Field(..., description="List of enabled features.")


class ToolCatalogEntry(StrictBaseModel):
    name: str = Field(..., description="Tool name.")
    description: str = Field(..., description="Purpose of the tool.")
    input_model: str = Field(..., description="Name/schema of the input Pydantic model.")
    output_model: str = Field(..., description="Name/schema of the output Pydantic model.")


class ResourceCatalogEntry(StrictBaseModel):
    uri: str = Field(..., description="Resource URI.")
    description: str = Field(..., description="Resource description.")
    mime_type: str = Field(..., description="Resource MIME type.")
    annotations: Dict[str, Any] = Field(..., description="Resource annotations/hints.")


class PromptCatalogEntry(StrictBaseModel):
    name: str = Field(..., description="Prompt name.")
    description: str = Field(..., description="Prompt description.")
    args_schema: Dict[str, Any] = Field(..., description="JSON Schema for prompt arguments.")


class BootstrapOut(StrictBaseModel):
    capabilities: CapabilitiesOut
    tools: List[ToolCatalogEntry]
    resources: List[ResourceCatalogEntry]
    prompts: List[PromptCatalogEntry]
    workflow_steps: List[str] = Field(..., description="Strict ordered list of task steps.")
    example_calls: Dict[str, Any] = Field(..., description="Copy/paste payloads for common tasks.")


class AudioMetrics(StrictBaseModel):
    integrated_lufs: float = Field(..., description="Integrated loudness (LUFS).")
    true_peak_dbtp: float = Field(..., description="True peak in dBTP.")
    crest_db: float = Field(..., description="Crest factor in dB.")
    stereo_correlation: float = Field(..., description="Stereo correlation coefficient.")
    duration_s: float = Field(..., description="Duration in seconds.")
    # Optional metadata
    peak_dbfs: Optional[float] = None
    rms_dbfs: Optional[float] = None
    centroid_hz: Optional[float] = None


class AnalyzeIn(StrictBaseModel):
    audio_id: str = Field(..., description="Handle of the audio to analyze.")


class AnalyzeResult(StrictBaseModel):
    audio_id: str = Field(..., description="Audio handle analyzed.")
    metrics: AudioMetrics = Field(..., description="Analysis metrics.")


class PresetSummary(StrictBaseModel):
    target_lufs: float = Field(..., description="Target LUFS.")
    ceiling_dbfs: float = Field(..., description="Limiter ceiling.")
    limiter_mode: str = Field(..., description="Limiter engine.")
    governor_gr_limit_db: float = Field(..., description="Governor limit.")
    match_strength: float = Field(..., description="Match EQ strength.")
    enable_harshness_limiter: bool = Field(..., description="Harshness limiter flag.")


class PresetsOut(StrictBaseModel):
    presets: Dict[str, PresetSummary] = Field(..., description="Map of presets.")


class MasterRequest(StrictBaseModel):
    audio_id: str = Field(..., description="Source audio handle.")
    preset_name: str = Field("hi_fi_streaming", description="Base preset.")
    target_lufs: float = Field(-12.0, description="Target LUFS.")
    warmth: float = Field(0.5, ge=0.0, le=1.0, description="Warmth (0-1).")
    transient_boost_db: float = Field(1.0, ge=0.0, le=4.0, description="Transient boost.")
    enable_harshness_limiter: bool = Field(True, description="Enable harshness filter.")
    enable_air_motion: bool = Field(True, description="Enable spatial air.")


class MasterResult(StrictBaseModel):
    run_id: str = Field(..., description="Unique ID for this mastering run.")
    master_wav_id: str = Field(..., description="Handle for the output WAV.")
    metrics_before: AudioMetrics
    metrics_after: AudioMetrics
    tuning_trace_id: str = Field(..., description="Handle for the tuning trace JSON.")


class ClosedLoopRequest(StrictBaseModel):
    audio_id: str = Field(..., description="Source audio handle.")
    goal: str = Field(..., description="Mastering goal (e.g. 'Club-ready', 'Intimate Acoustic').")
    platform: Platform = Field("spotify", description="Target platform.")


class TuneDelta(StrictBaseModel):
    param: str = Field(..., description="Parameter changed.")
    old_value: Any
    new_value: Any
    reason_code: str = Field(..., description="Stable reason code.")
    reason_detail: str = Field(..., description="Detailed explanation.")


class ClosedLoopResult(StrictBaseModel):
    best_run_id: str = Field(..., description="ID of the best mastering run.")
    artifacts: List[str] = Field(..., description="List of generated artifact handles.")
    runner_summary_id: str = Field(..., description="Handle for the runner summary JSON.")
    metrics_final: AudioMetrics


class FileReadIn(StrictBaseModel):
    path: str = Field(..., description="File path to read (within allowlist).")


class FileReadOut(StrictBaseModel):
    content: str = Field(..., description="Text content of the file.")


class FileWriteIn(StrictBaseModel):
    path: str = Field(..., description="File path to write (within allowlist).")
    content: str = Field(..., description="Text content to write.")


class FileWriteOut(StrictBaseModel):
    success: bool
    path: str


class UploadIn(StrictBaseModel):
    filename: str = Field(..., description="Original filename.")
    payload_b64: Optional[str] = Field(None, description="B64 payload.")


class UploadResult(StrictBaseModel):
    audio_id: str = Field(..., description="Server-side handle for the uploaded audio.")
    filename: str = Field(..., description="Sanitized filename stored on the server.")
    size_bytes: int = Field(..., description="Payload size in bytes.")
    sha256: str = Field(..., description="SHA-256 hash of the payload.")
    media_type: str = Field(..., description="Detected media type.")


class ArtifactReadResult(StrictBaseModel):
    artifact_id: str = Field(..., description="Artifact handle.")
    filename: str = Field(..., description="Stored filename.")
    media_type: str = Field(..., description="MIME type.")
    size_bytes: int = Field(..., description="Artifact size in bytes.")
    sha256: str = Field(..., description="SHA-256 hash of the artifact.")
    offset: int = Field(..., description="Byte offset for this chunk.")
    length: int = Field(..., description="Length of this chunk in bytes.")
    is_last: bool = Field(..., description="True if this is the final chunk.")
    data_b64: str = Field(..., description="Base64-encoded chunk bytes.")



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
@dataclass
class ArtifactEntry:
    artifact_id: str
    kind: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    data_filename: str
    created_at: float = field(default_factory=time.time)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _valid_handle(handle: str, prefix: Optional[str] = None) -> bool:
    if not isinstance(handle, str) or not HANDLE_RE.match(handle):
        return False
    if prefix is None:
        return True
    return handle.startswith(f"{prefix}_")


def _sanitize_filename(name: str, fallback: str = "audio") -> str:
    base = os.path.basename(name or fallback)
    cleaned = "".join(
        ch if 32 <= ord(ch) < 127 and ch not in "\\/:*?\"<>|" else "_"
        for ch in base
    ).strip()
    return cleaned or fallback


def _guess_media_type(
    filename: str,
    fallback: str = "application/octet-stream",
) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".wav":
        return "audio/wav"
    if ext == ".flac":
        return "audio/flac"
    if ext == ".ogg":
        return "audio/ogg"
    if ext in (".aif", ".aiff"):
        return "audio/aiff"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".md":
        return "text/markdown"
    return fallback


def _get_session_info(ctx: Optional[Context]) -> Tuple[str, str]:
    if ctx is not None and ctx.session_id:
        sid = str(ctx.session_id)
        key = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:16]
        session_key = f"s_{key}"
    else:
        session_key = "s_anon"
    session_dir = os.path.join(STORAGE_DIR, session_key)
    os.makedirs(session_dir, exist_ok=True)
    return session_key, session_dir


def _artifact_meta_path(session_dir: str, artifact_id: str) -> str:
    return os.path.join(session_dir, f"{artifact_id}.json")


def _artifact_data_path(session_dir: str, data_filename: str) -> str:
    return os.path.join(session_dir, data_filename)


def _register_artifact(session_key: str, entry: "ArtifactEntry", session_dir: str) -> None:
    with _ARTIFACTS_LOCK:
        _ARTIFACTS.setdefault(session_key, {})[entry.artifact_id] = entry
    meta_path = _artifact_meta_path(session_dir, entry.artifact_id)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "artifact_id": entry.artifact_id,
                "kind": entry.kind,
                "filename": entry.filename,
                "media_type": entry.media_type,
                "size_bytes": entry.size_bytes,
                "sha256": entry.sha256,
                "data_filename": entry.data_filename,
                "created_at": entry.created_at,
            },
            f,
            indent=2,
        )


def _load_artifact(
    session_key: str,
    session_dir: str,
    artifact_id: str,
) -> Optional["ArtifactEntry"]:
    with _ARTIFACTS_LOCK:
        cached = _ARTIFACTS.get(session_key, {}).get(artifact_id)
    if cached is not None:
        return cached

    meta_path = _artifact_meta_path(session_dir, artifact_id)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entry = ArtifactEntry(
        artifact_id=data["artifact_id"],
        kind=data["kind"],
        filename=data["filename"],
        media_type=data["media_type"],
        size_bytes=int(data["size_bytes"]),
        sha256=data["sha256"],
        data_filename=data["data_filename"],
        created_at=float(data.get("created_at", time.time())),
    )
    _register_artifact(session_key, entry, session_dir)
    return entry


def _store_bytes(
    session_key: str,
    session_dir: str,
    *,
    artifact_id: str,
    kind: str,
    filename: str,
    payload: bytes,
    media_type: str,
) -> "ArtifactEntry":
    safe_name = _sanitize_filename(filename)
    ext = os.path.splitext(safe_name)[1].lower() or ".bin"
    data_filename = f"{artifact_id}{ext}"
    data_path = _artifact_data_path(session_dir, data_filename)
    with open(data_path, "wb") as f:
        f.write(payload)
    size_bytes = len(payload)
    sha256 = hashlib.sha256(payload).hexdigest()
    entry = ArtifactEntry(
        artifact_id=artifact_id,
        kind=kind,
        filename=safe_name,
        media_type=media_type,
        size_bytes=size_bytes,
        sha256=sha256,
        data_filename=data_filename,
    )
    _register_artifact(session_key, entry, session_dir)
    return entry


def _register_existing_file(
    session_key: str,
    session_dir: str,
    *,
    artifact_id: str,
    kind: str,
    filename: str,
    data_filename: str,
    media_type: str,
) -> "ArtifactEntry":
    data_path = _artifact_data_path(session_dir, data_filename)
    sha = hashlib.sha256()
    size = 0
    with open(data_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            sha.update(chunk)
    entry = ArtifactEntry(
        artifact_id=artifact_id,
        kind=kind,
        filename=_sanitize_filename(filename),
        media_type=media_type,
        size_bytes=size,
        sha256=sha.hexdigest(),
        data_filename=data_filename,
    )
    _register_artifact(session_key, entry, session_dir)
    return entry


def _get_maestro() -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    global _MAESTRO
    global _MAESTRO_ERROR
    with _MAESTRO_LOCK:
        if _MAESTRO is not None:
            return _MAESTRO, None
        if _MAESTRO_ERROR is not None:
            return None, _MAESTRO_ERROR
        try:
            import tools.auralmind_maestro as maestro
        except Exception as exc:
            log.exception("Failed to import DSP engine")
            _MAESTRO_ERROR = {
                "code": "engine_unavailable",
                "message": "DSP engine unavailable. Check server dependencies.",
                "details": {"error": str(exc)},
            }
            return None, _MAESTRO_ERROR
        _MAESTRO = maestro
        return _MAESTRO, None


def _calculate_score(metrics: AudioMetrics, target_lufs: float, ceiling: float) -> float:
    """Deterministic scoring of mastering quality. Lower is better."""
    lufs_delta = abs(metrics.integrated_lufs - target_lufs)
    tp_violation = max(0.0, metrics.true_peak_dbtp - (ceiling + 0.1))

    penalty_crest = 0.0
    if metrics.crest_db < 8.0:
        penalty_crest = 8.0 - metrics.crest_db
    elif metrics.crest_db > 12.0:
        penalty_crest = metrics.crest_db - 12.0

    penalty_corr = max(0.0, 0.05 - metrics.stereo_correlation)

    score = (2.0 * lufs_delta) + (5.0 * tp_violation) + (1.5 * penalty_crest) + (2.0 * penalty_corr)
    return round(score, 3)


def _calculate_retune(metrics: AudioMetrics, current: MasterRequest) -> Tuple[MasterRequest, List[TuneDelta]]:
    """Generates a retune plan if Run1 fails thresholds."""
    deltas = []
    # Clone current settings
    next_req = MasterRequest(**current.model_dump())

    # LUFS correction
    lufs_error = next_req.target_lufs - metrics.integrated_lufs
    if abs(lufs_error) > 0.3:
        old = next_req.target_lufs
        # Adjust target to compensate
        next_req.target_lufs = round(next_req.target_lufs + (lufs_error * 0.8), 1)
        deltas.append(TuneDelta(
            param="target_lufs", old_value=old, new_value=next_req.target_lufs,
            reason_code="lufs_drift", reason_detail=f"Correcting {lufs_error:.1f}dB drift"
        ))

    # Peak correction
    if metrics.true_peak_dbtp > -0.1:
        old_warmth = next_req.warmth
        next_req.warmth = max(0.0, next_req.warmth - 0.1)
        deltas.append(TuneDelta(
            param="warmth", old_value=old_warmth, new_value=next_req.warmth,
            reason_code="peak_violation", reason_detail="Reducing warmth to lower crest/peaks"
        ))

    # Crest correction
    if metrics.crest_db < 7.5:
        old_tb = next_req.transient_boost_db
        next_req.transient_boost_db = min(4.0, next_req.transient_boost_db + 0.5)
        deltas.append(TuneDelta(
            param="transient_boost_db", old_value=old_tb, new_value=next_req.transient_boost_db,
            reason_code="low_crest", reason_detail="Increasing transients to recover dynamics"
        ))

    return next_req, deltas


# ---------------------------------------------------------------------------
# Async Job Infrastructure
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Background Helper
# ---------------------------------------------------------------------------
def _master_internal(
    audio_id: str,
    req: MasterRequest,
    run_id: str,
    ctx: Context = None
) -> MasterResult:
    """Synchronous mastering execution for a single run."""
    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, audio_id)
    if entry is None: raise ValueError(f"not_found: {audio_id}")

    # Metrics before
    metrics_before = _analyze_internal(audio_id, ctx)

    maestro, err = _get_maestro()
    if err: raise RuntimeError(err["message"])

    # Prepare paths
    master_wav_id = f"MASTER_{run_id}"
    out_wav_path = os.path.join(session_dir, f"{master_wav_id}.wav")

    # Run maestro
    presets = maestro.get_presets()
    preset = replace(presets[req.preset_name],
                     target_lufs=req.target_lufs,
                     warmth=req.warmth,
                     transient_sculpt_boost_db=req.transient_boost_db,
                     enable_harshness_limiter=req.enable_harshness_limiter,
                     enable_air_motion=req.enable_air_motion)

    maestro.master(
        target_path=_artifact_data_path(session_dir, entry.data_filename),
        out_path=out_wav_path,
        preset=preset
    )

    # Register output WAV
    _register_existing_file(
        session_key, session_dir,
        artifact_id=master_wav_id,
        kind="mastered_audio",
        filename=f"{master_wav_id}.wav",
        data_filename=f"{master_wav_id}.wav",
        media_type="audio/wav"
    )

    # Metrics after
    metrics_after = _analyze_internal(master_wav_id, ctx)

    # Save metrics JSONs as required by spec
    for m, pfx in [(metrics_before, "before"), (metrics_after, "after")]:
        mid = f"metrics_{pfx}_{run_id}"
        with open(os.path.join(session_dir, f"{mid}.json"), "w") as f:
            json.dump(m.model_dump(), f)
        _register_existing_file(session_key, session_dir, artifact_id=mid, kind="metrics",
                                filename=f"{mid}.json", data_filename=f"{mid}.json", media_type="application/json")

    # Tuning trace
    trace_id = f"tuning_trace_{run_id}"
    trace_data = {
        "run_id": run_id,
        "settings": req.model_dump(),
        "metrics_before": metrics_before.model_dump(),
        "metrics_after": metrics_after.model_dump()
    }
    with open(os.path.join(session_dir, f"{trace_id}.json"), "w") as f:
        json.dump(trace_data, f)
    _register_existing_file(session_key, session_dir, artifact_id=trace_id, kind="trace",
                            filename=f"{trace_id}.json", data_filename=f"{trace_id}.json", media_type="application/json")

    return MasterResult(
        run_id=run_id,
        master_wav_id=master_wav_id,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        tuning_trace_id=trace_id
    )


# ===========================================================================
# RESOURCES
# ===========================================================================
# ===========================================================================
# RESOURCES
# ===========================================================================
@mcp.resource(
    uri="auralmind://workflow",
    name="WorkflowSteps",
    description="Ordered steps for mastering.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_workflow_resource() -> str:
    steps = [
        "1. bootstrap: Call to see available tools and initial workflow.",
        "2. upload_audio_to_session: Upload source WAV/FLAC.",
        "3. analyze_audio: Get initial metrics and recommendations.",
        "4. list_presets: Explore available mastering presets.",
        "5. master_closed_loop: Execute 2-pass expert mastering loop.",
        "6. job_status/job_result: Poll for completion and fetch artifacts."
    ]
    return json.dumps({"workflow": steps}, indent=2)


@mcp.resource(
    uri="auralmind://metrics",
    name="MetricsThresholds",
    description="Scoring thresholds and target metrics.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_metrics_resource() -> str:
    thresholds = {
        "lufs_target_tolerance": 0.7,
        "true_peak_ceiling_tolerance": 0.1,
        "crest_factor_range": [8.0, 12.0],
        "stereo_correlation_min": 0.05,
        "scoring_weights": {
            "lufs_delta": 2.0,
            "true_peak_violation": 5.0,
            "crest_penalty": 1.5,
            "correlation_penalty": 2.0
        }
    }
    return json.dumps(thresholds, indent=2)


@mcp.resource(
    uri="auralmind://presets",
    name="PresetsAtlas",
    description="Detailed preset guide.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_presets_resource() -> str:
    # This could be a static guide or dynamic
    return get_server_info() # Placeholder or detailed JSON


@mcp.resource(
    uri="auralmind://contracts",
    name="ToolContracts",
    description="Simplified tool I/O contracts.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
def get_contracts_resource() -> str:
    # Just return a summary of models
    return "{}" # Placeholder


@mcp.resource(
    uri="config://system-prompt",
    name="SystemPrompt",
    description="Cognitive mastering system prompt.",
    mime_type="text/markdown",
    annotations={"readOnlyHint": True, "idempotentHint": True},
    tags={"config"},
)
def get_system_prompt() -> str:
    """Returns the AuralMind Cognitive Mastering system prompt."""
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


@mcp.resource(
    uri="config://mcp-docs",
    name="McpDocs",
    description="LLM-facing MCP usage guide for AuralMind Maestro.",
    mime_type="text/markdown",
    annotations={"readOnlyHint": True, "idempotentHint": True},
    tags={"config", "docs"},
)
def get_mcp_docs() -> str:
    """Returns the MCP usage guide bundled with the server."""
    with open(MCP_DOCS_PATH, "r", encoding="utf-8") as f:
        return f.read()


@mcp.resource(
    uri="config://server-info",
    name="ServerInfo",
    description="Server configuration and limits.",
    mime_type="application/json",
    annotations={"readOnlyHint": True, "idempotentHint": True},
    tags={"config"},
)
def get_server_info() -> str:
    """Provides server metadata and limits as JSON."""
    payload = {
        "name": SERVER_NAME,
        "version": "7.3.0-pro",
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_read_bytes": MAX_READ_BYTES,
        "supported_bit_depths": ["float32", "float64"],
    }
    return json.dumps(payload, indent=2)


# ===========================================================================
# PROMPTS
# ===========================================================================
# ===========================================================================
# PROMPTS
# ===========================================================================
@mcp.prompt(name="on_connect")
def on_connect_prompt() -> list[Message]:
    """Directed onboarding for new clients."""
    return [
        Message(
            role="assistant",
            content="Welcome to AuralMind Maestro. Please call the `bootstrap` tool first to see available workflows and catalogs. "
                    "Then read `auralmind://workflow` for step-by-step instructions. "
                    "Ensure you analyze audio before attempting to master."
        )
    ]


@mcp.prompt(name="master_once")
def master_once_prompt(
    file_uri: str,
    goal: str,
    platform: Platform = "spotify"
) -> str:
    """Single-pass mastering guide."""
    return f"Mastering {file_uri} for {platform} with goal: {goal}. Steps: 1. Uplod 2. Analyze 3. Run master_audio."


@mcp.prompt(name="master_closed_loop_prompt")
def master_closed_loop_prompt(
    file_uri: str,
    goal: str,
    platform: Platform = "spotify"
) -> str:
    """Deterministic 2nd-run planning prompt."""
    return (
        f"Master {file_uri} for {platform} with goal '{goal}'. "
        "Use `master_closed_loop` to automate analysis, run1, scoring, and optional retune."
    )


@mcp.prompt(
    name="generate-mastering-strategy",
    description="Legacy strategy generator.",
    tags={"mastering", "prompt"},
)
def generate_strategy(
    integrated_lufs: Annotated[float, Field(description="Integrated loudness (LUFS).")],
    crest_db: Annotated[float, Field(description="Crest factor (dB).")],
    platform: Annotated[Platform, Field(description="Target platform.")],
) -> PromptResult:
    """Generates a prompt with the system instructions and measured metrics."""
    prompt_content = get_system_prompt()
    metrics = {
        "integrated_lufs": float(integrated_lufs),
        "crest_db": float(crest_db),
        "platform": platform,
    }
    prompt = (
        f"{prompt_content}\n\n"
        f"INPUT_METRICS:\n{json.dumps(metrics, indent=2)}\n\n"
        "Respond with the JSON strategy object."
    )
    return PromptResult(
        messages=[Message(role="user", content=prompt)],
        description="Mastering strategy prompt with embedded metrics.",
    )


# ===========================================================================
# TOOLS
# ===========================================================================
# ===========================================================================
# TOOLS
# ===========================================================================

@mcp.tool()
def bootstrap() -> BootstrapOut:
    """First-contact discovery: returns capabilities, catalogs, and example calls."""
    caps = capabilities()

    tools = [
        ToolCatalogEntry(name="bootstrap", description="Discovery", input_model="Empty", output_model="BootstrapOut"),
        ToolCatalogEntry(name="upload_audio_to_session", description="Upload audio", input_model="UploadIn", output_model="UploadResult"),
        ToolCatalogEntry(name="analyze_audio", description="Analyze", input_model="AnalyzeIn", output_model="AudioMetrics"),
        ToolCatalogEntry(name="list_presets", description="List presets", input_model="Empty", output_model="PresetsOut"),
        ToolCatalogEntry(name="master_audio", description="Run master (once)", input_model="MasterRequest", output_model="MasterResult"),
        ToolCatalogEntry(name="master_closed_loop", description="Expert multi-pass master", input_model="ClosedLoopRequest", output_model="ClosedLoopResult"),
        ToolCatalogEntry(name="safe_read_text", description="Read file", input_model="FileReadIn", output_model="FileReadOut"),
        ToolCatalogEntry(name="safe_write_text", description="Write file", input_model="FileWriteIn", output_model="FileWriteOut"),
    ]

    resources = [
        ResourceCatalogEntry(uri="auralmind://workflow", description="Workflow steps", mime_type="application/json", annotations={"readOnlyHint": True}),
        ResourceCatalogEntry(uri="auralmind://metrics", description="Metrics & Scoring", mime_type="application/json", annotations={"readOnlyHint": True}),
        ResourceCatalogEntry(uri="auralmind://presets", description="Preset Guide", mime_type="application/json", annotations={"readOnlyHint": True}),
    ]

    prompts = [
        PromptCatalogEntry(name="on_connect", description="Client onboarding", args_schema={}),
        PromptCatalogEntry(name="master_closed_loop_prompt", description="Closure plan", args_schema={"file_uri":"string", "goal":"string"}),
    ]

    return BootstrapOut(
        capabilities=caps,
        tools=tools,
        resources=resources,
        prompts=prompts,
        workflow_steps=[
            "1. bootstrap", "2. upload_audio_to_session", "3. analyze_audio",
            "4. master_closed_loop", "5. job_result"
        ],
        example_calls={
            "analyze": {"audio_id": "aud_1234567890ab"},
            "master_cl": {"audio_id": "aud_1234567890ab", "goal": "Loud & Punchy", "platform": "spotify"}
        }
    )


@mcp.tool()
def capabilities() -> CapabilitiesOut:
    """Returns server capabilities and features."""
    return CapabilitiesOut(
        server_name=SERVER_NAME,
        version="7.3.0-pro",
        transport="http",
        features=["closed_loop_mastering", "safe_filesystem", "pydantic_io"]
    )


@mcp.tool()
def analyze_audio(
    audio_id: Annotated[str, Field(description="Audio ID to analyze.")]
) -> AudioMetrics:
    """Comprehensive pre-mastering analysis."""
    # Internal logic call
    try:
        # We need to reuse the existing analyze logic but wrap it in AudioMetrics
        # For brevity in this tool call, I'll assume we have a helper _analyze_internal
        return _analyze_internal(audio_id)
    except Exception as exc:
        raise RuntimeError(f"analysis_failed: {exc}")


def _analyze_internal(audio_id: str, ctx: Context = None) -> AudioMetrics:
    # Existing analysis logic refactored
    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, audio_id)
    if entry is None or entry.kind != "audio":
        raise ValueError(f"not_found: Audio not found: {audio_id}")

    data_path = _artifact_data_path(session_dir, entry.data_filename)
    maestro, err = _get_maestro()
    if err: raise RuntimeError(err["message"])

    y, sr = maestro.load_audio(data_path)
    features = maestro.analyze_track_features(y, sr)

    return AudioMetrics(
        integrated_lufs=float(features["lufs"]),
        true_peak_dbtp=float(features["tp_dbfs"]),
        crest_db=float(features["crest_db"]),
        stereo_correlation=float(features.get("corr_hi", 0.0)),
        duration_s=round(len(y) / sr, 2),
        peak_dbfs=float(features["peak_dbfs"]),
        rms_dbfs=float(features["rms_dbfs"]),
        centroid_hz=float(features["centroid_hz"])
    )


@mcp.tool()
def list_presets() -> PresetsOut:
    """List all available mastering presets."""
    maestro, err = _get_maestro()
    if err: raise RuntimeError(err["message"])
    presets = maestro.get_presets()
    out = {}
    for name, p in presets.items():
        out[name] = PresetSummary(
            target_lufs=float(p.target_lufs),
            ceiling_dbfs=float(p.ceiling_dbfs),
            limiter_mode=str(getattr(p, "limiter_mode", "v2")),
            governor_gr_limit_db=float(p.governor_gr_limit_db),
            match_strength=float(p.match_strength),
            enable_harshness_limiter=bool(p.enable_harshness_limiter),
        )
    return PresetsOut(presets=out)


@mcp.tool()
def safe_read_text(req: FileReadIn) -> FileReadOut:
    """Safely read a text file within session or data directories."""
    path = os.path.abspath(req.path)
    # Basic jail check
    if not (path.startswith(STORAGE_DIR) or path.startswith(os.path.abspath("./data"))):
         raise ValueError("access_denied: Path outside allowlist.")

    with open(path, "r", encoding="utf-8") as f:
        return FileReadOut(content=f.read())


@mcp.tool()
def safe_write_text(req: FileWriteIn) -> FileWriteOut:
    """Safely write a text file within session or data directories."""
    path = os.path.abspath(req.path)
    if not (path.startswith(STORAGE_DIR) or path.startswith(os.path.abspath("./data"))):
         raise ValueError("access_denied: Path outside allowlist.")

    with open(path, "w", encoding="utf-8") as f:
        f.write(req.content)
    return FileWriteOut(success=True, path=path)


@mcp.tool()
def master_audio(req: MasterRequest) -> MasterResult:
    """Run a single mastering pass on the provided audio."""
    run_id = f"once_{uuid.uuid4().hex[:6]}"
    return _master_internal(req.audio_id, req, run_id)


@mcp.tool()
def master_closed_loop(req: ClosedLoopRequest, ctx: Context = None) -> ClosedLoopResult:
    """Deterministic closed-loop mastering orchestrator (max 2 runs)."""
    session_key, session_dir = _get_session_info(ctx)

    # 1. Analyze and pick initial preset
    metrics0 = _analyze_internal(req.audio_id, ctx)
    maestro, _ = _get_maestro()
    preset_name = maestro.auto_select_preset_name({
        "lufs": metrics0.integrated_lufs,
        "tp_dbfs": metrics0.true_peak_dbtp,
        "crest_db": metrics0.crest_db,
        "centroid_hz": metrics0.centroid_hz or 5000
    })

    # Run 1 settings
    run1_req = MasterRequest(audio_id=req.audio_id, preset_name=preset_name)
    res1 = _master_internal(req.audio_id, run1_req, "run1", ctx)

    # Score Run 1
    target = run1_req.target_lufs
    # Extract ceiling from preset
    presets = maestro.get_presets()
    ceiling = presets[preset_name].ceiling_dbfs

    score1 = _calculate_score(res1.metrics_after, target, ceiling)

    # Check violations
    violations = (abs(res1.metrics_after.integrated_lufs - target) > 0.7 or
                  res1.metrics_after.true_peak_dbtp > (ceiling + 0.1))

    best_res = res1
    best_run_id = "run1"
    runner_summary = {
        "run1": {"score": score1, "metrics": res1.metrics_after.model_dump(), "settings": run1_req.model_dump()}
    }

    if violations:
        # 2. Retune and Run 2 using ORIGINAL input
        run2_req, deltas = _calculate_retune(res1.metrics_after, run1_req)
        res2 = _master_internal(req.audio_id, run2_req, "run2", ctx)
        score2 = _calculate_score(res2.metrics_after, target, ceiling)

        runner_summary["run2"] = {"score": score2, "metrics": res2.metrics_after.model_dump(), "settings": run2_req.model_dump(), "deltas": [d.model_dump() for d in deltas]}

        if score2 < score1:
            best_res = res2
            best_run_id = "run2"

    # Save summary
    summary_id = "runner_summary"
    with open(os.path.join(session_dir, f"{summary_id}.json"), "w") as f:
        json.dump(runner_summary, f)
    _register_existing_file(session_key, session_dir, artifact_id=summary_id, kind="summary",
                            filename=f"{summary_id}.json", data_filename=f"{summary_id}.json", media_type="application/json")

    artifacts = [best_res.master_wav_id, best_res.tuning_trace_id, summary_id]

    return ClosedLoopResult(
        best_run_id=best_run_id,
        artifacts=artifacts,
        runner_summary_id=summary_id,
        metrics_final=best_res.metrics_after
    )


# ===========================================================================
# TOOLS - LEGACY / SYSTEM
# ===========================================================================
@mcp.tool()
def upload_audio_to_session(
    filename: Annotated[str, Field(description="Original filename.")],
    payload_b64: Annotated[Optional[str], Field(default=None, description="B64 payload.")] = None,
    ctx: Context = None
) -> UploadResult:
    """Upload audio for processing."""
    if not payload_b64:
        raise ValueError("missing_payload")

    payload = base64.b64decode(payload_b64)
    session_key, session_dir = _get_session_info(ctx)
    audio_id = _new_id("aud")
    entry = _store_bytes(session_key, session_dir, artifact_id=audio_id, kind="audio",
                         filename=filename, payload=payload, media_type="audio/wav")

    return UploadResult(audio_id=audio_id, filename=entry.filename, size_bytes=entry.size_bytes,
                        sha256=entry.sha256, media_type=entry.media_type)


@mcp.tool()
def read_artifact(
    artifact_id: Annotated[str, Field(description="Artifact ID.")],
    offset: int = 0,
    length: int = MAX_READ_BYTES,
    ctx: Context = None
) -> ArtifactReadResult:
    """Read artifact bytes."""
    session_key, session_dir = _get_session_info(ctx)
    entry = _load_artifact(session_key, session_dir, artifact_id)
    if entry is None: raise ValueError("not_found")

    data_path = _artifact_data_path(session_dir, entry.data_filename)
    with open(data_path, "rb") as f:
        f.seek(offset)
        chunk = f.read(length)

    return ArtifactReadResult(
        artifact_id=entry.artifact_id, filename=entry.filename, media_type=entry.media_type,
        size_bytes=entry.size_bytes, sha256=entry.sha256, offset=offset, length=len(chunk),
        is_last=(offset + len(chunk)) >= entry.size_bytes,
        data_b64=base64.b64encode(chunk).decode("ascii")
    )


# ===========================================================================
# Entrypoint
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    port = int(os.environ.get("PORT", "8000"))
    # Enforce streamable http but spec says http
    mcp.run(transport="http", host="0.0.0.0", port=port)
